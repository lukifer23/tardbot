#!/usr/bin/env python3
"""Download and normalize processed datasets for SFT, tool-calling, and reasoning."""

import argparse
import ast
import json
import re
import sys
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Callable, Any

from datasets import load_dataset, get_dataset_config_names

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.data_config import DataConfig
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def _parse_jsonish(payload: str) -> Optional[Any]:
    payload = payload.strip()
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(payload)
        except Exception:
            return None


def _write_jsonl(records: Iterable[Dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if not record:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _dataset_iterator(spec: Dict[str, Any], limit: Optional[int]):
    split = spec.get("split", "train")
    streaming = spec.get("streaming", False)
    dataset = load_dataset(
        spec["path"],
        spec.get("name"),
        split=split,
        streaming=streaming,
    )
    iterator: Iterable[Any]
    iterator = dataset if streaming else iter(dataset)
    if limit and limit > 0:
        iterator = islice(iterator, limit)
    for example in iterator:
        yield example


def _format_alpaca(example: Dict[str, Any]) -> Dict[str, Any]:
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    if input_text:
        instruction = f"{instruction}\n\nInput: {input_text}".strip()
    output = (example.get("output") or "").strip()
    if not instruction or not output:
        return {}
    return {"instruction": instruction, "output": output}


def _normalize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    normalized = []
    for msg in messages:
        role = msg.get("role") or msg.get("from") or "user"
        role = role.lower()
        if role in ("human", "user"):
            role = "user"
        elif role in ("assistant", "gpt", "bot"):
            role = "assistant"
        content = (msg.get("content") or msg.get("value") or msg.get("markdown") or "").strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _format_sharegpt(example: Dict[str, Any]) -> Dict[str, Any]:
    messages = _normalize_messages(example.get("conversations") or [])
    if len(messages) < 2:
        return {}
    return {"messages": messages}


def _format_ultrachat(example: Dict[str, Any]) -> Dict[str, Any]:
    messages = _normalize_messages(example.get("messages") or [])
    if len(messages) < 2:
        return {}
    return {"messages": messages}


SECTION_PATTERN = re.compile(r"^(USER|ASSISTANT|FUNCTION RESPONSE):", re.MULTILINE)


def _split_sections(text: str) -> Iterator[tuple[str, str]]:
    matches = list(SECTION_PATTERN.finditer(text))
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        yield match.group(1), text[start:end].strip()


def _format_glaive_function_calling(example: Dict[str, Any]) -> Dict[str, Any]:
    chat = (example.get("chat") or "").replace("\r\n", "\n").strip()
    if not chat:
        return {}

    user_context: List[str] = []
    assistant_messages: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []

    pending_call_id: Optional[str] = None
    call_index = 0

    for label, section in _split_sections(chat):
        cleaned = section.replace("<|endoftext|>", "").strip()
        if label == "USER":
            if cleaned:
                user_context.append(cleaned)
        elif label == "ASSISTANT":
            if cleaned.startswith("<functioncall>"):
                payload = cleaned[len("<functioncall>"):].strip()
                call = _parse_jsonish(payload)
                if not isinstance(call, dict):
                    continue
                call_id = f"call_{call_index}"
                call_index += 1
                call["id"] = call_id
                arguments = call.get("arguments")
                if isinstance(arguments, str):
                    parsed_args = _parse_jsonish(arguments)
                    if parsed_args is not None:
                        call["arguments"] = parsed_args
                tool_calls.append(call)
                pending_call_id = call_id
            elif cleaned:
                assistant_messages.append(cleaned)
        elif label == "FUNCTION RESPONSE":
            result = _parse_jsonish(cleaned)
            payload = result if result is not None else {"content": cleaned}
            tool_results.append(
                {
                    "tool_call_id": pending_call_id,
                    "content": payload,
                }
            )
            pending_call_id = None

    if not tool_calls or not tool_results or not assistant_messages or not user_context:
        return {}

    return {
        "user_message": "\n\n".join(user_context).strip(),
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "assistant_response": assistant_messages[-1].strip(),
    }


def _split_reasoning_answer(answer: str) -> tuple[str, str]:
    answer = (answer or "").strip()
    if "####" in answer:
        reasoning, final = answer.split("####", 1)
        return reasoning.strip(), final.strip()
    return answer, answer


def _format_gsm8k(example: Dict[str, Any]) -> Dict[str, Any]:
    question = (example.get("question") or "").strip()
    reasoning, final = _split_reasoning_answer(example.get("answer") or "")
    if not question or not final:
        return {}
    return {
        "question": question,
        "reasoning": reasoning,
        "answer": final,
    }


def _format_aqua(example: Dict[str, Any]) -> Dict[str, Any]:
    question = (example.get("question") or "").strip()
    rationale = (example.get("rationale") or "").strip()
    options = example.get("options") or []
    correct = (example.get("correct") or "").strip()
    answer_text = ""
    for opt in options:
        if opt.startswith(correct):
            answer_text = opt.split(")", 1)[-1].strip()
            break
    if not answer_text:
        answer_text = correct
    if not question or not rationale:
        return {}
    return {
        "question": question,
        "reasoning": rationale,
        "answer": answer_text,
    }


BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")


def _format_hendrycks_math(example: Dict[str, Any]) -> Dict[str, Any]:
    question = (example.get("problem") or "").strip()
    solution = (example.get("solution") or "").strip()
    if not question or not solution:
        return {}
    match = BOXED_PATTERN.search(solution)
    final_answer = match.group(1).strip() if match else solution.splitlines()[-1].strip()
    return {
        "question": question,
        "reasoning": solution,
        "answer": final_answer,
    }


INSTRUCT_SOURCES: Dict[str, Dict[str, Any]] = {
    "alpaca": {
        "path": "yahma/alpaca-cleaned",
        "split": "train",
        "formatter": _format_alpaca,
    },
    "sharegpt": {
        "path": "zetavg/ShareGPT-Processed",
        "split": "train",
        "streaming": True,
        "formatter": _format_sharegpt,
    },
    "ultrachat": {
        "path": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "formatter": _format_ultrachat,
    },
}

TOOL_SOURCES: Dict[str, Dict[str, Any]] = {
    "tool_calling_examples": {
        "path": "glaiveai/glaive-function-calling-v2",
        "split": "train",
        "formatter": _format_glaive_function_calling,
    },
}


def _all_math_configs() -> List[str]:
    configs = get_dataset_config_names("EleutherAI/hendrycks_math")
    return configs


REASONING_SOURCES: Dict[str, Dict[str, Any]] = {
    "gsm8k": {
        "path": "gsm8k",
        "name": "main",
        "split": "train",
        "formatter": _format_gsm8k,
    },
    "aqua": {
        "path": "aqua_rat",
        "split": "train",
        "formatter": _format_aqua,
    },
    "math": {
        "path": "EleutherAI/hendrycks_math",
        "split": "train",
        "formatter": _format_hendrycks_math,
        "configs": _all_math_configs,
    },
}


def _prepare_instruct(datasets: List[str], output_dir: Path, limit: Optional[int]):
    def generator():
        for name in datasets:
            spec = INSTRUCT_SOURCES.get(name)
            if spec is None:
                logger.warning("Unknown instruction dataset alias '%s'; skipping", name)
                continue
            logger.info("Building instruction samples from %s", spec["path"])
            for example in _dataset_iterator(spec, limit):
                formatter: Callable[[Dict[str, Any]], Dict[str, Any]] = spec["formatter"]
                formatted = formatter(example)
                if formatted:
                    yield formatted

    output_path = output_dir / "instruct_data.jsonl"
    count = _write_jsonl(generator(), output_path)
    logger.info("Saved %d instruction examples to %s", count, output_path)


def _prepare_tool(datasets: List[str], output_dir: Path, limit: Optional[int]):
    def generator():
        for name in datasets:
            spec = TOOL_SOURCES.get(name)
            if spec is None:
                logger.warning("Unknown tool dataset alias '%s'; skipping", name)
                continue
            logger.info("Building tool-call samples from %s", spec["path"])
            for example in _dataset_iterator(spec, limit):
                formatter = spec["formatter"]
                formatted = formatter(example)
                if formatted:
                    yield formatted

    output_path = output_dir / "tool_data.jsonl"
    count = _write_jsonl(generator(), output_path)
    logger.info("Saved %d tool-calling examples to %s", count, output_path)


def _prepare_reasoning(datasets: List[str], output_dir: Path, limit: Optional[int]):
    def iterator_for_spec(spec: Dict[str, Any]):
        configs_provider = spec.get("configs")
        if configs_provider:
            configs = configs_provider()
            for config_name in configs:
                cfg_spec = {**spec, "name": config_name}
                for example in _dataset_iterator(cfg_spec, limit):
                    yield example
        else:
            for example in _dataset_iterator(spec, limit):
                yield example

    def generator():
        for name in datasets:
            spec = REASONING_SOURCES.get(name)
            if spec is None:
                logger.warning("Unknown reasoning dataset alias '%s'; skipping", name)
                continue
            logger.info("Building reasoning samples from %s", spec["path"])
            for example in iterator_for_spec(spec):
                formatter = spec["formatter"]
                formatted = formatter(example)
                if formatted:
                    yield formatted

    output_path = output_dir / "reasoning_data.jsonl"
    count = _write_jsonl(generator(), output_path)
    logger.info("Saved %d reasoning examples to %s", count, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare processed datasets for fine-tuning stages")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["instruct", "tool", "reasoning", "all"],
        default=["all"],
        help="Subset of dataset groups to build",
    )
    parser.add_argument("--instruct-datasets", nargs="*", help="Override instruction dataset aliases")
    parser.add_argument("--tool-datasets", nargs="*", help="Override tool dataset aliases")
    parser.add_argument("--reasoning-datasets", nargs="*", help="Override reasoning dataset aliases")
parser.add_argument("--instruct-limit", type=int, default=15000, help="Optional per-dataset limit for instruction data")
parser.add_argument("--tool-limit", type=int, default=3000, help="Optional per-dataset limit for tool data")
parser.add_argument("--reasoning-limit", type=int, default=2000, help="Optional per-dataset limit for reasoning data")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    data_config = DataConfig()
    processed_dir = Path(data_config.processed_data_dir)

    selected = set(args.stages)
    if "all" in selected:
        selected = {"instruct", "tool", "reasoning"}

    if "instruct" in selected:
        datasets = args.instruct_datasets or data_config.instruct_datasets
        _prepare_instruct(datasets, processed_dir, args.instruct_limit)

    if "tool" in selected:
        datasets = args.tool_datasets or data_config.tool_datasets
        _prepare_tool(datasets, processed_dir, args.tool_limit)

    if "reasoning" in selected:
        datasets = args.reasoning_datasets or data_config.reasoning_datasets
        _prepare_reasoning(datasets, processed_dir, args.reasoning_limit)


if __name__ == "__main__":
    main()
