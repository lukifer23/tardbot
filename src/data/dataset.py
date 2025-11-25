import torch
from torch.utils.data import Dataset, IterableDataset
from typing import List, Dict, Any, Optional, Iterator
import json
import random
from pathlib import Path
from .tokenizer import TardBotTokenizer


class PretrainDataset(Dataset):
    """
    Legacy in-memory dataset. Use StreamingPretrainDataset for large corpora.
    """
    def __init__(
        self,
        texts: List[str],
        tokenizer: TardBotTokenizer,
        max_length: int = 4096,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }


class StreamingPretrainDataset(IterableDataset):
    """
    Iterable dataset that streams either raw text or pre-tokenized JSONL shards.
    Supports on-the-fly tokenization and deterministic train/val splitting.
    """
    def __init__(
        self,
        file_paths: List[str],
        tokenizer: TardBotTokenizer,
        max_length: int = 4096,
        split: str = "train",
        val_ratio: float = 0.05,
        seed: int = 42,
        shuffle_files: bool = True,
        total_sequences: Optional[int] = None,
    ):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        self.shuffle_files = shuffle_files
        self._iteration = 0
        self._base_rng = random.Random(seed)
        self.total_sequences = total_sequences

        # Calculate skip interval for validation
        # If val_ratio is 0.05 (1/20), we pick every 20th item for val
        if val_ratio > 0:
            self.skip_interval = int(1 / val_ratio)
        else:
            self.skip_interval = 0

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()

        self._iteration += 1
        file_order = list(self.file_paths)
        if self.shuffle_files:
            base_seed = self.seed + self._iteration
            if worker_info is not None:
                worker_seed = base_seed + worker_info.id
            else:
                worker_seed = base_seed
            rng = random.Random(worker_seed)
            rng.shuffle(file_order)

        if worker_info is not None:
            file_order = file_order[worker_info.id :: worker_info.num_workers]

        for file_path in file_order:
            path = Path(file_path)
            if path.suffix.lower() in {".jsonl", ".json"}:
                yield from self._iterate_tokenized_file(path, worker_info)
            else:
                yield from self._iterate_text_file(path, worker_info)

    def __len__(self) -> int:
        if self.total_sequences is None:
            raise TypeError(
                "StreamingPretrainDataset does not have a defined length. "
                "Pass total_sequences when constructing the dataset to enable __len__."
            )
        if self.split == "val":
            frac = self.val_ratio
        else:
            frac = 1.0 - self.val_ratio
        frac = max(frac, 0.0)
        return max(int(self.total_sequences * frac), 1)

    def _line_in_split(self, line_index: int) -> bool:
        if self.skip_interval == 0:
            return self.split == "train"
        is_val = (line_index % self.skip_interval) == 0
        if self.split == "val":
            return is_val
        return not is_val

    def _iterate_text_file(self, path: Path, worker_info) -> Iterator[Dict[str, torch.Tensor]]:
        with path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue

                if not self._line_in_split(line_index):
                    continue

                encoded = self.tokenizer.encode(
                    line,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors=None,
                )

                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                yield {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(input_ids, dtype=torch.long),
                }

    def _iterate_tokenized_file(self, path: Path, worker_info) -> Iterator[Dict[str, torch.Tensor]]:
        with path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue

                if not self._line_in_split(line_index):
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                input_ids = record.get("input_ids")
                attention_mask = record.get("attention_mask")
                if not isinstance(input_ids, list):
                    continue

                if attention_mask is None:
                    attention_mask = [1] * len(input_ids)
                if len(attention_mask) != len(input_ids):
                    continue

                tensor_ids = torch.tensor(input_ids, dtype=torch.long)
                tensor_mask = torch.tensor(attention_mask, dtype=torch.long)

                yield {
                    "input_ids": tensor_ids,
                    "attention_mask": tensor_mask,
                    "labels": tensor_ids.clone(),
                }


class InstructDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: TardBotTokenizer,
        max_length: int = 4096,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        if "instruction" in item and "output" in item:
            prompt = f"<bos>Instruction: {item['instruction']}\nResponse: {item['output']}<eos>"
        elif "messages" in item:
            messages = item["messages"]
            prompt = "<bos>"
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"{role.capitalize()}: {content}\n"
            prompt += "<eos>"
        else:
            prompt = str(item)
        
        encoded = self.tokenizer.encode(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }


class ToolDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: TardBotTokenizer,
        max_length: int = 4096,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        user_message = item.get("user_message", "")
        tool_calls = item.get("tool_calls", [])
        tool_results = item.get("tool_results", [])
        assistant_response = item.get("assistant_response", "")
        
        prompt = f"<bos>User: {user_message}\n"
        
        if tool_calls:
            prompt += "<tool_call>\n"
            for tool_call in tool_calls:
                prompt += json.dumps(tool_call) + "\n"
            prompt += "</tool_call>\n"
        
        if tool_results:
            prompt += "<tool_result>\n"
            for result in tool_results:
                prompt += json.dumps(result) + "\n"
            prompt += "</tool_result>\n"
        
        prompt += f"Assistant: {assistant_response}<eos>"
        
        encoded = self.tokenizer.encode(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }


class ReasoningDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: TardBotTokenizer,
        max_length: int = 4096,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        question = item.get("question", "")
        reasoning = item.get("reasoning", "")
        answer = item.get("answer", "")
        
        prompt = f"<bos>Question: {question}\n<thinking>\n{reasoning}\n</thinking>\nAnswer: {answer}<eos>"
        
        encoded = self.tokenizer.encode(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }
