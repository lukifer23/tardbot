import os
import json
from typing import List, Optional, Union
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from transformers import PreTrainedTokenizerFast


class TardBotTokenizer:
    def __init__(
        self,
        vocab_size: int = 32000,
        tokenizer_path: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        self.vocab_size = vocab_size
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.tokenizer = None
        self.fast_tokenizer = None

        if tokenizer_path and os.path.exists(tokenizer_path):
            if os.path.isdir(tokenizer_path):
                self.load_from_model(tokenizer_path)
            else:
                self.load(tokenizer_path)
        elif model_path and os.path.exists(model_path):
            self.load_from_model(model_path)

    def train(
        self,
        files: List[str],
        output_path: str,
        vocab_size: Optional[int] = None,
    ):
        if vocab_size is None:
            vocab_size = self.vocab_size

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[
                "<pad>",
                "<bos>",
                "<eos>",
                "<unk>",
                "<mask>",
                "<tool_call>",
                "<tool_result>",
                "<thinking>",
                "</thinking>",
            ],
            min_frequency=2,
            show_progress=True,
        )

        tokenizer.train(files, trainer)
        tokenizer.save(output_path)
        self.tokenizer_path = output_path
        self.load(output_path)

    def load(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer_path = tokenizer_path

        self.fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            unk_token="<unk>",
            mask_token="<mask>",
            padding_side="right",
            truncation_side="right",
        )

        self.fast_tokenizer.pad_token_id = 0
        self.fast_tokenizer.bos_token_id = 1
        self.fast_tokenizer.eos_token_id = 2
        self.fast_tokenizer.unk_token_id = 3

    def load_from_model(self, model_path: str):
        tokenizer_path = os.path.join(model_path, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            self.load(tokenizer_path)

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        if self.tokenizer:
            tokenizer_path = os.path.join(output_dir, "tokenizer.json")
            self.tokenizer.save(tokenizer_path)

        if self.fast_tokenizer:
            self.fast_tokenizer.save_pretrained(output_dir)

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ):
        if self.fast_tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load() or train() first.")

        return self.fast_tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
    ):
        if self.fast_tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load() or train() first.")

        return self.fast_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    @property
    def vocab_size(self):
        if self.fast_tokenizer:
            return len(self.fast_tokenizer)
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size = value

    @property
    def pad_token_id(self):
        return self.fast_tokenizer.pad_token_id if self.fast_tokenizer else 0

    @property
    def bos_token_id(self):
        return self.fast_tokenizer.bos_token_id if self.fast_tokenizer else 1

    @property
    def eos_token_id(self):
        return self.fast_tokenizer.eos_token_id if self.fast_tokenizer else 2
