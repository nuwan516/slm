from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFC, Sequence, Strip
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def train_bpe_tokenizer(
    texts: Iterable[str],
    vocab_size: int = 16000,
    min_frequency: int = 2,
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFC(), Strip()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def encode_batch(tokenizer: Tokenizer, texts: List[str]) -> List[List[int]]:
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    output: List[List[int]] = []
    for text in texts:
        ids = tokenizer.encode(text).ids
        output.append([bos_id, *ids, eos_id])
    return output
