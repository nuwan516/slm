from __future__ import annotations

import csv
import json
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional

from datasets import load_dataset


SINHALA_CLEAN_MAP = {
    "\u200d": "",  # ZERO WIDTH JOINER
    "\u200c": "",  # ZERO WIDTH NON-JOINER
    "\ufeff": "",  # ZERO WIDTH NO-BREAK SPACE/BOM
}


def normalize_sinhala_text(text: str) -> str:
    """Normalize Sinhala text to a consistent Unicode representation."""
    if not text:
        return ""

    normalized = unicodedata.normalize("NFC", text)
    for src, dst in SINHALA_CLEAN_MAP.items():
        normalized = normalized.replace(src, dst)

    # Normalize whitespace
    normalized = " ".join(normalized.strip().split())
    return normalized


def _read_txt(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def _read_jsonl(path: Path, text_column: str) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            text = obj.get(text_column)
            if text:
                yield str(text)


def _read_csv(path: Path, text_column: str) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = row.get(text_column)
            if text:
                yield str(text)


def load_texts(
    dataset: str,
    text_column: str = "text",
    dataset_config: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[str]:
    """Load dataset texts from local path or Hugging Face dataset name."""
    dataset_path = Path(dataset)
    texts: List[str] = []

    if dataset_path.exists():
        suffix = dataset_path.suffix.lower()
        if suffix == ".txt":
            iterator = _read_txt(dataset_path)
        elif suffix in {".jsonl", ".json"}:
            iterator = _read_jsonl(dataset_path, text_column=text_column)
        elif suffix == ".csv":
            iterator = _read_csv(dataset_path, text_column=text_column)
        else:
            raise ValueError(
                f"Unsupported local dataset format: {suffix}. "
                "Use .txt, .jsonl, .json, or .csv"
            )

        for idx, text in enumerate(iterator):
            texts.append(normalize_sinhala_text(text))
            if max_samples and idx + 1 >= max_samples:
                break
        return [t for t in texts if t]

    hf_ds = load_dataset(dataset, dataset_config, split=split)
    for idx, row in enumerate(hf_ds):
        text = row.get(text_column)
        if text is None:
            continue
        texts.append(normalize_sinhala_text(str(text)))
        if max_samples and idx + 1 >= max_samples:
            break

    return [t for t in texts if t]
