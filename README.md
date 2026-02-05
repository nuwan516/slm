# Sinhala SLM (Small Language Model)

This repository provides an end-to-end Sinhala-focused SLM pipeline that:

1. Automatically loads datasets from local files (`.txt`, `.jsonl`, `.csv`) **or** Hugging Face datasets.
2. Normalizes Sinhala text into a single Unicode form (NFC) with light cleanup.
3. Trains a BPE tokenizer on the normalized corpus.
4. Trains a compact decoder-only language model (GPT-style) with a standard training loop.
5. Runs inference (text generation) from a prompt.

## Project structure

- `src/sinhala_slm/data.py` — dataset loading + Sinhala normalization
- `src/sinhala_slm/tokenizer.py` — tokenizer training + load/save helpers
- `src/sinhala_slm/model.py` — tiny decoder-only transformer LM
- `src/sinhala_slm/train.py` — training script
- `src/sinhala_slm/infer.py` — inference script

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Train

### Local file dataset

```bash
PYTHONPATH=src python -m sinhala_slm.train \
  --dataset ./data/sinhala_corpus.txt \
  --output_dir ./artifacts/sinhala_slm
```

### Hugging Face dataset

```bash
PYTHONPATH=src python -m sinhala_slm.train \
  --dataset oscar \
  --dataset_config unshuffled_deduplicated_si \
  --split train[:1%] \
  --text_column text \
  --output_dir ./artifacts/sinhala_slm
```

## 2) Inference

```bash
PYTHONPATH=src python -m sinhala_slm.infer \
  --checkpoint ./artifacts/sinhala_slm/model.pt \
  --tokenizer ./artifacts/sinhala_slm/tokenizer.json \
  --prompt "මගේ නම" \
  --max_new_tokens 60
```

## Notes

- Unicode normalization uses NFC and cleanup rules in `normalize_sinhala_text`.
- For bigger runs, increase `--max_steps`, `--batch_size`, and `--model_dim`.
- Training uses next-token prediction with cross-entropy.
