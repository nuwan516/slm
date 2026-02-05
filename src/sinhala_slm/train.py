from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .data import load_texts
from .model import TinyGPT
from .tokenizer import encode_batch, save_tokenizer, train_bpe_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sinhala SLM")
    parser.add_argument("--dataset", required=True, help="Path to file or Hugging Face dataset name")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--output_dir", default="./artifacts/sinhala_slm")
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--min_frequency", type=int, default=2)

    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--model_dim", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batch(
    tokenized_texts: List[List[int]],
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for _ in range(batch_size):
        seq = random.choice(tokenized_texts)
        if len(seq) < 2:
            continue

        if len(seq) > max_seq_len + 1:
            start = random.randint(0, len(seq) - (max_seq_len + 1))
            seq = seq[start : start + max_seq_len + 1]

        if len(seq) < max_seq_len + 1:
            pad_len = max_seq_len + 1 - len(seq)
            seq = seq + [0] * pad_len

        x = seq[:-1]
        y = seq[1:]
        xs.append(x)
        ys.append(y)

    if not xs:
        raise RuntimeError("Could not create a non-empty batch. Check dataset/tokenization.")

    x_t = torch.tensor(xs, dtype=torch.long, device=device)
    y_t = torch.tensor(ys, dtype=torch.long, device=device)
    return x_t, y_t


def evaluate(
    model: TinyGPT,
    tokenized_texts: List[List[int]],
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        x, y = make_batch(tokenized_texts, batch_size, max_seq_len, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
    model.train()
    return float(loss.item())


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    texts = load_texts(
        dataset=args.dataset,
        text_column=args.text_column,
        dataset_config=args.dataset_config,
        split=args.split,
        max_samples=args.max_samples,
    )
    if not texts:
        raise RuntimeError("No text samples found after loading + preprocessing.")
    print(f"Loaded {len(texts)} text samples")

    print("Training tokenizer...")
    tokenizer = train_bpe_tokenizer(texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency)
    tokenizer_path = output_dir / "tokenizer.json"
    save_tokenizer(tokenizer, str(tokenizer_path))

    tokenized = encode_batch(tokenizer, texts)
    vocab_size = tokenizer.get_vocab_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TinyGPT(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        dim=args.model_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("Starting training...")
    for step in range(1, args.max_steps + 1):
        x, y = make_batch(tokenized, args.batch_size, args.max_seq_len, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == 1:
            val_loss = evaluate(model, tokenized, args.max_seq_len, args.batch_size, device)
            print(f"step={step} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "max_seq_len": args.max_seq_len,
            "dim": args.model_dim,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "dropout": args.dropout,
        },
    }
    ckpt_path = output_dir / "model.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
