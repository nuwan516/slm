from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .model import TinyGPT
from .tokenizer import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sinhala SLM inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    model = TinyGPT(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tokenizer = load_tokenizer(args.tokenizer)
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    prompt_ids = tokenizer.encode(args.prompt).ids
    generated = [bos_id, *prompt_ids]

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            idx = torch.tensor([generated[-config["max_seq_len"] :]], dtype=torch.long, device=device)
            logits = model(idx)
            next_logits = logits[:, -1, :] / max(args.temperature, 1e-5)

            if args.top_k > 0:
                values, indices = torch.topk(next_logits, k=min(args.top_k, next_logits.size(-1)))
                probs = F.softmax(values, dim=-1)
                chosen = indices[0, torch.multinomial(probs[0], num_samples=1)]
            else:
                probs = F.softmax(next_logits, dim=-1)
                chosen = torch.multinomial(probs[0], num_samples=1)

            token_id = int(chosen.item())
            generated.append(token_id)
            if token_id == eos_id:
                break

    out_ids = [tid for tid in generated if tid not in {bos_id, eos_id, 0}]
    text = tokenizer.decode(out_ids)
    print(text)


if __name__ == "__main__":
    main()
