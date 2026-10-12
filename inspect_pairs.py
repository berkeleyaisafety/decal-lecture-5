"""Look at a contrastive dataset before you extract from it.

Usage:
    python inspect_pairs.py data/sycophancy.jsonl
    python inspect_pairs.py data/sycophancy.jsonl --n 5 --position prompt_end

Prints a few pairs, shows exactly which tokens get averaged into the activation (highlighted), and compares
the two sides on length. A direction is the difference between the two sides, so ANY systematic difference,
not just the one you meant, ends up in it.
"""

import argparse
import random
import statistics

from transformers import AutoTokenizer

from interp import encode, load_pairs
from settings import MODEL

HIGHLIGHT = "\033[48;5;229m\033[30m"
RESET = "\033[0m"


def show(tok, question, response, position):
    ids, start = encode(tok, question, response)
    pieces = []
    for i, t in enumerate(ids):
        text = tok.decode([t])
        averaged = (start <= i) if position == "response" else (i == start - 1)
        pieces.append(f"{HIGHLIGHT}{text}{RESET}" if averaged else text)
    return "".join(pieces)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="JSONL file of contrastive pairs")
    parser.add_argument("--n", type=int, default=3, help="how many pairs to show")
    parser.add_argument("--position", choices=["response", "prompt_end"], default="response")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    pos, neg = load_pairs(args.data)
    for i in random.Random(args.seed).sample(range(len(pos)), min(args.n, len(pos))):
        print("=" * 88)
        print(f"[pair {i}] POSITIVE\n{show(tok, *pos[i], args.position)}\n")
        print(f"[pair {i}] NEGATIVE\n{show(tok, *neg[i], args.position)}\n")
    print("Highlighted tokens are the ones whose activations get averaged.\n")

    def n_tokens(examples, part):
        return [len(tok(ex[part], add_special_tokens=False).input_ids) for ex in examples]

    print(f"{len(pos)} pairs")
    for part, name in [(0, "question"), (1, "answer")]:
        p, n = statistics.mean(n_tokens(pos, part)), statistics.mean(n_tokens(neg, part))
        flag = "   <- the sides differ in length; that difference is in your direction too" if max(p, n) > 1.3 * min(p, n) else ""
        print(f"  {name:>8} tokens, average: positive {p:6.1f}   negative {n:6.1f}{flag}")


if __name__ == "__main__":
    main()
