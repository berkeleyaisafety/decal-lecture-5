"""Is your direction really the trait? Compare it with other directions, and read it as words.

Usage:
    python compare_vectors.py vectors/g3-syco-v1.pt vectors/length.pt vectors/evil.pt
    python compare_vectors.py vectors/*.pt --layer 14
    python compare_vectors.py vectors/g3-syco-v1.pt vectors/g3-syco-v2.pt     # two vectors: also compared at every layer

Cosine similarity: 1 = same direction, 0 = unrelated, -1 = opposite. Two random directions in this model's
2560-dimensional space have a cosine of about 0 ± 0.02, so anything above ~0.1 is a real overlap.

Logit lens: push the direction through the model's output layer. Which tokens does adding it make more likely,
and which less? A "sycophancy" direction whose top tokens are all about length is a red flag.
"""

import argparse
import os

import torch

from interp import load_vector, logit_lens
from settings import DEFAULT_LAYER


def short(path):
    return os.path.splitext(os.path.basename(path))[0]


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vectors", nargs="+", help="vectors/<name>.pt files")
    parser.add_argument("--layer", type=int, default=None, help="layer to compare at (default: first vector's best layer)")
    parser.add_argument("--no-lens", action="store_true", help="skip the logit lens")
    parser.add_argument("--k", type=int, default=12, help="tokens to show in the logit lens")
    args = parser.parse_args()

    vecs = {short(p): load_vector(p) for p in args.vectors}
    names = list(vecs)
    layer = args.layer or vecs[names[0]].get("best_layer", DEFAULT_LAYER)
    width = max(10, max(len(n) for n in names) + 1)

    print(f"Cosine similarity at layer {layer}\n")
    print(" " * width + "".join(f"{n[:width - 1]:>{width}}" for n in names))
    for a in names:
        row = "".join(f"{cosine(vecs[a]['direction'][layer - 1], vecs[b]['direction'][layer - 1]).item():>{width}.2f}"
                      for b in names)
        print(f"{a:<{width}}{row}")

    if len(names) == 2:
        a, b = names
        per_layer = cosine(vecs[a]["direction"], vecs[b]["direction"]).tolist()
        print(f"\n{a} vs {b} at every layer:")
        for i in range(0, len(per_layer), 8):
            print("  " + "  ".join(f"L{j + 1:>2} {per_layer[j]:+.2f}" for j in range(i, min(i + 8, len(per_layer)))))

    if not args.no_lens:
        print(f"\nLogit lens at layer {layer}: tokens each direction promotes (+) and suppresses (-)")
        for name in names:
            up, down = logit_lens(vecs[name]["direction"][layer - 1], k=args.k)
            print(f"\n{name}\n  + {' | '.join(repr(t)[1:-1] for t in up)}\n  - {' | '.join(repr(t)[1:-1] for t in down)}")


if __name__ == "__main__":
    main()
