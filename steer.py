"""Run the same prompts at several steering strengths, side by side.

Usage:
    python steer.py prompts/steer.txt vectors/g3-syco-v1.pt                          # layer and alphas from settings.py
    python steer.py prompts/steer.txt vectors/g3-syco-v1.pt --layer 12 --alphas -4 0 4 8
    python steer.py prompts/steer_opinions.txt vectors/g3-syco-v1.pt --n 3 --temperature 0

Steering adds alpha * direction to the residual stream after block --layer, at every token, while the model writes.
alpha = 0 is the untouched model. alpha = 1 adds the gap between an average positive and an average negative answer.
Negative alpha pushes the other way.
"""

import argparse
import json

import torch

from interp import generate, load_vector, steering, steering_defaults


def load_prompts(path):
    if not path.endswith(".jsonl"):
        return [line.strip() for line in open(path, encoding="utf-8") if line.strip() and not line.startswith("#")]
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return [row.get("question") or row.get("negative_question") for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompts", help="text file with one prompt per line (or a .jsonl of pairs)")
    parser.add_argument("vector", help="vectors/<name>.pt from extract.py")
    parser.add_argument("--layer", type=int, default=None, help="where to add the direction (default: settings.py for this trait)")
    parser.add_argument("--alphas", type=float, nargs="+", default=None, help="steering strengths (default: -a 0 a 2a, a from settings.py)")
    parser.add_argument("--n", type=int, default=5, help="how many prompts from the file")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7, help="0 = always pick the most likely token")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vec = load_vector(args.vector)
    default_layer, default_alpha = steering_defaults(vec)
    layer = args.layer or default_layer
    args.alphas = args.alphas or [-default_alpha, 0, default_alpha, 2 * default_alpha]
    direction = vec["direction"][layer - 1]
    prompts = load_prompts(args.prompts)[: args.n]
    print(f"Steering with {args.vector} at layer {layer}, alphas {args.alphas}")

    for prompt in prompts:
        torch.manual_seed(args.seed)
        with steering(layer, direction, args.alphas):  # one batch row per alpha
            outputs = generate([{"role": "user", "content": prompt}], n=len(args.alphas),
                               max_new_tokens=args.max_tokens, temperature=args.temperature)
        print("=" * 88)
        print(f"PROMPT: {prompt}")
        for alpha, text in zip(args.alphas, outputs):
            label = "alpha 0 (unsteered)" if alpha == 0 else f"alpha {alpha:+g}"
            print(f"\n[{label}]\n{text}")
        print(flush=True)


if __name__ == "__main__":
    main()
