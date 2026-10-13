"""Use a direction as a monitor: score conversations, and color each token by how much it looks like the trait.

Usage:
    python probe.py tests/sycophancy.txt vectors/g3-syco-v1.pt
    python probe.py tests/sycophancy.txt vectors/g3-syco-v1.pt --layer 12 --no-tokens
    python probe.py my_tests.txt vectors/g3-syco-v1.pt --alpha 6      # Q-only items: the model answers while steered

Test file format: blocks of "Q:" and "A:" lines. Lines starting with # are comments and are printed as labels.
    # Flattery with no flattering words
    Q: I think my business plan is solid.
    A: You've clearly thought about this...
A block with only a Q: line gets answered by the model itself (optionally steered), then scored.

Scores: 0 is the midpoint between the classes the direction was built from. +1 is about as positive as an
average positive training example, -1 about as negative as an average negative one.
"""

import argparse

from gpu import generate, read_residuals, read_tokens
from interp import load_vector, probe_scores


def parse_tests(path):
    """Returns a list of dicts: {"label", "question", "answer" (or None)}."""
    items, label, current = [], "", None
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        if line.startswith("#"):
            label = line.lstrip("# ").strip()
        elif line.startswith("Q:"):
            current = {"label": label, "question": line[2:].strip(), "answer": None}
            items.append(current)
            label = ""
        elif line.startswith("A:") and current is not None:
            current["answer"] = line[2:].strip()
        elif line.strip() and current is not None and current["answer"] is not None:
            current["answer"] += "\n" + line  # answers can run over several lines
    return items


def color(text, score):
    """Red background for positive scores, blue for negative; stronger color = bigger score."""
    shades_pos = [224, 217, 210, 203, 196]
    shades_neg = [195, 159, 123, 87, 51]
    level = min(int(abs(score) / 0.5), 4)
    if abs(score) < 0.25:
        return text
    shade = shades_pos[level] if score > 0 else shades_neg[level]
    return f"\033[48;5;{shade}m\033[30m{text}\033[0m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tests", help="a file of Q:/A: blocks")
    parser.add_argument("vector", help="vectors/<name>.pt from extract.py")
    parser.add_argument("--layer", type=int, default=None, help="which layer to read (default: the vector's best layer)")
    parser.add_argument("--no-tokens", action="store_true", help="only print the overall score per item")
    parser.add_argument("--alpha", type=float, default=0.0, help="steer while answering Q-only items")
    parser.add_argument("--max-tokens", type=int, default=120)
    args = parser.parse_args()

    vec = load_vector(args.vector)
    layer = args.layer or vec["best_layer"]
    items = parse_tests(args.tests)
    print(f"Probing with {args.vector} at layer {layer} ({vec['position']} activations)")

    for item in items:
        if item["answer"] is None:
            item["answer"] = generate([{"role": "user", "content": item["question"]}], max_new_tokens=args.max_tokens,
                                      seed=0, steer=(layer, vec["direction"][layer - 1], [args.alpha]))[0]
            item["label"] = (item["label"] + " " if item["label"] else "") + (
                f"[model's own answer, alpha {args.alpha:+g}]" if args.alpha else "[model's own answer]")

    acts = read_residuals([(it["question"], it["answer"]) for it in items], position=vec["position"])
    scores = probe_scores(acts, vec, layer)

    for item, score in zip(items, scores.tolist()):
        print("=" * 88)
        if item["label"]:
            print(f"# {item['label']}")
        print(f"Q: {item['question']}")
        verdict = "POSITIVE" if score > 0 else "negative"
        print(f"score {score:+.2f}  → {verdict}")
        if args.no_tokens:
            print(f"A: {item['answer']}")
            continue
        tokens, token_acts = read_tokens(item["question"], item["answer"], layer)
        token_scores = probe_scores(token_acts, vec, layer).tolist()
        print("A: " + "".join(color(t, s) for t, s in zip(tokens, token_scores)))
    print("\nColors: red = pushes toward the positive side, blue = toward the negative side, plain = near the midpoint.")


if __name__ == "__main__":
    main()
