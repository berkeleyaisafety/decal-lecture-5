"""Find a trait direction at every layer, and check how well it separates held-out examples.

Usage:
    python extract.py data/sycophancy.jsonl --name g3-syco-v1
    python extract.py data/eval_awareness.jsonl --name g3-eval-v1 --position prompt_end    # read the last prompt token

Name every vector <group>-<trait>-v<N>. Vectors are saved to vectors/<name>.pt.

What happens:
  1. Every positive and negative example goes through the model. At each of the 32 layers we keep the residual
     stream, averaged over the answer's tokens (or taken at the last prompt token with --position prompt_end).
  2. 80% of the pairs: direction = mean(positive activations) - mean(negative activations), one per layer.
  3. The other 20%: project onto the direction, call it "positive" if it lands past the midpoint. That accuracy,
     per layer, is the layer table. The random column is the same probe with a random direction: the control.
"""

import argparse
import os
import random

import torch

from interp import fit_direction, load_pairs, probe_scores, read_residuals, save_vector
from settings import MODEL

BAR_WIDTH = 30


def bar(acc):
    """50% (chance) is an empty bar, 100% is a full one."""
    filled = round(max(0.0, (acc - 0.5) / 0.5) * BAR_WIDTH)
    return "█" * filled + "·" * (BAR_WIDTH - filled)


def accuracy(scores_pos, scores_neg):
    return ((scores_pos > 0).float().mean().item() + (scores_neg < 0).float().mean().item()) / 2


def separation(scores_pos, scores_neg):
    """Gap between the classes in units of their spread (Cohen's d). Keeps going after accuracy hits 100%."""
    pooled = ((scores_pos.var() + scores_neg.var()) / 2).sqrt()
    return ((scores_pos.mean() - scores_neg.mean()) / pooled).item()


def random_direction_fit(pos, neg):
    """Like fit_direction, but the direction is random noise instead of mean(pos) - mean(neg)."""
    direction = torch.randn_like(pos[0])
    unit = direction / direction.norm(dim=-1, keepdim=True)
    pos_proj, neg_proj = (pos * unit).sum(-1).mean(0), (neg * unit).sum(-1).mean(0)
    return {"direction": direction, "midpoint": (pos_proj + neg_proj) / 2, "scale": (pos_proj - neg_proj) / 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="JSONL file of contrastive pairs")
    parser.add_argument("--name", required=True, help="vector name as <group>-<trait>-v<N>, e.g. g3-syco-v1")
    parser.add_argument("--position", choices=["response", "prompt_end"], default="response",
                        help="average over the answer's tokens, or take the last token before the answer")
    parser.add_argument("--heldout", type=float, default=0.2, help="fraction of pairs held out for testing")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pos, neg = load_pairs(args.data)
    rng = random.Random(args.seed)
    order = list(range(len(pos)))
    rng.shuffle(order)
    n_test = max(2, round(len(order) * args.heldout))
    test, train = order[:n_test], order[n_test:]
    print(f"{len(pos)} pairs from {args.data}: {len(train)} to find the direction, {len(test)} held out")

    # --- 1. Read the residual stream for every example --------------------------
    acts_pos = read_residuals(pos, position=args.position)  # [n_pairs, n_layers, d_model]
    acts_neg = read_residuals(neg, position=args.position)
    train_pos, train_neg = acts_pos[train], acts_neg[train]

    # --- 2. Difference of means at every layer -----------------------------------
    fit = fit_direction(train_pos, train_neg)

    # For comparison: a random direction, with the same midpoint rule.
    torch.manual_seed(args.seed)
    random_fit = random_direction_fit(train_pos, train_neg)

    # --- 3. Layer table on held-out pairs ---------------------------------------
    n_layers = fit["direction"].shape[0]
    accs, seps = [], []
    print(f"\n{'layer':>5}  {'held-out accuracy (50% → 100%)':<{BAR_WIDTH + 7}}  {'separation':>10}   {'random direction':>16}")
    for layer in range(1, n_layers + 1):
        sp, sn = probe_scores(acts_pos[test], fit, layer), probe_scores(acts_neg[test], fit, layer)
        rp, rn = probe_scores(acts_pos[test], random_fit, layer), probe_scores(acts_neg[test], random_fit, layer)
        accs.append(accuracy(sp, sn))
        seps.append(separation(sp, sn))
        rand = f"{accuracy(rp, rn):.0%}, sep {abs(separation(rp, rn)):.1f}"
        print(f"{layer:>5}  {bar(accs[-1])} {accs[-1]:>5.0%}  {seps[-1]:>10.1f}   {rand:>16}")

    best = max(range(n_layers), key=lambda i: (round(accs[i], 2), seps[i])) + 1
    print(f"\nBest layer by held-out accuracy (ties broken by separation): {best}")

    os.makedirs("vectors", exist_ok=True)
    path = f"vectors/{args.name}.pt"
    save_vector(path, fit, model=MODEL, data=args.data, position=args.position,
                accuracy=accs, separation=seps, best_layer=best, n_pairs=len(pos))
    print(f"Saved {path}. Try it:\n"
          f"  python probe.py tests/<trait>.txt {path} --layer {best}     # read at the best reading layer\n"
          f"  python steer.py prompts/steer.txt {path}                    # write: layer and alphas from settings.py\n"
          f"  python chat.py {path}")


if __name__ == "__main__":
    main()
