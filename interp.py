"""The interpretability core. Every script in this folder is a thin wrapper around these functions.

Three places to read:
  1. read_residuals(): a forward hook copies the residual stream out of every block.
  2. fit_direction():  the direction is a difference of means; the probe is "which side of the midpoint are you on".
  3. steering():       a forward hook adds alpha * direction back into the residual stream at one block.
"""

import json
import os
from contextlib import contextmanager

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # keep the terminal readable

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import DEFAULTS, MODEL

_loaded = {}


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load():
    """Load the model once per process. Returns (tokenizer, model)."""
    if not _loaded:
        print(f"Loading {MODEL} on {device()}...", flush=True)
        transformers.logging.set_verbosity_error()  # hides "causal_conv1d not installed": harmless, just slower
        transformers.logging.disable_progress_bar()
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device()).eval()
        _loaded["tok"], _loaded["model"] = tok, model
    return _loaded["tok"], _loaded["model"]


def blocks(model):
    """The transformer blocks, in order. Block L writes the residual stream we call 'layer L' (1-indexed)."""
    return model.model.layers


def prompt_ids(tok, messages):
    """Chat template up to the point where the assistant starts talking."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False).input_ids


def load_pairs(path):
    """Contrastive pairs from a JSONL file. Returns (positive examples, negative examples) as (question, response) lists.

    Two line formats:
      {"question": ..., "positive": ..., "negative": ...}                   same question, the answers differ
      {"positive_question": ..., "negative_question": ..., "response": ...}  the questions differ, same answer
    """
    pos, neg = [], []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if "response" in row:
            pos.append((row["positive_question"], row["response"]))
            neg.append((row["negative_question"], row["response"]))
        else:
            pos.append((row["question"], row["positive"]))
            neg.append((row["question"], row["negative"]))
    return pos, neg


def encode(tok, question, response):
    """Token ids for [user: question][assistant: response], plus the index where the response starts."""
    p = prompt_ids(tok, [{"role": "user", "content": question}])
    r = tok(response, add_special_tokens=False).input_ids
    return p + r, len(p)


# --- 1. Reading the residual stream --------------------------------------------------------------


@contextmanager
def recording(model):
    """While active, every forward pass stores each block's output in the returned list (index 0 = block 1)."""
    store = [None] * len(blocks(model))
    hooks = []
    for i, block in enumerate(blocks(model)):
        def hook(module, inputs, output, i=i):
            store[i] = output  # [batch, tokens, d_model]: the residual stream after this block
        hooks.append(block.register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def read_residuals(examples, position="response", batch_size=8):
    """examples: list of (question, response) pairs.

    Returns a [n_examples, n_layers, d_model] tensor: the residual stream after every block, averaged over
    the response tokens (position="response") or taken at the last token before the response ("prompt_end").
    """
    tok, model = load()
    out = []
    for start in range(0, len(examples), batch_size):
        encoded = [encode(tok, q, r) for q, r in examples[start : start + batch_size]]
        length = max(len(ids) for ids, _ in encoded)
        input_ids = torch.full((len(encoded), length), tok.pad_token_id)
        mask = torch.zeros(len(encoded), length, dtype=torch.long)
        weights = torch.zeros(len(encoded), length)  # which positions get averaged
        for j, (ids, resp_start) in enumerate(encoded):
            input_ids[j, : len(ids)] = torch.tensor(ids)
            mask[j, : len(ids)] = 1  # padding goes on the right, so it can't affect earlier tokens
            if position == "response":
                weights[j, resp_start : len(ids)] = 1
            else:
                weights[j, resp_start - 1] = 1
        weights = (weights / weights.sum(1, keepdim=True)).to(device())
        with recording(model) as store:
            model(input_ids=input_ids.to(device()), attention_mask=mask.to(device()))
        out.append(torch.stack([torch.einsum("btd,bt->bd", h.float(), weights) for h in store], 1).cpu())
    return torch.cat(out)


@torch.no_grad()
def read_tokens(question, response, layer):
    """Per-token residual stream at one layer, for the response tokens only. Returns (token strings, [T, d])."""
    tok, model = load()
    ids, resp_start = encode(tok, question, response)
    with recording(model) as store:
        model(input_ids=torch.tensor([ids], device=device()))
    acts = store[layer - 1][0, resp_start:].float().cpu()
    return [tok.decode([t]) for t in ids[resp_start:]], acts


# --- 2. The direction and the probe --------------------------------------------------------------


def fit_direction(pos, neg):
    """pos, neg: [n, n_layers, d] activations. Returns per-layer directions and probe thresholds.

    direction[L] = mean(pos) - mean(neg). That's the whole method.
    The probe projects an activation onto the unit direction and compares it with the midpoint
    between the two classes. 'scale' (half the gap) lets scores read as -1 ≈ typical negative, +1 ≈ typical positive.
    """
    direction = pos.mean(0) - neg.mean(0)  # [n_layers, d]
    unit = direction / direction.norm(dim=-1, keepdim=True)
    pos_proj = (pos * unit).sum(-1).mean(0)
    neg_proj = (neg * unit).sum(-1).mean(0)
    return {"direction": direction, "midpoint": (pos_proj + neg_proj) / 2, "scale": (pos_proj - neg_proj) / 2}


def probe_scores(acts, vec, layer):
    """Score activations ([n, d] or [n, n_layers, d]) at one layer. > 0 means 'looks like the positive side'."""
    i = layer - 1
    if acts.dim() == 3:
        acts = acts[:, i]
    unit = vec["direction"][i] / vec["direction"][i].norm()
    return ((acts @ unit) - vec["midpoint"][i]) / vec["scale"][i]


def save_vector(path, fit, **info):
    torch.save({**fit, **info}, path)


def load_vector(path):
    return torch.load(path, weights_only=False)


def steering_defaults(vec):
    """(layer, alpha) to steer with: the tested default for this trait if there is one, else the best reading layer.
    The best layer for reading a trait is often not the best layer for writing it."""
    trait = os.path.splitext(os.path.basename(vec.get("data", "")))[0]
    for name, defaults in DEFAULTS.items():
        if name in trait:  # data/sycophancy.jsonl, data/g3-sycophancy-v2.jsonl, ...
            return defaults
    return vec["best_layer"], 4.0


# --- 3. Writing to the residual stream -----------------------------------------------------------


@contextmanager
def steering(layer, direction, alphas):
    """While active, add alpha * direction to the residual stream after block `layer`, at every position.

    alphas is one number per row of the batch, so a single generate() call can compare several strengths.
    alpha = 1 adds exactly the gap between the average positive and average negative answer.
    """
    _, model = load()
    add = torch.tensor(alphas, dtype=torch.float32).view(-1, 1, 1) * direction.view(1, 1, -1)
    add = add.to(device(), torch.bfloat16)

    def hook(module, inputs, output):
        return output + add  # returning a value from a forward hook replaces the block's output

    handle = blocks(model)[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def generate(messages, n=1, max_new_tokens=200, temperature=0.7):
    """Generate n completions of one conversation. Wrap in steering(...) to steer them."""
    tok, model = load()
    ids = torch.tensor([prompt_ids(tok, messages)] * n, device=device())
    sample = temperature > 0
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=sample,
        temperature=temperature if sample else None,
        top_p=0.95 if sample else None,
        top_k=None,
        pad_token_id=tok.pad_token_id,
    )
    return [tok.decode(row[ids.shape[1] :], skip_special_tokens=True).strip() for row in out]


# --- Reading a direction as words ----------------------------------------------------------------


@torch.no_grad()
def logit_lens(direction, k=12):
    """Push a direction through the final norm and the unembedding: which tokens does it promote and suppress?"""
    tok, model = load()
    v = direction.to(device(), torch.bfloat16).view(1, 1, -1)
    logits = model.lm_head(model.model.norm(v))[0, 0].float()
    top = logits.topk(k).indices.tolist()
    bottom = (-logits).topk(k).indices.tolist()
    return [tok.decode([t]) for t in top], [tok.decode([t]) for t in bottom]
