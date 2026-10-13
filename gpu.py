"""Run the model on a Modal GPU. The scripts run on your laptop and call these functions; the model never leaves Modal.

Setup, once per Modal workspace:
    modal deploy gpu.py

Every function in interp.py that touches the model (read_residuals, read_tokens, generate, steering, logit_lens)
runs inside a Modal container. Everything else (fitting the direction, scoring, saving vectors) runs locally.
Containers stay warm for SCALEDOWN_WINDOW after the last call, so only the first call after a break waits for a
cold start (~1 minute: start a GPU container, load 9 GB of weights from the cache volume).

Edited interp.py? Run `modal deploy gpu.py` again: the deployed copy doesn't see your local edits until you do.

Have your own GPU (or a Mac with 16 GB+)? `export DECAL5_LOCAL=1` skips Modal and runs the model in-process.
"""

import os

import modal

from settings import MODAL_APP

GPU = "L40S"  # 48 GB, similar speed to the A40 our timings in FACILITATOR.md were measured on. "A10G" or "L4" are cheaper
SCALEDOWN_WINDOW = 15 * 60  # seconds an idle container stays up (and bills) before it shuts down
MAX_CONTAINERS = 10  # caps the bill: at most this many GPUs at once, however many people are running scripts

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.9.1",
        "transformers>=5.19",
        "numpy",
        "flash-linear-attention",  # fast kernels for Qwen3.5's linear-attention layers
    )
    .env({"HF_HOME": "/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_python_source("interp", "settings")
)
hf_cache = modal.Volume.from_name(f"{MODAL_APP}-hf-cache", create_if_missing=True)  # weights download once, not per container

app = modal.App(MODAL_APP, image=image)


# One request at a time per container: steering() and recording() put hooks on the shared model, so two requests
# running at once in the same container would see each other's hooks. Modal starts more containers instead.
@app.cls(gpu=GPU, volumes={"/hf": hf_cache}, scaledown_window=SCALEDOWN_WINDOW, max_containers=MAX_CONTAINERS,
         timeout=20 * 60)
class Model:
    @modal.enter()
    def setup(self):
        import interp

        interp.load()
        hf_cache.commit()  # keep the downloaded weights for the next container

    @modal.method()
    def read_residuals(self, examples, position="response"):
        import interp

        return interp.read_residuals(examples, position=position)

    @modal.method()
    def read_tokens(self, question, response, layer):
        import interp

        return interp.read_tokens(question, response, layer)

    @modal.method()
    def generate(self, messages, n=1, max_new_tokens=200, temperature=0.7, seed=None, steer=None):
        return _generate(messages, n, max_new_tokens, temperature, seed, steer)

    @modal.method()
    def logit_lens(self, direction, k=12):
        import interp

        return interp.logit_lens(direction, k=k)


def _generate(messages, n, max_new_tokens, temperature, seed, steer):
    """interp.generate, seeded, and wrapped in interp.steering(...) if steer=(layer, direction, alphas)."""
    import torch

    import interp

    if seed is not None:
        torch.manual_seed(seed)
    if steer is None:
        return interp.generate(messages, n=n, max_new_tokens=max_new_tokens, temperature=temperature)
    layer, direction, alphas = steer
    with interp.steering(layer, direction, alphas):
        return interp.generate(messages, n=n, max_new_tokens=max_new_tokens, temperature=temperature)


# --- What the scripts call (on your laptop) ------------------------------------------------------


class _InProcess:
    """Same interface as the deployed Model, but runs interp right here (DECAL5_LOCAL=1)."""

    class _Call:
        def __init__(self, fn):
            self.remote = fn

    def __init__(self):
        import interp

        interp.load()
        self.read_residuals = self._Call(interp.read_residuals)
        self.read_tokens = self._Call(interp.read_tokens)
        self.generate = self._Call(lambda messages, n=1, max_new_tokens=200, temperature=0.7, seed=None, steer=None:
                                   _generate(messages, n, max_new_tokens, temperature, seed, steer))
        self.logit_lens = self._Call(interp.logit_lens)


_model = None


def model():
    """The deployed Model, looked up by name (or the in-process stand-in with DECAL5_LOCAL=1)."""
    global _model
    if _model is None and os.environ.get("DECAL5_LOCAL") == "1":
        _model = _InProcess()
    if _model is None:
        try:
            cls = modal.Cls.from_name(MODAL_APP, "Model")
            cls.hydrate()
        except modal.exception.AuthError:
            raise SystemExit("Not logged in to Modal. Run `modal setup` first.")
        except modal.exception.NotFoundError:
            raise SystemExit(f"No Modal app named {MODAL_APP!r} in this workspace. Run `modal deploy gpu.py` first.")
        print(f"Using the GPU on Modal (app {MODAL_APP!r}). After a break, the first call takes ~1 minute "
              f"while a container starts.", flush=True)
        _model = cls()
    return _model


def read_residuals(examples, position="response"):
    """interp.read_residuals on the GPU: [n_examples, n_layers, d_model]."""
    return model().read_residuals.remote(examples, position=position)


def read_tokens(question, response, layer):
    """interp.read_tokens on the GPU: (token strings, [T, d])."""
    return model().read_tokens.remote(question, response, layer)


def generate(messages, n=1, max_new_tokens=200, temperature=0.7, seed=None, steer=None):
    """interp.generate on the GPU. steer=(layer, direction, alphas) wraps it in interp.steering(...),
    with one alpha per completion."""
    if steer is not None:
        layer, direction, alphas = steer
        steer = (layer, direction.cpu(), list(alphas))
    return model().generate.remote(messages, n=n, max_new_tokens=max_new_tokens, temperature=temperature,
                                   seed=seed, steer=steer)


def logit_lens(direction, k=12):
    """interp.logit_lens on the GPU: (promoted tokens, suppressed tokens)."""
    return model().logit_lens.remote(direction.cpu(), k=k)
