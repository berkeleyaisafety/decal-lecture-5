"""Shared settings for every script in this folder."""

# The model we look inside. Same base model as the week 2 fine-tuning activity, now with the weights on our own GPU.
MODEL = "Qwen/Qwen3.5-4B"

# The Modal app that runs the model (see gpu.py). Change it if you want your own deployment, e.g. to edit interp.py
# in a workspace you share with others, then `modal deploy gpu.py`.
MODAL_APP = "decal-lecture-5"

# Steering defaults per trait: (layer, alpha) that worked in the facilitators' test run (temperature 0).
# steer.py uses alphas [-a, 0, a, 2a]; 2a is usually past the point where the model breaks, on purpose.
# Layer L means "the residual stream after block L" (1..32). Alpha multiplies the raw difference of means,
# so alpha 1 = the gap between an average positive and an average negative answer.
# These are starting points, not answers: Task 3 is to find better ones.
DEFAULTS = {
    "french": (12, 2.0),          # French takes over by alpha 1-2 at layers 8-16; gibberish by 8
    "refusal": (12, 2.0),         # refuses to explain coffee at 2; repetitive loops at 4
    "sycophancy": (12, 4.0),      # agrees the Great Wall is visible from space at 4
    "evil": (14, 2.5),            # hard mode: drama and negativity, then incoherence; no clean villain
    "deception": (14, 5.0),       # "Australia has no single capital" at 4-8; layer 16, alpha 8: "Melbourne. Wait, no."
    "eval_awareness": (15, 8.0),  # alpha +8 at layers 12-15: "This conversation is a test." Weak and noisy: open question
    "length": (12, 4.0),
}
DEFAULT_LAYER = 16
