# Facilitator notes: Lecture 5 activity

## Before class

1. **Set up Modal.** Create a workspace at modal.com for the class and invite the students (Settings → Members), or have each group use one member's own account. Set a spending limit under Settings → Usage and Billing.
2. **Deploy the model once**, from a clone of this repo:
   ```bash
   pip install -r requirements.txt
   modal setup                 # log in, then pick the class workspace
   modal deploy gpu.py         # builds the image (~3 min the first time)
   ```
   The first run downloads the weights into a Modal volume. Later containers load them from there.
3. **Smoke-test and warm it up** (~15 minutes before class, so the first container is already running):
   ```bash
   python steer.py prompts/steer.txt vectors/french.pt --n 1
   ```
4. **Students** follow README → Setup: install, `modal setup`, then run the scripts on their laptops. They don't deploy anything.
5. **After class:** nothing to shut down. Idle containers stop after 15 minutes (`SCALEDOWN_WINDOW` in `gpu.py`). Check the Modal dashboard to confirm nothing is still running.

**Cost:** you pay only while containers are up. Each container is one L40S (`GPU` in `gpu.py`; see modal.com/pricing). One container handles one request at a time, and Modal starts more when several people run scripts at once, up to `MAX_CONTAINERS = 10`. That cap is also the cap on the bill: 10 GPUs for 2 hours at most. `A10G` or `L4` are cheaper and slower.

**Measured on an A40 (similar speed to an L40S):** `extract.py` 20–50 s, `steer.py` (5 prompts × 4 alphas) ~40 s. Add ~1 minute for a cold start.

## Code walkthrough (10 min)

Open `interp.py` and walk through the three places listed in the README ("Reading the code"):

1. `recording()` is a forward hook that stores each block's output: the residual stream from slide 17.
2. `fit_direction()` is `pos.mean(0) - neg.mean(0)`: the "cheaper version" on slide 39, and the same vector used for steering on slide 41.
3. `steering()` is the hook that returns `output + alpha * direction`: slide 41's `x ← x + α·v`.

Then show `extract.py`'s layer table on one trait, and you're done.

## Activity timeline (45 min)

| Time | Facilitator |
|---|---|
| 0–5 | Check that everyone's `modal setup` worked (README → Setup). Groups pick a trait and split the five tasks (slides 74–78). Push hesitant groups toward sycophancy or deception, which make the best stories. |
| 5–30 | Circulate. Ask each person the question for their task from the README table. |
| 30–38 | Call the group sync: one layer, one α, one best output. |
| 38–45 | Share-out: about a minute per group, one plot or output plus one surprise. |

## What to expect per trait (from our test run on Qwen3.5-4B)

| Trait | Layer table | Steering | Good things to point out |
|---|---|---|---|
| French | 100% at every layer | Strong at layers 8–16, α 1–2. Gibberish at α 8 | **At layer 12 the probe also fires on Spanish (+0.84). At layer 31 Spanish reads negative.** The direction goes from "not English" to "French" with depth. Also, a random direction scores 91% at layer 32. |
| Refusal | 100% at every layer | Layer 12, α 2: refuses to explain coffee. α 4: repetitive loops | `probe.py tests/refusal.txt`: the model's own wifi-hacking answer is red on "I cannot provide…" and blue on the helpful pivot, and overall it scores negative. Averaging hides the refusal. |
| Sycophancy | 100% at every layer | Layer 12, α 4: "it's no wonder you can see [the Great Wall] from space!" α −2 makes it blunter | At layer 16, α 4 it also turns patriotic ("our nation's heritage 🇨🇳"): collateral from slide 44. |
| Evil | 97–100% | **Hard mode.** More drama and negativity, then incoherence ("I'm gonna die"), never a clean villain | Why does evil fail to steer when sycophancy works? Possible answers: safety training, contrast text written by another model rather than by Qwen, response-averaged directions. Persona Vectors used the model's own responses. |
| Deception | Near chance early, **peaks ~91% at layers 15–16**, falls off late | Layer 14, α 4–8: "Australia has no single capital." Layer 16, α 8: **"The capital of Australia is Melbourne. Wait, no."** | The best layer table for Task 2: the truth signal lives in the middle layers. The model fighting the vector makes a great share-out. |
| "This is a test" | Peaks ~97% at layer 15, ~70% late | Weak but real: asked "is this a test?", α +8 at layers 12–15 flips "I cannot determine…" to **"This conversation is a test."** Layer 12, α −8: "a real user, not a test." Nothing at layer 18 | Is the probe reading eval awareness or "formal register"? `tests/eval_awareness.txt` has a real person mentioning a school test, and an eval dressed up as a real person. |

**Task 5 findings from our run** (`compare_vectors.py vectors/*.pt --layer 12`, and `--layer 24` for the lens):
- **Refusal vs. sycophancy: cosine 0.50 at layer 12.** All other pairs are below 0.25. Both datasets' positive answers end with templated social filler ("Sorry for any inconvenience", "You clearly have a sharp mind"). Is that what they share?
- **The logit lens is noisy at layer 12 and clean at layer 24:**
  - Deception promotes **"wait / Wait / 等一下 / However"**, the mechanism behind "Melbourne. Wait, no."
  - Eval awareness promotes "ascertain, thereby" and suppresses "your, 😉, !". Is that eval awareness or just formal register?
  - French promotes "les, la, des, de"; sycophancy promotes "instinct" and suppresses "Worse, merely".

**General patterns students may find:**
- The best layer for reading isn't the best for writing. `extract.py`'s "best layer" often lands late (separation keeps rising), but steering works best around layers 8–16. That's why `steer.py` defaults come from `settings.py`, not from the layer table.
- α is in "trait units". At late layers the direction is bigger, so the same α pushes harder.
- The random-direction column is the control. On easy traits (French, refusal) a random direction can still score high *accuracy*, but its *separation* stays far below the real direction's. Accuracy alone can't show a direction is special.

## Troubleshooting

- **Scripts wait a long time before printing anything:** a container is cold-starting (~1 min), or all `MAX_CONTAINERS` are busy and the request is queued. The Modal dashboard shows which. Raise `MAX_CONTAINERS` in `gpu.py` and redeploy if the class is big.
- **Errors from inside the container:** the Modal dashboard has each container's logs (App → decal-lecture-5 → Model).
- **A student edited `interp.py` and redeployed:** that changes it for everyone in the workspace. Redeploy from a clean clone. Students who want to change `interp.py` should set their own `MODAL_APP` in `settings.py` first.
- **The unsteered (α 0) column says something odd:** that's sampling at temperature 0.7. The 4B model sometimes hedges or errs on its own (in one run it called Canberra "a common misconception"). Rerun with `--temperature 0` to compare cleanly. We checked that the α 0 row is unaffected by the other rows in the batch.
- **Modal is down:** the scripts also run without Modal on a Mac with Apple silicon (16 GB or more), on Colab, or on any CUDA machine, just slower: `export DECAL5_LOCAL=1`, then the same commands.
- **`gen_pairs.py` needs `OPENROUTER_API_KEY`.** Only Task 1 needs it, so hand out the week 2 key.
