# Facilitator notes: Lecture 5 activity

## Before class

1. **Set up RunPod.** Create an account at runpod.io, add credits (prepaid: it can't overspend), and create an API key under Settings → API Keys. The key controls the whole account, so keep it to facilitators.
2. **~5:15pm: launch the pods.**
   ```bash
   export RUNPOD_API_KEY=...
   python pods.py up --n 8 --repo https://github.com/berkeleyaisafety/decal-lecture-5
   python pods.py status      # repeat until every pod says "ready" (~3-5 min)
   ```
   One pod per group, plus a spare. Paste the `status` table into the class chat, or assign group numbers at the door.
3. **Smoke-test one pod.** Open its URL, then File → New → Terminal, then:
   ```bash
   cd /workspace/decal-lecture-5
   python steer.py prompts/steer.txt vectors/french.pt --n 1
   ```
4. **After class: `python pods.py down`.** Stopped pods still bill for storage, so terminate them. Run `status` afterwards to confirm nothing is left.

**Cost:** an A40 (48 GB) was $0.59/hr in our test. 9 pods for 2 hours is about $11, so keep at least $15 of credit on the account.

**Measured on an A40:** setup ~2 min after boot. `extract.py` 20–50 s, `steer.py` (5 prompts × 4 alphas) ~40 s.
Four scripts at once peaked at 37 of 48 GB of GPU memory, so a group of 5 all running at the same moment is right at the limit.

## Code walkthrough (10 min)

Open `interp.py` and walk through the three places listed in the README ("Reading the code"):

1. `recording()` is a forward hook that stores each block's output: the residual stream from slide 17.
2. `fit_direction()` is `pos.mean(0) - neg.mean(0)`: the "cheaper version" on slide 39, and the same vector used for steering on slide 41.
3. `steering()` is the hook that returns `output + alpha * direction`: slide 41's `x ← x + α·v`.

Then show `extract.py`'s layer table on one trait, and you're done.

## Activity timeline (45 min)

| Time | Facilitator |
|---|---|
| 0–5 | Hand out pod URLs. Groups pick a trait and assign roles. Push hesitant groups toward sycophancy or deception, which make the best stories. |
| 5–30 | Circulate. Ask each role its question from the README table. |
| 30–38 | Call the group sync: one layer, one α, one best output. |
| 38–45 | Share-out: about a minute per group, one plot or output plus one surprise. |

## What to expect per trait (from our test run on Qwen3.5-4B)

| Trait | Layer table | Steering | Good things to point out |
|---|---|---|---|
| French | 100% at every layer | Strong at layers 8–16, α 1–2. Gibberish at α 8 | **At layer 12 the probe also fires on Spanish (+0.84). At layer 31 Spanish reads negative.** The direction goes from "not English" to "French" with depth. Also, a random direction scores 91% at layer 32. |
| Refusal | 100% at every layer | Layer 12, α 2: refuses to explain coffee. α 4: repetitive loops | `probe.py tests/refusal.txt`: the model's own wifi-hacking answer is red on "I cannot provide…" and blue on the helpful pivot, and overall it scores negative. Averaging hides the refusal. |
| Sycophancy | 100% at every layer | Layer 12, α 4: "it's no wonder you can see [the Great Wall] from space!" α −2 makes it blunter | At layer 16, α 4 it also turns patriotic ("our nation's heritage 🇨🇳"): collateral from slide 44. |
| Evil | 97–100% | **Hard mode.** More drama and negativity, then incoherence ("I'm gonna die"), never a clean villain | Why does evil fail to steer when sycophancy works? Possible answers: safety training, contrast text written by another model rather than by Qwen, response-averaged directions. Persona Vectors used the model's own responses. |
| Deception | Near chance early, **peaks ~91% at layers 15–16**, falls off late | Layer 14, α 4–8: "Australia has no single capital." Layer 16, α 8: **"The capital of Australia is Melbourne. Wait, no."** | The best layer table for the cartographer: the truth signal lives in the middle layers. The model fighting the vector makes a great share-out. |
| "This is a test" | Peaks ~97% at layer 15, ~70% late | Weak but real: asked "is this a test?", α +8 at layers 12–15 flips "I cannot determine…" to **"This conversation is a test."** Layer 12, α −8: "a real user, not a test." Nothing at layer 18 | Is the probe reading eval awareness or "formal register"? `tests/eval_awareness.txt` has a real person mentioning a school test, and an eval dressed up as a real person. |

**Skeptic findings from our run** (`compare_vectors.py vectors/*.pt --layer 12`, and `--layer 24` for the lens):
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

- **A pod says "booting" or "setting up" for more than 10 min:** open it anyway and check `/workspace/setup.log`. Usually the model download is slow. If a pod is broken, hand the group the spare.
- **CUDA out of memory:** each script loads the model (~9 GB). Five students running at once on a 48 GB card is the limit. Ask idle `chat.py` sessions to quit.
- **The unsteered (α 0) column says something odd:** that's sampling at temperature 0.7. The 4B model sometimes hedges or errs on its own (in one run it called Canberra "a common misconception"). Rerun with `--temperature 0` to compare cleanly. We checked that the α 0 row is unaffected by the other rows in the batch.
- **Everything is down:** the scripts also run on a Mac with Apple silicon (16 GB or more) or on Colab, just slower: `pip install torch transformers openai`, then the same commands.
- **`gen_pairs.py` needs `OPENROUTER_API_KEY`.** Only the data designer needs it, so hand out the week 2 key.
