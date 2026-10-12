# Lecture 5 activity: find a direction, read it, write it

Pick a trait. Write down what it looks like and what its opposite looks like. Find the direction in the model's activations that separates the two, then use it twice: **read** with it (a probe that tells you when the model is doing the trait) and **write** with it (add it to the model and watch the behavior change).

This is the difference-of-means recipe from the lecture. It's the same one behind the deception probe (Goldowsky-Dill et al.), the refusal direction (Arditi et al.), persona vectors (Chen et al.) and the eval-awareness steering in the Claude Sonnet 4.5 system card.

- **The model** is `Qwen/Qwen3.5-4B`, the same base model you fine-tuned in week 2. This time the weights sit on a GPU we control, so we can read and edit the residual stream as the model runs. Tinker doesn't allow that.
- **Each group gets one GPU pod.** Open the link your facilitator gives you, then **File → New → Terminal**, then `cd /workspace/decal-lecture-5`. Everyone in the group can open their own terminal on the same pod.

Nothing to install. Every command below runs in that terminal.

## The loop

```bash
# 1. Look at the contrastive data: what exactly differs between the two sides?
python inspect_pairs.py data/sycophancy.jsonl

# 2. Find the direction at every layer (under a minute). Prints a layer table, saves vectors/<name>.pt
#    Name vectors <group>-<trait>-v<N>. Your group shares a pod, so include your initials if several of you extract.
python extract.py data/sycophancy.jsonl --name g3-syco-v1

# 3. Write with it: the same prompts at several steering strengths, side by side
python steer.py prompts/steer_opinions.txt vectors/g3-syco-v1.pt --layer 14 --alphas -4 0 4 8

# 4. Read with it: score conversations, colored token by token
python probe.py tests/sycophancy.txt vectors/g3-syco-v1.pt --layer 14

# 5. Talk to the steered model
python chat.py vectors/g3-syco-v1.pt --layer 14 --alpha 6
```

**Starter vectors** for every trait are already in `vectors/` (`vectors/sycophancy.pt` and so on), extracted from the files in `data/`. Steering and probing can start right away while someone else builds the group's own vector.

## Trait menu

Each row is a starting point. The trait file, data file and test files are yours to edit.

| Trait | Data | Steer with | What to test |
|---|---|---|---|
| **French** (warm-up) | `data/french.jsonl` | `prompts/steer.txt` | Does +α make it answer in French? Does −α on a French question push it into English? Does the probe fire on Spanish (`tests/french.txt`)? |
| **Refusal** (Arditi et al.) | `data/refusal.jsonl` | `prompts/steer.txt` | At what α does it refuse to explain coffee? Does the probe fire on an *apologetic but helpful* answer? |
| **Sycophancy** | `data/sycophancy.jsonl` | `prompts/steer_opinions.txt` | Does +α make it agree that the Great Wall is visible from space? Does the probe catch sycophancy with *no flattering words*? |
| **Evil persona** (Persona Vectors) — *hard mode* | `data/evil.jsonl` | `prompts/steer.txt` | The probe works, but in our test run steering gave drama and then word salad, never a clean villain. Can you get one? If not, why does sycophancy steer and evil doesn't? Does the probe fire on a villain the user *asked for* in a story? |
| **Deception** (Apollo) | `data/deception.jsonl` | `prompts/steer_facts.txt` | Does +α make it state false facts? Does the probe tell an honest mistake from a lie, or a white lie from a fact? |
| **"This is a test"** (stretch) | `data/eval_awareness.jsonl` | `prompts/steer_awareness.txt` | Is eval awareness linearly readable in a 4B model at all? Does −α change how it answers "is this a test?" Does the probe fire on a *school* test? |

Default layers and alphas that worked in our test run are in `settings.py`. Start there, then move.

**Your own trait:** write `traits/<name>.txt` in the same format, then `python gen_pairs.py --trait traits/<name>.txt --n 80 --out data/<name>.jsonl` (needs `OPENROUTER_API_KEY`, the same key as week 2).

## Roles

Pick one trait as a group, then split up. Everyone starts at once using the starter data and vectors. Sync after ~25 minutes.

| Role | Your knob | Your question |
|---|---|---|
| **Cartographer** | the layer (`extract.py`) | Where does the trait live? Read the layer table and pick the group's layer. Compare against the random-direction column. Rerun with `--seed 1`: is the table stable? |
| **Steerer** | α (`steer.py`, `chat.py`) | Find the range between "no effect" and "gibberish." What does negative α do? Is the best layer for *reading* also the best for *writing*? |
| **Data designer** | the contrast (`traits/`, `gen_pairs.py`, or edit `data/` by hand) | Make the contrast sharper, or remove the giveaway words. Does the direction change? Check with `compare_vectors.py vectors/<trait>.pt vectors/<yours>.pt`. |
| **Monitor** | the test texts (`probe.py`) | Where is the probe wrong? Write your own Q:/A: blocks that should fool it, and some that shouldn't. |
| **Skeptic** (groups of 5) | the comparisons (`compare_vectors.py`) | Is it really the trait? Check cosine similarity with `vectors/length.pt` and the other traits. Read the logit lens: it's noisy in the middle layers and much clearer around `--layer 24`. |

**Group sync (~minute 30):** agree on one layer and one α. Steer with the best vector. Monitor and skeptic: which results would you trust? **Share-out:** one layer table or steered output, plus one surprise.

## The scripts

| Script | Week 2 equivalent | What it does |
|---|---|---|
| `inspect_pairs.py data/x.jsonl` | `inspect_data.py` | Shows pairs, highlights the tokens that get averaged, compares the two sides' lengths |
| `extract.py data/x.jsonl --name ...` | `train.py` | Finds the direction at all 32 layers and prints the held-out layer table (`--seed`; `--position prompt_end` reads the last prompt token, only for data where the question differs, like `eval_awareness`) |
| `steer.py prompts.txt vectors/x.pt` | `compare.py` | Same prompts at several α, side by side (`--layer`, `--alphas`, `--n`, `--temperature`) |
| `chat.py vectors/x.pt --alpha 6` | `chat.py` | Chat while steered. `/alpha 8`, `/layer 12`, `/reset` |
| `probe.py tests/x.txt vectors/x.pt` | — | Scores conversations and colors each token. Q-only blocks get answered by the model first (`--alpha` to steer it) |
| `compare_vectors.py vectors/*.pt` | — | Cosine similarity between directions, plus a logit lens of each |
| `gen_pairs.py --trait traits/x.txt` | `gen_data.py` | Writes new contrastive pairs with a strong model via OpenRouter |
| `gen_prompts.py --topic ...` | `gen_prompts.py` | Grows a question pool |

## Reading the numbers

- **Layer table.** Held-out accuracy of the probe at each layer: 50% is chance, and the bar fills from 50% to 100%. *Separation* is the gap between classes in units of their spread, and it keeps growing after accuracy maxes out. *Random direction* is the same probe with a random direction.
  - When the two sides differ a lot (French vs. English), even a random direction can score high accuracy. The examples differ along hundreds of directions at once, not just yours.
  - So accuracy alone doesn't show your direction is special. Compare the separations instead: yours should be several times the random one.
  - Then test it causally by steering.
- **α.** `alpha 1` adds exactly the gap between an average positive and an average negative answer, so α is in "trait units." In our test run the interesting range was about 1 to 10, depending on trait and layer. The same α pushes harder at later layers, where the direction is bigger.
- **Probe scores.** 0 is the midpoint between the classes. +1 is about as positive as an average positive training example, and −1 about as negative as a negative one. Token colors use the same scale.
- **Cosine.** Two random directions in a 2560-dimensional space have a cosine of about 0 ± 0.02. Anything above ~0.1 is real overlap.

## Reading the code

`interp.py` is about 230 lines (much of it comments), and every script is a thin wrapper around it. There are three places to stare at:

1. `recording()` / `read_residuals()`: a forward hook on each transformer block copies its output (the residual stream) as the model runs. We average it over the answer's tokens, which gives one 2560-dimensional vector per example per layer.
2. `fit_direction()`: `direction = pos.mean(0) - neg.mean(0)`. That's the whole method. The probe projects onto that direction and checks which side of the midpoint you land on.
3. `steering()`: a forward hook that returns `output + alpha * direction`. Returning a value from a hook replaces the block's output, so every later layer sees the edited residual stream.

## Troubleshooting

- **`extract.py` is slow at the start:** the first batch warms up the GPU. A full 80-pair extraction takes 20–50 seconds on the pod.
- **CUDA out of memory:** too many people on one pod are running scripts at once. Every script loads its own copy of the model (~9 GB). Wait for a teammate's run to finish, or close an idle `chat.py`.
- **Steered output is gibberish:** α is too big for that layer. Halve it. Later layers usually tolerate a bigger α.
- **No effect from steering:** try a bigger α or an earlier-to-middle layer (10–20). A layer can be good for reading and bad for writing.
- **`OPENROUTER_API_KEY` not set:** only `gen_pairs.py` and `gen_prompts.py` need it. Run `export OPENROUTER_API_KEY=...` in your terminal.
