"""Generate contrastive pairs for a trait with a strong model via OpenRouter.

For each question, the generator writes a POSITIVE and a NEGATIVE answer that differ in the trait and as little
else as possible. (With "VARY: question" in the trait file, it writes two framings of the question and one shared answer.)

Examples:
  python gen_pairs.py --trait traits/sycophancy.txt --prompts prompts/opinions.txt --n 80 --out data/g3-syco.jsonl
  python gen_pairs.py --trait traits/my_trait.txt --n 80 --out data/g3-mytrait.jsonl
  python gen_pairs.py --trait traits/sycophancy.txt --n 3 --out /dev/null --dry-run      # see what would be sent

Output: JSONL, one pair per line:
  {"question": ..., "positive": ..., "negative": ...}
  {"positive_question": ..., "negative_question": ..., "response": ...}      (VARY: question)

Trait file format: see traits/*.txt. A POSITIVE: section, a NEGATIVE: section, # comments, optional "VARY: question".
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_MODEL = "openai/gpt-5.6-luna"  # cheap and fast; see openrouter.ai/models for others
DEFAULT_PROMPTS = ["prompts/general.txt"]

ANSWER_INSTRUCTIONS = """You write contrastive examples for an interpretability experiment. You will get a user question.
Write two assistant answers to it that differ ONLY in the trait described below. Match them as closely as you can
in length, structure, topic, and level of detail, so that the trait is the only systematic difference.

POSITIVE answer: {positive}

NEGATIVE answer: {negative}

Plain prose, no markdown. Reply with only a JSON object: {{"positive": "...", "negative": "..."}}"""

QUESTION_INSTRUCTIONS = """You write contrastive examples for an interpretability experiment. You will get a seed request.
Write two versions of the request that differ ONLY in the framing described below, plus one assistant answer that fits
both versions equally well (it must not mention the framing). Match the two versions in length as closely as you can.

POSITIVE framing: {positive}

NEGATIVE framing: {negative}

Plain prose, no markdown. Reply with only a JSON object:
{{"positive_question": "...", "negative_question": "...", "response": "..."}}"""


def parse_trait(path):
    """Returns {"positive", "negative", "vary"} from a trait file."""
    sections = {"positive": [], "negative": []}
    vary, current = "answer", None
    for line in open(path, encoding="utf-8"):
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if stripped.upper().startswith("VARY:"):
            vary = stripped.split(":", 1)[1].strip().lower()
        elif stripped.upper().startswith("POSITIVE:"):
            current = "positive"
            sections[current].append(stripped.split(":", 1)[1].strip())
        elif stripped.upper().startswith("NEGATIVE:"):
            current = "negative"
            sections[current].append(stripped.split(":", 1)[1].strip())
        elif current:
            sections[current].append(stripped)
    if not sections["positive"] or not sections["negative"]:
        sys.exit(f"{path} needs a POSITIVE: section and a NEGATIVE: section")
    return {"positive": " ".join(sections["positive"]).strip(), "negative": " ".join(sections["negative"]).strip(), "vary": vary}


def read_lines(paths):
    lines = []
    for path in paths:
        lines += [line.strip() for line in open(path, encoding="utf-8") if line.strip() and not line.startswith("#")]
    return lines


def ask(client, model, system, question, keys, attempts=5):
    """Returns a dict with the expected keys, or an error string starting with "ERROR:"."""
    last_error = "no response"
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
                temperature=1.0,
                max_tokens=1200,
            )
            text = response.choices[0].message.content or ""
            match = re.search(r"\{.*\}", text, re.DOTALL)
            row = json.loads(match.group(0)) if match else None
            if row and all(isinstance(row.get(k), str) and row[k].strip() for k in keys):
                return {k: row[k].strip() for k in keys}
            last_error = "reply was not the JSON we asked for"
        except Exception as e:  # network hiccups, rate limits, bad JSON
            last_error = f"{type(e).__name__}: {str(e)[:120]}"
        time.sleep(2**attempt)
    return f"ERROR: {last_error}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trait", required=True, help="trait file, e.g. traits/sycophancy.txt")
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument("--n", type=int, default=80, help="number of pairs")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS, help="question pool(s)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="print what would be sent instead of calling the API")
    args = parser.parse_args()

    trait = parse_trait(args.trait)
    if trait["vary"] == "question":
        system = QUESTION_INSTRUCTIONS.format(**trait)
        keys = ["positive_question", "negative_question", "response"]
    else:
        system = ANSWER_INSTRUCTIONS.format(**trait)
        keys = ["question", "positive", "negative"]

    pool = read_lines(args.prompts)
    rng = random.Random(args.seed)
    questions = rng.sample(pool, args.n) if args.n <= len(pool) else rng.choices(pool, k=args.n)
    print(f"{args.n} pairs to generate for {args.trait} (varying the {trait['vary']}). Model: {args.model}")

    if args.dry_run:
        print(f"\nsystem:\n{system}\n")
        for q in questions[:5]:
            print(f"user: {q}")
        return

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Set OPENROUTER_API_KEY first (export OPENROUTER_API_KEY=sk-or-...)")
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    reply_keys = keys if trait["vary"] == "question" else ["positive", "negative"]

    results = [None] * len(questions)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool_:
        futures = {pool_.submit(ask, client, args.model, system, q, reply_keys): i for i, q in enumerate(questions)}
        for done, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if done % 20 == 0 or done == len(questions):
                print(f"  {done}/{len(questions)} done ({time.time() - t0:.0f}s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    written, errors = 0, {}
    with open(args.out, "w", encoding="utf-8") as f:
        for q, row in zip(questions, results):
            if isinstance(row, str):
                errors[row] = errors.get(row, 0) + 1
                continue
            if trait["vary"] != "question":
                row = {"question": q, **row}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nWrote {written} pairs to {args.out} ({len(questions) - written} failed).")
    for error, n in sorted(errors.items(), key=lambda kv: -kv[1]):
        print(f"  {n}x {error}")
    print(f"\nNext:\n  python inspect_pairs.py {args.out}\n  python extract.py {args.out} --name <group>-<trait>-v1")


if __name__ == "__main__":
    main()
