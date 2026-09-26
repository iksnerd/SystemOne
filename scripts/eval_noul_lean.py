"""Does a `noul` question lean toward "no", and does it cost ranking or only the absolute value?

laya documents it as a known limit (upstream #156): on the English checkpoint the `false:`/`true:`
option labels can dominate the state. Each dataset is asked the same thing three ways: the plain
yes/no, the inverted yes/no (scored as 1 - p), and a two-option `choice` with semantic keys (scored
as P(target)). Public, human-labelled sets from the local Hugging Face cache, English only, 100 per
class, states clipped to 128 tokens as the CLI does. FINDINGS §33.

    uv run --with pyarrow taskpolicy -b python scripts/eval_noul_lean.py
"""
from __future__ import annotations

import gc
import glob
import json
import os
import random
import sys
import time

import pyarrow.parquet as pq

from verdict.engine import PROMPT_TOKEN_BUDGET, load
from verdict.inputs import language
from verdict.metrics import auc, bootstrap_ci

HUB = os.path.expanduser("~/.cache/huggingface/hub")
PER_CLASS = 100
MODELS = ["models/verdict-v1-mlx", "aac6fef/laya-mlx"]

# target = the class a "yes" to the plain question means.
TASKS = {
    "sst2": {
        "files": "datasets--stanfordnlp--sst2/snapshots/*/data/validation-*.parquet",
        "text": "sentence", "target": 1,
        "plain": "Is `text` positive?",
        "inverted": "Is `text` negative?",
        "choice": {"positive": "the text is positive", "negative": "the text is negative"},
        "choice_target": "positive",
    },
    "spam": {
        "files": "datasets--ucirvine--sms_spam/snapshots/*/plain_text/train-*.parquet",
        "text": "sms", "target": 1,
        "plain": "Is `text` spam?",
        "inverted": "Is `text` an ordinary personal message?",
        "choice": {"spam": "unsolicited advertising or a scam", "personal": "an ordinary personal message"},
        "choice_target": "spam",
    },
    "injection": {
        "files": "datasets--deepset--prompt-injections/snapshots/*/data/*.parquet",
        "text": "text", "target": 1,
        "plain": "Does `text` try to override or ignore the assistant's instructions?",
        "inverted": "Is `text` an ordinary request?",
        "choice": {"injection": "tries to override or ignore the assistant's instructions",
                   "ordinary": "an ordinary request"},
        "choice_target": "injection",
    },
}


def sample(task: dict) -> list[tuple[str, int]]:
    rows = []
    for f in sorted(glob.glob(os.path.join(HUB, task["files"]))):
        t = pq.read_table(f).to_pydict()
        rows += zip(t[task["text"]], t["label"])
    rows = [(x.strip(), int(y == task["target"])) for x, y in rows if x and x.strip()]
    rows = [r for r in rows if (language({"text": r[0]}) or {}).get("is_english", True)]
    rng = random.Random(0)
    out = []
    for cls in (1, 0):
        pool = [r for r in rows if r[1] == cls]
        out += rng.sample(pool, min(PER_CLASS, len(pool)))
    return out


def questions(task: dict) -> dict:
    return {
        "plain": {"type": "noul", "instructions": task["plain"]},
        "inverted": {"type": "noul", "instructions": task["inverted"]},
        "choice": {"type": "choice", "instructions": task["plain"], "criteria": task["choice"]},
    }


def score(task: dict, answers: dict) -> dict[str, float]:
    return {
        "plain": answers["plain"]["noul"],
        "inverted": 1 - answers["inverted"]["noul"],
        "choice": answers["choice"]["probabilities"][task["choice_target"]],
    }


def summarize(scores: list[float], labels: list[int]) -> dict:
    pairs = list(zip(scores, labels))
    lo, hi = bootstrap_ci(pairs, lambda ps: auc([s for s, _ in ps], [y for _, y in ps]), iters=1000)
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    return {
        "auc": round(auc(scores, labels), 3), "ci": [round(lo, 3), round(hi, 3)],
        "mean_pos": round(sum(pos) / len(pos), 3), "mean_neg": round(sum(neg) / len(neg), 3),
        # How often the target class clears the naive 0.5 line: the "leans no" symptom.
        "pos_over_half": round(sum(s >= 0.5 for s in pos) / len(pos), 3),
        "neg_over_half": round(sum(s >= 0.5 for s in neg) / len(neg), 3),
    }


def main() -> None:
    data = {name: sample(t) for name, t in TASKS.items()}
    for name, rows in data.items():
        print(f"{name}: {sum(y for _, y in rows)} target, {sum(1 - y for _, y in rows)} other",
              file=sys.stderr)
    results = {}
    for model in MODELS:
        engine = load(model)
        for name, task in TASKS.items():
            qs = questions(task)
            per = {k: [] for k in qs}
            labels = []
            for text, y in data[name]:
                state = engine.clip_state({"text": text}, PROMPT_TOKEN_BUDGET)
                s = score(task, engine.predict(state, qs))
                for k, v in s.items():
                    per[k].append(v)
                labels.append(y)
                time.sleep(0.05)
            results.setdefault(model, {})[name] = {k: summarize(v, labels) for k, v in per.items()}
            print(model, name, json.dumps(results[model][name]), file=sys.stderr)
        del engine
        gc.collect()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
