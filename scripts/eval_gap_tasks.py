"""Do the checkpoints read tasks Laya was never trained on? A baseline before any new data.

Three public, human-labelled sets from families outside Laya's training mix and its own
benchmarks (FINDINGS §41): GitHub issue type (NLBSE'24: bug, feature, question), conventional
commit type (rsh-raj/commit-classification-17k, the author's own prefix, stripped), and the
dialogue acts of DailyDialog (via eusip/silicone `dyda_da`: inform, question, directive,
commissive). Balanced samples, seed 0, states clipped to 128 tokens as the CLI does. Every
checkpoint named on the command line is scored, one at a time.

    uv run --with pyarrow taskpolicy -b python scripts/eval_gap_tasks.py models/verdict-v1-mlx aac6fef/laya-mlx

Accuracy with a 1,000-sample bootstrap interval for the choices, AUC for the yes/no forms.
Issue titles often carry their label ("Bug: ...", "[Feature Request]"); those prefixes are
stripped so the model has to read the text, not a tag.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_noul_lean import summarize  # noqa: E402

from verdict.metrics import bootstrap_ci  # noqa: E402
from verdict.engine import PROMPT_TOKEN_BUDGET, load  # noqa: E402
from verdict.schema import laya_questions  # noqa: E402

CACHE = Path("data/gap")
NLBSE = ("https://raw.githubusercontent.com/nlbse2024/issue-report-classification/main/"
         "data/issues_test.csv")

ISSUE_Q = {"type": {
    "type": "choice", "instructions": "What does `issue` report or ask for?",
    "criteria": {"bug": "something is broken, crashes or behaves wrongly",
                 "feature": "a new feature or an improvement to how something works",
                 "question": "how to do something, or help understanding something"}}}

COMMIT_TYPES = {
    "feat": "adds a new feature or capability", "fix": "fixes a bug",
    "docs": "changes documentation only", "refactor": "restructures code without changing behaviour",
    "test": "adds or changes tests", "chore": "maintenance: dependencies, config, tooling, releases",
    "ci": "changes the CI pipeline or build workflows"}
COMMIT_Q = {"type": {"type": "choice", "instructions": "What kind of change does `subject` describe?",
                     "criteria": COMMIT_TYPES}}

ACTS = {"inform": "states or tells something", "question": "asks something",
        "directive": "asks or tells the listener to do something: a request, suggestion or order",
        "commissive": "commits the speaker to something: a promise, offer, acceptance or refusal"}
INSTRUCTION = "Is this an instruction to perform an action, rather than a question?"
DIALOG_Q = {
    "act": {"type": "choice", "instructions": "What does `text` do?", "criteria": ACTS},
    "instr_plain": {"type": "noul", "instructions": INSTRUCTION},
    "instr_no_yes": {"type": "choice", "instructions": INSTRUCTION,
                     "criteria": {"no": "", "yes": ""}},
}

_TAG = re.compile(r"^\s*(\[[^\]]*\]\s*|(bug|feature( request)?|question|enhancement|"
                  r"feat|fix|\[?q\]?)\s*[:\-]\s*)+", re.I)


def balanced(rows, label, per_class, seed=0):
    by = {}
    for r in rows:
        by.setdefault(label(r), []).append(r)
    rng = random.Random(seed)
    out = []
    for k in sorted(by):
        rng.shuffle(by[k])
        out += by[k][:per_class]
    return out


def issues(per_class=100):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "nlbse24_issues_test.csv"
    if not f.exists():
        urllib.request.urlretrieve(NLBSE, f)
    csv.field_size_limit(10**8)
    rows = balanced(list(csv.DictReader(f.open())), lambda r: r["label"], per_class)
    return [({"issue": f"{_TAG.sub('', r['title']).strip()}\n{r['body'][:600]}"}, r["label"])
            for r in rows]


def commits(per_class=60):
    from datasets import load_dataset

    d = load_dataset("rsh-raj/commit-classification-17k", split="train")
    rows = [r for r in d if r["type"] in COMMIT_TYPES and r["masked_commit_message"].strip()]
    return [({"subject": r["masked_commit_message"].strip()[:300]}, r["type"])
            for r in balanced(rows, lambda r: r["type"], per_class)]


def dialog(per_class=100):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("eusip/silicone", "dyda_da/test/0000.parquet", repo_type="dataset",
                        revision="refs/convert/parquet")
    rows = pq.read_table(p).to_pylist()
    return [({"text": r["Utterance"]}, r["Dialogue_Act"])
            for r in balanced(rows, lambda r: r["Dialogue_Act"], per_class)]


def accuracy(pred, gold):
    pairs = list(zip(pred, gold))
    acc = lambda ps: sum(p == g for p, g in ps) / len(ps)  # noqa: E731
    lo, hi = bootstrap_ci(pairs, acc, iters=1000)
    labels = sorted(set(gold))
    recall = {k: round(sum(p == g for p, g in pairs if g == k) / gold.count(k), 3) for k in labels}
    return {"accuracy": round(acc(pairs), 3), "ci": [round(lo, 3), round(hi, 3)],
            "chance": round(1 / len(labels), 3), "recall": recall,
            "predicted": dict(Counter(pred).most_common())}


def ask(engine, items, bank):
    bank = laya_questions(bank)
    out = []
    for state, _ in items:
        out.append(engine.predict(engine.clip_state(state, PROMPT_TOKEN_BUDGET), bank))
        time.sleep(0.05)
    return out


def main() -> None:
    sets = {"issues": (issues(), ISSUE_Q), "commits": (commits(), COMMIT_Q),
            "dialog": (dialog(), DIALOG_Q)}
    for name, (items, _) in sets.items():
        print(f"{name}: {len(items)} items, {dict(Counter(g for _, g in items))}", file=sys.stderr)
    results = {}
    for model in sys.argv[1:] or ["models/verdict-v1-mlx"]:
        engine = load(model)
        r = results[model] = {}
        for name, (items, bank) in sets.items():
            answers = ask(engine, items, bank)
            gold = [g for _, g in items]
            if name == "dialog":
                r[name] = {"act": accuracy([a["act"]["choice"] for a in answers], gold)}
                is_dir = [g == "directive" for g in gold]
                r[name]["instruction_plain"] = summarize([a["instr_plain"]["noul"] for a in answers], is_dir)
                r[name]["instruction_no_yes"] = summarize(
                    [a["instr_no_yes"]["probabilities"]["yes"] for a in answers], is_dir)
            else:
                r[name] = accuracy([a["type"]["choice"] for a in answers], gold)
            print(model, name, json.dumps(r[name]), file=sys.stderr)
        del engine
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
