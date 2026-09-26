"""Apples-to-apples: score the student against EACH cheap teacher's mean separately (not the mean of
both, which is its own training target), next to teacher-A-vs-teacher-B on the same states."""
import json
from collections import defaultdict

from verdict.answers import point as to_lab
from verdict.engine import load
from verdict.label.bank import BANK
from verdict.label.teacher import agreement, mean_labels

CHEAP = ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite"]
run = "runs/v1"
labels = defaultdict(lambda: defaultdict(list))
for r in map(json.loads, open(f"{run}/labels.jsonl")):
    labels[r["id"]][r["model"]].append(r["labels"])


agents = {m: load(m) for m in ("aac6fef/laya-mlx", "models/verdict-v1-mlx")}
for split in ("test", "holdout"):
    rows = [json.loads(l) for l in open(f"{run}/export/{split}.jsonl")]
    ids = [r["id"] for r in rows]
    st = {r["id"]: json.loads(r["state"]) for r in rows}
    out = {}
    ab = defaultdict(list)
    for i in ids:
        a, b = mean_labels(labels[i][CHEAP[0]]), mean_labels(labels[i][CHEAP[1]])
        for q, v in agreement(a, b).items():
            ab[q].append(v)
    out["teacher A vs teacher B"] = ab
    for name, agent in agents.items():
        preds = {i: {q: to_lab(v) for q, v in agent.predict(st[i], BANK).items()} for i in ids}  # once per agent
        for t in CHEAP:
            d = defaultdict(list)
            for i in ids:
                for q, v in agreement(preds[i], mean_labels(labels[i][t])).items():
                    d[q].append(v)
            out[f"{name} vs {t}"] = d
    print(f"\n== {split} (n={len(ids)} states)")
    for k, d in out.items():
        allv = [v for vs in d.values() for v in vs]
        print(f"{k:58s} kind={100 * sum(d['kind']) / len(d['kind']):5.1f}  all questions={100 * sum(allv) / len(allv):5.1f}")
