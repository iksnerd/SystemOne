"""The real check: does the fine-tuned model type REAL multi-agent log messages better than the base?

Real messages (data/hub_real_360.jsonl, 60 per type, private, never sent anywhere) carry the type the
author-agent assigned. Labels are noisy, so this is a sanity test of usefulness, not ground truth.
Only `kind` is scored; 'other' counts as a miss for the six real types."""
import json
from collections import Counter

from verdict.engine import load
from verdict.label.bank import BANK
from verdict.metrics import macro_f1, mean_ci

SIX = ["action", "synthesis", "decision", "thought", "draft", "note"]
rows = [json.loads(l) for l in open("data/hub_real_360.jsonl")]


for name in ("aac6fef/laya-mlx", "models/verdict-v1-mlx"):
    agent = load(name)
    y, p = [], []
    for r in rows:
        st = {"room_topic": r["room"], "author": r["author"], "message": r["text"]}
        p.append(agent.predict(st, {"kind": BANK["kind"]})["kind"]["choice"])
        y.append(r["gold"])
    hits = [int(a == b) for a, b in zip(y, p)]
    lo, hi = mean_ci(hits, iters=1000)
    print(f"{name:26s} n={len(rows)} acc={sum(hits) / len(hits):.3f} [{lo:.2f}-{hi:.2f}] macroF1={macro_f1(y, p, SIX):.3f}  preds={dict(Counter(p).most_common(4))}", flush=True)
