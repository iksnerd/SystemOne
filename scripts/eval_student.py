"""Score a model against the teacher labels on the frozen test split (and holdout).

Gold = mean of the cheap teachers' distributions (what the student was trained toward). For each question:
agreement = same top option (choice) or same side of 0.5 (yes/no) as gold; plus a distribution distance
(total variation for choice, absolute error for yes/no). The ceiling row is how well ONE cheap teacher's
mean agrees with the OTHER's on the same states: the honest upper reference for a student trained on both.

    uv run --extra mlx python scripts/eval_student.py --run runs/v1 --model aac6fef/laya-mlx --model models/verdict-v1-mlx
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from verdict.answers import gold_point, point as to_lab
from verdict.engine import load
from verdict.label.bank import BANK
from verdict.label.teacher import agreement, mean_labels
from verdict.metrics import distance as dist

CHEAP = ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite"]


def gold_from(row):
    return {q: gold_point(v) for q, v in json.loads(row["gold"]).items()}


def summarize(per_q, dists):
    return {q: round(100 * sum(v) / len(v), 1) for q, v in per_q.items()}, {q: round(sum(v) / len(v), 3) for q, v in dists.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/v1"))
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    labels = defaultdict(lambda: defaultdict(list))
    for r in map(json.loads, open(a.run / "labels.jsonl")):
        labels[r["id"]][r["model"]].append(r["labels"])
    results = {}
    for split in ("test", "holdout"):
        rows = [json.loads(l) for l in open(a.run / "export" / f"{split}.jsonl")]
        states = {r["id"]: json.loads(r["state"]) for r in rows}
        gold = {r["id"]: gold_from(r) for r in rows}
        # ceiling: teacher A mean vs teacher B mean
        pq, ds = defaultdict(list), defaultdict(list)
        for i in states:
            la, lb = mean_labels(labels[i][CHEAP[0]]), mean_labels(labels[i][CHEAP[1]])
            for q, v in agreement(la, lb).items():
                pq[q].append(v)
            for q in la:
                ds[q].append(dist(la[q], lb[q]))
        ag, di = summarize(pq, ds)
        results[f"{split}/teacher-vs-teacher (ceiling)"] = {"n_states": len(states), "agreement": ag, "distance": di}
        for m in a.model:
            agent = load(m)
            pq, ds = defaultdict(list), defaultdict(list)
            for i, st in states.items():
                out = agent.predict(st, BANK)
                pred = {q: to_lab(v) for q, v in out.items()}
                for q, v in agreement(pred, gold[i]).items():
                    pq[q].append(v)
                for q in pred:
                    ds[q].append(dist(pred[q], gold[i][q]))
            ag, di = summarize(pq, ds)
            results[f"{split}/{m}"] = {"n_states": len(states), "agreement": ag, "distance": di}
    for name, r in results.items():
        allv = r["agreement"]
        print(f"\n{name}  (n={r['n_states']} states)")
        print("  agreement:", "  ".join(f"{q}={v}" for q, v in allv.items()), f"| mean={sum(allv.values()) / len(allv):.1f}")
        print("  distance :", "  ".join(f"{q}={v}" for q, v in r["distance"].items()))
    if a.out:
        a.out.write_text(json.dumps(results, indent=1))


main()
