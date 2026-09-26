"""Teacher-agreement metric: how often do the cheap teachers land on the same decision as the
reference teacher? This is the ceiling a student trained on their labels can hope to reach.

Scored per question (same argmax for a choice, same side of 0.5 for a yes/no), averaged, as a share
out of 100 so APOL can use an integer perfect score."""
from __future__ import annotations

from collections import defaultdict

from .teacher import agreement, mean_labels


def _group(rows: list[dict]) -> dict:
    by: dict = defaultdict(lambda: defaultdict(list))
    for r in sorted(rows, key=lambda r: (r["id"], r["model"], r["repeat"])):
        by[r["id"]][r["model"]].append(r["labels"])
    return by


def teacher_agreement(rows: list[dict], cheap: list[str], reference: str) -> dict:
    by = _group(rows)
    per_q: dict[str, list[float]] = defaultdict(list)
    per_model: dict[str, list[float]] = {m: [] for m in cheap}
    selfs: list[float] = []
    ids = [i for i, m in by.items() if reference in m and all(c in m for c in cheap)]
    for i in ids:
        ref = by[i][reference][0]
        for m in cheap:
            runs = by[i][m]
            for qid, v in agreement(mean_labels(runs), ref).items():
                per_q[qid].append(v)
                per_model[m].append(v)
            if len(runs) >= 2:
                selfs += list(agreement(runs[0], runs[1]).values())

    def pct(xs):
        return round(100 * sum(xs) / len(xs), 2) if xs else float("nan")

    allv = [v for xs in per_q.values() for v in xs]
    return {
        "score": pct(allv),
        "n_states": len(ids),
        "per_question": {q: pct(v) for q, v in per_q.items()},
        "per_model": {m: pct(v) for m, v in per_model.items()},
        "self_agreement": pct(selfs),
        "reference": reference,
        "cheap": cheap,
    }
