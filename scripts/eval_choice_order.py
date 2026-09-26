"""Does a two-option choice depend on which option is listed first? (laya-mlx #5 found 27-33% flips
on the multilingual checkpoint, Chinese, world-knowledge questions.) Same samples and questions as
`eval_noul_lean.py`, our checkpoint, each choice asked in both orders. FINDINGS §33.

    uv run --with pyarrow taskpolicy -b python scripts/eval_choice_order.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_noul_lean import TASKS, sample, summarize  # noqa: E402

from verdict.engine import PROMPT_TOKEN_BUDGET, load  # noqa: E402


def main() -> None:
    engine = load("models/verdict-v1-mlx")
    results = {}
    for name, task in TASKS.items():
        target = task["choice_target"]
        crit = task["choice"]
        orders = {"target first": crit, "target second": dict(reversed(list(crit.items())))}
        assert list(orders["target first"])[0] == target
        per = {k: [] for k in orders}
        picks = {k: [] for k in orders}
        labels = []
        for text, y in sample(task):
            state = engine.clip_state({"text": text}, PROMPT_TOKEN_BUDGET)
            for k, c in orders.items():
                a = engine.predict(state, {"q": {"type": "choice", "instructions": task["plain"],
                                                  "criteria": c}})["q"]
                per[k].append(a["probabilities"][target])
                picks[k].append(a["choice"])
                time.sleep(0.05)
            labels.append(y)
        mean = [(a + b) / 2 for a, b in zip(per["target first"], per["target second"])]
        flips = sum(a != b for a, b in zip(picks["target first"], picks["target second"]))
        results[name] = {**{k: summarize(v, labels) for k, v in per.items()},
                         "averaged": summarize(mean, labels),
                         "flip_rate": round(flips / len(labels), 3)}
        print(name, json.dumps(results[name]), file=sys.stderr)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
