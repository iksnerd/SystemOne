"""Can the CLI turn any yes/no into a choice mechanically, with no sides to write?

§33 found a named two-option choice best or tied everywhere, and a noul relabelled to `no`/`yes`
as good on the collapsed SST-2 question. A tool rewriting an arbitrary question cannot name its
sides ("positive"/"negative"), but it can always offer `no` and `yes` as choice options. That is
not the relabel patch: the type token says `choice`, not `noul`. This measures it, on the same
samples as `eval_noul_lean.py`, our checkpoint only. The score is P(yes).

    uv run --with pyarrow taskpolicy -b python scripts/eval_noul_as_choice.py [MODEL]

MODEL defaults to our fine-tune; `aac6fef/laya-mlx` is base laya (FINDINGS §40).
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
    engine = load(sys.argv[1] if len(sys.argv) > 1 else "models/verdict-v1-mlx")
    results = {}
    for name, task in TASKS.items():
        bank = {
            "noul": {"type": "noul", "instructions": task["plain"]},
            "no_yes": {"type": "choice", "instructions": task["plain"],
                       "criteria": {"no": "", "yes": ""}},
        }
        scores: dict[str, list[float]] = {"noul": [], "no_yes": []}
        labels = []
        for text, y in sample(task):
            state = engine.clip_state({"text": text}, PROMPT_TOKEN_BUDGET)
            a = engine.predict(state, {**{k: {"instructions": "", **q} for k, q in bank.items()}})
            scores["noul"].append(a["noul"]["noul"])
            scores["no_yes"].append(a["no_yes"]["probabilities"]["yes"])
            labels.append(y)
            time.sleep(0.05)
        results[name] = {arm: summarize(s, labels) for arm, s in scores.items()}
        print(name, json.dumps(results[name]), file=sys.stderr)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
