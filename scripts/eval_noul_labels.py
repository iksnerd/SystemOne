"""Would upstream's `labels` fix (#156) rescue a yes/no that collapses? A test before a port.

laya 0.3.20 lets a `noul` show the model other words than `false:` / `true:` for its two slots;
laya-mlx 0.2.0 does not have it. This patches laya-mlx's option renderer in-process to emulate it,
on the same samples as `eval_noul_lean.py`, with the plain question, our checkpoint only.
FINDINGS §33.

    uv run --with pyarrow taskpolicy -b python scripts/eval_noul_labels.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import laya_mlx.agent
import laya_mlx.common

sys.path.insert(0, str(Path(__file__).parent))
from eval_noul_lean import TASKS, sample, summarize  # noqa: E402

from verdict.engine import PROMPT_TOKEN_BUDGET, load  # noqa: E402

_original = laya_mlx.common.render_options
#: (false slot, true slot). Upstream's documented override is true -> A, false -> B.
PAIRS = {"false/true": None, "B/A": ("B", "A"), "no/yes": ("no", "yes")}
current = {"pair": None}


def render(q):
    opts = _original(q)
    pair = current["pair"]
    if q["t"] != "noul" or pair is None:
        return opts
    return [pair[0] + opts[0][len("false"):], pair[1] + opts[1][len("true"):]]


laya_mlx.common.render_options = render
laya_mlx.agent.render_options = render


def main() -> None:
    engine = load("models/verdict-v1-mlx")
    results = {}
    for name, task in TASKS.items():
        q = {"plain": {"type": "noul", "instructions": task["plain"]}}
        rows = sample(task)
        for label, pair in PAIRS.items():
            current["pair"] = pair
            scores, labels = [], []
            for text, y in rows:
                state = engine.clip_state({"text": text}, PROMPT_TOKEN_BUDGET)
                scores.append(engine.predict(state, q)["plain"]["noul"])
                labels.append(y)
                time.sleep(0.05)
            results.setdefault(name, {})[label] = summarize(scores, labels)
            print(name, label, json.dumps(results[name][label]), file=sys.stderr)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
