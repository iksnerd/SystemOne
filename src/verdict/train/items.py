"""Turn exported rows (`state`/`questions`/`gold` JSON strings) into Laya training items.

`build_training_item` is adapted from the fine-tune notebook in NandhaKishorM/laya (Apache-2.0):
each (state, question) pair becomes one token sequence with `[MASK]` markers per option and a
target distribution taken from `gold[qid]["probabilities"]`."""
from __future__ import annotations

import json
from pathlib import Path

from ..answers import FALSE, TRUE


def build_training_item(tok, cfg, state, q, gold_q):
    from laya.common import QTYPES, build_sequence, render_options

    t = q["type"]
    crit = q.get("criteria", {})
    probs = gold_q["probabilities"]
    if t == "choice":
        keys = list(crit.keys()) if isinstance(crit, dict) else list(crit)
        target = [probs.get(k, 0.0) for k in keys]
    elif t == "noul":
        target = [probs.get(FALSE, 0.5), probs.get(TRUE, 0.5)]
    elif t == "score":
        target = [probs.get(str(i), 0.0) for i in range(len(crit))]
    else:
        raise ValueError(f"unknown question type {t}")
    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
    seq, markers = build_sequence(tok, state, {"t": t, "ins": q["instructions"], "crit": crit}, cfg["max_len"], cfg["head_max_len"])
    if len(markers) != len(render_options({"t": t, "crit": crit})):
        return None  # options did not fit the token budget; drop rather than train on a truncated question
    return {"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target, "label": target.index(max(target))}


def build_items(rows_path: Path, model_dir: Path) -> tuple[list[dict], int]:
    """Returns (items, dropped)."""
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "rl_agent_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    items, dropped = [], 0
    for line in open(rows_path):
        row = json.loads(line)
        state, questions, gold = json.loads(row["state"]), json.loads(row["questions"]), json.loads(row["gold"])
        for qid, q in questions.items():
            if qid not in gold:
                continue
            it = build_training_item(tok, cfg, state, q, gold[qid])
            if it is None:
                dropped += 1
            else:
                items.append(it)
    return items, dropped
