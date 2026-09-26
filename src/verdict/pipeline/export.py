"""Export labelled states in the `LocalLLaMA/typed-decisions` schema that Laya's fine-tune
notebook reads: one row per case with `state`, `questions` and `gold` as JSON strings, where
`gold[qid]["probabilities"]` is the target distribution.

Targets are the mean of the cheap teachers' distributions. The reference teacher is left out on
purpose so it stays an independent judge. Rules carried over from an earlier corpus pipeline:
the test split is written apart from the training files and never named as one, states that leak
into held-out splits are excluded, and the write order is a salted hash of the id so it is
uncorrelated with generation order (a positional validation split would otherwise be one topic)."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ..answers import noul_probs
from ..label.teacher import mean_labels
from .audit import held_out_leak
from .config import PipelineConfig
from .stages import write_manifest
from .store import JsonlStore, write_json_atomic


def _gold(bank: dict, mean: dict) -> dict:
    out = {}
    for qid, q in bank.items():
        v = mean[qid]
        out[qid] = {"probabilities": v if q["type"] != "noul" else noul_probs(v)}
    return out


def export_stage(cfg: PipelineConfig, run_dir: Path, bank: dict, salt: str = "0") -> dict:
    run_dir = Path(run_dir)
    states = JsonlStore(run_dir / "states.jsonl", key=lambda r: r["id"]).rows()
    split = json.loads((run_dir / "split.json").read_text())
    per: dict = defaultdict(lambda: defaultdict(list))
    for r in JsonlStore(run_dir / "labels.jsonl", key=lambda r: f'{r["id"]}|{r["model"]}|{r["repeat"]}').rows():
        if r["model"] in cfg.label.models:  # the reference is excluded on purpose
            per[r["id"]][r["model"]].append(r["labels"])
    leaked = set(held_out_leak(states, split))
    questions = json.dumps(bank, ensure_ascii=False)
    by_id = {s["id"]: s for s in states}
    counts, excluded = {}, {"leaked": len(leaked), "unlabelled": 0}
    for name in ("train", "holdout", "test"):
        rows = []
        for i in split[name]:
            if (name == "train" and i in leaked):
                continue
            teachers = per.get(i, {})
            if not all(m in teachers for m in cfg.label.models) or not teachers:
                excluded["unlabelled"] += 1
                continue
            mean = mean_labels([lab for runs in teachers.values() for lab in runs])
            rows.append({"id": i, "workflow": cfg.domain, "state": json.dumps(by_id[i]["state"], ensure_ascii=False),
                         "questions": questions, "gold": json.dumps(_gold(bank, mean))})
        rows.sort(key=lambda r: hashlib.sha256(f"{salt}|{r['id']}".encode()).hexdigest())
        path = run_dir / "export" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        counts[name] = len(rows)
    write_manifest(run_dir, "export", cfg, ["states.jsonl", "labels.jsonl", "split.json"],
                   [f"export/{n}.jsonl" for n in ("train", "holdout", "test")], counts, excluded=excluded, salt=salt)
    return counts
