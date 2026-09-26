"""Build throwaway training items for the Modal timing probe from runs/dev, using the Pro reference
labels as targets. These items exist only to give the probe realistic sequence lengths: the real run
trains on the cheap teachers' labels and never on the reference."""
import json
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from verdict.label.bank import BANK
from verdict.label.teacher import mean_labels
from verdict.pipeline.export import _gold
from verdict.train.items import build_items

run = Path("runs/dev")
ref = defaultdict(list)
for l in open(run / "labels.jsonl"):
    r = json.loads(l)
    if r["model"] == "gemini-3.1-pro-preview":
        ref[r["id"]].append(r["labels"])
rows = []
for l in open(run / "states.jsonl"):
    s = json.loads(l)
    if s["id"] in ref:
        rows.append({"id": s["id"], "workflow": "probe", "state": json.dumps(s["state"]), "questions": json.dumps(BANK),
                     "gold": json.dumps(_gold(BANK, mean_labels(ref[s["id"]])))})
(run / "probe_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
md = snapshot_download("convaiinnovations/laya", allow_patterns=["tokenizer/*", "rl_agent_config.json", "encoder/*"])
items, dropped = build_items(run / "probe_rows.jsonl", Path(md))
torch.save(items, run / "probe_items.pt")
lens = sorted(len(i["ids"]) for i in items)
print(f"{len(rows)} states -> {len(items)} items ({dropped} dropped); tokens min/median/max = {lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")
