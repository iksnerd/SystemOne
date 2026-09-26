"""One-off: turn the first pilot's ad-hoc files (data/pilot_states_try*.jsonl, data/labels_pilot.jsonl)
into a pipeline run directory, so scorecards read the same layout the pipeline writes.
No API calls. Token counts were not kept for these rows, so spend is recorded in the manifest
from the run's own printed total instead."""
import json
from pathlib import Path

from verdict.pipeline.store import JsonlStore, sha256_file, write_json_atomic

out = Path("runs/pilot")
states = JsonlStore(out / "states.jsonl", key=lambda r: r["id"])
for p in ("data/pilot_states_try1.jsonl", "data/pilot_states_try2.jsonl"):
    for line in open(p):
        c = json.loads(line)
        if c["id"] not in states.done:
            states.append(c)
labels = JsonlStore(out / "labels.jsonl", key=lambda r: f'{r["id"]}|{r["model"]}|{r["repeat"]}')
for line in open("data/labels_pilot.jsonl"):
    r = json.loads(line)
    for model, runs in r["labels"].items():
        for rep, lab in enumerate(runs):
            row = {"id": r["id"], "model": model, "repeat": rep, "labels": lab, "tokens_in": 0, "tokens_out": 0, "imported": True}
            if f'{r["id"]}|{model}|{rep}' not in labels.done:
                labels.append(row)
write_json_atomic(out / "import.manifest.json", {
    "stage": "import",
    "note": "imported from data/pilot_states_try1/2.jsonl and data/labels_pilot.jsonl; per-row token counts not kept",
    "spend_usd_reported_by_original_run": 0.3862,
    "teachers": sorted({r["model"] for r in labels.rows()}),
    "counts": {"states": len(states.rows()), "label_rows": len(labels.rows())},
    "outputs": {n: sha256_file(out / n) for n in ("states.jsonl", "labels.jsonl")},
})
print(len(states.rows()), "states,", len(labels.rows()), "label rows ->", out)
