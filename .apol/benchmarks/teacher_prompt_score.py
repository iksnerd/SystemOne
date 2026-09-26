"""APOL judge for the teacher-prompt benchmark: label one split with a cheap teacher under the
candidate preamble and print the agreement with the Pro reference as [APOL_SCORE: X/100].

Runs from any working directory (the optimizer runs it from a sandbox); data and code come from this
repo. Needs GOOGLE_API_KEY (or GEMINI_API_KEY) in the environment. Spend is capped by a persistent
ledger (VERDICT_OPT_CAP, default $0.90) and repeated candidates are cached."""
import argparse
import json
import os
from pathlib import Path

from verdict.label.bank import BANK
from verdict.label.prompt_eval import SpendLedger, score_preamble
from verdict.label.teacher import Teacher, gemini_transport, load_key

REPO = Path(__file__).resolve().parents[2]
REFERENCE = "gemini-3.1-pro-preview"

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["train", "holdout", "test"], required=True)
ap.add_argument("--preamble", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--run", type=Path, default=REPO / "runs/dev")
ap.add_argument("--model", default="gemini-2.5-flash-lite")
a = ap.parse_args()

split = json.loads((a.run / "split.json").read_text())
states = {r["id"]: r for r in map(json.loads, open(a.run / "states.jsonl"))}
ref = {}
for r in map(json.loads, open(a.run / "labels.jsonl")):
    if r["model"] == REFERENCE and r["repeat"] == 0:
        ref[r["id"]] = r["labels"]
ids = [i for i in split[a.split] if i in ref]

teacher = Teacher(a.model, gemini_transport(load_key(), temperature=0.0), preamble=a.preamble.read_text())
result = score_preamble(
    [states[i] for i in ids], ref, teacher, BANK,
    cache_path=a.run / "prompt_cache.jsonl",
    ledger=SpendLedger(a.run / "optimizer_spend.json", cap=float(os.environ.get("VERDICT_OPT_CAP", "0.90"))),
)
result["split"] = a.split
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(result, indent=1))
print(f"{a.split}: n={result['n']} cached={result['cached']} spend_this_call=${result['spend_usd_this_call']} total=${result['spend_usd_total']}")
print(f"[APOL_SCORE: {result['score']}/100]")
