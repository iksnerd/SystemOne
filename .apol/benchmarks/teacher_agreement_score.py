"""Score script for the teacher-agreement benchmark. Imports the project's own metric; it never
reimplements it, so the scorecard and the code cannot disagree about the number."""
import argparse
import json
from pathlib import Path

from verdict.label.scorecard import teacher_agreement

CHEAP = ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite"]
REFERENCE = "gemini-3.1-pro-preview"

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, default=Path("runs/pilot"))
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

rows = [json.loads(line) for line in open(a.run / "labels.jsonl")]
result = teacher_agreement(rows, cheap=CHEAP, reference=REFERENCE)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(result, indent=1))
print(f"teacher agreement with {REFERENCE}: {result['score']} over {result['n_states']} states "
      f"(self-agreement {result['self_agreement']})")
