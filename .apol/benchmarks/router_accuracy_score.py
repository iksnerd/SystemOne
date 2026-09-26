"""Score script for the router-accuracy benchmark. Imports the project's own metric; it never
reimplements it, so the scorecard and the code cannot disagree about the number.

Replays `data/router_probs.json` when it is present, and measures the prompts it is missing.
The model is deterministic, so a rerun on a warm cache returns the same score.
"""
import argparse
import json
from pathlib import Path

from verdict.router_eval import evaluate, load_prompts, measure

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/verdict-v1-mlx")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

rows = load_prompts()
probs, _ = measure(rows, a.model)
result = evaluate(rows, probs)
result["model"] = a.model
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(result, indent=1))

lo, hi = result["held_out_ci_95"]
ho = result["held_out"]
print(f"held-out accuracy {result['score']}% over {result['n_test']} prompts "
      f"(95% CI {lo} to {hi}, chance {result['chance']}), "
      f"hard->small {ho['hard_to_small']}, cost {ho['cost']:.3f}/prompt")
