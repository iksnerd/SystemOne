"""Score the big-vs-small switch on `evals/router_prompts.jsonl`. Reproduces the shipped numbers.

    uv run python scripts/eval_router.py            # report on the held-out split
    uv run python scripts/eval_router.py --refit    # also print thresholds to paste into router.py
    uv run python scripts/eval_router.py --json out.json

A thin command line over `verdict.router_eval`, which the APOL scorecard `router-accuracy` also
calls, so the two cannot disagree about the number.

Thresholds are fitted on the **train** split and reported on a **test** split the fit never saw.
That is the point of the file: §12 fitted and scored on the same 24 prompts and reported 83.3%,
which meant nothing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st

from verdict.router import ROUTER_BANK, T_DIFFICULTY, T_SENSITIVE
from verdict.router_eval import CACHE, DEFAULT_MODEL, evaluate, fit, load_prompts, measure, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()

    rows = load_prompts()
    probs, times = measure(rows, args.model)
    train = [r for r in rows if r["split"] == "train"]
    result = evaluate(rows, probs)

    print(f"model={args.model}  prompts={len(rows)} (train {result['n_train']}, test {result['n_test']})")
    if times:
        print(f"measured {len(times)} prompts, {st.median(times):.0f} ms p50, {len(ROUTER_BANK)} questions")
    else:
        print(f"all {len(rows)} prompts served from {CACHE} (delete it to re-measure)")

    ho, (lo, hi) = result["held_out"], result["held_out_ci_95"]
    print(f"\n=== shipped thresholds (difficulty {T_DIFFICULTY}, sensitive {T_SENSITIVE}) ===")
    print(f"  train      {result['train_accuracy']:.1f}%")
    print(f"  HELD-OUT   {ho['right']}/{ho['n']} = {result['score']:.1f}%   "
          f"95% CI {lo:.1f}% to {hi:.1f}%   (chance {result['chance']:.0f}%)")
    print(f"  hard->small {ho['hard_to_small']}   easy->big {ho['easy_to_big']}   "
          f"cost {ho['cost']:.3f}/prompt")

    print(f"\n=== rank signal, all {len(rows)} prompts (threshold-free) ===")
    for q, v in result["auc"].items():
        print(f"  {q:14s} AUC {v:.2f}")

    if result["mistakes"]:
        print(f"\n=== held-out mistakes ({len(result['mistakes'])}) ===")
        for w in result["mistakes"]:
            print(f"  {w}")

    if args.refit:
        c, td, ts = fit(train, probs)
        refit = evaluate(rows, probs, td, ts)
        rlo, rhi = refit["held_out_ci_95"]
        print(f"\n=== refit on train, minimising cost (hard->small counts 3x) ===")
        print(f"  best on train   cost {c:.3f}/prompt at difficulty {td}, sensitive {ts}")
        print(f"  that on test    {refit['score']:.1f}%  95% CI {rlo:.1f}% to {rhi:.1f}%   "
              f"hard->small {refit['held_out']['hard_to_small']}  "
              f"easy->big {refit['held_out']['easy_to_big']}")
        print(f"  paste into router.py:  T_DIFFICULTY = {td}   T_SENSITIVE = {ts}")

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(result, indent=1))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
