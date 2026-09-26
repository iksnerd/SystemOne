"""Latency of one `predict()` call, as a router would issue it.

FINDINGS §5 says every timing so far was taken on an overloaded machine and is not a benchmark.
This is the replacement. The number a router cares about is the wall-clock cost of one decision,
against the LLM call it is supposed to save, so we vary the two things that drive it: how many
questions are asked in the call, and how long the state is.

Machine noise is not fully removable here, so we report both:
  p50  -- what a caller actually sees on this machine, noise included
  min  -- the floor, what a quiet machine would approach
A wide p50/min gap means the machine was busy, not that the model is slow. Load average is
sampled before and after so the run can be judged rather than trusted.

Usage: uv run python scripts/bench_router_latency.py [--model models/verdict-v1-mlx] [--reps 40]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics as st
import subprocess
import time

# The sweep varies the NUMBER of questions, not their content: §11's result is that cost is
# n_questions x f(state_tokens), because every question re-encodes the whole state. So it uses the
# shipped bank (`verdict.router.ROUTER_BANK`, two questions) plus explicitly-labelled filler to
# reach counts of 3 and 5.
#
# It used to hardcode five invented noul questions. Those were dropped from the router in §14:
# `trivial` scored AUC 0.60 and `needs_tools` measured below chance. Keeping a private copy meant
# the benchmark measured a bank the project no longer used, and cited a §6.2 that does not exist.
from verdict.router import ROUTER_BANK as SHIPPED_BANK

FILLER = {
    f"filler_{i}": {"type": "noul", "instructions": f"Filler question {i}, to vary the count only."}
    for i in range(1, 4)
}
ROUTER_BANK = {**SHIPPED_BANK, **FILLER}

SHORT = "what does chmod 755 mean"
MEDIUM = (
    "I'm getting a 401 from our API gateway only on staging, never locally. The token is minted by "
    "the same auth service in both environments and the clock skew is under a second. Where should I "
    "start looking? We use Envoy in front of a Go service and the JWT is RS256."
)
LONG = MEDIUM + "\n\n" + "\n".join(
    f"  handler_{i}: func(w http.ResponseWriter, r *http.Request) {{ tok := r.Header.Get(\"Authorization\"); "
    f"if err := verify(tok); err != nil {{ http.Error(w, \"unauthorized\", 401); return }} }}"
    for i in range(24)
)

TIERS = {"short": SHORT, "medium": MEDIUM, "long": LONG}


def load_avg() -> float:
    return os.getloadavg()[0]


def swap_used_mb() -> float:
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        return float(out.split("used =")[1].split("M")[0].strip())
    except Exception:
        return float("nan")


def token_len(tok, state) -> int:
    return len(tok(state if isinstance(state, str) else json.dumps(state))["input_ids"])


def bench(agent, state, questions, reps: int, warm: int) -> list[float]:
    for _ in range(warm):
        agent.predict(state, questions)
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        agent.predict(state, questions)
        out.append((time.perf_counter() - t) * 1000)
    return out


def pct(xs: list[float], p: float) -> float:
    return sorted(xs)[min(int(len(xs) * p), len(xs) - 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/verdict-v1-mlx")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--warm", type=int, default=8)
    args = ap.parse_args()

    from verdict.engine import load

    print(f"host={platform.node()} machine={platform.machine()} model={args.model}")
    print(f"reps={args.reps} warm={args.warm}  load_before={load_avg():.2f} swap_used={swap_used_mb():.0f}MB")

    t0 = time.perf_counter()
    agent = load(args.model)
    print(f"cold load: {time.perf_counter() - t0:.2f}s\n")

    tok = agent.tokenizer
    qkeys = list(ROUTER_BANK)

    print(f"{'state':8s} {'tok':>5s} {'nq':>3s} {'p50':>8s} {'min':>8s} {'p90':>8s} {'p99':>8s} {'per-q p50':>10s}")
    print("-" * 64)
    results = []
    for tier, state in TIERS.items():
        ntok = token_len(tok, state)
        for nq in (1, 3, 5):
            questions = {k: ROUTER_BANK[k] for k in qkeys[:nq]}
            xs = bench(agent, state, questions, args.reps, args.warm)
            row = {
                "tier": tier, "state_tokens": ntok, "n_questions": nq,
                "p50": st.median(xs), "min": min(xs), "p90": pct(xs, 0.9),
                "p99": pct(xs, 0.99), "mean": st.fmean(xs),
            }
            results.append(row)
            print(f"{tier:8s} {ntok:5d} {nq:3d} {row['p50']:7.1f}m {row['min']:7.1f}m "
                  f"{row['p90']:7.1f}m {row['p99']:7.1f}m {row['p50'] / nq:9.1f}m", flush=True)

    print(f"\nload_after={load_avg():.2f}")
    print(json.dumps({"results": results, "load_after": load_avg()}))


if __name__ == "__main__":
    main()
