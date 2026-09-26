"""Do laya-mlx's opt-in options (compile, pad_to_multiple, cache_prompts, batch_size) help on OUR
workload, and do they change the answers?

Workload: 16 frozen-test states x the 5 ledger questions, one call per state. Each config runs 5 passes;
passes 1-2 are warm-up (compile cost lives there). For each state we keep the FASTEST post-warm-up time,
which is robust to other processes stealing the machine, and report the median over states. Answers from
the last pass are compared with the default config's."""
import json
import statistics as st
import time

from verdict.engine import load
from verdict.label.bank import BANK

rows = [json.loads(l) for l in open("runs/v1/export/test.jsonl")][:16]
states = [json.loads(r["state"]) for r in rows]
CONFIGS = {
    "default (fp16, no options)": {},
    "pad_to_multiple=16": {"pad_to_multiple": 16},
    "compile + pad16": {"compile": True, "pad_to_multiple": 16},
    "compile + pad16 + cache_prompts": {"compile": True, "pad_to_multiple": 16, "cache_prompts": True},
    "batch_size=64": {"batch_size": 64},
}
PASSES, WARM = 5, 2


def flat(res):
    out = []
    for q, a in sorted(res["answers"].items()):
        out += list(a["probabilities"].values()) if a["type"] != "noul" else [a["noul"]]
    return out


ref = None
for name, kw in CONFIGS.items():
    t0 = time.perf_counter()
    agent = load("models/verdict-v1-mlx", **kw)
    load_s = time.perf_counter() - t0
    per_state = [[] for _ in states]
    first = []
    outs = []
    for p in range(PASSES):
        outs = []
        for i, s in enumerate(states):
            t = time.perf_counter()
            r = {"answers": agent.predict(s, BANK)}
            ms = (time.perf_counter() - t) * 1000
            outs.append(flat(r))
            if p == 0:
                first.append(ms)
            if p >= WARM:
                per_state[i].append(ms)
    if ref is None:
        ref = outs
    diff = max(abs(a - b) for x, y in zip(outs, ref) for a, b in zip(x, y))
    best = [min(v) for v in per_state]
    print(f"{name:34s} first-pass p50={st.median(first):7.1f}  steady best-of-{PASSES - WARM} p50={st.median(best):6.1f} ms  "
          f"p90={sorted(best)[int(len(best) * 0.9)]:6.1f}  load={load_s:4.1f}s  max prob diff={diff:.4f}", flush=True)
