"""Does batching several states into one forward pass beat one `predict` per state, and does it
give the same answers? FINDINGS §34 (not shipped).

§18 left this undone: laya-mlx's `predict` takes one state, and a path assembled around its
internals is not covered by the port's parity checks. Here the assembly is minimal. `predict`
already chunks, collates, runs and post-processes its question rows; this hands it the rows of
many states at once through laya-mlx's own `prepare`, so every number still comes from laya-mlx
code. Answers are compared against the one-state-at-a-time path, row by row.

    uv run python scripts/bench_batch_states.py
"""
from __future__ import annotations

import json
import random
import sys
import time

from verdict import bench
from verdict.engine import PROMPT_TOKEN_BUDGET, load

N = 256
BANKS = {
    "1 question": {"q": {"type": "noul", "instructions": "Is `text` positive?"}},
    "2 questions": {
        "q": {"type": "noul", "instructions": "Is `text` positive?"},
        "c": {"type": "choice", "instructions": "Is `text` positive?",
              "criteria": {"positive": "the text is positive", "negative": "the text is negative"}},
    },
}


def predict_many(agent, states, questions, batch_size, sort):
    """Every state's rows through one `predict` call, regrouped per state."""
    order = sorted(range(len(states)), key=lambda i: len(json.dumps(states[i]))) if sort \
        else list(range(len(states)))
    flat = {f"{i}\x1f{qid}": q for i in order for qid, q in questions.items()}
    original = agent.prepare

    def prepare(_state, _questions):
        items, internal = [], []
        for i in order:
            it, inn = original(states[i], questions)
            items += it
            internal += inn
        return items, internal

    agent.prepare, saved = prepare, agent.batch_size
    agent.batch_size = batch_size
    try:
        answers = agent.predict(None, flat)["answers"]
    finally:
        del agent.prepare
        agent.batch_size = saved
    out = [dict() for _ in states]
    for key, a in answers.items():
        i, qid = key.split("\x1f")
        out[int(i)][qid] = a
    return out


def diff(a, b):
    """Largest absolute probability difference, and whether the decision agrees."""
    worst, same = 0.0, True
    for x, y in zip(a, b):
        for qid in x:
            if x[qid]["type"] == "noul":
                worst = max(worst, abs(x[qid]["noul"] - y[qid]["noul"]))
            else:
                worst = max(worst, max(abs(x[qid]["probabilities"][k] - y[qid]["probabilities"][k])
                                       for k in x[qid]["probabilities"]))
                same &= x[qid]["choice"] == y[qid]["choice"]
    return worst, same


def main() -> None:
    suite = next(s for s in bench.load_suites() if s.name == "sst2-choice")
    items = bench.sample(suite, bench.fetch(suite))
    random.Random(0).shuffle(items)
    engine = load("models/verdict-v1-mlx")
    states = [engine.clip_state({"text": t}, PROMPT_TOKEN_BUDGET) for t, _ in items[:N]]
    agent = engine.agent
    for name, qs in BANKS.items():
        engine.predict(states[0], qs)  # warm
        t = time.perf_counter()
        base = [engine.predict(s, qs) for s in states]
        one = (time.perf_counter() - t) / len(states) * 1000
        print(f"{name}: one predict per state {one:.1f} ms/state", file=sys.stderr)
        for sort in (False, True):
            for bs in (16, 32, 64):
                predict_many(agent, states[:8], qs, bs, sort)  # warm this shape
                t = time.perf_counter()
                got = predict_many(agent, states, qs, bs, sort)
                ms = (time.perf_counter() - t) / len(states) * 1000
                worst, same = diff(base, got)
                print(f"  batch {bs:3d} sorted={sort!s:5}  {ms:5.1f} ms/state  {one / ms:4.1f}x  "
                      f"max prob diff {worst:.4f}  same choices {same}", file=sys.stderr)


if __name__ == "__main__":
    main()
