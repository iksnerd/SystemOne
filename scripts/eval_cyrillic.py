"""How do the English and the multilingual checkpoints read Cyrillic? FINDINGS §37.

The bench's four Cyrillic suites (Bulgarian store reviews, Russian product reviews; each as a named
choice and as a yes/no), asked of the fine-tuned English checkpoint and of laya's multilingual
one. One model resident at a time: English first, then unloaded, then multilingual. States are
clipped to 128 tokens as the CLI does.

    uv run --extra bench python scripts/eval_cyrillic.py
"""
from __future__ import annotations

import gc
import json
import sys
import time

from verdict import bench, config
from verdict.engine import PROMPT_TOKEN_BUDGET, load


def run(engine, suites, items_by) -> dict:
    out = {}
    for s in suites:
        def ask(state, questions):
            time.sleep(0.05)
            return {"answers": engine.predict(engine.clip_state(state, PROMPT_TOKEN_BUDGET),
                                              questions)}
        r = bench.score(s, items_by[s.name], ask)
        out[s.name] = {k: r[k] for k in ("metric", "value", "ci") if k in r}
        if "auc" in r:
            out[s.name]["auc"] = r["auc"]
        if "target_over_half" in r:
            out[s.name]["target_over_half"] = r["target_over_half"]
        print(engine.name, s.name, json.dumps(out[s.name]), file=sys.stderr, flush=True)
    return out


def main() -> None:
    suites = [s for s in bench.load_suites() if s.lang == "multi"]
    items_by = {s.name: bench.sample(s, bench.fetch(s)) for s in suites}
    settings = config.load()
    results = {}
    for label, model in (("english", settings.model_path), ("multilingual", settings.multilingual_path)):
        resolved, _ = config.resolve_model(model)
        engine = load(resolved)
        results[label] = run(engine, suites, items_by)
        del engine
        gc.collect()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
