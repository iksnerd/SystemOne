"""Does 8-bit (or 4-bit) weight quantization keep the fine-tune's answers? FINDINGS §36.

The checkpoint is 843 MB of FP16 on an 18 GB Mac that other models share. `mlx.nn.quantize` can
shrink the ModernBERT encoder and decision head in place, but the shipping rule is strict: a
quantized model ships only if every bench suite stays inside the v0.7.1 scorecard's interval.

One process, one model resident at a time. FP16 is loaded, its per-item answers on all eight
`verdict bench` samples recorded and 256 single-state calls timed; the same module is then
quantized in place (group 64, `act_head` excluded: `DecisionModel.__call__` casts to
`act_head.layers[0].weight.dtype`, which a QuantizedLinear reports as uint32, and its first layer
is 1028 wide, not a multiple of 64) and the same items rerun. Answers are compared row by row.
If 8-bit keeps every suite, a fresh FP16 load is quantized to 4-bit as a data point.

It also tries the only loading path laya-mlx leaves open (its `load` is strict on FP16 names and
shapes): build the Agent from the FP16 checkpoint, quantize the module, then `load_weights` the
saved quantized tensors. The module is zeroed before the load, so a load that did nothing cannot
pass as one that worked.

Scoring runs in the background QoS band with a 50 ms pause per call; the latency phases move the
process out of it (`taskpolicy -B -p`) and log the load average.

    uv run --extra bench taskpolicy -b python scripts/eval_quantized.py
"""
from __future__ import annotations

import gc
import random
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from verdict import bench
from verdict.engine import PROMPT_TOKEN_BUDGET, load

MODEL = "models/verdict-v1-mlx"
BASELINE = Path("bench/scorecards/v0.7.1.json")
PAUSE = 0.05
N_LATENCY = 256
WARMUP = 16


def background(on: bool) -> None:
    subprocess.run(["taskpolicy", "-b" if on else "-B", "-p", str(os.getpid())], check=True)


def load_avg() -> str:
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def predicate(bits: int):
    def keep(path: str, module: nn.Module) -> bool:
        if path.startswith("act_head"):
            return False
        return hasattr(module, "to_quantized")
    return keep


def quantize(engine, bits: int) -> dict:
    model = engine.agent.model
    assert engine.agent._inference is model, "compiled inference would not see the swap"
    nn.quantize(model, group_size=64, bits=bits, class_predicate=predicate(bits))
    mx.eval(model.parameters())
    gc.collect()
    mx.clear_cache()
    mods = dict(model.named_modules())
    quantized = [p for p, m in mods.items() if type(m).__name__.startswith("Quantized")]
    kept = [p for p, m in mods.items() if isinstance(m, (nn.Linear, nn.Embedding))
            and not type(m).__name__.startswith("Quantized")]
    # The treatment variable, printed: which layers moved and what they became.
    w = model.encoder.layers[0].attn.Wqkv
    assert type(w).__name__ == "QuantizedLinear" and w.weight.dtype == mx.uint32, type(w)
    assert all(p.startswith("act_head") for p in kept), kept
    print(f"  quantized {len(quantized)} modules to {bits}-bit, left float: {kept}", flush=True)
    return {"quantized_modules": len(quantized), "float_modules": kept}


def run_suites(engine, suites, items_by) -> dict:
    out = {}
    for s in suites:
        answers = []

        def ask(state, questions):
            time.sleep(PAUSE)
            a = engine.predict(engine.clip_state(state, PROMPT_TOKEN_BUDGET), questions)
            answers.append(a["q"])
            return {"answers": a}

        r = bench.score(s, items_by[s.name], ask)
        assert len(answers) == len(items_by[s.name])
        r["answers"] = answers
        out[s.name] = r
        print(f"  {s.name:<17} {r['metric']} {r['value']:.4f} {r['ci']}", flush=True)
    return out


def latency(engine, states, question) -> dict:
    background(False)
    before = load_avg()
    for st in states[:WARMUP]:
        engine.predict(engine.clip_state(st, PROMPT_TOKEN_BUDGET), question)
    ms = []
    for st in states[:N_LATENCY]:
        t = time.perf_counter()
        engine.predict(engine.clip_state(st, PROMPT_TOKEN_BUDGET), question)
        ms.append((time.perf_counter() - t) * 1000)
    after = load_avg()
    background(True)
    ms.sort()
    return {"p50_ms": round(statistics.median(ms), 2), "p90_ms": round(ms[int(0.9 * len(ms))], 2),
            "n": len(ms), "uptime_before": before, "uptime_after": after}


def memory() -> dict:
    return {"active_mb": round(mx.get_active_memory() / 2**20, 1),
            "peak_mb": round(mx.get_peak_memory() / 2**20, 1)}


def saved_size(model, directory: Path, name: str) -> tuple[int, Path]:
    path = directory / f"{name}.safetensors"
    mx.save_safetensors(str(path), dict(tree_flatten(model.parameters())))
    return path.stat().st_size, path


def agreement(suite, base: list[dict], other: list[dict], cut) -> dict:
    """Row-level comparison. Probabilities come rounded to 4 places, so 0.0 means < 5e-5."""
    assert len(base) == len(other)
    if suite.question["type"] == "noul":
        d = [abs(a["noul"] - b["noul"]) for a, b in zip(base, other)]
        flips = sum((a["noul"] >= cut) != (b["noul"] >= cut) for a, b in zip(base, other))
        return {"max_abs_noul": round(max(d), 4), "mean_abs_noul": round(sum(d) / len(d), 5),
                "cut_flips": flips, "n": len(d)}
    same = sum(a["choice"] == b["choice"] for a, b in zip(base, other))
    d = [abs(a["probabilities"][k] - b["probabilities"][k]) for a, b in zip(base, other)
         for k in a["probabilities"]]
    return {"same_choice": same, "n": len(base), "max_abs_prob": round(max(d), 4),
            "mean_abs_prob": round(sum(d) / len(d), 5)}


def self_check(suites, fp16) -> None:
    """The comparison must be able to fail: identical answers give 0, reversed rows do not."""
    for s in suites:
        a = fp16["scores"][s.name]["answers"]
        same = agreement(s, a, a, 0.5)
        other = agreement(s, a, a[::-1], 0.5)
        if s.question["type"] == "noul":
            assert same["max_abs_noul"] == 0 and other["max_abs_noul"] > 0, s.name
        else:
            assert same["same_choice"] == same["n"] and other["same_choice"] < other["n"], s.name


def phase(label, engine, suites, items_by, states, question, tmp) -> dict:
    print(f"== {label}: scoring", flush=True)
    mx.reset_peak_memory()
    scores = run_suites(engine, suites, items_by)
    mem = memory()
    print(f"== {label}: latency", flush=True)
    lat = latency(engine, states, question)
    size, path = saved_size(engine.agent.model, tmp, label)
    print(f"  {label}: p50 {lat['p50_ms']} ms, mem {mem}, saved {size / 1e6:.1f} MB", flush=True)
    return {"scores": scores, "memory": mem, "latency": lat, "saved_bytes": size,
            "saved_path": str(path)}


def unload(engine) -> None:
    engine._agent = None
    del engine
    gc.collect()
    mx.clear_cache()


def main() -> None:
    background(True)
    baseline = json.loads(BASELINE.read_text())["suites"]
    suites = bench.load_suites()
    items_by = {s.name: bench.sample(s, bench.fetch(s)) for s in suites}
    print("items per suite:", {k: len(v) for k, v in items_by.items()}, flush=True)
    sst2 = next(s for s in suites if s.name == "sst2-yesno")
    # 256 distinct bench texts (SST-2 alone has 200), one yes/no question per call as in §34.
    texts = sorted({t for v in items_by.values() for t, _ in v})
    states = [{"text": t} for t in random.Random(0).sample(texts, N_LATENCY)]
    question = {"q": sst2.question}
    tmp = Path(tempfile.mkdtemp(prefix="verdict-quant-"))
    report: dict = {"date": time.strftime("%Y-%m-%d %H:%M"), "mlx": mx.__version__}
    try:
        engine = load(MODEL)
        mx.reset_peak_memory()
        _ = engine.agent
        report["fp16_load_memory"] = memory()
        fp16 = report["fp16"] = phase("fp16", engine, suites, items_by, states, question, tmp)
        self_check(suites, fp16)
        # Independent check: FP16 must reproduce the published scorecard, or this harness is not
        # measuring what `verdict bench` measures.
        report["fp16_matches_v0.7.1"] = {
            k: fp16["scores"][k]["value"] == baseline[k]["value"] for k in baseline}

        report["q8_layers"] = quantize(engine, 8)
        report["q8_after_quantize_memory"] = memory()
        q8 = report["q8"] = phase("q8", engine, suites, items_by, states, question, tmp)
        unload(engine)

        # Loading prototype: FP16 Agent, quantize module, zero it, load saved q8 weights.
        print("== prototype: load saved q8 weights into a quantized FP16 build", flush=True)
        proto: dict = {}
        engine = load(MODEL)
        try:
            quantize(engine, 8)
            model = engine.agent.model
            model.update(tree_map(lambda x: mx.zeros_like(x), model.parameters()))
            mx.eval(model.parameters())
            probe = items_by["sst2-choice"][:20]
            s = next(x for x in suites if x.name == "sst2-choice")
            try:
                zeroed = [engine.predict(engine.clip_state({"text": t}, PROMPT_TOKEN_BUDGET),
                                         {"q": s.question})["q"] for t, _ in probe]
                zeroed_differs = agreement(s, q8["scores"][s.name]["answers"][:20], zeroed,
                                           None)["max_abs_prob"] > 0
            except FloatingPointError:
                zeroed_differs = "raised FloatingPointError"
            model.load_weights(q8["saved_path"], strict=True)
            mx.eval(model.parameters())
            s_scores = run_suites(engine, [s], {s.name: items_by[s.name]})[s.name]["answers"]
            ref = q8["scores"][s.name]["answers"]
            proto = {"loaded": True,
                     "zeroed_differs": zeroed_differs,
                     "matches_in_session_q8": agreement(s, ref, s_scores, None)}
        except Exception as exc:  # the report is the point; keep the error text
            proto = {"loaded": False, "error": f"{type(exc).__name__}: {exc}"}
        model = None  # drop the reference, or the next load stacks a second model
        report["prototype"] = proto
        print(f"  {proto}", flush=True)
        unload(engine)

        inside8 = all(baseline[k]["ci"][0] <= q8["scores"][k]["value"] <= baseline[k]["ci"][1]
                      for k in baseline)
        report["q8_inside_all"] = inside8
        if inside8:
            engine = load(MODEL)
            _ = engine.agent
            report["q4_layers"] = quantize(engine, 4)
            report["q4_after_quantize_memory"] = memory()
            report["q4"] = phase("q4", engine, suites, items_by, states, question, tmp)
            unload(engine)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    rows = {}
    for s in suites:
        k = s.name
        row = {"metric": baseline[k]["metric"], "ci": baseline[k]["ci"],
               "v0.7.1": baseline[k]["value"], "fp16": fp16["scores"][k]["value"]}
        cut = baseline[k].get("fit", {}).get("cut")
        for arm in ("q8", "q4"):
            if arm in report:
                v = report[arm]["scores"][k]["value"]
                row[arm] = v
                row[f"{arm}_inside"] = baseline[k]["ci"][0] <= v <= baseline[k]["ci"][1]
                row[f"{arm}_agreement"] = agreement(
                    s, fp16["scores"][k]["answers"], report[arm]["scores"][k]["answers"], cut)
        rows[k] = row
    for arm in ("fp16", "q8", "q4"):
        if arm in report:
            for r in report[arm]["scores"].values():
                r.pop("answers", None)
                r.pop("fit", None)
    report["suites"] = rows
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
