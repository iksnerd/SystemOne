"""Could a small decoder, scored the way a Jev-style model can be, fill Laya's holes? Zero-shot.

Jev appears to be a post-trained language model (its tokenizer looks like Qwen3.5's), and such a
model need not generate: it reads the state, the question and lettered options once, and each
option's probability is the next-token probability of its letter. This scores a small Qwen3.5 that
way on the same three real sets as FINDINGS §41, with no training, and times each forward pass on
this Mac. Chat template with thinking off; no tokens are generated. FINDINGS §43.

    uv run --with mlx-lm --with pyarrow taskpolicy -b python scripts/eval_decoder_zeroshot.py \\
        mlx-community/Qwen3.5-2B-MLX-4bit
"""
from __future__ import annotations

import json
import string
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_gap_tasks import (ACTS, COMMIT_TYPES, INSTRUCTION, ISSUE_Q, accuracy,  # noqa: E402
                            commits, dialog, issues)
from eval_noul_lean import summarize  # noqa: E402

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402

TASKS = {
    "issues": ("What does this GitHub issue report or ask for?", ISSUE_Q["type"]["criteria"], "issue"),
    "commits": ("What kind of change does this commit message describe?", COMMIT_TYPES, "subject"),
    "dialog": ("What does the speaker do in this utterance?", ACTS, "text"),
}


class Scorer:
    def __init__(self, model_id: str):
        self.model, self.tok = load(model_id)
        self.times: list[float] = []

    def letter_ids(self, n: int) -> list[int]:
        return [self.tok.encode(c, add_special_tokens=False)[0] for c in string.ascii_uppercase[:n]]

    def probs(self, text: str, question: str, options: dict[str, str]) -> dict[str, float]:
        keys = list(options)
        lines = "\n".join(f"{string.ascii_uppercase[i]}. {k}: {options[k]}" for i, k in enumerate(keys))
        prompt = (f"Text:\n{text}\n\nQuestion: {question}\n\nOptions:\n{lines}\n\n"
                  "Reply with the letter of the best option only.")
        messages = [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           enable_thinking=False, tokenize=True)
        t = time.perf_counter()
        logits = self.model(mx.array([ids]))[0, -1]
        chosen = logits[mx.array(self.letter_ids(len(keys)))]
        p = mx.softmax(chosen.astype(mx.float32)).tolist()
        self.times.append((time.perf_counter() - t) * 1000)
        return dict(zip(keys, p))


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3.5-2B-MLX-4bit"
    scorer = Scorer(model_id)
    data = {"issues": issues(), "commits": commits(), "dialog": dialog()}
    results = {}
    for name, (question, options, field) in TASKS.items():
        items = data[name]
        rows = []
        for state, _ in items:
            rows.append(scorer.probs(state[field], question, options))
            time.sleep(0.05)
        gold = [g for _, g in items]
        results[name] = accuracy([max(r, key=r.get) for r in rows], gold)
        print(model_id, name, json.dumps(results[name]), file=sys.stderr)
    # The library's is_instruction wording, as a yes/no, on the dialogue utterances.
    yes = [scorer.probs(s["text"], INSTRUCTION, {"yes": "yes", "no": "no"})["yes"]
           for s, _ in data["dialog"]]
    results["dialog_instruction"] = summarize(yes, [g == "directive" for _, g in data["dialog"]])
    print(model_id, "dialog_instruction", json.dumps(results["dialog_instruction"]), file=sys.stderr)
    ts = sorted(scorer.times)
    results["latency_ms"] = {"p50": round(ts[len(ts) // 2], 1), "p95": round(ts[int(len(ts) * 0.95)], 1),
                             "calls": len(ts)}
    print(json.dumps({model_id: results}, indent=2))


if __name__ == "__main__":
    main()
