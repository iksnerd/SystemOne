"""A bank of typed questions about a state: `verdict decide` and `verdict ask`."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import inference, support
from ..engine import maybe_truncated


def _read_fits(path: str | None) -> dict:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    return data.get("questions", data)


def _decide_cmd(args: argparse.Namespace) -> int:
    """State plus a bank of typed questions in, one answer per question out, in one pass."""
    from .. import calibrate, inputs
    from ..schema import laya_questions

    if args.jsonl and not args.questions:
        return support._fail("--jsonl reads states from stdin, so --questions is required")
    try:
        questions = (inputs.load_questions(args.questions) if args.questions
                     else json.load(sys.stdin))
        questions = laya_questions(questions)
        fits = _read_fits(args.calibration)
    except ValueError as exc:
        return support._fail(str(exc))
    if args.jsonl:
        # A generator, not a list: a live pipe never reaches EOF, and each line is answered as it
        # arrives.
        states = (inputs.parse_state(line) for line in sys.stdin if line.strip())
    elif args.state:
        states = [inputs.read_state(args.state)]
    else:
        return support._fail("give a state, or --questions and --jsonl with states on stdin")

    ask = inference._Asker(args)
    # A long criterion is cut mid-sentence with no error. The character check is free; --check
    # pays about 1.3 s for the tokenizer to count exactly.
    warnings = (inference.load(ask.main_path).option_overflow(questions) if args.check
                else maybe_truncated(questions))
    for line in warnings:
        ask.warn(line)

    # Per yes/no question: count, low, high. Running, not a list: a live pipe never ends.
    seen: dict[str, tuple[int, float, float]] = {}
    for i, state in enumerate(states):
        if i and args.pause:
            time.sleep(args.pause)
        result = ask(state, questions)
        if fits:
            result["answers"] = calibrate.apply(result["answers"], fits)
        for qid, a in result["answers"].items():
            if a.get("type") == "noul":
                n, lo, hi = seen.get(qid, (0, a["noul"], a["noul"]))
                seen[qid] = (n + 1, min(lo, a["noul"]), max(hi, a["noul"]))
        if args.jsonl:
            print(json.dumps({"state": state, **result}), flush=True)
        else:
            print(json.dumps(result, indent=2))
    if args.jsonl:
        for line in _spread_report(seen):
            print(f"verdict: {line}", file=sys.stderr)
    return 0


#: Below this range across a batch, a yes/no question ranks almost nothing. A status triage's
#: `blocked` question scored 0.58 to 0.65 over 137 rooms while it was reading linter
#: boilerplate, and read 0.12 to 0.63 once the inputs were fixed (FINDINGS §30).
NARROW_SPREAD = 0.1
SPREAD_MIN_STATES = 10


def _spread_report(seen: dict[str, tuple[int, float, float]]) -> list[str]:
    """One line per yes/no question: its range over the batch, flagged when too narrow to rank.
    `seen` maps each question to `(count, low, high)`.

    The scores are uncalibrated, so a batch is only useful for its ordering, and an ordering
    squeezed into a few hundredths is noise. Usually the inputs are at fault (all near-identical,
    or all boilerplate), not the model, so the fix is to look at them.
    """
    out = []
    for qid, (n, lo, hi) in seen.items():
        if n < SPREAD_MIN_STATES:
            continue
        line = f"spread {qid}: {lo:.2f} to {hi:.2f} over {n} states"
        if hi - lo < NARROW_SPREAD:
            line += (" -- narrow range, it ranks almost nothing; check the inputs "
                     "(near-identical? boilerplate?) before trusting the order")
        out.append(line)
    return out


def _ask_cmd(args: argparse.Namespace) -> int:
    """One question, inline. Prints one line; `--json` for everything."""
    from .. import calibrate, inputs

    try:
        question = inputs.inline_question(args.question, args.options, args.levels,
                                          args.true, args.false)
        cut, fit = None, None
        if args.cut is not None:
            try:
                cut = float(args.cut)
            except ValueError:
                fits = _read_fits(args.cut)
                fit = fits.get("q") or (next(iter(fits.values())) if len(fits) == 1 else None)
                if fit is None:
                    raise ValueError(f"{args.cut} holds more than one fit and none named 'q'")
                cut = fit.get("cut")
    except (ValueError, OSError) as exc:
        return support._fail(str(exc))

    state = inputs.read_state(args.text)
    result = inference._Asker(args)(state, {"q": question})
    if fit:
        result["answers"] = calibrate.apply(result["answers"], {"q": fit})
    answer = result["answers"]["q"]

    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    if answer["type"] == "choice":
        # The chosen option's probability, calibrated when a fit is given. Not laya's `confidence`,
        # which for a choice is 1 - H(p)/log(k): a 0.50/0.50 split would print "A 0.00".
        p = answer.get("calibrated", answer["probabilities"])[answer["choice"]]
        print(f"{answer['choice']} {p:.2f}")
        return 0
    if answer["type"] == "score":
        top = max(answer["probabilities"], key=answer["probabilities"].get)
        print(f"{answer['score']:.2f}/{len(answer['legend']) - 1} {answer['legend'][top]}")
        return 0
    score = answer["noul"]
    if cut is None:
        print(f"{score:.2f}")
        return 0
    yes = score >= cut
    print(f"{'yes' if yes else 'no'} {score:.2f}")
    return 0 if yes else 1
