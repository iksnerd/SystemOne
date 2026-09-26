"""Does rewording fix the holes §41 found, before any new training data? A free check.

Same samples as `eval_gap_tasks.py`. For each task, the question as §41 asked it and alternatives
that each test one idea: option descriptions that say what the failing class looks like, one
no/yes choice per class with the highest P(yes) winning (one-vs-rest), and for issues the title
alone, in case the templates in issue bodies drown the question. Reports accuracy and the
failing class's recall (GitHub questions, commissive acts, `chore` commits). FINDINGS §42.

    uv run --with pyarrow taskpolicy -b python scripts/eval_gap_wording.py MODEL [MODEL ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_gap_tasks import (COMMIT_Q, COMMIT_TYPES, DIALOG_Q, ISSUE_Q,  # noqa: E402
                            accuracy, ask, commits, dialog, issues)

from verdict.engine import load  # noqa: E402

ISSUE_RICH = {"type": {
    "type": "choice", "instructions": "Is `issue` a bug report, a feature request, or a question?",
    "criteria": {
        "bug": "a bug report: something is broken, crashes, errors or behaves wrongly",
        "feature": "a feature request: asks for something new or for a change to how it works",
        "question": "a question: the writer asks how to do something, why something happens, or "
                    "for help, rather than reporting a defect"}}}
ISSUE_OVR = {
    "bug": "Does `issue` report that something is broken, crashes or behaves wrongly?",
    "feature": "Does `issue` ask for a new feature or a change to how something works?",
    "question": "Does `issue` ask how to do something, why something happens, or for help?",
}

DIALOG_RICH = {"act": {"type": "choice", "instructions": "What does the speaker do in `text`?",
                       "criteria": {
    "inform": "tells or states something",
    "question": "asks a question",
    "directive": "asks, tells or suggests that the listener do something (please..., let's..., "
                 "you should..., why don't you...)",
    "commissive": "commits to something: promises, offers, agrees, accepts or refuses (sure, "
                  "I will, no thanks, of course)"}}}
DIALOG_OVR = {
    "inform": "Does `text` state or tell something?",
    "question": "Does `text` ask a question?",
    "directive": "Does `text` ask, tell or suggest that the listener do something?",
    "commissive": "Does `text` promise, offer, agree to, accept or refuse something?",
}

COMMIT_RICH = {"type": {"type": "choice", "instructions": "What kind of change does `subject` describe?",
                        "criteria": {**COMMIT_TYPES,
    "chore": "maintenance with no code behaviour change: bumping dependencies or versions, "
             "releases, build or tooling config, formatting, housekeeping",
    "fix": "fixes a bug: something was broken or wrong and now works"}}}
COMMIT_OVR = {k: f"Does `subject` describe a change that {v}?" for k, v in COMMIT_RICH["type"]["criteria"].items()}


def one_vs_rest(questions: dict[str, str]) -> dict:
    return {k: {"type": "choice", "instructions": v, "criteria": {"no": "", "yes": ""}}
            for k, v in questions.items()}


def argmax_yes(answer: dict) -> str:
    return max(answer, key=lambda k: answer[k]["probabilities"]["yes"])


def run(engine, items, variants, field):
    gold = [g for _, g in items]
    out = {}
    for name, (bank, pick, transform) in variants.items():
        answers = ask(engine, [(transform(s), g) for s, g in items], bank)
        out[name] = accuracy([pick(a) for a in answers], gold)
        print(f"  {field} {name}: {json.dumps(out[name])}", file=sys.stderr)
    return out


def title_only(state):
    return {"issue": state["issue"].split("\n", 1)[0]}


def main() -> None:
    same = lambda s: s  # noqa: E731
    first = lambda key: (lambda a: a[key]["choice"])  # noqa: E731
    tasks = {
        "issues": (issues(), {
            "as_41": (ISSUE_Q, first("type"), same),
            "rich": (ISSUE_RICH, first("type"), same),
            "one_vs_rest": (one_vs_rest(ISSUE_OVR), argmax_yes, same),
            "rich_title_only": (ISSUE_RICH, first("type"), title_only),
            "one_vs_rest_title_only": (one_vs_rest(ISSUE_OVR), argmax_yes, title_only),
        }),
        "dialog": (dialog(), {
            "as_41": ({"act": DIALOG_Q["act"]}, first("act"), same),
            "rich": (DIALOG_RICH, first("act"), same),
            "one_vs_rest": (one_vs_rest(DIALOG_OVR), argmax_yes, same),
        }),
        "commits": (commits(), {
            "as_41": (COMMIT_Q, first("type"), same),
            "rich": (COMMIT_RICH, first("type"), same),
            "one_vs_rest": (one_vs_rest(COMMIT_OVR), argmax_yes, same),
        }),
    }
    results = {}
    for model in sys.argv[1:]:
        engine = load(model)
        print(model, file=sys.stderr)
        results[model] = {t: run(engine, items, variants, t) for t, (items, variants) in tasks.items()}
        del engine
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
