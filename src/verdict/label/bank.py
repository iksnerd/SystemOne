"""The question bank for the engineering-room ledger domain, in the wire format of
verdict.schema (laya's). `kind` is the one exclusive choice; the noul questions are
properties that can hold together, which is the shape Laya is strongest at."""
from ..gen.domains import EMOTION_TYPES, TYPES

SIX = ["action", "synthesis", "decision", "thought", "draft", "note"]

BANK = {
    "kind": {
        "type": "choice",
        "instructions": "What kind of ledger entry is `message`?",
        "criteria": {t: TYPES[t] for t in SIX} | {"other": "none of the other kinds fits"},
    },
    "records_decision": {"type": "noul", "instructions": "Does `message` record a choice that was made?"},
    "reports_shipped_work": {"type": "noul", "instructions": "Does `message` report work that was completed or shipped?"},
    "leaves_open_question": {"type": "noul", "instructions": "Does `message` leave a question or problem unresolved?"},
    "is_proposal": {"type": "noul", "instructions": "Is `message` a proposal put forward for feedback or approval?"},
}


EMOTION_BANK = {
    "emotion": {
        "type": "choice",
        "instructions": "Which emotion does the writer of `text` express?",
        "criteria": {e: f"the writer {d}" for e, d in EMOTION_TYPES.items()},
    },
}
