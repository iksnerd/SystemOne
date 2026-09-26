import pytest
from pydantic import ValidationError

from verdict.schema import DecideRequest


def req(questions):
    return DecideRequest.model_validate({"state": "hello", "questions": questions})


def test_wire_format_matches_laya_predict_args():
    r = req(
        {
            "dept": {"type": "choice", "instructions": "Which team?", "criteria": {"a": "x", "b": "y"}},
            "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
            "spam": {"type": "noul", "instructions": "Is it spam?"},
        }
    )
    assert set(r.questions) == {"dept", "urgency", "spam"}


def test_choice_criteria_may_be_a_list():
    r = req({"q": {"type": "choice", "instructions": "?", "criteria": ["a", "b"]}})
    assert r.questions["q"].option_keys() == ["a", "b"]


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "choice", "instructions": "?", "criteria": {"only": "one"}},
        {"type": "score", "instructions": "?", "criteria": ["single"]},
        {"type": "choice", "instructions": "?"},
        {"type": "noul", "instructions": "?", "criteria": ["a", "b"]},
        {"type": "mystery", "instructions": "?"},
    ],
)
def test_rejects_malformed_questions(bad):
    with pytest.raises(ValidationError):
        req({"q": bad})


def test_requires_at_least_one_question():
    with pytest.raises(ValidationError):
        req({})
