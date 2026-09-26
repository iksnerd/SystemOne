"""One place that ties a domain to its generator, question bank and teacher preamble."""
from __future__ import annotations

from dataclasses import dataclass

from .gen.domains import EMOTION, ISSUES, ISSUES_SHAPES, MESSAGE_TYPING, Domain
from .label.bank import BANK, EMOTION_BANK, ISSUE_BANK

EMOTION_PREAMBLE = (
    "You label one short social media post. For each question, give your honest probability distribution "
    "over its options, reflecting real uncertainty (do not force 0 or 1 unless you are certain). "
    "Probabilities for a question must sum to 1."
)

ISSUE_PREAMBLE = (
    "You label one GitHub issue. For each question, give your honest probability distribution over "
    "its options, reflecting real uncertainty (do not force 0 or 1 unless you are certain). A "
    "question that mentions an error is still a question if the writer asks how or why rather than "
    "reporting a defect. Probabilities for a question must sum to 1."
)


@dataclass(frozen=True)
class DomainSpec:
    domain: Domain
    bank: dict
    preamble: str | None


DOMAINS = {
    "hub_message_typing": DomainSpec(MESSAGE_TYPING, BANK, None),
    "emotion": DomainSpec(EMOTION, EMOTION_BANK, EMOTION_PREAMBLE),
    "github_issue_typing": DomainSpec(ISSUES, ISSUE_BANK, ISSUE_PREAMBLE),
    "github_issue_shapes": DomainSpec(ISSUES_SHAPES, ISSUE_BANK, ISSUE_PREAMBLE),
}


def get_domain(name: str) -> DomainSpec:
    return DOMAINS[name]
