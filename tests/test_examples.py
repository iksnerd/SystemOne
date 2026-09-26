"""The examples are documentation that can rot, so they are tested like code: every bank must
validate against the wire schema, every input must parse, and no question may name a field its
inputs lack. No model loads; this checks shape, not answers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verdict import inputs
from verdict.schema import DecideRequest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
BANKS = sorted(EXAMPLES.glob("*/bank.json"))


def test_there_are_examples():
    assert len(BANKS) >= 3


@pytest.mark.parametrize("bank_path", BANKS, ids=lambda p: p.parent.name)
def test_bank_validates_and_inputs_match_it(bank_path):
    bank = json.loads(bank_path.read_text())
    files = sorted(bank_path.parent.glob("*.jsonl"))
    assert files, f"{bank_path.parent.name} has no inputs"
    for inputs_file in files:
        rows = [json.loads(l) for l in inputs_file.read_text().splitlines() if l.strip()]
        assert rows, f"{inputs_file} is empty"
        for row in rows:
            state = row.get("state", row)
            DecideRequest(state=state, questions=bank)
            assert not inputs.missing_fields(state, bank), (inputs_file.name, state)


def test_every_example_is_listed_in_the_examples_readme():
    readme = (EXAMPLES / "README.md").read_text()
    for bank in BANKS:
        assert f"{bank.parent.name}/" in readme, bank.parent.name


def test_the_documented_install_line_names_the_current_version():
    """The README and the guide pin a release tag in their install line. A release that bumps the
    version without updating them leaves a copy-paste install of the old version, silently."""
    import tomllib

    root = EXAMPLES.parent
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    for doc in ("README.md", "docs/guide.md"):
        text = (root / doc).read_text()
        pins = set(__import__("re").findall(r"verdict\.git@v(\d+\.\d+\.\d+)", text))
        assert pins == {version}, f"{doc} installs {sorted(pins)}, pyproject says {version}"
