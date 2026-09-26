"""Every test sees an empty weights directory and no network for the weights download.

Two `update` tests passed on this Mac and failed in CI: here the fine-tuned weights really sit in
~/.local/share/verdict/models, so `update`'s weights check found them and did nothing; on a clean
runner it tried a real download. A test must not depend on what the machine has
installed, and the suite needs no network (CLAUDE.md), so both are pinned here for every test.

Settings too: with `lang = "multi"` in ~/.config/verdict/config.toml, four language tests failed
in the pre-commit export, and passed in the checkout only because its gitignored verdict.toml
shadowed the user file. No config file and no VERDICT_* variable reaches a test unless it sets one."""

from __future__ import annotations

import os

import pytest

from verdict import config, weights


@pytest.fixture(autouse=True)
def _no_machine_weights(monkeypatch, tmp_path_factory):
    monkeypatch.setattr(weights, "DATA_DIR", tmp_path_factory.mktemp("weights-data"))

    def no_network(*args, **kwargs):
        raise AssertionError(f"a test tried to run {args[0] if args else '?'}; the weights "
                             "download must be faked (pass `download`, or patch weights.fetch)")

    # At the Hub call, not at `hf_download`, so tests of `hf_download` itself still run it.
    monkeypatch.setattr(weights, "_snapshot_download", no_network)


@pytest.fixture(autouse=True)
def _no_machine_config(monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATHS", ())
    for name in [n for n in os.environ if n.startswith("VERDICT_")]:
        monkeypatch.delenv(name)


def typed(questions, noul_answer):
    """A fake server's answers, typed as a real server types them. `noul_answer(qid)` is the
    yes/no answer the fake would give; a question the CLI rewrote as a no/yes choice gets the
    same number back as a choice, so fakes written for yes/no keep their meaning."""
    out = {}
    for qid, q in questions.items():
        a = noul_answer(qid)
        if q.get("type") == "choice" and list(q.get("criteria") or {}) == ["no", "yes"]:
            p = a["noul"]
            a = {"type": "choice", "choice": "yes" if p >= 0.5 else "no", "confidence": a["confidence"],
                 "probabilities": {"no": 1 - p, "yes": p}}
        out[qid] = a
    return out
