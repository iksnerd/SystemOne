"""Trivial cross-cutting helpers shared by every command group: resolved settings, the one place
that prints a usage failure and its exit code, the user config path, and the installed version
string. Nothing here loads a model; see `inference.py` for that."""
from __future__ import annotations

import sys


def _settings():
    """Resolved settings. Loaded inside a command rather than at import: `verdict.config` costs
    7.7 ms and `verdict --help` should not pay it."""
    from .. import config

    return config.load()


def _fail(message: str) -> int:
    print(f"verdict: {message}", file=sys.stderr)
    return 2


def _user_config():
    from ..config import USER_CONFIG

    return USER_CONFIG


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"verdict {version('verdict')}"
    except PackageNotFoundError:
        return "verdict (not installed)"
