"""`python -m verdict.cli` is a second entry point beside the installed `verdict` command
(pyproject's `verdict = "verdict.cli:main"`), and needs its own `__main__.py` now that
`verdict.cli` is a package rather than a single module."""
from __future__ import annotations

import subprocess
import sys


def test_python_dash_m_verdict_cli_runs_main():
    result = subprocess.run(
        [sys.executable, "-m", "verdict.cli", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Usage: verdict" in result.stdout
