"""Crash-safe files. A long labeling run must survive being killed: rows are appended and
fsynced one at a time, a torn last line is dropped on reopen, and whole-file writes go through
a temp file and rename so a reader never sees half a JSON document."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class JsonlStore:
    def __init__(self, path: str | Path, key: Callable[[dict], str]):
        self.path, self.key = Path(path), key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict] = []
        if self.path.exists():
            self._load()
        self.done: set[str] = {key(r) for r in self._rows}

    def _load(self) -> None:
        raw = self.path.read_text()
        lines = raw.split("\n")
        torn = not raw.endswith("\n")
        good = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                good.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines) - 1 and not (torn and i == len(lines) - 1):
                    raise
        self._rows = good
        if torn:  # rewrite without the partial tail so the next append starts on a fresh line
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in good))
            os.replace(tmp, self.path)

    def append(self, row: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._rows.append(row)
        self.done.add(self.key(row))

    def rows(self) -> list[dict]:
        return list(self._rows)
