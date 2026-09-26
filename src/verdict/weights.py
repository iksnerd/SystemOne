"""The fine-tuned checkpoint, fetched from its private Hugging Face repo.

The weights (843 MB) are too large for git or a wheel, and they stay private: the fine-tune was
trained on Gemini teacher labels, and Google's terms bar using the service to build competing
models, a call made in favour of not redistributing them. They live in the private model repo
`iksnerd/verdict-v1-mlx`, pinned here by revision, and `model.safetensors` is checked by sha256
before anything is put in place. Access needs a Hugging Face token that can read the repo.

Until 0.9.0 they were a GitHub release asset (`weights-verdict-v1`), fetched with `gh`; that moved
here before the code repo went public.

Nothing here runs on a verdict: `verdict weights`, `init` and `update` call it, all maintenance.
"""
from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Callable

REPO = "iksnerd/verdict-v1-mlx"
REVISION = "9a1e2d3e1c1f721c9312d126c5efe4ee596f147c"
#: sha256 of `model.safetensors` at REVISION.
SHA256 = "564ce35bc55a5abf5b143790df966164c468ac4b324778ab528d380bf1c23ec4"
DIRNAME = "verdict-v1-mlx"
#: The checkpoint, as laya-mlx loads it. The repo's README and .gitattributes are not part of it.
FILES = ("model.safetensors", "mlx_config.json", "rl_agent_config.json", "encoder/config.json",
         "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")
#: Where a `uv tool` install keeps its weights (CLAUDE.md), and where a fetch lands by default.
DATA_DIR = Path.home() / ".local" / "share" / "verdict" / "models"


def target_for(model_path: str) -> Path:
    """Where to put the checkpoint: a configured absolute path, else the data dir. The built-in
    default is relative to the current directory, which is no place for 843 MB."""
    p = Path(model_path).expanduser()
    if p.is_absolute():
        return p
    if model_path.count("/") == 1 and not model_path.startswith(("models/", ".")):
        raise ValueError(f"model.path is the Hub id {model_path!r}; there is nothing to fetch")
    return DATA_DIR / DIRNAME


def present(target: Path) -> bool:
    return (target / "model.safetensors").is_file()


def _snapshot_download(repo_id: str, **kwargs) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, **kwargs)


def hf_download(dest: Path) -> Path:
    """The pinned revision of the private repo into `dest`. The token comes from $HF_TOKEN or a
    prior `hf auth login`, as huggingface_hub reads it."""
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        _snapshot_download(REPO, revision=REVISION, local_dir=str(dest),
                           allow_patterns=list(FILES))
    except (RepositoryNotFoundError, GatedRepoError) as exc:
        # Signed out, the Hub reports a private repo as not found, which reads as if the weights
        # did not exist.
        raise ValueError(f"cannot read {REPO}: the weights are private. Set HF_TOKEN to a token "
                         f"with access, or run `hf auth login`") from exc
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(target: Path, *, download: Callable[[Path], Path] | None = None,
          sha256: str = SHA256) -> str:
    """Download and check the checkpoint, then move it to `target`. It lands beside the target and
    moves only when complete, so a failure leaves nothing half-written."""
    if present(target):
        return f"{target} already has the weights"
    download = download or hf_download
    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        got_dir = work / DIRNAME
        got_dir.mkdir()
        download(got_dir)
        missing = [f for f in FILES if not (got_dir / f).is_file()]
        if missing:
            raise ValueError(f"{REPO} at {REVISION[:8]} is missing {', '.join(missing)}")
        got = _sha256(got_dir / "model.safetensors")
        if got != sha256:
            raise ValueError(f"model.safetensors sha256 is {got}, expected {sha256}; not installed")
        shutil.rmtree(got_dir / ".cache", ignore_errors=True)  # the Hub's bookkeeping
        if target.is_symlink():
            target.unlink()
        got_dir.rename(target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return f"fetched {REPO}@{REVISION[:8]} into {target}"
