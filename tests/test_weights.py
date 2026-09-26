"""`verdict weights`: fetch the fine-tuned checkpoint from its private Hugging Face repo, pinned by
revision and checked by sha256 before it is put in place. No network here: the download is a
function that writes the files we choose."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from verdict import cli, config, weights

WEIGHTS = b"weights"
GOOD = {"model.safetensors": WEIGHTS, "mlx_config.json": b"{}", "rl_agent_config.json": b"{}",
        "encoder/config.json": b"{}", "tokenizer/tokenizer.json": b"{}",
        "tokenizer/tokenizer_config.json": b"{}"}
SHA = hashlib.sha256(WEIGHTS).hexdigest()


def downloader(files=GOOD, extra_cache=True):
    """A stand-in for the Hub download: writes the checkpoint into `dest`, as snapshot_download
    does with local_dir, including its `.cache` bookkeeping folder."""
    def download(dest: Path) -> Path:
        for name, data in files.items():
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            (dest / name).write_bytes(data)
        if extra_cache:
            (dest / ".cache" / "huggingface").mkdir(parents=True)
        return dest
    return download


def test_fetch_checks_and_places_the_checkpoint(tmp_path):
    target = tmp_path / "models" / "verdict-v1-mlx"
    msg = weights.fetch(target, download=downloader(), sha256=SHA)
    assert (target / "model.safetensors").read_bytes() == WEIGHTS
    assert (target / "tokenizer" / "tokenizer.json").exists()
    assert not (target / ".cache").exists(), "the Hub's bookkeeping is not part of the checkpoint"
    assert "verdict-v1-mlx" in msg
    assert [p.name for p in target.parent.iterdir()] == ["verdict-v1-mlx"], "no temp dirs left"


def test_a_wrong_checksum_leaves_nothing_behind(tmp_path):
    target = tmp_path / "models" / "verdict-v1-mlx"
    with pytest.raises(ValueError, match="sha256"):
        weights.fetch(target, download=downloader(), sha256="0" * 64)
    assert not target.exists()
    assert list(target.parent.iterdir()) == []


def test_a_missing_file_is_refused(tmp_path):
    target = tmp_path / "models" / "verdict-v1-mlx"
    partial = {k: v for k, v in GOOD.items() if k != "tokenizer/tokenizer.json"}
    with pytest.raises(ValueError, match="tokenizer/tokenizer.json"):
        weights.fetch(target, download=downloader(partial), sha256=SHA)
    assert not target.exists()


def test_an_existing_checkpoint_is_not_downloaded_again(tmp_path):
    target = tmp_path / "verdict-v1-mlx"
    target.mkdir()
    (target / "model.safetensors").write_bytes(b"mine")

    def download(dest):
        raise AssertionError("must not download")
    assert "already" in weights.fetch(target, download=download)
    assert (target / "model.safetensors").read_bytes() == b"mine"


def test_the_download_is_the_pinned_revision_of_the_private_repo(monkeypatch, tmp_path):
    seen = {}

    def snapshot(repo_id, **kw):
        seen.update(repo_id=repo_id, **kw)
        return kw["local_dir"]
    monkeypatch.setattr(weights, "_snapshot_download", snapshot)
    weights.hf_download(tmp_path)
    assert seen["repo_id"] == weights.REPO
    assert seen["revision"] == weights.REVISION and len(weights.REVISION) == 40
    assert set(seen["allow_patterns"]) == set(weights.FILES)


def test_no_access_says_how_to_get_it(monkeypatch, tmp_path):
    """Signed out, the Hub reports a private repo as not found, which reads as if the weights
    did not exist. The error says what to do instead."""
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError

    def snapshot(repo_id, **kw):
        request = httpx.Request("GET", f"https://huggingface.co/api/models/{repo_id}")
        raise RepositoryNotFoundError("404 Client Error. Repository Not Found",
                                      response=httpx.Response(404, request=request))
    monkeypatch.setattr(weights, "_snapshot_download", snapshot)
    with pytest.raises(ValueError, match="HF_TOKEN") as err:
        weights.hf_download(tmp_path)
    assert weights.REPO in str(err.value)


@pytest.mark.parametrize("configured,expected", [
    ("/opt/ckpt/verdict-v1-mlx", Path("/opt/ckpt/verdict-v1-mlx")),
    ("models/verdict-v1-mlx", None),  # the data dir, read at run time (the fixture moves it)
])
def test_the_target_is_the_configured_absolute_path_or_the_data_dir(configured, expected):
    assert weights.target_for(configured) == (expected or weights.DATA_DIR / "verdict-v1-mlx")


def test_a_hub_id_has_nothing_to_fetch():
    with pytest.raises(ValueError, match="Hub"):
        weights.target_for("aac6fef/laya-mlx")


def test_the_resolver_finds_fetched_weights_for_the_built_in_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(weights, "DATA_DIR", tmp_path / "data")
    (tmp_path / "data" / "verdict-v1-mlx").mkdir(parents=True)
    model, warning = config.resolve_model("models/verdict-v1-mlx")
    assert model == str(tmp_path / "data" / "verdict-v1-mlx") and warning is None


def test_the_fallback_warning_says_how_to_fetch(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(weights, "DATA_DIR", tmp_path / "data")
    _, warning = config.resolve_model("/nowhere/verdict-v1-mlx")
    assert "verdict weights" in warning


def test_weights_check_reports_without_downloading(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(weights, "DATA_DIR", tmp_path)
    monkeypatch.setenv("VERDICT_MODEL", "models/verdict-v1-mlx")
    assert cli.main(["weights", "--check"]) == 1
    assert "missing" in capsys.readouterr().out
