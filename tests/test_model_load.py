"""The saved joblib resolves locally, then from the DVC remote cache."""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import pytest

from resolution_classifier import model as model_module


def _bundle_bytes() -> bytes:
    buffer = io.BytesIO()
    joblib.dump({"model": "marker", "encoder": "enc"}, buffer)
    return buffer.getvalue()


def _isolate_model_lookup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(model_module, "_session", None)
    monkeypatch.setattr(model_module, "MODEL_PATH", tmp_path / "absent.joblib")
    monkeypatch.setenv("RESOLUTION_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("RESOLUTION_MODEL", raising=False)
    monkeypatch.delenv("RESOLUTION_REV", raising=False)
    monkeypatch.delenv("RESOLUTION_REPO", raising=False)


def test_missing_named_model_stays_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fail(repo: str, rev: str | None) -> str:
        raise AssertionError((repo, rev))

    monkeypatch.setattr(model_module, "_remote_md5", fail)
    with pytest.raises(FileNotFoundError, match="resolution model not found"):
        model_module.load_model(tmp_path / "missing.joblib")


def test_local_artifact_skips_the_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_path = tmp_path / "xgb_classifier.joblib"
    model_path.write_bytes(_bundle_bytes())
    monkeypatch.setattr(model_module, "_session", None)
    monkeypatch.setattr(model_module, "MODEL_PATH", model_path)
    monkeypatch.delenv("RESOLUTION_MODEL", raising=False)

    def fail(repo: str, rev: str | None) -> str:
        raise AssertionError((repo, rev))

    monkeypatch.setattr(model_module, "_remote_md5", fail)
    bundle = model_module.load_model()
    assert bundle["model"] == "marker"
    assert bundle["encoder"] == "enc"
    scorer_model, scorer_encoder = model_module.load_scorer()
    assert scorer_model == "marker"
    assert scorer_encoder == "enc"


def test_cache_miss_reads_with_dvc_api_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import dvc.api

    _isolate_model_lookup(monkeypatch, tmp_path)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    payload = _bundle_bytes()
    calls = {"open": 0}

    def md5(repo: str, rev: str | None) -> str:
        assert repo == model_module._MODEL_REPO
        assert rev is None
        return "abc123"

    def fake_open(path: str, **kwargs: object) -> io.BytesIO:
        calls["open"] += 1
        assert path == model_module._MODEL_RELATIVE
        assert kwargs["repo"] == model_module._MODEL_REPO
        assert kwargs["remote"] == "neon"
        assert kwargs["mode"] == "rb"
        return io.BytesIO(payload)

    monkeypatch.setattr(model_module, "_remote_md5", md5)
    monkeypatch.setattr(model_module, "_remote_url", lambda name: "s3://bucket/prefix")
    monkeypatch.setattr(model_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(dvc.api, "open", fake_open)
    bundle = model_module.load_model()
    again = model_module.load_model()
    assert bundle["model"] == "marker"
    assert again["encoder"] == "enc"
    assert calls["open"] == 1
    assert (tmp_path / "cache" / "abc123.joblib").is_file()


def test_s3_cache_miss_requires_aws_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _isolate_model_lookup(monkeypatch, tmp_path)
    monkeypatch.setattr(model_module, "_remote_md5", lambda repo, rev: "abc123")
    monkeypatch.setattr(model_module, "_remote_url", lambda name: "s3://bucket/prefix")
    monkeypatch.setattr(model_module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    def fail(*args: object, **kwargs: object) -> bytes:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(model_module, "_read_remote_model", fail)
    with pytest.raises(RuntimeError, match="AWS_ACCESS_KEY_ID"):
        model_module.load_model()
