"""Load the saved same-event pair classifier.

A local joblib is used when one is already on disk. Otherwise the latest fit is
read from the DVC remote in ``.dvc/config`` with ``dvc.api.open`` and cached by
md5. The ``neon`` remote is ``s3://`` and needs ``AWS_ACCESS_KEY_ID`` and
``AWS_SECRET_ACCESS_KEY`` (also the Neon ``endpointurl`` already in ``.dvc/config``).

The bundle is a joblib dict with keys ``model`` (Platt-calibrated classifier)
and ``encoder`` (the fitted :class:`~sklearn.compose.ColumnTransformer`).
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
from dotenv import load_dotenv

from resolution_classifier.paths import MODEL_PATH, ROOT

_MODEL_REPO = "https://github.com/Sidel-Meril/xgb-resolution-classifier-smbo.git"
_MODEL_REMOTE = "neon"
_MODEL_RELATIVE = "models/pairs_same_event/xgb_classifier.joblib"
_session: tuple[str, str | None, Path] | None = None


def _env(name: str) -> str:
    return os.getenv(f"RESOLUTION_{name}", "").strip()


def _cache_dir() -> Path:
    override = _env("CACHE")
    if override:
        return Path(override)
    local = os.getenv("LOCALAPPDATA", "").strip()
    if local:
        return Path(local) / "resolution-classifier"
    xdg = os.getenv("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "resolution-classifier"


def _repo_and_rev() -> tuple[str, str | None]:
    repo = _env("REPO") or _MODEL_REPO
    rev = _env("REV") or None
    return repo, rev


def _checksum(info: dict[str, object]) -> str:
    sources: list[dict[str, object]] = [info]
    nested = info.get("dvc_info")
    if isinstance(nested, dict):
        sources.append(nested)
    for source in sources:
        raw = source.get("md5")
        if not isinstance(raw, str) or not raw.strip():
            raw = source.get("checksum")
        if isinstance(raw, str) and raw.strip():
            text = raw.strip()
            if text.startswith("md5:"):
                text = text[4:]
            return text.split(".", 1)[0]
    raise RuntimeError("DVC model info has no md5 checksum")


def _clone_error(repo: str) -> RuntimeError:
    return RuntimeError(
        f"could not clone {repo}. Authenticate to GitHub so this process can read it "
        "(gh auth login, or a credential helper for https://github.com)."
    )


def _remote_url(name: str) -> str:
    """URL of one remote in ``.dvc/config``."""
    config = (ROOT / ".dvc" / "config").read_text(encoding="utf-8")
    headers = {f"['remote \"{name}\"']", f'[remote "{name}"]'}
    in_section = False
    for line in config.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped in headers
            continue
        if not in_section:
            continue
        key, sep, raw = stripped.partition("=")
        if sep and key.strip() == "url":
            value = raw.strip().strip('"').strip("'")
            if value:
                return value
    raise RuntimeError(f"DVC remote {name} has no url in .dvc/config")


def _remote_md5(repo: str, rev: str | None) -> str:
    from dvc.api import DVCFileSystem
    from dvc.scm import CloneError

    os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")
    filesystem = DVCFileSystem(repo, rev=rev) if rev else DVCFileSystem(repo)
    try:
        info = filesystem.info(_MODEL_RELATIVE)
    except CloneError as exc:
        raise _clone_error(repo) from exc
    return _checksum(info)


def _read_remote_model(repo: str, rev: str | None) -> bytes:
    import dvc.api
    from dvc.scm import CloneError

    os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")
    kwargs: dict[str, object] = {"repo": repo, "remote": _MODEL_REMOTE, "mode": "rb"}
    if rev:
        kwargs["rev"] = rev
    try:
        with dvc.api.open(_MODEL_RELATIVE, **kwargs) as source:
            data = source.read()
    except CloneError as exc:
        raise _clone_error(repo) from exc
    if not data:
        raise RuntimeError(f"DVC remote returned an empty {_MODEL_RELATIVE}")
    return data


def _require_object_storage_credentials() -> None:
    url = _remote_url(_MODEL_REMOTE)
    if not url.startswith("s3://"):
        return
    load_dotenv(ROOT / ".env")
    key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    secret = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    if key and secret:
        return
    raise RuntimeError(
        "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required to download "
        f"the model from the {_MODEL_REMOTE} DVC remote."
    )


def fetch_cached_model() -> Path:
    """Md5-cached joblib. Download with ``dvc.api.open`` when the hash is new."""
    global _session
    repo, rev = _repo_and_rev()
    cached = _session
    if cached is not None and cached[0] == repo and cached[1] == rev and cached[2].is_file():
        return cached[2]
    digest = _remote_md5(repo, rev)
    dest = _cache_dir() / f"{digest}.joblib"
    if not dest.is_file():
        _require_object_storage_credentials()
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_suffix(".joblib.partial")
        try:
            partial.write_bytes(_read_remote_model(repo, rev))
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"could not download {_MODEL_RELATIVE} from the {_MODEL_REMOTE} DVC remote"
            ) from exc
        partial.replace(dest)
    _session = (repo, rev, dest)
    return dest


def ensure_model_file(path: Path | str | None = None) -> Path:
    """Named file, env file, local artifact, then the DVC md5 cache."""
    if path is not None and str(path).strip():
        chosen: Path | None = Path(path)
    else:
        override = _env("MODEL")
        chosen = Path(override) if override else None
    if chosen is not None:
        if not chosen.is_file():
            raise FileNotFoundError(f"resolution model not found: {chosen}")
        return chosen
    if MODEL_PATH.is_file():
        return MODEL_PATH
    return fetch_cached_model()


def _register_saved_encoder() -> None:
    """The training notebook ran from ``__main__``; register the custom encoder there."""
    import sys

    from resolution_classifier.pair_encoder import DayOfYearEncoder

    sys.modules["__main__"].DayOfYearEncoder = DayOfYearEncoder


def load_model(path: Path | str | None = None) -> dict[str, object]:
    """Load the saved joblib. Keys are ``model`` and ``encoder``."""
    _register_saved_encoder()
    bundle = joblib.load(ensure_model_file(path))
    if not isinstance(bundle, dict) or "model" not in bundle or "encoder" not in bundle:
        raise TypeError(
            "resolution model must be a joblib dict with model and encoder, "
            f"got {type(bundle).__name__}"
        )
    return bundle


def load_scorer(path: Path | str | None = None) -> tuple[object, object]:
    """Classifier and the encoder it was fit with, for the scoring pipeline."""
    bundle = load_model(path)
    return bundle["model"], bundle["encoder"]
