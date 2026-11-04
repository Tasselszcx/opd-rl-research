"""Small, dependency-free helpers for durable experiment state."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

UPSTREAM_COMMIT = "37371a4c31ad7947746200d234161769191f4748"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path | str) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_digest(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value: Any, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def project_path(value: str, repository_root: Path = REPOSITORY_ROOT) -> Path:
    result = Path(os.path.expanduser(value))
    return (repository_root / result).resolve() if not result.is_absolute() else result.resolve()
