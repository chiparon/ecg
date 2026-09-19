"""Namespace-confined I/O; shared metadata only, never shared analysis caches."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "overnight_supplements_v2"
VENDOR = OUT / "shared/vendor"
if VENDOR.is_dir() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_json(path):
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def file_info(path, schema=None, source=None):
    path = resolve(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    result = {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    if schema is not None:
        result["schema"] = schema
    if source is not None:
        result["source"] = source
    return result


def write_json(path, value):
    path = resolve(path)
    path.resolve().relative_to(OUT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(value, training_invocations=0, inference_invocations=0)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat()


def config():
    return read_json(OUT / "freeze/run_config.json")
