"""Consolidate pre-consumption identities and independently rehash protected inputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import zipfile

from overnight_supplements_v2.shared.common import ROOT, OUT, config, file_info, now, read_json, resolve, write_json
import numpy as np


def numpy_header(handle):
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        raise ValueError(f"Unsupported frozen numpy format: {version}")
    return {"shape": list(shape), "dtype": str(dtype), "fortran_order": bool(fortran)}


def schema(path):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        with path.open("rb") as handle:
            return {"format": "npy", **numpy_header(handle)}
    if suffix == ".npz":
        fields = {}
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith(".npy"):
                    raise ValueError(f"Unexpected NPZ member: {name}")
                with archive.open(name) as handle:
                    fields[name[:-4]] = numpy_header(handle)
        return {"format": "npz", "fields": fields, "allow_pickle": False}
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            columns = next(csv.reader(handle))
        return {"format": "csv", "columns": columns, "numeric_parsing": "Analysis uses unrounded serialized values; raw CSV has no embedded dtype schema"}
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {"format": "json", "root_type": type(payload).__name__, "top_level_fields": {key: type(value).__name__ for key, value in payload.items()} if isinstance(payload, dict) else None}
    if suffix in (".pt", ".pth"):
        return {"format": "opaque checkpoint bytes", "loaded": False, "inspection": "SHA-256 and size only; no model or tensor deserialization"}
    return {"format": suffix.lstrip("."), "interpretation": "Immutable source/configuration/document text or opaque metadata; no model execution"}


def consolidate():
    old = read_json(OUT / "freeze/source_manifest.json")
    producers = [read_json(OUT / "did/sources.json"), read_json(OUT / "snrp/sources.json")]
    third = read_json(OUT / "third_architecture/gate0_manifest.json")
    original = old.get("inputs", [])
    all_sources = original + [entry for document in producers for entry in document["sources"]] + third["sources"] + third["public_sources"]
    grouped = {}
    for entry in all_sources:
        path = entry["path"]
        if path.startswith("overnight_supplements_v2/"):
            continue  # New analysis artifacts are versioned separately, not historical inputs.
        source = entry.get("source", "Historical analysis source")
        origins = {str(value) for value in source} if isinstance(source, list) else {str(source)}
        if path in grouped:
            previous = grouped[path]
            if (previous["bytes"], previous["sha256"]) != (entry["bytes"], entry["sha256"]):
                raise ValueError(f"Input identity conflict across analysis slices: {path}")
            previous["sources"].update(origins)
        else:
            grouped[path] = dict(entry, sources=origins)
    entries = []
    for path, entry in sorted(grouped.items()):
        origins = sorted(entry.pop("sources"))
        if path.startswith("https://"):
            entry["source"] = origins
            entry["verification"] = "Public immutable-commit bytes fetched and hashed by static inventory; not refetched during local protection scan"
        else:
            actual = file_info(path)
            if any(actual[key] != entry[key] for key in ("bytes", "sha256")):
                raise ValueError(f"Protected input changed since pre-consumption snapshot: {path}")
            entry = dict(actual, schema=schema(resolve(path)), source=origins)
        entries.append(entry)
    manifest = {
        "status": "sealed", "initial_frozen_at": old.get("initial_frozen_at", old.get("frozen_at")), "sealed_at": now(),
        "baseline_commit": config()["baseline_commit"],
        "freeze_policy": "All producer sources were hashed before derivation; consolidated identity collisions are rejected and every local historical source is rehashed at seal.",
        "source_indexes": [file_info(OUT / "did/sources.json"), file_info(OUT / "snrp/sources.json")],
        "n_sources": len(entries), "n_local_sources": sum(not e["path"].startswith("https://") for e in entries),
        "inputs": entries,
    }
    write_json(OUT / "freeze/source_manifest.json", manifest)
    code = []
    for directory in ("freeze", "shared", "did", "snrp", "qa", "reports", "third_architecture"):
        for path in sorted((OUT / directory).glob("*.py")):
            code.append(file_info(path))
    write_json(OUT / "freeze/code_manifest.json", {"status": "frozen", "recorded_at": now(), "baseline_commit": config()["baseline_commit"], "scripts": code, "run_config": file_info(OUT / "freeze/run_config.json")})
    return {"status": "sealed", "n_sources": len(entries), "n_scripts": len(code), "source_manifest": file_info(OUT / "freeze/source_manifest.json")}


def verify():
    started = time.perf_counter()
    manifest = read_json(OUT / "freeze/source_manifest.json")
    if manifest["status"] != "sealed":
        raise ValueError("Source manifest must be sealed before final verification")
    failures = []
    checked = 0
    for entry in manifest["inputs"]:
        if entry["path"].startswith("https://"):
            continue
        actual = file_info(entry["path"])
        if any(actual[key] != entry[key] for key in ("bytes", "sha256")):
            failures.append({"expected": entry, "actual": actual})
        checked += 1
    code = read_json(OUT / "freeze/code_manifest.json")
    for entry in code["scripts"] + [code["run_config"]]:
        actual = file_info(entry["path"])
        if any(actual[key] != entry[key] for key in ("bytes", "sha256")):
            failures.append({"expected": entry, "actual": actual})
    result = {"status": "passed" if not failures else "failed", "verified_at": now(), "source_manifest": file_info(OUT / "freeze/source_manifest.json"), "historical_local_sources_checked": checked, "frozen_code_and_config_checked": len(code["scripts"]) + 1, "failures": failures, "elapsed_seconds": time.perf_counter() - started, "scope": "All consumed historical source bytes plus every frozen new script/config; not a claim of hashing every unconsumed file on disk"}
    result["code_manifest"] = file_info(OUT / "freeze/code_manifest.json")
    write_json(OUT / "qa/source_protection.json", result)
    if failures:
        raise ValueError("Source protection failed; affected results cannot be released")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.seal == args.verify:
        parser.error("Choose exactly one of --seal or --verify")
    result = consolidate() if args.seal else verify()
    print(json.dumps(dict(result, training_invocations=0, inference_invocations=0), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
