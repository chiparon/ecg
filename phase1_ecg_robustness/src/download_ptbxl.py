"""Restart-safe, SHA256-verified downloads from official PhysioNet endpoints.

Run ``python -m src.download_ptbxl --help`` for all options. Auto transport first
tries a seekable HTTP Range ZIP reader; it downloads only blocks containing
requested members, not the full 100/500 Hz archive. Direct fallback uses one
persistent requests.Session per worker and bounded concurrency.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PTB_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
ZIP_URL = "https://physionet.org/content/ptb-xl/get-zip/1.0.3/"
NST_URL = "https://physionet.org/files/nstdb/1.0.0/"
_local = threading.local()


def now():
    return datetime.now(timezone.utc).isoformat()


def session():
    if not hasattr(_local, "session"):
        s = requests.Session()
        retry = Retry(
            total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
        )
        s.mount(
            "https://",
            HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2),
        )
        s.headers.update(
            {"User-Agent": "phase1-ecg-robustness/1.0", "Accept-Encoding": "identity"}
        )
        _local.session = s
    return _local.session


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def checksums(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+)$", line)
        if match:
            result[match[2].removeprefix("./")] = match[1].lower()
    if not result:
        raise ValueError(f"No SHA256 entries in {path}")
    return result


def inventory(path, relative, url, expected, transport, cached=False):
    actual = sha256(path)
    if expected and actual != expected:
        raise ValueError(f"SHA256 mismatch for {relative}: {actual} != {expected}")
    return {
        "path": str(path),
        "relative_path": relative,
        "url": url,
        "bytes": Path(path).stat().st_size,
        "sha256": actual,
        "expected_sha256": expected,
        "verified": bool(expected),
        "verification": (
            "official SHA256 match"
            if expected
            else "local SHA256 only; no published checksum"
        ),
        "transport": transport,
        "cached": cached,
        "checked_at": now(),
    }


def download_file(root, relative, base_url, expected=None, url=None):
    target = Path(root) / relative
    url = url or base_url + relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if (
        target.is_file()
        and target.stat().st_size
        and (not expected or sha256(target) == expected)
    ):
        return inventory(target, relative, url, expected, "existing", cached=True)
    temporary = target.with_name(target.name + ".part")
    # Re-download incomplete small members. Completed files are retained on every restart.
    with session().get(url, stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        with temporary.open("wb") as out:
            for block in response.iter_content(1024 * 1024):
                out.write(block)
    row = inventory(temporary, relative, url, expected, "direct")
    os.replace(temporary, target)
    row["path"] = str(target)
    return row


class RangeZipReader(io.RawIOBase):
    """Seekable block cache with bounded prefetch for requested ZIP members."""

    def __init__(self, url, block_size=8 * 1024 * 1024):
        super().__init__()
        self.url, self.block_size = url, block_size
        self.position, self.block_start, self.block = 0, -1, b""
        self.requests, self.bytes_received = 0, 0
        self.pool, self.pending, self.queued = None, {}, iter(())
        self._stats_lock = threading.Lock()
        with session().get(
            url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(20, 180)
        ) as r:
            if r.status_code != 206:
                raise OSError(
                    f"ZIP endpoint does not honor HTTP Range (status {r.status_code})"
                )
            match = re.fullmatch(r"bytes 0-0/(\d+)", r.headers.get("Content-Range", ""))
            if not match:
                raise OSError("ZIP endpoint returned invalid Content-Range")
            self.size = int(match[1])
            self.resolved_url = r.url
            self.etag = r.headers.get("ETag")
            if len(r.content) != 1:
                raise OSError("ZIP Range probe length mismatch")
            self.requests += 1
            self.bytes_received += 1

    def _fetch_block(self, start):
        end = min(start + self.block_size, self.size) - 1
        headers = {"Range": f"bytes={start}-{end}"}
        if self.etag:
            headers["If-Range"] = self.etag
        with session().get(
            self.resolved_url, headers=headers, stream=True, timeout=(20, 180)
        ) as r:
            if (
                r.status_code != 206
                or r.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}"
            ):
                raise OSError("ZIP Range response changed or was ignored")
            data = r.content
        if len(data) != end - start + 1:
            raise OSError("Truncated ZIP range")
        with self._stats_lock:
            self.requests += 1
            self.bytes_received += len(data)
        return data

    def prefetch(self, members, workers=6):
        starts = set()
        for member in members:
            start = member.header_offset // self.block_size * self.block_size
            end = (
                member.header_offset
                + 30
                + len(member.filename.encode())
                + len(member.extra)
                + member.compress_size
                + 1024
            )
            starts.update(
                range(
                    start,
                    min(end, self.size - 1) // self.block_size * self.block_size + 1,
                    self.block_size,
                )
            )
        self.queued = iter(sorted(starts))
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.window = workers
        self._fill_window()

    def _fill_window(self):
        while len(self.pending) < self.window:
            start = next(self.queued, None)
            if start is None:
                break
            if start == self.block_start:
                continue
            self.pending[start] = self.pool.submit(self._fetch_block, start)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        super().close()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = (
            offset
            if whence == 0
            else self.position + offset if whence == 1 else self.size + offset
        )
        if position < 0:
            raise ValueError("Negative ZIP seek")
        self.position = position
        return position

    def read(self, size=-1):
        remaining = (
            max(0, self.size - self.position)
            if size < 0
            else min(size, max(0, self.size - self.position))
        )
        chunks = []
        while remaining:
            start = (self.position // self.block_size) * self.block_size
            if self.block_start != start:
                if start in self.pending:
                    self.block = self.pending.pop(start).result()
                    self._fill_window()
                else:
                    self.block = self._fetch_block(start)
                self.block_start = start
            offset = self.position - start
            count = min(remaining, len(self.block) - offset)
            chunks.append(self.block[offset : offset + count])
            self.position += count
            remaining -= count
        return b"".join(chunks)


def extract_ranges(root, paths, hashes, record, manifest):
    with RangeZipReader(ZIP_URL) as remote, zipfile.ZipFile(remote) as archive:
        members = {}
        wanted = set(paths)
        for member in archive.infolist():
            # Official ZIP has a dataset/version top directory. Match only exact requested suffixes.
            parts = member.filename.split("/")
            for n in range(len(parts)):
                candidate = "/".join(parts[n:])
                if candidate in wanted:
                    if candidate in members:
                        raise ValueError(f"Ambiguous ZIP member {candidate}")
                    members[candidate] = member
                    break
        missing = wanted - members.keys()
        if missing:
            raise ValueError(
                f"Official ZIP lacks requested records: {sorted(missing)[:5]}"
            )
        missing_members = [
            members[p]
            for p in paths
            if not (root / p).is_file() or sha256(root / p) != hashes[p]
        ]
        remote.prefetch(missing_members)
        for relative in sorted(paths, key=lambda p: members[p].header_offset):
            target = root / relative
            expected = hashes[relative]
            if target.is_file() and sha256(target) == expected:
                record(
                    inventory(
                        target, relative, PTB_URL + relative, expected, "existing", True
                    )
                )
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            with archive.open(members[relative]) as source, temporary.open("wb") as out:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    out.write(block)
            row = inventory(
                temporary, relative, PTB_URL + relative, expected, "zip-range"
            )
            row.update(
                {"archive_url": ZIP_URL, "archive_member": members[relative].filename}
            )
            os.replace(temporary, target)
            row["path"] = str(target)
            record(row)
        remote.close()
        manifest["zip_range"] = {
            "url": ZIP_URL,
            "resolved_url": remote.resolved_url,
            "archive_bytes": remote.size,
            "http_requests": remote.requests,
            "transferred_bytes": remote.bytes_received,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/ptb-xl-1.0.3"),
        help="Existing/new directory containing ptbxl_database.csv and records100/",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="First N rows of official metadata for smoke; omitted means ALL records",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Direct-download keepalive workers (default: 6)",
    )
    parser.add_argument(
        "--transport",
        choices=["auto", "zip-range", "direct"],
        default="auto",
        help="Auto tries selective ZIP HTTP Range before direct fallback",
    )
    parser.add_argument(
        "--nstdb",
        action="store_true",
        help="Also fetch NSTDB 1.0.0 bw/ma/em .hea/.dat and license/checksums",
    )
    parser.add_argument("--nstdb-dir", type=Path, default=Path("data/raw/nstdb-1.0.0"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/download_manifest.json"),
        help="Summary JSON; sibling .inventory.jsonl records every checked file",
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    inv_path = args.manifest.with_suffix(".inventory.jsonl")
    manifest = {
        "started_at": now(),
        "status": "running",
        "ptbxl_version": "1.0.3",
        "ptbxl_url": PTB_URL,
        "raw_dir": str(args.raw_dir.resolve()),
        "requested_limit": args.limit,
        "transport_requested": args.transport,
        "inventory": str(inv_path),
        "file_count": 0,
        "bytes": 0,
        "verified_file_count": 0,
        "errors": [],
    }
    write_json(args.manifest, manifest)
    # Append inventory across retries/runs for an auditable history; summary is this invocation.
    with inv_path.open("a", encoding="utf-8") as log:
        seen = {}

        def record(row):
            row["run_started_at"] = manifest["started_at"]
            log.write(json.dumps(row) + "\n")
            log.flush()
            previous = seen.get(row["path"])
            if previous is not None:
                manifest["bytes"] -= previous["bytes"]
                manifest["verified_file_count"] -= int(previous["verified"])
            else:
                manifest["file_count"] += 1
            seen[row["path"]] = row
            manifest["bytes"] += row["bytes"]
            manifest["verified_file_count"] += int(row["verified"])
            if manifest["file_count"] % 100 == 0:
                write_json(args.manifest, manifest)
                print(f"Checked {manifest['file_count']} files", flush=True)

        try:
            record(download_file(args.raw_dir, "SHA256SUMS.txt", PTB_URL))
            hashes = checksums(args.raw_dir / "SHA256SUMS.txt")
            for relative in ["ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt"]:
                record(
                    download_file(args.raw_dir, relative, PTB_URL, hashes.get(relative))
                )
            with (args.raw_dir / "ptbxl_database.csv").open(
                newline="", encoding="utf-8"
            ) as f:
                rows = list(csv.DictReader(f))
            manifest["metadata_records"] = len(rows)
            selected = rows[: args.limit] if args.limit else rows
            paths = [
                row["filename_lr"] + ext for row in selected for ext in (".hea", ".dat")
            ]
            manifest["requested_records"] = len(selected)
            manifest["requested_ecg_ids"] = [int(row["ecg_id"]) for row in selected]
            absent_hashes = set(paths) - hashes.keys()
            if absent_hashes:
                raise ValueError(
                    f"Official checksums missing {len(absent_hashes)} requested waveforms"
                )
            write_json(args.manifest, manifest)
            direct = args.transport == "direct"
            if not direct:
                try:
                    extract_ranges(args.raw_dir, paths, hashes, record, manifest)
                    manifest["transport_used"] = "zip-range"
                except Exception as error:
                    manifest["errors"].append(
                        {"at": now(), "stage": "zip-range", "error": repr(error)}
                    )
                    if args.transport == "zip-range":
                        raise
                    direct = True
            if direct:
                manifest["transport_used"] = "direct"
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = {
                        pool.submit(
                            download_file, args.raw_dir, p, PTB_URL, hashes[p]
                        ): p
                        for p in paths
                    }
                    failed = []
                    for future in as_completed(futures):
                        try:
                            record(future.result())
                        except Exception as error:
                            failed.append(futures[future])
                            manifest["errors"].append(
                                {
                                    "at": now(),
                                    "file": futures[future],
                                    "error": repr(error),
                                }
                            )
                    if failed:
                        raise RuntimeError(
                            f"Failed {len(failed)} files; rerun to reuse completed verified files"
                        )
            if args.nstdb:
                manifest["nstdb"] = {
                    "version": "1.0.0",
                    "url": NST_URL,
                    "raw_dir": str(args.nstdb_dir.resolve()),
                    "records": ["bw", "ma", "em"],
                    "interpretation": "structured replay approximation, not original 12-lead electrode recordings",
                }
                record(download_file(args.nstdb_dir, "SHA256SUMS.txt", NST_URL))
                noise_hashes = checksums(args.nstdb_dir / "SHA256SUMS.txt")
                for name in [
                    kind + ext
                    for kind in ("bw", "ma", "em")
                    for ext in (".hea", ".dat")
                ]:
                    record(
                        download_file(args.nstdb_dir, name, NST_URL, noise_hashes[name])
                    )
                record(
                    download_file(
                        args.nstdb_dir,
                        "LICENSE.html",
                        NST_URL,
                        url="https://physionet.org/content/nstdb/view-license/1.0.0/",
                    )
                )
            manifest["status"] = "complete"
        except BaseException as error:
            manifest["status"] = "failed"
            manifest["errors"].append(
                {"at": now(), "stage": "download", "error": repr(error)}
            )
            raise
        finally:
            manifest["finished_at"] = now()
            write_json(args.manifest, manifest)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("status", "file_count", "bytes", "verified_file_count")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
