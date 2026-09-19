"""Execute a benchmark command, preserving complete logs and an exit record."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required after --")
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"command": command, "started_at": datetime.datetime.now().astimezone().isoformat(),
              "status": "running", "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result_path = args.output / "command.json"
    result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    started = time.perf_counter()
    code = 1
    try:
        with (args.output / "stdout.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       encoding="utf-8", errors="replace", bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        record.update(status="passed" if code == 0 else "failed", exit_code=code)
    except Exception as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        record.update(wall_time_sec=time.perf_counter()-started,
                      finished_at=datetime.datetime.now().astimezone().isoformat())
        result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
