#!/usr/bin/env python3
"""Run one command and persist wall time plus child-process resource usage."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    completed = subprocess.run(command, check=False)
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {
        "command": command,
        "exit_code": completed.returncode,
        "real_seconds": elapsed,
        "user_seconds": after.ru_utime - before.ru_utime,
        "sys_seconds": after.ru_stime - before.ru_stime,
        "max_rss_kib": after.ru_maxrss,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
