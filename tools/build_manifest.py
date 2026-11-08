#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest for the shareable repository tree."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--output", default="MANIFEST.sha256")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = root / args.output
    rows = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or ".git" in path.parts
            or ".cache" in path.parts
            or "__pycache__" in path.parts
            or path == output
        ):
            continue
        rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {output} with {len(rows)} entries")


if __name__ == "__main__":
    main()
