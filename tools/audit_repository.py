#!/usr/bin/env python3
"""Fail closed on common private-repository packaging mistakes."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


MAX_BYTES = 25 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".bin", ".ckpt", ".joblib", ".jsonl", ".ndjson", ".npy", ".npz",
    ".onnx", ".pickle", ".pkl", ".pt", ".pth", ".safetensors",
    ".tar", ".tgz", ".zip",
}
FORBIDDEN_NAMES = {
    ".env", "id_rsa", "id_ed25519", "credentials", "credentials.json",
}
TEXT_SUFFIXES = {
    ".cfg", ".csv", ".ini", ".json", ".md", ".mjs", ".py", ".sh",
    ".toml", ".tsv", ".txt", ".yaml", ".yml",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "generic assigned secret": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|password|passwd|secret)\b"
        r"\s*[:=]\s*['\"][^'\"\n]{8,}['\"]"
    ),
}


def iter_files(root: Path):
    for path in root.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or ".cache" in path.parts
            or "__pycache__" in path.parts
        ):
            continue
        yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    problems: list[str] = []
    checked = 0

    for path in iter_files(root):
        checked += 1
        rel = path.relative_to(root)
        lower_name = path.name.lower()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden artifact: {rel}")
        if lower_name in FORBIDDEN_NAMES or lower_name.startswith(".env."):
            problems.append(f"credential-like filename: {rel}")
        if path.stat().st_size > MAX_BYTES:
            problems.append(f"file larger than 25 MiB: {rel}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"Makefile", "LICENSE"}:
            continue
        # The audit source contains the detection expressions themselves.
        if rel == Path("tools/audit_repository.py"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"possible {label}: {rel}")

    if problems:
        print("Repository audit failed:")
        for problem in sorted(set(problems)):
            print(f"- {problem}")
        return 1
    print(f"Repository audit passed: {checked} files checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

