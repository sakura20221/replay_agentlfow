#!/usr/bin/env python3
"""Verify frozen dataset row counts and exact-byte hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "shared" / "data"


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest["file_integrity"]["files"]
    failures = []
    for name, record in expected.items():
        path = DATA / name
        if not path.is_file():
            failures.append(f"{name}: missing")
            continue
        rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ok = rows == record["rows"] and digest == record["sha256"]
        print(f"[{'ok' if ok else 'FAIL'}] {name}: rows={rows} sha256={digest[:16]}")
        if not ok:
            failures.append(
                f"{name}: expected rows={record['rows']} sha256={record['sha256']}"
            )
    extras = sorted(path.name for path in DATA.glob("*.jsonl") if path.name not in expected)
    if extras:
        failures.append("unlisted JSONL files: " + ", ".join(extras))
    if failures:
        raise SystemExit("data verification failed:\n  " + "\n  ".join(failures))
    print(f"verified {len(expected)} frozen data files")


if __name__ == "__main__":
    main()
