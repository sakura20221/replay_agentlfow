#!/usr/bin/env python3
"""Clone pinned author repositories and apply the local reproduction shims.

The author repositories are deliberately not vendored into this Git repository.
Their exact revisions live in upstreams.lock.json; the shims are the reviewable
description of every local source modification.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "upstreams.lock.json"
THIRD_PARTY = ROOT / "third_party"
INSTALLERS = (
    "shims/maas_family/install.py",
    "shims/aflow/install.py",
    "shims/diverseflow/install.py",
    "shims/gdesigner_family/install.py",
    "shims/masrouter/install.py",
)


def run(command: list[str], *, cwd: Path = ROOT, capture: bool = False) -> str:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )
    return completed.stdout.strip() if capture else ""


def git(path: Path, *args: str, capture: bool = False) -> str:
    return run(["git", "-C", str(path), *args], capture=capture)


def entries() -> list[dict]:
    data = json.loads(LOCK.read_text(encoding="utf-8"))
    return data["repositories"]


def clone_one(entry: dict) -> None:
    target = THIRD_PARTY / entry["name"]
    expected = entry["commit"]
    if target.exists():
        if not (target / ".git").is_dir():
            raise SystemExit(f"refusing to replace non-Git path: {target}")
        origin = git(target, "remote", "get-url", "origin", capture=True)
        head = git(target, "rev-parse", "HEAD", capture=True)
        dirty = git(target, "status", "--porcelain", "--untracked-files=no", capture=True)
        if origin != entry["url"]:
            raise SystemExit(f"{target}: origin is {origin!r}, expected {entry['url']!r}")
        if head != expected:
            detail = " and has local changes" if dirty else ""
            raise SystemExit(
                f"{target}: HEAD is {head}, expected {expected}{detail}; "
                "move it aside and rerun"
            )
        print(f"[ok] {entry['name']} at {expected[:12]}")
        return

    target.mkdir(parents=True)
    try:
        run(["git", "init", "--quiet", str(target)])
        git(target, "remote", "add", "origin", entry["url"])
        excluded = entry.get("sparse_exclude", [])
        if excluded:
            git(target, "config", "core.sparseCheckout", "true")
            git(target, "config", "core.sparseCheckoutCone", "false")
            sparse_file = target / ".git" / "info" / "sparse-checkout"
            patterns = ["/*", *[f"!/{path.strip('/')}/" for path in excluded]]
            sparse_file.write_text("\n".join(patterns) + "\n", encoding="utf-8")
        # Blob filtering plus sparse checkout prevents DAAO's bundled 2+ GB copy
        # of MiniLM from being downloaded; dl_model.sh supplies one shared copy.
        git(target, "fetch", "--depth", "1", "--filter=blob:none", "origin", expected)
        git(target, "checkout", "--quiet", "--detach", "FETCH_HEAD")
        head = git(target, "rev-parse", "HEAD", capture=True)
        if head != expected:
            raise RuntimeError(f"resolved {head}, expected {expected}")
        print(f"[ok] cloned {entry['name']} at {expected[:12]}")
    except Exception:
        print(f"clone failed; partial directory retained for inspection: {target}", file=sys.stderr)
        raise


def verify_commits() -> None:
    failures = []
    for entry in entries():
        target = THIRD_PARTY / entry["name"]
        if not (target / ".git").is_dir():
            failures.append(f"{entry['name']}: missing")
            continue
        head = git(target, "rev-parse", "HEAD", capture=True)
        if head != entry["commit"]:
            failures.append(f"{entry['name']}: {head} != {entry['commit']}")
        else:
            print(f"[ok] {entry['name']} {head[:12]}")
    if failures:
        raise SystemExit("upstream verification failed:\n  " + "\n  ".join(failures))


def apply_shims(check_only: bool) -> None:
    flag = ["--check"] if check_only else []
    for installer in INSTALLERS:
        run([sys.executable, installer, *flag])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clone-only", action="store_true", help="do not apply shims")
    parser.add_argument("--check", action="store_true", help="only verify commits and installed shims")
    args = parser.parse_args()

    if args.check:
        verify_commits()
        apply_shims(check_only=True)
        return

    THIRD_PARTY.mkdir(exist_ok=True)
    for entry in entries():
        clone_one(entry)
    verify_commits()
    if not args.clone_only:
        apply_shims(check_only=False)
        apply_shims(check_only=True)


if __name__ == "__main__":
    main()
