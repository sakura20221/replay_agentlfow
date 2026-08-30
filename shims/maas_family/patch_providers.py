#!/usr/bin/env python3
"""Make the MaAS-family provider package importable without every vendor SDK.

`provider/__init__.py` eagerly imports thirteen LLM backends. The bake-off only
ever uses the OpenAI-compatible one (all traffic goes through the local proxy),
but a single unavailable or broken vendor package -- sparkai does not import on
Python 3.12 -- makes the entire repo unimportable. Rewriting the import block to
skip backends that are not installed avoids pulling in a dozen unused SDKs while
leaving the providers we do use, and every method's own logic, untouched.

Idempotent; safe to re-run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARKER = "# --- shared-layer shim: tolerant provider imports ---"

TARGETS = {
    "maas": ROOT / "third_party" / "maas" / "maas" / "provider" / "__init__.py",
    "daao": ROOT / "third_party" / "daao" / "daao" / "provider" / "__init__.py",
}

IMPORT_RE = re.compile(r"^from\s+(\w+)\.provider\.(\w+)\s+import\s+(\w+)\s*$", re.MULTILINE)

problems: list[str] = []


def patch(label: str, path: Path) -> None:
    if not path.exists():
        problems.append(f"{label}: {path} missing")
        print(f"  [FAIL] {label}: file missing")
        return

    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"  [ok] {label}: already patched")
        return

    pairs = [(m.group(2), m.group(3)) for m in IMPORT_RE.finditer(text)]
    package = next((m.group(1) for m in IMPORT_RE.finditer(text)), None)
    if not pairs or package is None:
        problems.append(f"{label}: no provider imports found")
        print(f"  [FAIL] {label}: no provider imports found")
        return

    block = "\n".join(
        [
            MARKER,
            "import importlib as _importlib",
            "",
            f"_PROVIDERS = {pairs!r}",
            "",
            "for _module_name, _class_name in _PROVIDERS:",
            "    try:",
            f'        _module = _importlib.import_module(f"{package}.provider.{{_module_name}}")',
            "        globals()[_class_name] = getattr(_module, _class_name)",
            "    except Exception:",
            "        # Backend not installed or not importable on this Python; the",
            "        # bake-off does not use it, so skip rather than fail the package.",
            "        pass",
            "",
            "__all__ = [_class_name for _, _class_name in _PROVIDERS if _class_name in globals()]",
        ]
    )

    # Drop the original eager imports and the hand-written __all__ that follows.
    text_without_imports = IMPORT_RE.sub("", text)
    text_without_all = re.sub(r"__all__\s*=\s*\[[^\]]*\]", "", text_without_imports, count=1, flags=re.DOTALL)
    path.write_text(text_without_all.rstrip() + "\n\n" + block + "\n", encoding="utf-8")
    print(f"  [ok] {label}: {len(pairs)} providers made optional")


def main() -> None:
    print("patching provider packages")
    for label, path in TARGETS.items():
        patch(label, path)
    if problems:
        for item in problems:
            print("  -", item)
        sys.exit(1)
    print("provider patch OK")


if __name__ == "__main__":
    main()
