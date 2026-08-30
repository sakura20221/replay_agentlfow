#!/usr/bin/env python3
"""Install exactly the packages a vendored repo needs, by following ImportErrors.

The author repos ship fully pinned freezes of their whole machine -- MaAS's is
269 lines including django, boto3 and torch==2.1.0+cu118 -- so installing them
wholesale drags in conflicts and an old CUDA build for packages the experiment
never touches. Instead this imports the target module, installs whatever the
resulting ImportError names, and repeats until the import succeeds.

    python resolve_imports.py --python envs/maas/bin/python \
        --cwd third_party/maas --module maas.ext.maas.benchmark.shared_shim
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Import names that differ from their distribution name.
MODULE_TO_PACKAGE = {
    "docx": "python-docx",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python-headless",
    "fitz": "pymupdf",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "pptx": "python-pptx",
    "tree_sitter": "tree-sitter",
    "tree_sitter_python": "tree-sitter-python",
    "attr": "attrs",
    "jwt": "pyjwt",
    "OpenSSL": "pyopenssl",
    "pkg_resources": "setuptools",
    "google": "google-generativeai",
    "qdrant_client": "qdrant-client",
    "gymnasium": "gymnasium",
    "faiss": "faiss-cpu",
    "pydantic_core": "pydantic",
    "git": "GitPython",
    "libcst": "libcst",
    "nbformat": "nbformat",
    "nbclient": "nbclient",
    "typer": "typer",
    "rich": "rich",
    "websockets": "websockets",
    "httpx": "httpx",
    "jinja2": "Jinja2",
    "semantic_version": "semantic-version",
    "gitdb": "gitdb",
    "ta": "ta",
    "curl_cffi": "curl-cffi",
    "socksio": "socksio",
    "zhipuai": "zhipuai",
    "regex": "regex",
    "tiktoken": "tiktoken",
    "networkx": "networkx",
    "aiofiles": "aiofiles",
    "loguru": "loguru",
    "tenacity": "tenacity",
    "openai": "openai",
    "anthropic": "anthropic",
    "torch_geometric": "torch_geometric",
    "sentence_transformers": "sentence-transformers",
    "transformers": "transformers",
    "datasets": "datasets",
}

MISSING_MODULE_RE = re.compile(r"No module named '([A-Za-z0-9_\.]+)'")


def run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--uv", default="envs/tools/bin/uv")
    parser.add_argument("--max-rounds", type=int, default=60)
    args = parser.parse_args()

    root = Path.cwd()
    # Deliberately not .resolve(): a venv's bin/python is a symlink to the base
    # interpreter, and following it makes uv target the uv-managed installation
    # (which refuses writes as "externally managed") and probes imports with the
    # wrong interpreter entirely.
    python = str(root / args.python)
    uv = str(root / args.uv)
    cwd = root / args.cwd

    installed: list[str] = []
    failed: list[str] = []
    attempted: set[str] = set()

    for round_number in range(1, args.max_rounds + 1):
        probe = run([python, "-c", f"import {args.module}; print('IMPORT_OK')"], cwd=cwd)
        if "IMPORT_OK" in probe.stdout:
            print(f"\nimport succeeded after {round_number - 1} installs")
            print("installed:", " ".join(installed) if installed else "(nothing)")
            if failed:
                print("could not install:", " ".join(failed))
            return

        text = probe.stderr
        match = MISSING_MODULE_RE.search(text)
        if not match:
            print(f"\nround {round_number}: import failed for a reason other than a missing module")
            print(text[-1500:])
            if installed:
                print("\ninstalled so far:", " ".join(installed))
            sys.exit(2)

        module = match.group(1)
        top = module.split(".")[0]
        package = MODULE_TO_PACKAGE.get(module) or MODULE_TO_PACKAGE.get(top) or top
        if package in failed:
            print(f"\nround {round_number}: {package} already failed to install; stopping")
            print(text[-800:])
            sys.exit(3)
        if module in attempted:
            # The package installed fine yet the module is still missing (e.g.
            # sparkai installs but sparkai.core does not import on 3.12). Retrying
            # is an infinite loop, so stop and let the caller decide -- usually by
            # making that import optional rather than satisfying it.
            print(f"\nround {round_number}: installing {package} did not make {module!r} importable; stopping")
            print(text[-800:])
            sys.exit(5)
        attempted.add(module)

        print(f"round {round_number}: missing {module!r} -> installing {package}", flush=True)
        result = run([uv, "pip", "install", "--python", python, "--quiet", package], cwd=root)
        if result.returncode != 0:
            print(f"  install failed: {result.stderr.strip()[-300:]}")
            failed.append(package)
        else:
            installed.append(package)

    print("\nhit max rounds; installed:", " ".join(installed))
    sys.exit(4)


if __name__ == "__main__":
    main()
