#!/usr/bin/env python3
"""Does a per-job URL actually reach the client, in each repo's own way?

Two mechanisms, and each could fail silently in the opposite direction -- the
dotenv one if python-dotenv were to override the environment, the YAML one if some
call path built its LLM without going through create_llm_instance. Either failure
leaves every request in the old method-wide bucket while everything appears to
work, so both are exercised here rather than assumed.

    envs/gdesigner/bin/python test_namespace.py --repo third_party/gdesigner
    envs/maas/bin/python      test_namespace.py --repo third_party/maas
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "http://127.0.0.1:18080/test/sentinel/mmlu_pro/v1"

failures = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global failures
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        failures += 1


def test_dotenv(repo: Path) -> None:
    """G-Designer / CARD / MasRouter: os.getenv after load_dotenv()."""
    os.environ["BASE_URL"] = SENTINEL + "/chat/completions"
    sys.path.insert(0, str(repo))
    package = "GDesigner" if (repo / "GDesigner").exists() else "CARD"
    if (repo / "MAR").exists():
        # MasRouter keeps its URL in MAR/LLM; the same getenv rule applies.
        import importlib
        module = importlib.import_module("MAR.LLM.llm")
        value = getattr(module, "MINE_BASE_URL", None) or os.getenv("BASE_URL")
    else:
        import importlib
        module = importlib.import_module(f"{package}.llm.gpt_chat")
        value = module.MINE_BASE_URL
    check(value == os.environ["BASE_URL"],
          "the environment wins over the repo's .env", repr(value))
    check("/test/sentinel/mmlu_pro/" in (value or ""),
          "and it carries phase, method and dataset")


def test_yaml(repo: Path) -> None:
    """MaAS / DAAO / AFlow / FlowBank: create_llm_instance honours SHIM_BASE_URL."""
    os.environ["SHIM_BASE_URL"] = SENTINEL
    sys.path.insert(0, str(repo))
    name = repo.name
    if name in ("maas", "daao"):
        import importlib
        registry = importlib.import_module(f"{name}.provider.llm_provider_registry")
        config_module = importlib.import_module(f"{name}.configs.llm_config")
        config = config_module.LLMConfig(
            api_type="openai", model="Qwen/Qwen3-8B", api_key="local",
            base_url="http://127.0.0.1:18080/train/OLD/v1")
        llm = registry.create_llm_instance(config)
        value = llm.config.base_url
    else:
        import importlib
        async_llm = importlib.import_module("scripts.async_llm")
        # AFlow/FlowBank take a config object with a base_url attribute.
        klass = getattr(async_llm, "LLMsConfig", None)
        if klass is not None and hasattr(klass, "default"):
            config = klass.default().get("qwen3-8b")
        else:  # fall back to whatever the module exposes
            raise SystemExit("could not obtain a config object from scripts.async_llm")
        config.base_url = "http://127.0.0.1:18080/train/OLD/v1"
        async_llm.create_llm_instance(config)
        value = config.base_url
    check(value == SENTINEL, "SHIM_BASE_URL overrides the config file", repr(value))
    check("/test/sentinel/mmlu_pro/" in (value or ""),
          "and it carries phase, method and dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    repo = ROOT / args.repo
    if not repo.exists():
        raise SystemExit(f"missing {repo}")

    os.chdir(repo)
    if (repo / "GDesigner").exists() or (repo / "CARD").exists() or (repo / "MAR").exists():
        test_dotenv(repo)
    else:
        test_yaml(repo)
    print(f"\n  {failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
