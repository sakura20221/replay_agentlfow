"""Let one job override the proxy URL, so accounting reaches (phase, method, dataset).

The proxy derives its accounting namespace from the URL path, and every repo reads
that URL from a file it keeps in its own checkout -- one file per repo, shared by
all of that method's jobs. So `/train/daao/v1` was as fine-grained as the numbers
could get: DAAO's four datasets and its search and test phases all landed in one
bucket, and "which cell cost what" and "which cell has stopped making requests"
were both unanswerable.

Two mechanisms, because the repos split into two kinds:

* G-Designer, CARD and MasRouter read `os.getenv("BASE_URL")` after
  `load_dotenv()`, and python-dotenv does not override variables that already
  exist in the environment. So sweep.py exporting BASE_URL is enough -- no patch,
  and nothing in those repos changes.
* MaAS, DAAO, AFlow and FlowBank read a YAML config through a MetaGPT-style
  loader with no environment override. Each has exactly one
  `create_llm_instance(config)`, which is the single place every provider is
  built, so honouring `SHIM_BASE_URL` there covers every call path -- including
  the optimizer's own LLM, which is a different config object from the executor's.

The override is opt-in: with SHIM_BASE_URL unset, the config file wins and
behaviour is exactly as before.
"""

from __future__ import annotations

from pathlib import Path

MARKER = "# --- shared-layer shim (agent_wf_v2) --- namespace override v1"

OVERRIDE_BLOCK = '''
    # --- shared-layer shim (agent_wf_v2) --- namespace override v1
    # The proxy reads its accounting namespace off the URL path, and only the
    # launching process knows which dataset and phase this job is. See
    # shims/namespace_patch.py.
    import os as _shim_ns_os

    _shim_ns_url = _shim_ns_os.getenv("SHIM_BASE_URL")
    if _shim_ns_url:
        try:
            {config}.base_url = _shim_ns_url
        except Exception:  # noqa: BLE001 - a frozen config must not stop the run
            pass
'''


def patch_file(path: Path, function: str, config_arg: str) -> str:
    """Insert the override at the top of `function`'s body. Idempotent."""
    if not path.exists():
        return "missing"
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return "already"

    needle = f"def {function}("
    start = text.find(needle)
    if start == -1:
        return "anchor-missing"
    # Insert after the signature line and after a docstring if there is one, so the
    # docstring stays the first statement.
    cursor = text.index("\n", start) + 1
    remainder = text[cursor:]
    stripped = remainder.lstrip()
    if stripped.startswith(('"""', "'''")):
        quote = stripped[:3]
        offset = len(remainder) - len(stripped)
        closing = remainder.find(quote, offset + 3)
        if closing != -1:
            cursor += remainder.index("\n", closing) + 1

    block = OVERRIDE_BLOCK.format(config=config_arg)
    text = text[:cursor] + block + text[cursor:]
    path.write_text(text, encoding="utf-8")
    return "patched"


def verify_file(path: Path) -> list[str]:
    if not path.exists():
        return [f"{path.name}: missing"]
    text = path.read_text(encoding="utf-8")
    if MARKER not in text:
        return [f"{path.name}: namespace override not installed"]
    return []
