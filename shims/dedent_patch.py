PATCH_FUNCTION = '''

def _shim_dedent_code(text):
    """Strip a uniform leading indent the model may have added to a whole reply.

    The upstream sanitize() keeps only top-level ``def``/``class`` statements whose
    name matches the requested entry point. Qwen3-8B, especially when asked to
    revise a previous attempt, frequently emits the whole function indented by two
    spaces; every line is shifted, so the ``def`` is no longer top level and
    sanitize() returns an empty string -- the model's correct code is discarded and
    the item is scored zero with no error anywhere. Measured on the mbpp test split:
    468 of 1618 code replies (28.9%) had an indented first ``def``, and 138 of the
    142 empty predictions had a reply that did define the required function.

    This only removes indentation that is common to every non-blank line, i.e. it
    restores the original column zero and never changes relative structure. Code
    that is already top level is returned unchanged.
    """
    if not text or "def " not in text:
        return text
    import re as _re
    import textwrap as _tw
    body = text
    fence = _re.search(r"```(?:python|py)?\\s*\\n(.*?)```", body, _re.S)
    if fence:
        body = fence.group(1)
    first = _re.search(r"^([ \\t]*)(?:async\\s+def|def|class)\\s+\\w+", body, _re.M)
    if not first or not first.group(1):
        return text
    dedented = _tw.dedent(body)
    check = _re.search(r"^(?:async\\s+def|def|class)\\s+\\w+", dedented, _re.M)
    if not check:
        return text
    return dedented

'''

CALL_OLD = "        extracted_code = sanitize(code=content, entrypoint=function_name)"
CALL_NEW = ("        content = _shim_dedent_code(content)\n"
            "        extracted_code = sanitize(code=content, entrypoint=function_name)")


SANITIZE_FUNCTION = r'''
# --- shared-layer shim (agent_wf_v2) --- mbpp uniform-indent v1
def _shim_dedent_code(text):
    """Remove one common outer indent from a complete code reply only."""
    if not isinstance(text, str) or not text:
        return text
    import re as _re
    import textwrap as _textwrap

    def _fix_body(body):
        first = _re.search(
            r"^([ \t]+)(?:async\s+def|def|class)\s+\w+", body, _re.M
        )
        if not first:
            return body
        prefix = first.group(1)
        lines = body.splitlines(True)
        nonblank = [line for line in lines if line.strip()]
        if not nonblank or any(not line.startswith(prefix) for line in nonblank):
            return body
        fixed = _textwrap.dedent(body)
        if not _re.search(
            r"^(?:async\s+def|def|class)\s+\w+", fixed, _re.M
        ):
            return body
        return fixed

    tick = _re.escape(chr(96))
    fenced = _re.search(
        tick * 3 + r"(?:python|py)?[ \t]*\n(.*?)" + tick * 3,
        text, _re.S | _re.I
    )
    if fenced:
        body = fenced.group(1)
        fixed = _fix_body(body)
        if fixed != body:
            return text[:fenced.start(1)] + fixed + text[fenced.end(1):]
        return text
    return _fix_body(text)

'''

SANITIZE_CALL_OLD = "    code = code_extract(code)"
SANITIZE_CALL_NEW = ("    code = _shim_dedent_code(code)\n"
                     "    code = code_extract(code)")


def patch_file(path):
    """Insert the dedent helper and route code_fill through it. Idempotent."""
    text = path.read_text(encoding="utf-8")
    if "_shim_dedent_code(content)" in text:
        return "already"
    if CALL_OLD not in text:
        return "anchor-missing"
    if "def _shim_dedent_code" not in text:
        # place the helper just before the class that owns code_fill
        marker = "\nclass ActionNode"
        idx = text.find(marker)
        if idx < 0:
            return "anchor-missing"
        text = text[:idx] + PATCH_FUNCTION + text[idx:]
    text = text.replace(CALL_OLD, CALL_NEW, 1)
    path.write_text(text, encoding="utf-8")
    return "patched"


def patch_sanitize_file(path):
    """Route an upstream sanitize() through the same conservative normalizer."""
    text = path.read_text(encoding="utf-8")
    if "mbpp uniform-indent v1" in text or "_shim_dedent_code(code)" in text:
        return "already"
    if SANITIZE_CALL_OLD not in text:
        return "anchor-missing"
    marker = "\ndef sanitize("
    idx = text.find(marker)
    if idx < 0:
        return "anchor-missing"
    text = text[:idx] + SANITIZE_FUNCTION + text[idx:]
    text = text.replace(SANITIZE_CALL_OLD, SANITIZE_CALL_NEW, 1)
    path.write_text(text, encoding="utf-8")
    return "patched"
