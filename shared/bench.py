#!/usr/bin/env python3
"""The single scoring authority for the bake-off.

Every method under comparison must route its answers through `score()` here.
If each repo kept its own evaluator, differences between methods would partly
reflect differences in answer extraction and grading, which is exactly the
confound this bake-off exists to remove.

The MATH/AMC and MBPP paths retain AFlow's published evaluators as their base,
with shared extraction and benchmark-contract fixes around them. DROP uses the
official token-F1 definition because AFlow's simplified copy differs from the
published metric. MMLU-Pro has no AFlow evaluator and is implemented here.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import string
from collections import Counter
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "shared" / "data"
AFLOW_DIR = ROOT / "third_party" / "aflow"

DATASETS = ("math", "amc", "mbpp", "drop", "mmlu_pro")

# Answer-length budgets. These are part of the frozen protocol: the cap drives
# both truncation rate and total wall-clock, so it is pinned per task type
# rather than left to each repo's default.
MAX_TOKENS = {name: 8192 for name in ("math", "amc", "mbpp", "drop", "mmlu_pro")}
# One uniform cap rather than a per-dataset table. Truncation is the confound to
# kill: a cut-off reply scores zero for reasons unrelated to the method, and
# methods do not truncate at equal rates, so any dataset-specific cap becomes
# something to defend. The cap costs nothing unless a model actually generates
# that far -- measured averages run 54-1637 tokens -- so setting it high is
# cheap insurance. Successive smoke runs at 1024/2048/4096 truncated 55%/30%/15%
# of AMC replies respectively.

MMLU_PRO_LETTERS = "ABCDEFGHIJ"

# Extraction health, per dataset. Populated by score(); read via extraction_stats().
_extraction_stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)


def _assert_latex_equivalence_available() -> None:
    """Refuse to grade maths where sympy cannot parse LaTeX.

    AFlow's math_equal tries string equality, then numeric closeness, then sympy.
    Without antlr4 installed, sympy's parse_latex raises and only the first two
    survive -- so "14/3" scores zero against a gold of "\\frac{14}{3}". Measured on
    the real gold answers: 47 of 47 fraction rewrites, 24 of 24 decimal rewrites
    and 5 of 5 fraction-for-decimal rewrites went from failing to passing once the
    dependency was present.

    It was installed in the gdesigner and pyg environments and missing from maas,
    which meant four methods were graded by string comparison on MATH while three
    were graded by symbolic equivalence -- a difference between methods produced
    entirely by which virtualenv happened to have a transitive dependency. Failing
    loudly is the only way that does not silently become a result.
    """
    global _LATEX_CHECKED
    if _LATEX_CHECKED:
        return
    try:
        from sympy.parsing.latex import parse_latex

        parse_latex(r"\frac{1}{2}")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "maths grading needs sympy with LaTeX support, and this interpreter "
            f"cannot parse LaTeX ({type(exc).__name__}). Install "
            "'antlr4-python3-runtime==4.11.1' into this environment, or run the "
            "collector with envs/maas/bin/python. Grading without it silently "
            "marks correct answers wrong."
        ) from exc
    _LATEX_CHECKED = True


_LATEX_CHECKED = False


def _ensure_aflow_importable() -> None:
    path = str(AFLOW_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


@lru_cache(maxsize=None)
def _aflow_scorer(name: str) -> Any:
    """Instantiate an AFlow benchmark purely for its grading methods.

    BaseBenchmark.__init__ only stores the three strings, so the dummy paths are
    never touched; we only call calculate_score / extract_model_answer /
    check_solution, which are synchronous and side-effect free.
    """
    _ensure_aflow_importable()
    if name == "math":
        from benchmarks.math import MATHBenchmark

        return MATHBenchmark("MATH", "unused", "unused")
    if name == "amc":
        from benchmarks.amc import AMCBenchmark

        return AMCBenchmark("AMC", "unused", "unused")
    if name == "mbpp":
        from benchmarks.mbpp import MBPPBenchmark

        return MBPPBenchmark("MBPP", "unused", "unused")
    if name == "drop":
        from benchmarks.drop import DROPBenchmark

        return DROPBenchmark("DROP", "unused", "unused")
    raise KeyError(f"no AFlow scorer for {name}")


_MBPP_CALL = re.compile(r"([A-Za-z_]\w*)\s*\(")
# Names that appear in an assert without being the function under test.
_MBPP_NOT_ENTRY = {
    "assert", "print", "set", "len", "list", "tuple", "dict", "str", "int",
    "float", "abs", "round", "sorted", "sum", "min", "max", "range", "math",
    "type", "bool", "map", "filter", "zip", "all", "any", "isinstance",
}


_MBPP_UNIFORM_DEDENT_VERSION = "v1"
_MBPP_TEST_SETUP_VERSION = "v1"


def _dedent_uniform_mbpp_code(text: str) -> str:
    """Remove only a common outer indent from a complete code reply.

    Some model replies contain a valid function shifted right as one block. The
    normalizer is deliberately conservative: every non-blank line must carry the
    same prefix as the first definition, and the result must expose a top-level
    definition. Existing top-level replies are returned byte-for-byte unchanged.
    """
    if not isinstance(text, str) or not text:
        return text

    import textwrap as _textwrap

    def _fix_body(body: str) -> str:
        first = re.search(
            r"^([ \t]+)(?:async\s+def|def|class)\s+\w+", body, re.M
        )
        if not first:
            return body
        prefix = first.group(1)
        lines = body.splitlines(True)
        nonblank = [line for line in lines if line.strip()]
        if not nonblank or any(not line.startswith(prefix) for line in nonblank):
            return body
        fixed = _textwrap.dedent(body)
        if not re.search(r"^(?:async\s+def|def|class)\s+\w+", fixed, re.M):
            return body
        return fixed

    fenced = re.search(
        r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S | re.I
    )
    if fenced:
        body = fenced.group(1)
        fixed = _fix_body(body)
        if fixed != body:
            return text[:fenced.start(1)] + fixed + text[fenced.end(1):]
        return text
    return _fix_body(text)


def _mbpp_test_setup(row: dict) -> str:
    """Dataset-supplied setup statements that must run before MBPP asserts."""
    blocks = row.get("test_imports") or ()
    return "\n".join(
        str(block).rstrip() for block in blocks if str(block).strip()
    )


def _mbpp_retain_setup_definitions(
    code: str, setup: str, entrypoint: str
) -> tuple[str, str]:
    """Keep candidate definitions referenced only by the official test setup.

    AFlow's sanitizer retains definitions reachable from the solution entrypoint.
    MBPP/367 is different: its candidate defines ``Node``, while the entrypoint
    never mentions that class; only test_setup_code does. Preserve exactly the
    candidate's top-level definitions named by setup, using a synthetic dependency
    root that the existing sanitizer can process normally.
    """
    if not setup:
        return code, entrypoint

    try:
        import ast as _ast

        _ensure_aflow_importable()
        from scripts.utils.sanitize import sanitize as _aflow_sanitize

        all_definitions = _aflow_sanitize(code=code, entrypoint=None)
        code_tree = _ast.parse(all_definitions)
        setup_tree = _ast.parse(setup)
    except Exception:  # noqa: BLE001 - the normal scorer reports malformed code
        return code, entrypoint

    definition_names = {
        node.name
        for node in code_tree.body
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef))
    }
    for node in code_tree.body:
        if isinstance(node, (_ast.Assign, _ast.AnnAssign)):
            targets = node.targets if isinstance(node, _ast.Assign) else [node.target]
            definition_names.update(
                target.id for target in targets if isinstance(target, _ast.Name)
            )
    setup_names = {
        node.id for node in _ast.walk(setup_tree) if isinstance(node, _ast.Name)
    }
    required = sorted(definition_names & setup_names)
    if not required:
        return code, entrypoint

    helper = "_mbpp_entry_with_setup_dependencies"
    while helper in definition_names:
        helper += "_"
    dependencies = [entrypoint] + [name for name in required if name != entrypoint]
    helper_code = (
        f"\ndef {helper}():\n"
        f"    return ({', '.join(dependencies)},)\n"
    )
    return all_definitions.rstrip() + helper_code, helper


def _mbpp_entry_point(row: dict) -> str:
    """The function MBPP's own tests call.

    The splits were built with `code.split("def ", 1)` -- the FIRST definition in
    the reference solution. For 20 of the 500 items that is a helper, not the
    entry point: item mbpp/18 defines str_to_list, lst_to_string and
    get_char_count_array before remove_dirty_chars, which is what every assert
    calls. Grading those items against the first definition fails every solution,
    including the reference one, which is how this was found -- replaying the gold
    code through the scorer.

    The tests are the contract, so the name is taken from them, and the stored
    value is used only when the tests name nothing recognisable.
    """
    stored = str(row.get("entry_point") or "")
    called = []
    for test in row.get("test_list") or []:
        for name in _MBPP_CALL.findall(str(test)):
            if name not in _MBPP_NOT_ENTRY and name not in called:
                called.append(name)
    if not called:
        return stored
    if stored in called:
        return stored
    return called[0]


@lru_cache(maxsize=None)
def load(name: str) -> tuple[dict, ...]:
    """Load a canonical split. Immutable so callers cannot reorder the set."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name}; expected one of {DATASETS}")
    path = DATA_DIR / f"{name}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle if line.strip())


# A uniform answer-format request, appended to the shared task statement.
#
# This belongs here and nowhere else. `question_text` is the one piece of text
# all seven methods receive identically, so a requirement added here is applied
# evenly; editing any method's own prompts would change the thing being measured.
# It reduces, but cannot remove, format mismatch: a multi-agent method's final
# decision node may rewrite the answer and drop the format, which is why the
# tolerant extractors below stay in place as well.
# A uniform answer-format request with a worked format example, appended to the
# shared task statement.
#
# The instruction alone was not enough. Measured across realistic output shapes,
# the graders accepted only 3 of 6 for MATH/AMC and 4 of 6 for MMLU-Pro: replies
# ending "Thus, 142." or "Option J is correct" scored zero despite being right.
# Two fixes are needed and they attack opposite ends of the same problem -- this
# one raises the share of replies that arrive in the expected shape, and the
# tiered extractors below catch what still does not.
#
# The examples demonstrate *format only*. They deliberately carry no reasoning,
# no worked solution and no domain hint, so no method gains a solving advantage
# from them -- and because they live in question_text, all seven methods receive
# exactly the same text.
ANSWER_FORMAT = {
    # No braces in these strings beyond the LaTeX the example needs: several repos
    # build prompts with str.format(), where a bare {} becomes a replacement field
    # and raises. The math examples are pre-escaped for that reason.
    "math": (
        "Your reply MUST end with the final answer inside a LaTeX boxed "
        "expression, on its own last line, and nothing after it.\n"
        "Format example (format only, unrelated to this problem):\n"
        "  ... reasoning ...\n"
        "  \\boxed{{42}}"
    ),
    "amc": (
        "Your reply MUST end with the final answer inside a LaTeX boxed "
        "expression, on its own last line, and nothing after it.\n"
        "Format example (format only, unrelated to this problem):\n"
        "  ... reasoning ...\n"
        "  \\boxed{{42}}"
    ),
    "mbpp": (
        "Return self-contained Python code in a single code block. It must "
        "define the requested entry-point function and may include any helper "
        "functions or classes it needs. Include no explanation after the code.\n"
        "Format example (format only, unrelated to this problem):\n"
        "```python\n"
        "def example_name(x):\n"
        "    return x\n"
        "```"
    ),
    "drop": (
        "Your reply MUST end with a line of the form 'Answer: <answer>', where "
        "<answer> is the concise answer (a span, number, date, or list as "
        "appropriate) and nothing else "
        "follows it.\n"
        "Format example (format only, unrelated to this passage):\n"
        "  ... reasoning ...\n"
        "  Answer: 57"
    ),
    "mmlu_pro": (
        "Your reply MUST end with a line of the form 'Answer: (X)', where X is a "
        "single option letter and nothing follows it.\n"
        "Format example (format only, unrelated to this question):\n"
        "  ... reasoning ...\n"
        "  Answer: (C)"
    ),
}

REQUIRE_ANSWER_FORMAT = os.getenv("BENCH_REQUIRE_ANSWER_FORMAT", "1") not in {"0", "false", "False"}


def protocol_fingerprint() -> dict[str, str]:
    """What the model was shown, and how the reply was graded.

    The common prompt, complete scorer source, and all method-adaptation sources
    are hashed separately. The broad source hash is deliberate: a role-prompt or
    orchestration change in one shim is just as protocol-relevant as changing the
    common answer-format suffix.

    Written into every job directory by sweep.py and checked by collect.py, so a
    leftover artefact from an earlier protocol is reported rather than silently
    averaged in. Wiping the old files is still the right thing to do -- this is
    the check that catches it when the wipe misses something.
    """
    prompt_material = json.dumps(
        {"answer_format": ANSWER_FORMAT, "required": REQUIRE_ANSWER_FORMAT},
        sort_keys=True, ensure_ascii=False,
    )
    scorer_material = Path(__file__).read_bytes()
    adapter_paths = [
        ROOT / "sweep.py",
        ROOT / "flowbank_pipeline.py",
        ROOT / "aflow_test.py",
        ROOT / "vllm_proxy.py",
        ROOT / "launch_vllm.sh",
        ROOT / "upstreams.lock.json",
    ]
    adapter_paths.extend(
        path for path in sorted((ROOT / "shims").rglob("*"))
        if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".yml"}
    )
    adapter_hash = hashlib.sha256()
    for path in adapter_paths:
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        adapter_hash.update(len(relative).to_bytes(4, "big"))
        adapter_hash.update(relative)
        contents = path.read_bytes()
        adapter_hash.update(len(contents).to_bytes(8, "big"))
        adapter_hash.update(contents)
    return {
        "prompt": hashlib.sha256(prompt_material.encode("utf-8")).hexdigest()[:16],
        "scorer": hashlib.sha256(scorer_material).hexdigest()[:16],
        "adapter": adapter_hash.hexdigest()[:16],
    }


def data_fingerprint(name: str) -> str:
    """Hash the exact search and evaluation bytes used by one dataset cell."""
    if name not in DATASETS:
        raise KeyError(name)
    digest = hashlib.sha256()
    for suffix in ("_search.jsonl", ".jsonl"):
        path = DATA_DIR / f"{name}{suffix}"
        contents = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()[:16]


def question_text(name: str, row: dict) -> str:
    """The canonical problem statement, with no method-specific scaffolding.

    Methods add their own roles, operators and instructions on top of this; the
    boundary keeps the *task* identical across methods while leaving prompting
    strategy as part of each method.
    """
    suffix = f"\n\n{ANSWER_FORMAT[name]}" if REQUIRE_ANSWER_FORMAT else ""
    if name in ("math", "amc"):
        return row["problem"] + suffix
    if name == "mbpp":
        # MBPP's text alone does not pin the function name, so a model has to
        # guess it and the asserts then fail for a reason unrelated to whether
        # it solved the problem. Showing the tests is the standard MBPP
        # protocol and is part of the task, not of any method's prompting.
        tests = "\n".join(row["test_list"])
        return f"{row['prompt']}\n\nYour code must pass these tests:\n{tests}" + suffix
    if name == "drop":
        return row["context"] + suffix
    if name == "mmlu_pro":
        options = "\n".join(f"({MMLU_PRO_LETTERS[i]}) {opt}" for i, opt in enumerate(row["options"]))
        return f"{row['question']}\n\nOptions:\n{options}" + suffix
    raise KeyError(name)


def gold(name: str, row: dict) -> str:
    if name in ("math", "amc"):
        return row["answer"]
    if name == "mbpp":
        return row["code"]
    if name == "drop":
        return row["ref_text"]
    if name == "mmlu_pro":
        return row["answer"]
    raise KeyError(name)


_MATH_LEAD_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*", re.IGNORECASE)

# Additional shapes a model actually produces when it does not use \boxed. Each is
# tried only after the explicit lead-in above fails, and each is counted so the
# rate is reportable rather than folded silently into the score.
_MATH_FALLBACKS = (
    # "**142**" -- models bold the figure they settled on
    ("bold", re.compile(r"\*\*([^*\n]{1,80}?)\*\*")),
    # "Thus, 142." / "Therefore 142" / "So the result is 142"
    # Dots are allowed inside the capture: excluding them broke on decimals --
    # "Thus, 142.0." captured nothing and the whole sentence went to the grader.
    ("connective", re.compile(r"(?:thus|therefore|hence|so)\s*,?\s*(?:the\s+\w+\s+is\s*)?"
                              r"([^\n]{1,80}?)\s*\.?\s*$", re.IGNORECASE | re.MULTILINE)),
)

# A trailing line is trusted as the answer only while it is short.
_MATH_SHORT_TAIL = 40


# Markdown emphasis around an answer. Models routinely bold the figure they
# settled on ("The answer is **142**"), and the asterisks travel into the graded
# string: `**142**` compares unequal to `142` for MATH/AMC, which grade by string
# and symbolic equality. DROP happens to survive it because its F1 normalisation
# strips punctuation -- so this bit only ever showed up on the maths datasets.
_EMPHASIS = re.compile(r"^[*`_\s]+|[*`_\s]+$")


def _clean_candidate(text: str) -> str:
    """Strip markdown emphasis, math delimiters and trailing punctuation.

    Iterated to a fixed point rather than applied once, because the layers
    interleave: in "The answer is **142**." the trailing full stop sits outside
    the emphasis, so one pass strips the leading asterisks, is blocked from the
    trailing ones by the dot, and leaves `142**`.
    """
    out = str(text)
    for _ in range(4):
        before = out
        out = _EMPHASIS.sub("", out)
        out = out.strip().strip("$").strip()
        out = out.rstrip(".\u3002,\uff0c;\uff1b").strip()
        if out == before:
            break
    return out


def _boxed_content(text: str) -> str | None:
    """The contents of the last \\boxed{...}, with braces matched properly.

    AFlow's extractor uses the regex ((?:[^{}]|{[^{}]*})*), which handles exactly
    one level of nesting. A second level does not match at all, so the extractor
    falls through to "the last sentence" and returns the whole "\\boxed{...}"
    string, which then compares unequal to the bare gold. Replaying the MATH gold
    answers through the scorer, 5 of them fail this way -- every answer of the
    shape \\frac{3\\sqrt{3}}{4}, where the numerator itself contains a group.

    Scanning for the matching brace has no depth limit and no regex to get wrong.
    """
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    index = start + len(marker)
    depth = 1
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(marker): index].strip()
        index += 1
    return None


def _normalise_math_prediction(prediction: str) -> str:
    """Wrap a stated answer in \\boxed{} when the model did not.

    AFlow's MATH extractor looks for \\boxed{...} and otherwise falls back to the
    reply's last *sentence*, so a correct "The answer is 259" is compared as the
    whole string "The answer is 259" against "259" and scored wrong. In a smoke
    run of G-Designer this made the score a pure function of whether the model
    happened to use \\boxed: 2 of 12 replies contained it and exactly those 2
    scored correct.

    Output format follows each method's own prompts, which are deliberately left
    as the authors wrote them, so normalising here is what keeps the metric
    comparable instead of rewarding whichever prompt style matches the scorer.
    The metric itself (math_equal / symbolic_equal) is untouched.
    """
    if not prediction or "\\boxed" in prediction:
        return prediction

    matches = list(_MATH_LEAD_RE.finditer(prediction))
    if not matches:
        counts = _extraction_stats["math_norm"]
        for tier, pattern in _MATH_FALLBACKS:
            found = pattern.findall(prediction)
            if found:
                candidate = _clean_candidate(found[-1])
                if candidate:
                    counts[f"norm_{tier}"] += 1
                    return prediction + f"\n\\boxed{{{candidate}}}"
        lines = [line.strip() for line in prediction.splitlines() if line.strip()]
        if lines and len(lines[-1]) <= _MATH_SHORT_TAIL:
            counts["norm_short_tail"] += 1
            return prediction + f"\n\\boxed{{{_clean_candidate(lines[-1])}}}"
        counts["norm_unextracted"] += 1
        return prediction

    tail = prediction[matches[-1].end():]
    # Keep the answer only: stop at a line break, and drop trailing punctuation
    # and stray math delimiters.
    answer = tail.splitlines()[0] if tail.splitlines() else ""
    answer = _clean_candidate(answer)
    if not answer:
        return prediction
    return f"{prediction}\n\\boxed{{{answer}}}"


# Ordered extraction tiers for a stated answer. Explicit markers first, then
# progressively weaker structural cues.
#
# The single-pattern version of this ("answer" followed by a mandatory colon or
# dash, else the last line) measured format compliance rather than capability:
# "Answer: 57" scored F1 1.000 while "The answer is 57." scored 0.500 and
# "...the relevant figure appears to be 57, based on..." scored 0.143, because the
# whole sentence went to the token-level F1. That systematically punished methods
# whose agents talk in prose -- MasRouter's Commonsense roles (Scientist, Critic,
# KnowledgeExpert) sat at 0.19 on DROP against 0.63-0.69 for the others, and most
# of that gap was extraction, not answers.
#
# None of these tiers may look at the gold answer. Choosing the extraction that
# happens to match would be scoring the grader, not the model.
_SPAN_TIERS = (
    ("boxed", re.compile(r"\\boxed\{([^{}]*)\}")),
    # "Answer: 57", "Final answer = 57", "answer - 57"
    #
    # The terminator is a full stop that ENDS the answer -- one not sitting between
    # digits. Written as a bare "." it truncated every decimal: "Answer: 87.9"
    # extracted "87", which scores 0 against a gold of 87.9. 161 of the 1000 DROP
    # items have a decimal answer.
    #
    # The bug was invisible for as long as the DROP prompts asked for \boxed{}: the
    # boxed tier runs first and caught those replies. Adapting those prompts to ask
    # for "Answer: <span>" removed the cover, and the cost surfaced as a 3.5-point
    # "regression" in DAAO that was entirely grading -- the model had answered 87.9
    # and been marked wrong.
    # Capture to the end of the line, and decide where the answer stops
    # afterwards (see _trim_trailing_sentence). No character makes a reliable
    # terminator on its own: a bare "." truncated every decimal ("Answer: 87.9" ->
    # "87"), and requiring whitespace after it still cut initials ("Answer: T.J.
    # Duckett" -> "T.J"). Both were measured on stored replies, 833 and 36 items.
    ("marker_punct", re.compile(r"(?:final\s+)?answer\s*(?:is\s*)?[:=\-]\s*([^\n]+)",
                                re.IGNORECASE)),
    # "The answer is 57", "the answer was 57" -- no punctuation after the verb.
    ("marker_verb", re.compile(r"(?:the\s+)?(?:final\s+)?answer\s+(?:is|was|are|were)\s+"
                               r"([^\n]+)", re.IGNORECASE)),
    # Models very often bold the figure they settled on.
    ("bold", re.compile(r"\*\*([^*\n]{1,60}?)\*\*")),
    # "Thus, 57." / "Therefore 57" -- scored 0.667 before, because the connective
    # stayed in the span and token-level F1 charged for it.
    # \b on both sides of the connective. Without the leading boundary "so"
    # matched inside ordinary words, and any answer containing it was eaten:
    # "Somalia" extracted as "malia", "Derrick Mason" as "n", "Massoud Barzani"
    # as "ud Barzani". Found by replaying the gold answers through the scorer --
    # 45 of the 7,461 DROP golds failed to score 1.0 against themselves.
    ("connective", re.compile(r"\b(?:thus|therefore|hence|so)\b\s*,?\s*"
                              r"(?:the\s+\w+\s+is\s*)?"
                              r"([^\n]{1,60}?)\s*\.?\s*$", re.IGNORECASE | re.MULTILINE)),
)

# A trailing fragment is only trusted when it is short: DROP's F1 is token-level
# and precision-sensitive, so handing it a sentence costs more than it gains.
_SHORT_SPAN = 60


_DROP_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_DROP_PUNCT = set(string.punctuation)


def _drop_is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _official_drop_normalise(answer: str) -> list[str]:
    """The DROP paper's own normalisation, transcribed from its evaluation script.

    Three steps that AFlow's copy omits and that decide whether a correct figure
    scores: tokens are split on hyphens as well as spaces; punctuation is stripped
    from words but NOT from numbers; and every numeric token is passed through
    float() so that ".08", "0.08" and "0.080" are the same token.
    """
    tokens = []
    # \s rather than a literal space: models emit non-breaking spaces, and "8.0\xa0in"
    # stayed one token, so it was not a number, so its punctuation was stripped to
    # "80in" and a correct answer of 8 scored zero.
    for token in re.split(r"[\s\-]+", answer.lower()):
        # Markdown emphasis rides through the whitespace split ("**4.4%**" is one
        # token) and defeats the number test, after which punctuation-stripping
        # mangles the figure to "44". Same for a percent sign on an embedded
        # figure ("4.4%" inside a sentence-long span). Both found 2026-08-24 in
        # flowbank/search records whose correct figures scored 0.
        token = token.strip("*_`")
        if token.endswith("%") and _drop_is_number(token[:-1]):
            token = token[:-1]
        if not _drop_is_number(token):
            token = "".join(ch for ch in token if ch not in _DROP_PUNCT)
        if _drop_is_number(token):
            token = str(float(token))
        token = " ".join(_DROP_ARTICLES.sub(" ", token).split())
        if token.strip():
            tokens.append(token.strip())
    return tokens


def _official_drop_f1(gold_answer: str, prediction: str) -> float:
    """Token-level F1 as DROP defines it. Partial credit is intended."""
    predicted = _official_drop_normalise(prediction)
    truth = _official_drop_normalise(gold_answer)
    if not predicted or not truth:
        return 0.0
    common = Counter(predicted) & Counter(truth)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(predicted)
    recall = same / len(truth)
    return 2 * precision * recall / (precision + recall)


def _as_number(text: str) -> float | None:
    """The numeric value of an answer, or None if it is not a number.

    Tolerates the decorations a model puts on a figure without changing it: a
    trailing percent sign or unit-free currency mark, thousands separators, and
    surrounding whitespace or brackets. "9.5%", "$1,234" and " 0.08 " are numbers;
    "9.5 million" and "two" are not.
    """
    if not text:
        return None
    cleaned = text.strip().strip("()[]{}").strip()
    cleaned = cleaned.lstrip("$€£").rstrip("%").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


# AFlow and FlowBank operators ask the model for an XML envelope --
# "<thought>...</thought><answer>...</answer>" -- and the answer field is the only
# part that is an answer. Left in place the tags travel into the graded span:
# "<answer>Britain</answer>" normalises to the single token "answerbritainanswer"
# and scores 0 against a gold of "Britain", and on a reply whose <thought> is long
# the extractor picks a sentence out of the reasoning instead. Both were seen in
# the live transcripts of search/flowbank/drop.
_XML_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_XML_THOUGHT = re.compile(r"<thought>.*?</thought>\s*", re.DOTALL | re.IGNORECASE)


def _strip_xml_envelope(text: str) -> str:
    """The <answer> payload if the reply uses the envelope, else the reply."""
    if not text or "<answer>" not in text.lower():
        # A reply may still carry a <thought> block without an <answer> one.
        return _XML_THOUGHT.sub("", text) if text else text
    found = _XML_ANSWER.findall(text)
    if found:
        return found[-1].strip()
    return _XML_THOUGHT.sub("", text)


def _extract_drop_span(prediction: str) -> str:
    """The answer span, with formatting that carries no meaning removed.

    Wraps the tier ladder below so that canonicalisation happens on every
    return path rather than being repeated at each one. The metric is the
    official DROP F1 and is not relaxed; what is cleaned here is our own
    output -- turning "Answer: 9.5%" into the figure the question asked for
    is the extractor's job, not the metric's.
    """
    return _canonical_drop_span(
        _extract_drop_span_raw(_strip_xml_envelope(prediction)))


def _trim_trailing_sentence(span: str) -> str:
    """Cut explanation that follows the answer, and nothing else.

    The answer line is often just the answer, sometimes the answer plus a
    sentence of justification. Splitting on the first full stop handles the second
    case and destroys the first whenever the answer itself contains one -- a
    decimal, or a name written with initials. Length decides instead: a fragment
    of three words or fewer after a full stop is part of the answer ("T.J.
    Duckett"), more than that is a new sentence ("57. This is because the passage
    says ...").

    Only a TRAILING full stop is stripped. Using str.strip(".") removed the
    leading one too, turning ".08" into "08".
    """
    text = span.strip()
    # A full stop right after a single capital letter is an initial, not a
    # sentence end: "RB T.J. Duckett and RB Kevin Jones" was split at "T.J",
    # the five words after it read as a new sentence, and the kept span "RB T.J"
    # scored 0 against "Kevin Jones" (found 2026-08-24, aflow/search).
    parts = re.split(r"(?<!\b[A-Z])(?<!\b[a-z])\.\s+", text)
    if len(parts) > 1 and len(parts[-1].split()) >= 4:
        text = ". ".join(parts[:-1])
    return text.rstrip(".").strip()


def _canonical_drop_span(span: str) -> str:
    """Strip decoration from a figure; leave text spans untouched."""
    if not span:
        return span
    # Markdown emphasis first: models bold the figure they settled on, and
    # "**99.8%**" left as-is is not recognised as a number, so the % survives
    # into the official normalisation, which then strips the punctuation of a
    # non-numeric token and grades "998" against a gold of "99.8". Measured: 34
    # correct answers marked wrong in one search cell alone.
    stripped = span.strip().strip("*_`").strip()
    stripped = stripped.strip("()[]{}").strip()
    # Only rewrite when the result is unambiguously the same number. A span
    # like "9.5 million" or "the 1980s" keeps every character.
    bare = stripped.lstrip("$€£").rstrip("%").replace(",", "").strip()
    if _as_number(bare) is not None:
        return bare
    return span


def _extract_drop_span_raw(prediction: str) -> str:
    """Reduce a reply to the answer span before F1 scoring.

    DROP's F1 is token-level and precision-sensitive, so grading a whole
    chain-of-thought against a gold span like "57" drives precision (and the
    score) to nearly zero -- a smoke run scored 3.2 F1 that way.
    """
    if not prediction:
        return ""
    counts = _extraction_stats["drop"]

    for tier, pattern in _SPAN_TIERS:
        found = pattern.findall(prediction)
        if found:
            span = _trim_trailing_sentence(found[-1])
            if span:
                counts[f"span_{tier}"] += 1
                return span

    # Last line, then last sentence -- accepted only while short.
    lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    if lines and len(lines[-1]) <= _SHORT_SPAN:
        counts["span_short_last_line"] += 1
        return _trim_trailing_sentence(lines[-1])

    tail = re.split(r"(?<=[.!?])\s+", prediction.strip())
    if tail and len(tail[-1]) <= _SHORT_SPAN:
        counts["span_short_last_sentence"] += 1
        return _trim_trailing_sentence(tail[-1])

    # Nothing structural to hold on to. Counted so the rate of it is reportable
    # rather than silently folded into the score.
    counts["span_unextracted"] += 1
    return lines[-1] if lines else prediction.strip()


def _extract_mmlu_pro_letter(prediction: str, num_options: int) -> str | None:
    """Pull a single option letter out of free-form text.

    Ordered from most explicit to least so that a model which both reasons and
    concludes is graded on its conclusion, not on an option it mentioned while
    thinking.
    """
    valid = MMLU_PRO_LETTERS[:num_options]
    # The (?![A-Za-z]) lookaheads exist because of a measured failure, not
    # pedantry: "### Final Answer:\nAnswer: (C)" -- the first "Answer:" satisfies
    # the lead-in, the \s* eats the newline, and without the guard the capture
    # takes the leading "A" OF THE WORD "Answer", never reaching "(C)". On
    # daao/mmlu_pro this mis-graded 168 of 1120 items (0.5196 -> 0.6696).
    patterns = (
        ("answer_lead", rf"answer\s*(?:is|:|=)\s*\(?([{valid}])\)?(?![A-Za-z])"),
        ("answer_near", rf"\banswer\b[^\n{valid}]{{0,20}}\(([{valid}])\)"),
        ("boxed", rf"\\boxed\{{\s*\(?([{valid}])\)?\s*\}}"),
        # "Option J is correct", "choice (J) is right", "select J"
        ("option_word", rf"(?:option|choice|select(?:ed)?)\s*\(?([{valid}])\)?(?![A-Za-z])"),
        # "**(J)**" or "**J**"
        ("bold", rf"\*\*\s*\(?([{valid}])\)?\s*\*\*"),
        ("alone", rf"^\s*\(?([{valid}])\)?\s*$"),
        ("paren_end", rf"\(([{valid}])\)\s*$"),
        ("letter_end", rf"\b([{valid}])\b\s*$"),
    )
    counts = _extraction_stats["mmlu_pro"]
    for tier, pattern in patterns:
        found = re.findall(pattern, prediction, re.IGNORECASE | re.MULTILINE)
        if found:
            counts[f"letter_{tier}"] += 1
            return found[-1].upper()
    counts["letter_unextracted"] += 1
    return None


# Every repo's operators return the same envelope -- {"response": ...} -- and the
# author-written seed workflows unwrap it before returning. A generated workflow
# that forgets to is common with a small optimiser: FlowBank's round 2 returned
# the raw dict from all three of its steps and scored exactly 0.0000 on all 256
# problems, which is indistinguishable in a results table from a workflow that
# genuinely answered everything wrong.
#
# Unwrapping here is mechanical and method-agnostic -- it is a container, not an
# answer -- and it is deliberately *not* the authors' remedy: DiverseFlow calls an
# LLM to extract an answer from whatever the workflow returned, which would also
# rescue this case but grades FlowBank on a more permissive scale than the other
# six methods. Each unwrap is counted so the rate is reportable.
_ENVELOPE_KEYS = ("response", "answer", "output", "solution")


def _unwrap_prediction(name: str, prediction):
    if not isinstance(prediction, dict):
        return prediction
    for key in _ENVELOPE_KEYS:
        value = prediction.get(key)
        if isinstance(value, str):
            _extraction_stats[name]["unwrapped_envelope"] += 1
            return value
    _extraction_stats[name]["unwrappable_dict"] += 1
    return str(prediction)


# Purely typographic LaTeX, i.e. macros that change how an expression is set but
# not what it denotes. Applied to prediction and gold alike, so it can never make
# two different answers compare equal in one direction only.
_LATEX_COSMETIC = (
    (re.compile(r"\\left\s*|\\right\s*"), ""),        # \left( x \right) -> ( x )
    (re.compile(r"\\[dt]frac"), r"\\frac"),           # display/text fraction -> fraction
    (re.compile(r"\\(?:!|,|;|:|quad|qquad)"), " "),   # spacing macros
    (re.compile(r"\\text\s*\{([^{}]*)\}"), r"\1"),    # \text{ cm} -> " cm"; content kept
    (re.compile(r"\\mbox\s*\{([^{}]*)\}"), r"\1"),
    # Escaped \$ before bare $: gold "\$36" vs a model's "36" survived every
    # tier because stripping the bare dollar left the backslash behind
    # (2026-08-24 audit, one item per G-Designer-family math cell).
    (re.compile(r"\\\$"), ""),
    (re.compile(r"[$]"), ""),                         # inline/display math delimiters
    # All whitespace, not merely runs of it. `\left( 3, x \right)` relaxes to
    # `( 3, x )` while the model writes `(3, x)`: collapsing runs leaves those two
    # different strings, and sympy's parse_latex cannot parse a tuple, so the
    # comparison had no way left to succeed. Removing spacing entirely is safe
    # because it is applied to gold and prediction alike -- it can only equate two
    # answers that differ by whitespace, never two that differ in content.
    (re.compile(r"\s+"), ""),
)


# Trailing unit symbols and leading variable assignments -- the two dressings a
# correct answer actually arrives in, measured on the live 2026-08-24 L5 run
# (audits/correct_but_zero: gold "28\\%" vs model "28" four times, gold
# "-2+\\sqrt{3}" vs model "a = -2+\\sqrt{3}"). Stripped from BOTH sides, only at
# the string's edge, and only inside the last-chance retry below, so it can
# recover a point but never move one between two genuinely different answers:
# "28\\%" vs "0.28" still differ after the strip and still score 0.
_UNIT_TAIL = re.compile(r"(?:\\%|%|\\?\u00b0|\^\s*(?:\\circ|\{\\circ\})|degrees?)\s*$",
                        re.IGNORECASE)
_ASSIGN_HEAD = re.compile(r"^\s*\(?\s*[a-zA-Z](?:_\{?[a-zA-Z0-9]+\}?)?"
                          r"(?:\s*,\s*[a-zA-Z](?:_\{?[a-zA-Z0-9]+\}?)?)*\s*\)?\s*=\s*")


_RATIO = re.compile(r"^(\d+)\s*[::]\s*(\d+)$")
_OUTER_PARENS = re.compile(r"^\((.+)\)$")


_BASE_SUFFIX = re.compile(r"^(.*?)_\{?(\d+)\}?$")
_PLAIN_SQRT_ALLOWED = re.compile(r"[A-Za-z0-9_+*/().^{}\[\]\\\s-]+")


def _plain_sqrt_to_latex(text: str) -> str | None:
    """Mechanically convert a narrow, balanced plain ``sqrt(...)`` form.

    This never evaluates model text. Prose, code, existing LaTeX, unsupported
    characters and malformed parentheses are rejected and remain untouched.
    """
    if ("sqrt(" not in text or r"\sqrt(" in text
            or not _PLAIN_SQRT_ALLOWED.fullmatch(text)):
        return None

    def convert(value: str) -> str | None:
        output: list[str] = []
        cursor = 0
        while cursor < len(value):
            start = value.find("sqrt(", cursor)
            if start < 0:
                output.append(value[cursor:])
                break
            output.append(value[cursor:start])
            depth = 1
            end = start + len("sqrt(")
            while end < len(value) and depth:
                if value[end] == "(":
                    depth += 1
                elif value[end] == ")":
                    depth -= 1
                end += 1
            if depth:
                return None
            inner = convert(value[start + len("sqrt("):end - 1])
            if inner is None or not inner.strip():
                return None
            output.append(r"\sqrt{" + inner + "}")
            cursor = end
        return "".join(output)

    converted = convert(text)
    return converted if converted != text else None


def _drop_onesided_base_suffix(a: str, b: str) -> tuple[str, str]:
    """gold "-221_3" vs a model's "-221": drop the base annotation only when
    exactly ONE side carries it. Both-sided suffixes are kept, so "101_2" vs
    a gold "101_3" (a genuinely wrong base) still compares unequal."""
    ma, mb = _BASE_SUFFIX.match(a), _BASE_SUFFIX.match(b)
    if ma and not mb:
        return ma.group(1), b
    if mb and not ma:
        return a, mb.group(1)
    return a, b


def _sympy_equal_direct(a: str, b: str) -> bool:
    """Equivalence through an UNMANGLED parse_latex.

    AFlow's amc.py strips every backslash before parsing ("\\frac{3}{8}" ->
    "frac{3}{8}"), so its symbolic_equal has never once succeeded -- found
    2026-08-24 when the innocence sweep flagged 67 stored AMC records whose
    final answer is sympy-equal to gold yet scored 0 (\\frac38 vs \\frac{3}{8},
    2\\sqrt{21} vs \\sqrt{84}, 5/8 vs 0.625, \\sqrt{x} vs x^{1/2}). The MATH
    benchmark's copy does not have that line, which is why the math column's
    symbolic rescues worked. This tier restores the equivalence the authors
    intended, from bench itself, for both datasets.
    """
    a, b = _drop_onesided_base_suffix(a.strip(), b.strip())
    if not a or not b:
        return False
    if a == b:
        return True
    # parse_latex parses a valid PREFIX and silently drops the rest: "4,24" and
    # "4,12" both come back as 4, "101_3" and "101_2" both as 101 -- which would
    # equate genuinely different tuples and base-annotated numerals. Multi-part
    # and subscripted answers are therefore never judged by this tier; the
    # string-level tiers above already handle their legitimate variants.
    if any(ch in a or ch in b for ch in ",;_=<>"):
        return False
    a = _plain_sqrt_to_latex(a) or a
    b = _plain_sqrt_to_latex(b) or b
    try:
        from sympy import N, simplify
        from sympy.parsing.latex import parse_latex
        ea, eb = parse_latex(a), parse_latex(b)
        if simplify(ea - eb) == 0:
            return True
        from math import isclose
        return isclose(float(N(ea)), float(N(eb)), abs_tol=1e-6)
    except Exception:  # noqa: BLE001 - unparseable forms simply do not match
        return False


def _split_top_level_commas(text: str) -> list[str] | None:
    """Split on commas that sit outside every (), [], {} nesting level."""
    depth = 0
    parts: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return None
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    return parts if depth == 0 else None


def _answer_elem_equal(name: str, a: str, b: str) -> bool:
    """One list/vector ELEMENT vs another: author scorer, then direct sympy."""
    if a == b:
        return True
    value, _ = _aflow_scorer(name).calculate_score(a, b)
    if float(value):
        return True
    return _sympy_equal_direct(_strip_answer_dressing(a),
                               _strip_answer_dressing(b))


_MATRIX_ENV = re.compile(r"^\\begin\{[pb]matrix\}(.*)\\end\{[pb]matrix\}$",
                         re.DOTALL)


def _vector_entries(text: str) -> list[str] | None:
    """A whole-string pmatrix/bmatrix VECTOR -> its entries, in order.

    vmatrix is excluded on purpose (vertical bars denote a determinant -- a
    scalar, not a vector), as is anything with a nested environment or a true
    2-D body (both '\\\\' row breaks and '&' column breaks present).
    """
    match = _MATRIX_ENV.match(text.strip())
    if not match:
        return None
    body = match.group(1)
    if "\\begin" in body:
        return None
    rows = [r.strip() for r in re.split(r"\\\\", body)]
    rows = [r for r in rows if r]
    if len(rows) == 1:
        rows = [c.strip() for c in rows[0].split("&")]
    if any("&" in r for r in rows):
        return None
    if not 2 <= len(rows) <= 6 or any(not r for r in rows):
        return None
    return rows


def _tuple_entries(text: str) -> list[str] | None:
    """A whole-string parenthesised tuple "(a,b,c)" -> its parts, in order."""
    s = text.strip()
    if not s or s[0] != "(" or not _enclosed_by_brackets(s):
        return None
    parts = _split_top_level_commas(s[1:-1])
    if not parts or not 2 <= len(parts) <= 6:
        return None
    if any(not p or "\\begin" in p for p in parts):
        return None
    return parts


def _vector_tuple_equal(name: str, expected: str, prediction: str) -> bool:
    """Sixth tier: column/row-vector notation vs an ordered tuple, ORDERED.

    Found 2026-08-25 in the whole-store audit: gold "(7,21,35)" vs a model's
    "\\begin{pmatrix} 7 \\\\ 21 \\\\ 35 \\end{pmatrix}" scored 0 -- the same
    ordered triple, one written as MATH's tuple, one as a column vector.

    At least one side must be matrix notation (bare tuples never reach here:
    earlier tiers own them), lengths must agree, and the comparison is
    strictly POSITIONAL -- a permuted vector stays wrong, unlike the bare
    comma lists of the fifth tier, because vectors and ordered tuples order.
    """
    vec_e, vec_p = _vector_entries(expected), _vector_entries(prediction)
    if vec_e is None and vec_p is None:
        return False
    parts_e = vec_e if vec_e is not None else _tuple_entries(expected)
    parts_p = vec_p if vec_p is not None else _tuple_entries(prediction)
    if not parts_e or not parts_p or len(parts_e) != len(parts_p):
        return False
    return all(_answer_elem_equal(name, g, p)
               for g, p in zip(parts_e, parts_p))


def _enclosed_by_brackets(s: str) -> bool:
    """True when ONE bracket pair spans the whole string: "(a,b)", "[0,1]".

    Checking the last character alone is wrong -- "\\frac{3}{4}" ends in "}"
    without being enclosed, and "(1,2),(3,4)" starts with "(" without being
    enclosed. Enclosure means the opener at position 0 closes exactly at the
    final character. Malformed/unbalanced strings report True so the multiset
    tier conservatively skips them.
    """
    if not s or s[0] not in "([{":
        return False
    depth = 0
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i == len(s) - 1
            if depth < 0:
                return True
    return True


def _comma_multiset_equal(name: str, expected: str, prediction: str) -> bool:
    """Fifth tier: BARE comma lists compared as multisets, order-free.

    Found 2026-08-25 in the overnight incremental audit: gold
    "\\frac{3}{4}, -\\frac{3}{4}" vs a model's "-\\frac{3}{4}, \\frac{3}{4}"
    scored 0 -- the same solution set, permuted. Neither the author scorer
    (string/numeric compare of the whole list) nor _sympy_equal_direct (rejects
    anything containing a comma) can see the equivalence.

    Deliberately compares the RELAXED forms, not the dressed ones:
    _strip_answer_dressing removes one outer parenthesis pair, so at the
    dressed level the coordinate pair "(-7,10)" is indistinguishable from a
    bare solution list and a flipped "(10,-7)" would wrongly match. Requiring
    both sides to be bracket-free before dressing keeps ordered tuples,
    intervals and sets out of this tier entirely.
    """
    for side in (expected, prediction):
        s = side.strip()
        if not s or _enclosed_by_brackets(s):
            return False
    parts_e = _split_top_level_commas(expected.strip())
    parts_p = _split_top_level_commas(prediction.strip())
    if (not parts_e or not parts_p or len(parts_e) != len(parts_p)
            or not 2 <= len(parts_e) <= 5):
        return False
    if any(not p for p in parts_e + parts_p):
        return False

    matrix = [[_answer_elem_equal(name, g, p) for p in parts_p]
              for g in parts_e]
    used = [False] * len(parts_p)

    def assign(i: int) -> bool:
        if i == len(parts_e):
            return True
        for j, ok in enumerate(matrix[i]):
            if ok and not used[j]:
                used[j] = True
                if assign(i + 1):
                    return True
                used[j] = False
        return False

    return assign(0)


def _strip_answer_dressing(text: str) -> str:
    out = str(text).strip()
    out = _ASSIGN_HEAD.sub("", out, count=1)
    for _ in range(2):
        out = _UNIT_TAIL.sub("", out).strip()
    # Ratio colon notation and a redundant outer parenthesis pair -- both found
    # as real correct-but-zero cases in the 2026-08-24 full-layer audit:
    # gold "16:3" vs a model's "\\frac{16}{3}", gold "(-1,2)" vs "-1, 2".
    # Symmetric (applied to both sides) and shape-anchored, so 16:3 vs 3:16
    # still differ and (a,b) vs (b,a) still differ.
    out = _RATIO.sub(r"\\frac{\1}{\2}", out)
    match = _OUTER_PARENS.match(out)
    if match:
        out = match.group(1).strip()
    return out


def _relax_latex(text: str) -> str:
    """Strip typography from a LaTeX answer, keeping its mathematical content.

    Only removes markup with no semantic weight. Notably it does *not* touch units,
    signs, or the contents of \\text{} -- `5 cm` and `5` still differ here, because
    deciding those are equal would be a judgement about the task rather than about
    formatting.
    """
    out = str(text)
    for pattern, replacement in _LATEX_COSMETIC:
        out = pattern.sub(replacement, out)
    return out.strip()


def score(name: str, row: dict, prediction: str) -> tuple[float, str]:
    """Grade one prediction. Returns (score in [0,1], extracted answer)."""
    if prediction is None:
        return 0.0, ""
    prediction = _unwrap_prediction(name, prediction)
    if not isinstance(prediction, str):
        prediction = str(prediction)
    # The AFlow/FlowBank operators wrap replies in <thought>/<answer> XML. The
    # operators normally unwrap it themselves before anything is stored, but any
    # path that hands the raw reply to the scorer -- a parse fallback, a direct
    # replay of a transcript -- would otherwise grade the tags as answer text:
    # "<answer>Britain</answer>" normalises to the single token
    # "answerbritainanswer" and scores 0 against "Britain". Stripping here, once,
    # covers all five datasets; it is a no-op when no envelope is present.
    prediction = _strip_xml_envelope(prediction)

    if name in ("math", "amc"):
        _assert_latex_equivalence_available()
        prediction = _strip_xml_envelope(prediction)
        normalised = _normalise_math_prediction(prediction)
        counts = _extraction_stats[name]
        counts["scored"] += 1
        if normalised != prediction:
            # The model stated an answer but not in \boxed{}; re-wrapped for the metric.
            counts["reformatted"] += 1
        elif "\\boxed" not in (prediction or ""):
            # Neither \boxed{} nor an "answer is" lead-in: AFlow's extractor will
            # fall back to the reply's last sentence and compare that whole string
            # against the gold answer. This is the path that silently scored 3 of
            # 12 correct MATH answers as wrong, so it is counted, not ignored.
            counts["no_format"] += 1
        expected = gold(name, row)
        # Hand the scorer the boxed CONTENT, not the wrapper. Its own extractor
        # cannot parse two levels of nesting and would return the whole
        # "\\boxed{...}" string; the content compares correctly against the bare
        # gold, and for a prediction with no nesting this changes nothing.
        content = _boxed_content(normalised)
        for_scoring = content if content is not None else normalised
        value, extracted = _aflow_scorer(name).calculate_score(expected, for_scoring)
        if not float(value):
            # Second chance, on cosmetically normalised LaTeX. AFlow's math_equal
            # tries exact string match, then numeric isclose, then sympy -- and
            # sympy's parse_latex cannot parse a coordinate pair or an interval, so
            # `(3, \frac{\pi}{2})` against the gold `\left( 3, \frac{\pi}{2} \right)`
            # falls through all three and scores 0 for a correct answer. Measured on
            # our splits, 26 of 500 MATH golds (5.2%) carry a purely typographic
            # difference of this kind -- larger than the 1-4 point gaps this bake-off
            # is trying to resolve.
            #
            # Deliberately a *retry* rather than a replacement: the author's own
            # verdict is taken first and is never overridden downwards, so this can
            # only recover points, never remove one. The counter makes the size of
            # the effect reportable instead of invisible.
            relaxed_prediction = _relax_latex(normalised)
            relaxed_expected = _relax_latex(expected)
            if (relaxed_prediction, relaxed_expected) != (normalised, expected):
                retry_value, retry_extracted = _aflow_scorer(name).calculate_score(
                    relaxed_expected, relaxed_prediction)
                if float(retry_value):
                    counts["recovered_latex_form"] += 1
                    value, extracted = retry_value, retry_extracted
        if not float(value):
            # Third tier: answer dressing (trailing units, leading assignment).
            # Same contract as the tier above: symmetric, retry-only, counted.
            dressed_prediction = _strip_answer_dressing(_relax_latex(for_scoring))
            dressed_expected = _strip_answer_dressing(_relax_latex(expected))
            if dressed_prediction and (dressed_prediction, dressed_expected) != (
                    for_scoring, expected):
                retry_value, retry_extracted = _aflow_scorer(name).calculate_score(
                    dressed_expected, dressed_prediction)
                if float(retry_value):
                    counts["recovered_answer_dressing"] += 1
                    value, extracted = retry_value, retry_extracted
        if not float(value):
            # Fourth tier: sympy through an unmangled parse_latex (see
            # _sympy_equal_direct). Same contract: retry-only, counted.
            dressed_prediction = _strip_answer_dressing(_relax_latex(for_scoring))
            dressed_expected = _strip_answer_dressing(_relax_latex(expected))
            if _sympy_equal_direct(dressed_prediction, dressed_expected):
                counts["recovered_sympy_direct"] += 1
                value, extracted = 1, dressed_prediction
        if not float(value):
            # Fifth tier: bare comma-list answers as order-free multisets (see
            # _comma_multiset_equal). Same contract: retry-only, counted.
            relaxed_prediction = _relax_latex(for_scoring)
            relaxed_expected = _relax_latex(expected)
            if _comma_multiset_equal(name, relaxed_expected, relaxed_prediction):
                counts["recovered_comma_multiset"] += 1
                value, extracted = 1, relaxed_prediction
        if not float(value):
            # Sixth tier: [pb]matrix vector vs ordered tuple, positional (see
            # _vector_tuple_equal). Same contract: retry-only, counted.
            relaxed_prediction = _relax_latex(for_scoring)
            relaxed_expected = _relax_latex(expected)
            if _vector_tuple_equal(name, relaxed_expected, relaxed_prediction):
                counts["recovered_vector_notation"] += 1
                value, extracted = 1, relaxed_prediction
        if not str(extracted).strip():
            counts["no_answer"] += 1
        return float(value), str(extracted)

    if name == "drop":
        # DROP ships several annotator answers per question, joined by "|".
        # They must be split before grading: AFlow's normalize_answer strips
        # punctuation, so passing the joined string would fuse "57|57" into the
        # single token "5757" and score a correct answer as 0. AFlow itself
        # splits in evaluate_problem and keeps the best candidate.
        span = _extract_drop_span(prediction)
        counts = _extraction_stats["drop"]
        counts["scored"] += 1
        if not span.strip():
            counts["no_answer"] += 1
        candidates = [c for c in gold("drop", row).split("|") if c.strip()]
        # Graded by the OFFICIAL DROP F1, not by AFlow's simplified copy of it.
        #
        # The metric itself is left exactly as the DROP paper defines it -- token
        # F1, partial credit, max over the annotators' alternative answers. What is
        # replaced is an implementation that departs from it in three ways, each of
        # which turns a right answer into a zero:
        #
        #   * it strips punctuation from every token including numbers, so "81.70"
        #     becomes "8170" while its gold "81.7" becomes "817";
        #   * it has no float normalisation, so ".08" never matches "0.08";
        #   * it splits only on spaces, while the official script also splits on
        #     hyphens, so "10-0" is one token to it and two to the metric.
        #
        # Measured over 8,039 stored replies, the two implementations disagreed on
        # 839 of them. Fixing the answer *format* instead -- being lenient about a
        # stray "%" during comparison -- was tried and rejected: that changes the
        # metric, and a metric of our own invention is not comparable with published
        # DROP numbers. Unit symbols are handled where they belong, in extraction.
        best_value = 0.0
        for candidate in candidates or [""]:
            best_value = max(best_value, _official_drop_f1(candidate, span))
        if best_value <= 0.0:
            counts["zero"] += 1
        return best_value, span

    if name == "mbpp":
        scorer = _aflow_scorer("mbpp")
        entry = _mbpp_entry_point(row)
        counts = _extraction_stats["mbpp"]
        code_text = _dedent_uniform_mbpp_code(prediction)
        if code_text != prediction:
            counts["uniform_indent_removed"] += 1
        setup_text = _mbpp_test_setup(row)
        test_text = row["test"]
        if entry == "check":
            # Three MBPP items ask for a function literally named `check`, which is
            # also the name AFlow gives its test wrapper: it execs the solution,
            # then execs `def check(): assert check(...)`, and the wrapper shadows
            # the solution, so the assert calls the zero-argument wrapper and every
            # solution fails. Renaming the entry point in the solution and in the
            # asserts -- but not the wrapper's own def line -- removes the
            # collision without touching the author's harness.
            alias = "_mbpp_entry_under_test"
            code_text = re.sub(r"\bcheck\b", alias, code_text)
            setup_text = re.sub(r"\bcheck\b", alias, setup_text)
            head, _, body = test_text.partition("\n")
            test_text = head + "\n" + re.sub(r"\bcheck\b", alias, body)
            entry = alias
        if setup_text:
            # The official MBPP field is named test_imports, but it can contain
            # arbitrary fixture construction (mbpp/367 creates three trees).
            # Execute it in the same namespace as the solution before check().
            test_text = setup_text + "\n" + test_text
            code_text, entry = _mbpp_retain_setup_definitions(
                code_text, setup_text, entry
            )
        result = scorer.check_solution(code_text, test_text, entry)
        # check_solution returns (PASS|FAIL, message) in AFlow.
        status = result[0] if isinstance(result, (tuple, list)) else result
        counts["scored"] += 1
        return (1.0 if str(status).upper().endswith("PASS") else 0.0), str(status)

    if name == "mmlu_pro":
        letter = _extract_mmlu_pro_letter(prediction, len(row["options"]))
        counts = _extraction_stats["mmlu_pro"]
        counts["scored"] += 1
        if letter is None:
            counts["no_answer"] += 1
        return (1.0 if letter == row["answer"].upper() else 0.0), letter or ""

    raise KeyError(name)


def extraction_stats() -> dict:
    """Per-dataset extraction health, so a zero is attributable.

    Without this, "the model got it wrong" and "we could not find an answer in
    the reply" are the same number. A scoring bug found in a smoke run cost 25
    points of absolute accuracy on MATH while looking exactly like a weak model,
    which is the failure mode these counters exist to make visible.

    Keys per dataset: scored, no_answer (nothing extractable),
    reformatted (the stated answer had to be re-wrapped for the metric).
    """
    return {
        dataset: dict(counts)
        for dataset, counts in sorted(_extraction_stats.items())
    }


def reset_extraction_stats() -> None:
    _extraction_stats.clear()


def metric_name(name: str) -> str:
    return {"mbpp": "pass@1", "drop": "f1"}.get(name, "accuracy")


def summary() -> dict:
    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return {
        name: {
            "n": len(load(name)),
            "metric": metric_name(name),
            "max_tokens": MAX_TOKENS[name],
            "sha256": manifest.get("datasets", {}).get(name, {}).get("sha256"),
        }
        for name in DATASETS
    }


if __name__ == "__main__":
    for dataset, info in summary().items():
        print(f"{dataset:10s} n={info['n']:5d} metric={info['metric']:9s} "
              f"max_tokens={info['max_tokens']:5d} sha={info['sha256']}")
