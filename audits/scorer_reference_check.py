#!/usr/bin/env python3
"""Score every stored reply twice -- with our scorer and with an independent
reference -- and list only the disagreements.

A scorer cannot be verified by reading it. The DROP decimal truncation survived
review, a hand-written test suite and a --check assertion, and was found only when
a real answer of "87.9" was graded wrong. What catches that class of error is a
second implementation that was written from a different source and does not share
the first one's assumptions.

References used, one per dataset:

  drop      the official DROP evaluation from the paper's own script: tokenise on
            space AND hyphen, skip punctuation removal for numeric tokens, and
            normalise every numeric token through float(). The implementation this
            project delegates to (AFlow's benchmarks/drop.py) omits all three, so
            ".08" and "0.08" score zero against each other.
  mmlu_pro  exact option-letter equality. Nearly unambiguous, so a disagreement
            here means the extractor read the wrong letter.
  math/amc  symbolic equivalence via sympy, which accepts 1/2 == 0.5 == \\frac12
            where string comparison does not.

Every disagreement is meant to be READ and adjudicated, not auto-applied: the
reference can be wrong too. Adjudicated cases belong in the frozen regression file
so the same mistake cannot come back.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
import bench  # noqa: E402


# --------------------------------------------------------------------------
# Reference implementation 1: official DROP F1
# --------------------------------------------------------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _official_normalise(answer: str) -> list[str]:
    tokens = []
    for token in re.split(r"[ \-]", answer.lower()):
        # Punctuation is stripped from words but NOT from numbers: removing it
        # turns ".08" into "08" and "1,234" into "1234", which then fail to match
        # their own gold.
        if not _is_number(token):
            token = "".join(ch for ch in token if ch not in _PUNCT)
        if _is_number(token):
            token = str(float(token))
        token = _ARTICLES.sub(" ", token)
        token = " ".join(token.split())
        if token.strip():
            tokens.append(token.strip())
    return tokens


def official_drop_f1(gold: str, prediction: str) -> float:
    """Max F1 over the pipe-separated gold alternatives."""
    best = 0.0
    predicted = _official_normalise(prediction)
    for alternative in [a for a in gold.split("|") if a.strip()]:
        truth = _official_normalise(alternative)
        common = collections.Counter(predicted) & collections.Counter(truth)
        same = sum(common.values())
        if same == 0 or not predicted or not truth:
            continue
        precision = same / len(predicted)
        recall = same / len(truth)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


# --------------------------------------------------------------------------
# Reference implementation 2: symbolic equality for maths
# --------------------------------------------------------------------------
def sympy_equal(gold: str, predicted: str) -> bool | None:
    """True/False, or None when neither side parses (no opinion)."""
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        from sympy.parsing.sympy_parser import parse_expr
    except Exception:  # noqa: BLE001
        return None

    def parse(text: str):
        text = text.strip().strip("$").replace("\\!", "").replace("\\,", "")
        for parser in (parse_expr, parse_latex):
            try:
                return parser(text)
            except Exception:  # noqa: BLE001
                continue
        return None

    left, right = parse(gold), parse(predicted)
    if left is None or right is None:
        return None
    try:
        return bool(simplify(left - right) == 0)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
def unwrap(reply) -> str:
    if isinstance(reply, list):
        return str(reply[0]) if reply else ""
    text = str(reply or "")
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except (ValueError, SyntaxError):
            pass
    return text


def stored_pairs(dataset: str) -> list[tuple[str, str]]:
    """(gold, reply) for every stored item of this dataset."""
    key = f"SHARED_{dataset.upper().replace('_', '')}"
    pairs: list[tuple[str, str]] = []
    for path in sorted(ROOT.glob(f"third_party/*/result/v3/*{dataset}*.json")):
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(records, dict):
            records = list(records.values())
        for record in records or []:
            if record.get("Answer") is not None and record.get("Response") is not None:
                pairs.append((str(record["Answer"]), unwrap(record["Response"])))
    for path in sorted(ROOT.glob(f"third_party/*/*/ext/maas/scripts/optimized/{key}/"
                                 f"*/round_*/0.*.csv")):
        try:
            for row in csv.DictReader(path.open(newline="", errors="replace")):
                if row.get("expected_output") and row.get("prediction"):
                    pairs.append((row["expected_output"], row["prediction"]))
        except OSError:
            continue
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True,
                        choices=["drop", "math", "amc", "mmlu_pro"])
    parser.add_argument("--show", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="cap items, for a quick pass")
    args = parser.parse_args()

    pairs = stored_pairs(args.dataset)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(f"no stored {args.dataset} replies found")

    ours_total = ref_total = 0.0
    disagreements = []
    for gold, reply in pairs:
        ours, extracted = bench.score(args.dataset,
                                      {"ref_text": gold, "answer": gold, "code": gold},
                                      reply)
        if args.dataset == "drop":
            reference = official_drop_f1(gold, extracted)
        elif args.dataset in ("math", "amc"):
            verdict = sympy_equal(gold, extracted)
            if verdict is None:
                continue
            reference = 1.0 if verdict else 0.0
        else:
            reference = 1.0 if extracted.strip().upper() == gold.strip().upper() else 0.0
        ours_total += ours
        ref_total += reference
        if abs(ours - reference) > 1e-6:
            disagreements.append((reference - ours, gold, extracted, reply.strip()[-90:]))

    n = len(pairs)
    print(f"  {args.dataset}: {n:,} stored replies")
    print(f"  our scorer      {ours_total / n:.4f}")
    print(f"  reference       {ref_total / n:.4f}   "
          f"({(ref_total - ours_total) / n * 100:+.2f} points)")
    print(f"  disagreements   {len(disagreements):,}  ({len(disagreements) / n:.2%})\n")

    grouped: collections.Counter = collections.Counter()
    for delta, gold, extracted, _tail in disagreements:
        grouped["reference higher (we may be marking correct answers wrong)"
                if delta > 0 else "we are higher (reference may be stricter)"] += 1
    for label, count in grouped.most_common():
        print(f"    {label}: {count:,}")

    print("\n  examples, largest disagreement first:")
    for delta, gold, extracted, tail in sorted(disagreements, key=lambda d: -abs(d[0]))[: args.show]:
        print(f"    {delta:+.2f}  gold={gold[:28]!r:<32} we extracted={extracted[:28]!r}")


if __name__ == "__main__":
    main()
