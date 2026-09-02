#!/usr/bin/env python3
"""The gold answer, rewritten in an equivalent form. Does it still score?

test_gold_roundtrip.py replays the gold string verbatim, so it proves the pipeline
does not corrupt an answer written exactly as the reference writes it. Models do
not do that: they write 0.5 for 1/2, "6" for "six", 2*sqrt(3) for 2\\sqrt{3}. Those
are the same answer and a grader that rejects them is marking correct work wrong.

Each case below is generated from a real gold value, so the equivalences are not
invented -- and each failure is printed with both forms, because some of them are
the metric behaving as its authors intended rather than a defect. DROP in
particular compares tokens, so "6" against a gold of "six" is a genuine zero under
the published metric and is reported rather than silently patched.
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
import bench  # noqa: E402

WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
         "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
         "eleven": "11", "twelve": "12", "zero": "0"}


def math_variants(gold: str) -> list[tuple[str, str]]:
    """Equivalent renderings a model plausibly produces for a maths answer."""
    out: list[tuple[str, str]] = []
    plain = gold.strip()

    frac = re.fullmatch(r"\\d?frac\{(-?\d+)\}\{(\d+)\}", plain)
    if frac:
        numerator, denominator = int(frac.group(1)), int(frac.group(2))
        out.append(("a/b", f"{numerator}/{denominator}"))
        out.append(("\\dfrac", f"\\dfrac{{{numerator}}}{{{denominator}}}"))
        value = numerator / denominator
        if abs(value - round(value, 4)) < 1e-9:
            out.append(("decimal", f"{round(value, 4)}"))
    simple = re.fullmatch(r"(-?\d+)/(\d+)", plain)
    if simple:
        out.append(("\\frac", f"\\frac{{{simple.group(1)}}}{{{simple.group(2)}}}"))
        out.append(("decimal", str(round(int(simple.group(1)) / int(simple.group(2)), 4))))
    if re.fullmatch(r"-?\d+", plain):
        out.append(("with .0", f"{plain}.0"))
        out.append(("in \\text", f"\\text{{{plain}}}"))
    if re.fullmatch(r"-?\d+\.\d+", plain):
        as_fraction = Fraction(plain).limit_denominator(10000)
        out.append(("as fraction", f"\\frac{{{as_fraction.numerator}}}{{{as_fraction.denominator}}}"))
    if "\\sqrt" in plain:
        # Replace only radical braces. A global right-brace replacement also
        # corrupts unrelated fractions and does not produce an equivalent form.
        variant = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", plain)
        if variant != plain and "\\sqrt" not in variant:
            out.append(("sqrt() form", variant))
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", plain):
        out.append(("no separators", plain.replace(",", "")))
    return out


def drop_variants(gold: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    plain = gold.strip()
    lowered = plain.lower()
    if lowered in WORDS:
        out.append(("digit for word", WORDS[lowered]))
    if re.fullmatch(r"-?\d+", plain):
        for word, digit in WORDS.items():
            if digit == plain:
                out.append(("word for digit", word))
        out.append(("with separator", f"{int(plain):,}") if abs(int(plain)) >= 1000 else
                   ("with .0", f"{plain}.0"))
    if re.fullmatch(r"-?\d+\.\d+", plain):
        out.append(("percent sign", f"{plain}%"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["math", "amc", "drop"])
    parser.add_argument("--show", type=int, default=14)
    args = parser.parse_args()

    rows = bench.load(args.dataset)
    tried = 0
    failures: list[tuple[str, str, str, float]] = []
    by_kind: collections.Counter = collections.Counter()
    tried_kind: collections.Counter = collections.Counter()

    for row in rows:
        golds = ([g.strip() for g in str(bench.gold(args.dataset, row)).split("|") if g.strip()]
                 if args.dataset == "drop" else [str(bench.gold(args.dataset, row))])
        for gold in golds[:1]:
            builder = drop_variants if args.dataset == "drop" else math_variants
            for kind, variant in builder(gold):
                reply = (f"Some reasoning.\n\nAnswer: {variant}" if args.dataset == "drop"
                         else f"Some reasoning.\n\\boxed{{{variant}}}")
                tried += 1
                tried_kind[kind] += 1
                value, _extracted = bench.score(args.dataset, row, reply)
                if value < 0.999:
                    failures.append((kind, gold, variant, value))
                    by_kind[kind] += 1

    print(f"  {args.dataset}: {tried:,} equivalent rewrites of real gold answers, "
          f"{len(failures):,} scored below 1.0 ({len(failures) / max(tried, 1):.1%})\n")
    print(f"  {'rewrite':<18}{'tried':>8}{'not accepted':>15}{'rate':>9}")
    for kind, count in tried_kind.most_common():
        bad = by_kind.get(kind, 0)
        print(f"  {kind:<18}{count:>8,}{bad:>15,}{bad / count:>8.0%}")
    print("\n  examples:")
    for kind, gold, variant, value in failures[: args.show]:
        print(f"    [{kind:<16}] gold={gold[:26]!r:<30} model wrote={variant[:26]!r:<30} {value:.2f}")


if __name__ == "__main__":
    main()
