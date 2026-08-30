#!/usr/bin/env python3
"""Is the DROP extractor right? Measured against every reply the run stored.

Hand-written shapes only prove the cases you thought of. The decimal truncation
survived a suite of them because nobody wrote "Answer: 87.9" down. So this audits
the real replies instead, and separates the two ways an item can score zero:

  MODEL MISS      the gold string does not appear in the reply at all -- the method
                  genuinely did not find the answer, and the grader is right.
  EXTRACTION MISS the gold IS in the reply, and often in the final answer line, but
                  the extractor returned something else. That is our bug, and every
                  one of these is a point the method earned and did not get.

The second number is the one that says whether the regexes are correct. It is
reported overall, per tier that fired, and with the worst examples printed so the
pattern behind them can be read rather than guessed.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "shared"))
import bench  # noqa: E402


def unwrap(reply) -> str:
    """G-Designer stores the reply as the repr of a one-element list."""
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


def load_pairs(paths: list[Path]) -> list[tuple[str, str, str]]:
    """(gold, reply, source) for every stored DROP item."""
    pairs = []
    for path in paths:
        if path.suffix == ".json":
            try:
                records = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(records, dict):
                records = list(records.values())
            for record in records or []:
                gold, reply = record.get("Answer"), record.get("Response")
                if gold is not None and reply is not None:
                    pairs.append((str(gold), unwrap(reply), path.parent.parent.name))
        else:
            try:
                rows = list(csv.DictReader(path.open(newline="", errors="replace")))
            except OSError:
                continue
            for row in rows:
                gold, reply = row.get("expected_output"), row.get("prediction")
                if gold and reply:
                    pairs.append((str(gold), str(reply), path.parts[1]))
    return pairs


def which_tier(reply: str) -> str:
    for name, pattern in bench._SPAN_TIERS:
        if pattern.findall(reply):
            return name
    return "fallback"


def as_number(text: str):
    """float(text) if the token is a number in any surface form, else None."""
    cleaned = text.strip().rstrip("%").replace(",", "").replace("$", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def classify(extracted: str, alternatives: list[str]) -> str:
    """Why did this miss? The categories decide what to fix."""
    value = as_number(extracted)
    for alt in alternatives:
        other = as_number(alt)
        if value is not None and other is not None and abs(value - other) < 1e-9:
            # Same number, different surface form: ".08" vs "0.08", "1,234" vs
            # "1234", "5.0" vs "5". The official DROP metric normalises numbers
            # before comparing; the implementation we delegate to does not.
            return "same number, different spelling"
    for alt in alternatives:
        if alt and alt.lower() in extracted.lower():
            return "extracted too much (gold is inside it)"
        if extracted and extracted.lower() in alt.lower():
            return "extracted too little (inside the gold)"
    return "unrelated span extracted"


def normalise(text: str) -> str:
    """The comparison the F1 scorer effectively makes, for 'is the gold present'."""
    return re.sub(r"[^a-z0-9.]+", " ", text.lower()).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    paths = sorted(ROOT.glob("third_party/*/result/v3/*drop*.json"))
    paths += sorted(ROOT.glob("third_party/*/*/ext/maas/scripts/optimized/SHARED_DROP/"
                              "test/round_*/0.*.csv"))
    pairs = load_pairs(paths)
    if not pairs:
        raise SystemExit("no stored DROP replies found")

    tiers: collections.Counter = collections.Counter()
    miss_tiers: collections.Counter = collections.Counter()
    scored = extraction_miss = model_miss = correct = 0
    causes: collections.Counter = collections.Counter()
    examples = []

    for gold, reply, source in pairs:
        row = {"ref_text": gold}
        value, extracted = bench.score("drop", row, reply)
        scored += 1
        tier = which_tier(reply)
        tiers[tier] += 1
        if value >= 0.999:
            correct += 1
            continue
        # Does any of the gold alternatives appear in the reply as written?
        alternatives = [a.strip() for a in gold.split("|") if a.strip()]
        flat = normalise(reply)
        present = any(normalise(a) and normalise(a) in flat for a in alternatives)
        if present and not any(normalise(a) == normalise(extracted) for a in alternatives):
            extraction_miss += 1
            miss_tiers[tier] += 1
            causes[classify(extracted, alternatives)] += 1
            examples.append((value, gold, extracted, reply.strip()[-110:], tier,
                             classify(extracted, alternatives)))
        else:
            model_miss += 1

    print(f"  {scored:,} stored DROP replies from {len(paths)} file(s)\n")
    print(f"  fully correct                      {correct:>7,}  {correct / scored:6.1%}")
    print(f"  model did not find the answer      {model_miss:>7,}  {model_miss / scored:6.1%}")
    print(f"  EXTRACTION MISS (ours to fix)      {extraction_miss:>7,}  "
          f"{extraction_miss / scored:6.1%}")
    print(f"\n  {'tier that fired':<20}{'used':>9}{'extraction misses':>20}{'miss rate':>12}")
    for tier, count in tiers.most_common():
        misses = miss_tiers.get(tier, 0)
        print(f"  {tier:<20}{count:>9,}{misses:>20,}{misses / count:>11.1%}")

    print(f"\n  worst extraction misses (gold is in the reply, we extracted otherwise):")
    print(f"\n  {'why the miss happened':<42}{'count':>8}{'share of misses':>18}")
    for cause, count in causes.most_common():
        print(f"  {cause:<42}{count:>8,}{count / max(extraction_miss, 1):>17.1%}")

    for value, gold, extracted, tail, tier, cause in sorted(
            examples, key=lambda e: (e[5], e[0]))[: args.show]:
        flat_tail = re.sub(r"\s+", " ", tail)
        print(f"    [{cause}] gold={gold[:24]!r:<28} extracted={extracted[:24]!r}")
        print(f"                    reply ends: ...{flat_tail}")


if __name__ == "__main__":
    main()
