#!/usr/bin/env python3
"""Adopt FlowBank's shipped AMC split verbatim (user decision, 2026-08-24).

The evaluation set becomes FlowBank's amc_test.jsonl (655 rows) and the search
set its amc_validate.jsonl (165 rows), replacing AI-MO/aimo-validation-amc (83
rows) and the borrow-MATH-search substitution. Motive: comparability -- FlowBank
and AFlow report AMC numbers on exactly these rows.

Deviation from verbatim, by user decision (2026-08-24): test rows whose content
also appears in amc_validate are REMOVED from the evaluation set -- those are
problems the optimisers get to see during search, and a test set must not
contain them. The removed rows are archived, the count is declared, and the
published-number comparison carries a ~1% caveat (their numbers include the
leaked items). The remaining defects are inherited and declared rather than
fixed: content duplicates inside each file and two golds that are a bare choice
letter for a problem whose options are not in the text.

Problem/solution text is copied byte-for-byte; only the fields our harness needs
are added: uid, question (= problem, AFlow's amc benchmark reads "question"),
and answer = the last \\boxed of the shipped solution, which is exactly how
FlowBank's own scorer derives gold.

    envs/tools/bin/python adopt_flowbank_amc.py          # dry run
    envs/tools/bin/python adopt_flowbank_amc.py --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "shared" / "data"
SRC = ROOT / "third_party" / "flowbank" / "datasets"


def boxed_tail(solution: str) -> str:
    index = solution.rfind("\\boxed")
    if index < 0:
        return ""
    tail = solution[index + len("\\boxed"):]
    if not tail.startswith("{"):
        return tail.strip().split("$")[0].strip()
    depth = 0
    for position, char in enumerate(tail):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return tail[1:position]
    return ""


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def convert(split: str, filename: str) -> list[dict]:
    rows = []
    for i, line in enumerate((SRC / filename).open(encoding="utf-8")):
        raw = json.loads(line)
        answer = boxed_tail(raw["solution"])
        if not answer:
            raise SystemExit(f"{filename} row {i}: no \\boxed gold; refusing")
        rows.append({
            "uid": f"amc/fb-{split}-{i}",
            "problem": raw["problem"],
            "question": raw["problem"],
            "answer": answer,
            "solution": raw["solution"],
            "level": raw.get("level"),
            "type": raw.get("type"),
        })
    return rows


def write(name: str, rows: list[dict]) -> dict:
    path = DATA / f"{name}.jsonl"
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write(line + "\n")
            digest.update(line.encode("utf-8"))
    return {"file": path.name, "n": len(rows), "sha256": digest.hexdigest()[:16]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    test = convert("test", "amc_test.jsonl")
    validate = convert("validate", "amc_validate.jsonl")

    # The defects being inherited, measured so the declaration has numbers.
    test_norms = [norm(r["problem"]) for r in test]
    val_norms = [norm(r["problem"]) for r in validate]
    letter = re.compile(r"^\(?([A-Ea-e])\)?$")
    stats = {
        "test_dup_contents": len(test_norms) - len(set(test_norms)),
        "validate_dup_contents": len(val_norms) - len(set(val_norms)),
        "validate_test_overlap": len(set(val_norms) & set(test_norms)),
        "letter_golds_test": sum(1 for r in test if letter.fullmatch(r["answer"].strip())),
        "letter_golds_validate": sum(1 for r in validate if letter.fullmatch(r["answer"].strip())),
        "levels_test": dict(Counter(str(r["level"]) for r in test)),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # The user-decided leak filter: no test row may share content with the
    # search pool. Matching is on normalised problem text, so a leaked problem
    # that is ALSO duplicated inside the test file loses every copy.
    val_set = set(val_norms)
    removed = [r for r in test if norm(r["problem"]) in val_set]
    test = [r for r in test if norm(r["problem"]) not in val_set]
    print(f"leak filter: removed {len(removed)} test row(s) seen in validate, "
          f"{len(test)} remain")
    for r in removed:
        print(f"  - {r['uid']} [{r['level']}/{r['type']}] answer={r['answer'][:20]!r} "
              f"{r['problem'][:70]!r}")

    if not args.apply:
        print(f"dry run: would write amc.jsonl ({len(test)}) + amc_search.jsonl "
              f"({len(validate)}), retiring the AI-MO 83-item set. --apply to do it.")
        return

    old = DATA / "amc.jsonl"
    if old.exists() and "aimo" not in old.read_text(encoding="utf-8")[:400].lower():
        # Already swapped once; keep the archive from the first run.
        print("amc.jsonl is not the AI-MO file; skipping the backup step")
    elif old.exists():
        backup = ROOT / "archive" / "amc83_aimo"
        backup.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(backup / "amc.jsonl"))
        (backup / "WHY_RETIRED.txt").write_text(
            "AI-MO/aimo-validation-amc (83 items, the 2022-2023 AMC12 papers).\n"
            "Retired 2026-08-24: the user chose FlowBank's shipped AMC split so the\n"
            "AMC column is directly comparable with its published numbers, accepting\n"
            "that this set is easier (measured 0.70 vs 0.55 single-shot on Qwen3-8B)\n"
            "and carries the defects declared in shared/data/manifest.json.\n",
            encoding="utf-8")
        print(f"archived the AI-MO set -> {backup.relative_to(ROOT)}/")

    keep = ROOT / "archive" / "amc_leak_removed.jsonl"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                            for r in removed), encoding="utf-8")

    info_test = write("amc", test)
    info_val = write("amc_search", validate)

    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = ("third_party/flowbank/datasets (shipped with the FlowBank repo, "
                  "origin undocumented upstream; historical AMC8/10/12 problems, "
                  "none from 2022-2023)")
    manifest["datasets"]["amc"] = {
        **info_test, "source": provenance,
        "sampling": ("their amc_test.jsonl minus the rows whose content appears in "
                     "amc_validate (user decision 2026-08-24; "
                     "archive/amc_leak_removed.jsonl)"),
        "adopted": time.strftime("%Y-%m-%d"),
        "declared_defects": stats,
        "replaces": "AI-MO/aimo-validation-amc n=83 (archive/amc83_aimo/)",
    }
    manifest.setdefault("search_splits", {})["amc_search"] = {
        **info_val, "source": provenance,
        "sampling": "none (their amc_validate.jsonl verbatim)",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"wrote {info_test} and {info_val}; manifest updated")


if __name__ == "__main__":
    main()
