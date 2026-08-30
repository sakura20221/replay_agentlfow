"""Grade realistic reply shapes, not just the one the instruction asks for.

A multi-agent method's last node rewrites the answer in its own words, so the
grader meets far more shapes than `\\boxed{}` / `Answer: X`. Every shape here was
produced by Qwen3-8B during the smoke runs. Before the tiered extractors, MATH,
AMC and MMLU-Pro accepted only about half of them -- a correct answer phrased
"Thus, 142." scored zero, which penalises exactly the methods that paraphrase.
"""
import sys
sys.path.insert(0, ".")
import bench

# (dataset, gold, reply, label) -- every reply states the gold answer correctly,
# so a correct grader must score all of them 1.0.
CASES = [
    ("math", "142", "Work: ...\nThus, 142.", "connective 'Thus,'"),
    ("math", "142", "Work: ...\nTherefore 142", "connective 'Therefore'"),
    ("math", "142", "Work: ...\n\\boxed{142}", "boxed"),
    ("math", "142", "Work: ...\nAnswer: 142", "Answer: lead"),
    ("math", "142", "Work: ...\nThe answer is **142**.", "bold"),
    ("math", "142", "142", "bare answer only"),
    ("amc", "142.0", "Work: ...\nThus, 142.0", "connective with decimal"),
    ("amc", "142.0", "Work: ...\nTherefore 142.0", "connective 'Therefore'"),
    ("amc", "142.0", "Work: ...\n\\boxed{142.0}", "boxed"),
    ("amc", "142.0", "Work: ...\nThe answer is 142.0.", "answer-is"),
    ("amc", "142.0", "Reasoning.\n\nFinal answer: 142.0", "final-answer lead"),
    ("amc", "142.0", "142.0", "bare answer only"),
    ("drop", "57", "Counting: ...\nThus, 57.", "connective 'Thus,'"),
    ("drop", "57", "Counting: ...\nTherefore 57", "connective 'Therefore'"),
    ("drop", "57", "Counting: ...\nAnswer: 57", "Answer: lead"),
    ("drop", "57", "Counting: ...\nThe answer is **57**", "bold"),
    ("drop", "57", "Counting: ...\n\\boxed{57}", "boxed"),
    ("drop", "57", "57", "bare span only"),
    ("mmlu_pro", "J", "Reasoning ...\nAnswer: (J)", "Answer: (X)"),
    ("mmlu_pro", "J", "Reasoning ...\nAnswer: J", "Answer: X"),
    ("mmlu_pro", "J", "Reasoning ...\nThe answer is (J).", "answer-is"),
    ("mmlu_pro", "J", "Reasoning ...\nOption J is correct.", "option-word"),
    ("mmlu_pro", "J", "Reasoning ...\n**J**", "bold letter"),
    ("mmlu_pro", "J", "Reasoning ...\n\\boxed{J}", "boxed letter"),
]

# mmlu_pro needs options present to grade a letter; the others read the gold field.
ROW = {
    "math": lambda g: {"problem": "x", "answer": g},
    "amc": lambda g: {"problem": "x", "answer": g},
    "drop": lambda g: {"context": "x", "ref_text": g},
    "mmlu_pro": lambda g: {"question": "x", "options": ["a"] * 10,
                           "answer": g, "answer_index": bench.MMLU_PRO_LETTERS.index(g)},
}


def main() -> int:
    passed = 0
    for dataset, gold, reply, label in CASES:
        try:
            value, extracted = bench.score(dataset, ROW[dataset](gold), reply)
        except Exception as exc:  # noqa: BLE001
            value, extracted = f"RAISED {type(exc).__name__}: {exc}", ""
        hit = value == 1.0
        passed += hit
        print(f"  {'OK  ' if hit else 'FAIL'} {dataset:<9} {label:<26} "
              f"score={value} extracted={extracted!r}")
    print(f"\n  -> {passed}/{len(CASES)} realistic shapes graded correctly")
    for dataset in ("math", "amc", "drop", "mmlu_pro"):
        stats = dict(bench.extraction_stats().get(dataset, {}))
        print(f"     {dataset:<9} {stats}")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
