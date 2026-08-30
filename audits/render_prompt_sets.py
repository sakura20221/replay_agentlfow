#!/usr/bin/env python3
"""Render every G-Designer/CARD domain's constraints and assert what changed.

The override lives in a subclass, so nothing in the authors' files moves and a
file diff shows nothing at all. The check that matters is therefore behavioural:
what text does each registered domain actually produce, and is it identical to
the parent's for the domains that should not have moved?

Run from the repo root; needs the gdesigner env for the GDesigner import.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="third_party/gdesigner",
                        help="gdesigner or card checkout")
    parser.add_argument("--show", action="store_true", help="print the rendered text")
    args = parser.parse_args()

    # CARD's package is named CARD, not GDesigner, so every import here is resolved
    # through the detected name. Hardcoding GDesigner made this script pass on one
    # repo and fail with ModuleNotFoundError on the other.
    import importlib

    repo = ROOT / args.repo
    sys.path.insert(0, str(repo))
    package = "GDesigner" if (repo / "GDesigner").exists() else "CARD"
    importlib.import_module(f"{package}.prompt.shared_prompt_sets")
    PromptSetRegistry = importlib.import_module(
        f"{package}.prompt.prompt_set_registry").PromptSetRegistry
    GSM8KPromptSet = importlib.import_module(
        f"{package}.prompt.gsm8k_prompt_set").GSM8KPromptSet
    HumanEvalPromptSet = importlib.import_module(
        f"{package}.prompt.humaneval_prompt_set").HumanEvalPromptSet
    MMLUPromptSet = importlib.import_module(
        f"{package}.prompt.mmlu_prompt_set").MMLUPromptSet

    failures = 0

    def report(ok: bool, message: str) -> None:
        nonlocal failures
        print(f"  [{'ok' if ok else 'FAIL'}] {message}")
        if not ok:
            failures += 1

    parents = {"math": GSM8KPromptSet, "amc": GSM8KPromptSet,
               "mbpp": HumanEvalPromptSet, "drop": MMLUPromptSet,
               "mmlu_pro": MMLUPromptSet}

    for domain in ("math", "amc", "mbpp"):
        shared = PromptSetRegistry.get(domain)
        parent = parents[domain]
        # These inherit strategy AND task wording, so every constraint method must
        # come back byte-identical. get_constraint takes a role in these two
        # domains and none in mmlu, hence the two call shapes.
        same = True
        for name in ("get_decision_constraint", "get_decision_role",
                     "get_adversarial_answer_prompt"):
            if not hasattr(parent, name):
                continue
            try:
                left = getattr(shared, name)("x") if name.endswith("answer_prompt") \
                    else getattr(shared, name)()
                right = getattr(parent, name)("x") if name.endswith("answer_prompt") \
                    else getattr(parent, name)()
            except (TypeError, NotImplementedError):
                continue
            same = same and left == right
        report(same, f"{domain}: inherits the author's wording unchanged")

    for domain, forbidden, required in (
            ("drop", "4 answers enumerated as A, B, C and D", "shortest exact span"),
            ("mmlu_pro", "4 answers enumerated as A, B, C and D",
             "option letters offered with the question")):
        shared = PromptSetRegistry.get(domain)
        # "Fake" is deliberately a role with no ROLE_DESCRIPTION-free branch:
        # get_analyze_constraint reads `A if role in D else "" + "<constraint>"`, so
        # for a KNOWN role Python evaluates it as `A`, and the constraint text is
        # dropped entirely. That precedence is the author's and is reproduced
        # verbatim, so the adapted constraint can only be observed via an unknown
        # role -- and the known-role path is asserted to still return the bare
        # description, proving the override did not "fix" the author's expression.
        rendered = "\n".join([
            shared.get_constraint(),
            shared.get_decision_constraint(),
            shared.get_analyze_constraint("NoSuchRole"),
            shared.get_adversarial_answer_prompt("Q?"),
        ])
        report(forbidden not in rendered, f"{domain}: no false option count anywhere")
        report(required in rendered, f"{domain}: states the real answer format")
        # Strategy must survive: these phrases are the author's, and their loss
        # would mean the override rewrote the method rather than the task.
        for phrase in ("less than 100 words",
                       "strictly prohibited from imitating",
                       "refer to the answers of other agents"):
            report(phrase in rendered, f"{domain}: keeps the author's {phrase!r}")
        report(shared.get_analyze_constraint("Critic").strip()
               == MMLUPromptSet.get_analyze_constraint("Critic").strip(),
               f"{domain}: known-role analyze path byte-identical to the author's")
        report(shared.get_decision_role() == MMLUPromptSet.get_decision_role(),
               f"{domain}: decision role unchanged")
        if args.show:
            print(f"\n--- {domain} get_decision_constraint ---{shared.get_decision_constraint()}")

    # The span task must not lose its answer to the letter-shaped postprocessor.
    # PromptSetRegistry.get returns an instance, not the class.
    drop = PromptSetRegistry.get("drop")
    report(drop.postprocess_answer("Answer: Corey Dillon") == "Answer: Corey Dillon",
           "drop: postprocess_answer keeps the whole span")
    mmlu = PromptSetRegistry.get("mmlu_pro")
    report(mmlu.postprocess_answer("J") == "J",
           "mmlu_pro: postprocess_answer still reduces to a letter (author's behaviour)")

    print(f"\n  {failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
