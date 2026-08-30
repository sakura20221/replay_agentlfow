#!/usr/bin/env python3
"""Evaluate AFlow's best searched workflow on the held-out test split.

AFlow has no reachable held-out evaluation. `Optimizer.test()` exists, but:

* `run.py` exposes no flag that reaches it -- the authors left
  `# optimizer.optimize("Test")` commented out next to the Graph call;
* `test()` hardcodes `rounds = [1]`, with the comment "You can choose the rounds
  you want to test here";
* it loads the graph from `{root_path}/workflows_test`, a directory the repo never
  creates.

So a plain `run.py` invocation only ever reports validation scores on the 256-item
search split. Claiming those as held-out numbers would be wrong, which is what an
earlier revision of the sweep did by marking AFlow's test phase as "not needed".

This driver supplies the three missing pieces: it picks the best round from the
search history, materialises it as `workflows_test/round_1`, and calls
`optimize("Test")`. The evaluation itself is entirely the authors' --
`graph_evaluate(..., is_test=True)` sets `va_list = None`, i.e. the whole test
split -- so nothing about how AFlow is measured is invented here.

    cd third_party/aflow && python ../../aflow_test.py --dataset SHARED_MATH
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# The module path a copied graph must import from. graph.py refers to its
# operators and prompts by absolute module path
# (`import workspace.SHARED_MATH.workflows.round_7.prompt as prompt_custom`), so
# the copies have to be rewritten to point inside workflows_test/round_1 -- left
# alone they would import the *search* tree, and any later edit there would
# silently change what the test run executes.
WORKFLOW_IMPORT = re.compile(r"\bworkspace\.([A-Za-z0-9_]+)\.workflows\.(round_\d+|template)\.")


def best_round(results_path: Path) -> tuple[int, float]:
    if not results_path.exists():
        raise SystemExit(f"no search history at {results_path}; run the search phase first")
    data = json.loads(results_path.read_text(encoding="utf-8"))
    scored = [r for r in data if isinstance(r.get("score"), (int, float))]
    if not scored:
        raise SystemExit(f"{results_path} has no scored round")
    best = max(scored, key=lambda r: r["score"])
    return int(best["round"]), float(best["score"])


def rewrite(text: str, key: str) -> str:
    def repl(match: re.Match) -> str:
        part = match.group(2)
        return f"workspace.{key}.workflows_test.{'round_1' if part.startswith('round_') else part}."
    return WORKFLOW_IMPORT.sub(repl, text)


def materialise(workspace: Path, key: str, round_number: int) -> None:
    source = workspace / "workflows"
    target = workspace / "workflows_test"
    if target.exists():
        # Rebuilt every time: a stale round_1 from a previous search would be
        # evaluated silently, and its score would look like a fresh result.
        shutil.rmtree(target)
    target.mkdir(parents=True)

    (target / "__init__.py").write_text("", encoding="utf-8")
    shutil.copytree(source / "template", target / "template")
    round_dir = target / "round_1"
    round_dir.mkdir()
    (round_dir / "__init__.py").write_text("", encoding="utf-8")
    for name in ("graph.py", "prompt.py"):
        src = source / f"round_{round_number}" / name
        if src.exists():
            shutil.copyfile(src, round_dir / name)
        elif name == "graph.py":
            raise SystemExit(f"round {round_number} has no graph.py")

    rewritten = 0
    for path in sorted(target.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        new_text = rewrite(text, key)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            rewritten += 1
    print(f"  workflows_test/round_1 <- round_{round_number} "
          f"({rewritten} file(s) had import paths rewritten)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="e.g. SHARED_MATH")
    parser.add_argument("--opt_model_name", default="qwen3-8b")
    parser.add_argument("--exec_model_name", default="qwen3-8b")
    parser.add_argument("--sample", type=int, default=4)
    parser.add_argument("--optimized_path", default="workspace")
    parser.add_argument("--round", type=int, default=None,
                        help="force this round instead of the stored-score winner. "
                             "results.json keeps the scores the search-era scorer "
                             "produced; pass the regrade_rounds winner so selection "
                             "reflects the current scorer.")
    args = parser.parse_args()

    sys.path.insert(0, str(Path.cwd()))
    import run as aflow_run  # noqa: E402  - reads EXPERIMENT_CONFIGS, incl. the shim's
    from scripts.async_llm import LLMsConfig  # noqa: E402
    from scripts.optimizer import Optimizer  # noqa: E402

    config = aflow_run.EXPERIMENT_CONFIGS.get(args.dataset)
    if config is None:
        raise SystemExit(f"{args.dataset} is not registered in run.py")

    workspace = Path(args.optimized_path) / args.dataset
    if args.round is not None:
        round_number = args.round
        print(f"  forced round = {round_number} (regraded-winner override)")
    else:
        round_number, score = best_round(workspace / "workflows" / "results.json")
        print(f"  best searched round = {round_number} (validation score {score:.4f})")
    materialise(workspace, args.dataset, round_number)

    models = LLMsConfig.default()
    optimizer = Optimizer(
        dataset=config.dataset,
        question_type=config.question_type,
        opt_llm_config=models.get(args.opt_model_name),
        exec_llm_config=models.get(args.exec_model_name),
        operators=config.operators,
        optimized_path=args.optimized_path,
        sample=args.sample,
        # test() reads self.round when building the round directory, so it has to
        # be 1 to match the round_1 materialised above.
        initial_round=1,
        max_rounds=1,
        validation_rounds=1,
        check_convergence=False,
    )
    optimizer.optimize("Test")

    results = workspace / "workflows_test" / "results.json"
    if results.exists():
        data = json.loads(results.read_text(encoding="utf-8"))
        for entry in data:
            print(f"  HELD-OUT {args.dataset}: score={entry.get('score')} "
                  f"(from searched round {round_number})")
    else:
        raise SystemExit("test() produced no results.json")


if __name__ == "__main__":
    main()
