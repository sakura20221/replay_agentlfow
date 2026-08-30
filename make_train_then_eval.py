#!/usr/bin/env python3
"""Build the concatenated train-then-eval files G-Designer and CARD need.

Their runner takes a single --dataset_json and switches from training to
evaluation after --num_iterations batches:

    if i_batch + 1 == args.num_iterations:
        args.optimized_spatial = False      # the optimiser step is guarded by this
        total_solved = 0                    # accuracy counters reset
        graph.gcn.eval()

Two facts make this exploitable rather than a problem: `dataloader` is plain
sequential slicing (`data_list[i*bs : i*bs+bs]`, no shuffle), and the gradient
step is guarded by `optimized_spatial`. So a file whose first N items are the
search split and whose remainder is the evaluation split gives exactly one
process that trains on the search split and then evaluates, with the learned
topology, on held-out data.

Running search and test as two separate processes -- the earlier design -- was
wrong twice over: the test process started from a freshly initialised GCN, so it
discarded everything the search learned, and it ran backward()/step() on the
evaluation split.

The boundary is exact by construction: 256 search items / batch_size 4 = 64
batches, so --num_iterations 64 consumes the search split and nothing else.

    python make_train_then_eval.py            # writes into shared/data/
    python make_train_then_eval.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "shared" / "data"
BATCH_SIZE = 4

# dataset -> (search split, eval split). AMC uses FlowBank's shipped split
# verbatim, adopted 2026-08-24 for comparability with its published numbers.
PAIRS = {
    "math": ("math_search.jsonl", "math.jsonl"),
    "amc": ("amc_search.jsonl", "amc.jsonl"),
    "mbpp": ("mbpp_search.jsonl", "mbpp.jsonl"),
    "drop": ("drop_search.jsonl", "drop.jsonl"),
    "mmlu_pro": ("mmlu_pro_search.jsonl", "mmlu_pro.jsonl"),
}


def read(path: Path) -> list[dict]:
    # Iterated rather than splitlines(): str.splitlines() also breaks on \x85,
    # U+2028 and U+2029, which occur inside MMLU-Pro question text.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build(check: bool) -> int:
    problems = 0
    for name, (search_file, eval_file) in PAIRS.items():
        search_path, eval_path = DATA / search_file, DATA / eval_file
        if not search_path.exists() or not eval_path.exists():
            print(f"  [FAIL] {name}: missing {search_file} or {eval_file}")
            problems += 1
            continue

        search_rows, eval_rows = read(search_path), read(eval_path)
        target = DATA / f"{name}_train_then_eval.jsonl"

        n_train_batches, remainder = divmod(len(search_rows), BATCH_SIZE)
        if remainder:
            # Trim to a whole number of batches so the switch cannot land inside
            # a batch that mixes search and evaluation items.
            search_rows = search_rows[: n_train_batches * BATCH_SIZE]

        if check:
            if not target.exists():
                print(f"  [FAIL] {name}: {target.name} missing")
                problems += 1
                continue
            rows = read(target)
            head_ok = [r.get("uid") for r in rows[: len(search_rows)]] == \
                      [r.get("uid") for r in search_rows]
            tail_ok = [r.get("uid") for r in rows[len(search_rows):]] == \
                      [r.get("uid") for r in eval_rows]
            ok = head_ok and tail_ok
            print(f"  [{'ok' if ok else 'FAIL'}] {name}: {len(rows)} rows, "
                  f"train batches={n_train_batches}, eval items={len(eval_rows)}")
            problems += 0 if ok else 1
            continue

        with target.open("w", encoding="utf-8") as handle:
            for row in search_rows + eval_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                        separators=(",", ":")) + "\n")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()[:16]
        dropped = len(eval_rows) % BATCH_SIZE
        note = f", {dropped} eval item(s) fall in a partial final batch" if dropped else ""
        print(f"  [ok] {target.name}: {len(search_rows)} train + {len(eval_rows)} eval, "
              f"--num_iterations {n_train_batches}, sha256={digest}{note}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print("[train_then_eval] " + ("verifying" if args.check else "building"))
    problems = build(args.check)
    print("=" * 60)
    if problems:
        raise SystemExit(f"{problems} problem(s)")
    print("train-then-eval splits OK")


if __name__ == "__main__":
    main()
