#!/usr/bin/env python3
"""做对却 0 分的比例 —— 对每个 (方法, 数据集),从存下来的逐题记录直接量。

这是判分正确性的**最终**度量,在真正被判分的那一层上算(不是 transcript 的
中间层 —— XML 信封那次的教训:算子返回前已解析信封,transcript 里的原始回复
不是判分看到的东西)。每条记录用**当前**判分器重判,然后按三类分桶:

  满分 / 部分分     正常
  0 分且确实答错    正常(模型的问题)
  0 分但答案在      我们的问题 —— 抽出来的 span 里含 gold,却得了 0

第三类的比例应接近 0;每个非零案例都打出来供人工裁决(子串启发式会误报,
gold "males" 是 "females" 的子串这种按用户裁决不算问题)。

各方法的存储格式不同,全部覆盖:
  gdesigner 系   result/<tag>/*.json      Question/Answer/Response/Solved
  maas 系        optimized/KEY/*/round_*/0.*.csv    question/prediction/expected_output/score
  aflow          workspace/KEY/workflows*/round_*/0.*.csv   同上
  flowbank       workspace/KEY/workflows/round_*/0.*.jsonl  同上
  masrouter      logs/scored_items_<dataset>.jsonl

    envs/maas/bin/python audits/correct_but_zero.py --since '2026-08-23 08:00'
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import glob
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
import bench  # noqa: E402

KEY = {d: f"SHARED_{d.upper().replace('_', '')}" for d in bench.DATASETS}
# gold 代码 -> 数据集行,mbpp 重判时回连 test 字段用。
_MBPP_BY_CODE = {str(r.get("code", "")).strip(): r for r in bench.load("mbpp")}


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


def sources(dataset: str, tag: str) -> list[tuple[str, str, Path]]:
    """(method, phase, path) for every per-item file of this dataset."""
    key = KEY[dataset]
    out: list[tuple[str, str, Path]] = []
    for method in ("gdesigner", "card", "gdesigner_authordefault", "card_authordefault"):
        repo = "card" if "card" in method else "gdesigner"
        for p in ROOT.glob(f"third_party/{repo}/result/{tag}/{method}_{dataset}_r*.json"):
            out.append((method, "train+eval", p))
    for method in ("maas", "daao"):
        base = ROOT / f"third_party/{method}/{method}/ext/maas/scripts/optimized/{key}"
        for phase in ("train", "test"):
            for p in base.glob(f"{phase}/round_*/0.*.csv"):
                out.append((method, phase, p))
    for p in (ROOT / f"third_party/aflow/workspace/{key}").glob("workflows*/round_*/0.*.csv"):
        out.append(("aflow", "search" if "workflows/" in str(p) else "test", p))
    for p in (ROOT / f"third_party/flowbank/DiverseFlow/workspace/{key}").glob(
            "workflows*/round_*/0.*.jsonl"):
        out.append(("flowbank", "search", p))
    return out


def records(path: Path):
    """(gold, prediction, stored_score) from any of the three storage formats."""
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(data, dict):
            data = list(data.values())
        for r in data or []:
            if r.get("Answer") is not None and r.get("Response") is not None:
                yield str(r["Answer"]), unwrap(r["Response"]), r.get("Solved")
    elif path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("expected_output") and r.get("prediction") is not None:
                yield str(r["expected_output"]), str(r["prediction"]), r.get("score")
    else:
        try:
            rows = list(csv.DictReader(path.open(newline="", errors="replace")))
        except OSError:
            return
        for r in rows:
            if r.get("expected_output") and r.get("prediction") is not None:
                yield str(r["expected_output"]), str(r["prediction"]), r.get("score")


def flat(text: str) -> str:
    return re.sub(r"[^a-z0-9.]+", " ", text.lower()).strip()


_MMLU_TAIL = re.compile(r"answer\s*(?:is)?\s*[::]?\s*\(?([A-J])\)?(?![A-Za-z])",
                        re.IGNORECASE)


def gold_in(extracted: str, gold: str, dataset: str, prediction: str = "") -> bool:
    """极简的"答案在里面"判断,只做人工裁决的预筛。

    mmlu_pro 单独用"回复末尾声明的字母 == gold"判断:子串式预筛要求
    len(gold) > 2,单字母 gold 永远不会被标记 —— 这正是 2026-08-24 的
    字母抽取缺陷能带着 168 个错判分潜伏到 daao 全量跑完的原因。
    """
    if dataset == "mmlu_pro":
        found = _MMLU_TAIL.findall(prediction or "")
        return bool(found) and found[-1].upper() == gold.strip().upper()
    alts = [a.strip() for a in gold.split("|") if a.strip()] if dataset == "drop" else [gold]
    body = flat(extracted)
    for a in alts:
        fa = flat(a)
        if fa and (fa == body or (len(fa) > 2 and fa in body)):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["drop", "math", "mmlu_pro"])
    parser.add_argument("--tag", default="v4")
    parser.add_argument("--since", default=None,
                        help="只看这个时刻之后写的文件,挡住作者仓库自带的 2025 年产物")
    parser.add_argument("--show", type=int, default=6)
    parser.add_argument("--fail-on-suspect", action="store_true",
                        help="exit non-zero when any candidate needs manual review")
    args = parser.parse_args()
    floor = time.mktime(time.strptime(args.since, "%Y-%m-%d %H:%M")) if args.since else 0.0

    suspect_total = 0
    for dataset in args.datasets:
        print(f"\n{'=' * 88}\n### {dataset}")
        cells: dict[str, dict] = collections.defaultdict(
            lambda: {"n": 0, "full": 0, "partial": 0, "zero": 0, "suspect": [],
                     "stored_sum": 0.0, "fresh_sum": 0.0})
        found_any = False
        for method, phase, path in sources(dataset, args.tag):
            if path.stat().st_mtime < floor:
                continue
            found_any = True
            cell = cells[f"{method}/{phase}"]
            for gold, pred, stored in records(path):
                # mmlu_pro 的判分要看选项数(字母空间上限)。逐题落盘里没有
                # options,而字母抽取只需要知道有几个选项 —— 给满 10 个,与
                # 数据集构造一致;gold 本身就是正确字母。
                #
                # mbpp 的判分要执行测试用例,落盘里没有 test 字段 —— 按 gold
                # 代码回连数据集行(参考解在划分内唯一),连不上就跳过并计数,
                # 而不是构造假 row 让判分器崩掉。
                if dataset == "mbpp":
                    row = _MBPP_BY_CODE.get(gold.strip())
                    if row is None:
                        cell["unmatched"] = cell.get("unmatched", 0) + 1
                        continue
                else:
                    row = {"ref_text": gold, "answer": gold, "code": gold,
                           "options": list("ABCDEFGHIJ")}
                value, extracted = bench.score(dataset, row, pred)
                cell["n"] += 1
                cell["fresh_sum"] += value
                try:
                    cell["stored_sum"] += float(stored or 0.0)
                except (TypeError, ValueError):
                    pass
                if value >= 0.999:
                    cell["full"] += 1
                elif value > 0:
                    cell["partial"] += 1
                else:
                    cell["zero"] += 1
                    if gold_in(extracted, gold, dataset, pred):
                        cell["suspect"].append((gold, extracted))
        if not found_any:
            print("  (还没有本轮的逐题落盘)")
            continue
        print(f"  {'cell':<32}{'n':>7}{'满分':>7}{'部分':>7}{'0分':>7}"
              f"{'做对却0分':>10}{'重判均分':>10}{'落盘均分':>10}")
        for name, c in sorted(cells.items()):
            if not c["n"]:
                continue
            print(f"  {name:<32}{c['n']:>7,}{c['full']:>7,}{c['partial']:>7,}"
                  f"{c['zero']:>7,}{len(c['suspect']):>10,}"
                  f"{c['fresh_sum'] / c['n']:>10.4f}{c['stored_sum'] / c['n']:>10.4f}")
        suspects = [(name, g, e) for name, c in cells.items() for g, e in c["suspect"]]
        suspect_total += len(suspects)
        if suspects:
            print(f"\n  待人工裁决的「做对却 0 分」案例(前 {args.show} 条):")
            for name, gold, extracted in suspects[: args.show]:
                print(f"    [{name}] gold={gold[:30]!r:<34} extracted={extracted[:40]!r}")
        else:
            print("\n  没有「做对却 0 分」的案例")
    print()
    if args.fail_on_suspect and suspect_total:
        raise SystemExit(f"{suspect_total} correct-but-zero candidate(s) need review")


if __name__ == "__main__":
    main()
