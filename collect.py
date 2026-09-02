#!/usr/bin/env python3
"""Collect held-out scores and measured budget for every finished job.

Each repo reports its result differently, so this is a per-method extractor
registry rather than one parser. Two rules apply throughout:

1. **Recompute rather than trust a printed number.** G-Designer and CARD run
   training and evaluation in one process over one concatenated file and print a
   running accuracy, so the figure on screen mixes the two phases at the moment it
   was printed. Their per-item records are re-scored here, keeping only items that
   belong to the evaluation split -- matched by stable item uid against
   shared/data, not by question text or by assuming a record offset.

2. **Say "pending" or "unavailable", never guess.** A missing number is a phase
   that has not finished or an extractor that does not apply; inventing one is how
   a plausible-looking table stops meaning anything.

Budget comes from each job's own stdout: the processes are separate, so their
cumulative token counters are per job, whereas the proxy's namespaces are per
method and cannot separate concurrent datasets.

    python collect.py
    python collect.py --json runs/summary_table.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = Path(os.getenv("SWEEP_RUNS", str(ROOT / "runs")))
# The tag the sweep actually ran under, read from the file sweep.py leaves beside
# the runs. Re-deriving it from the directory name here was wrong whenever the
# sweep was launched with SWEEP_TAG set, and the symptom -- "no result file yet" --
# is indistinguishable from a job that failed to produce one.
def _run_tag() -> str:
    marker = RUNS / "TAG"
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip()
    return os.getenv("SWEEP_TAG", RUNS.name)


RUN_TAG = os.getenv("SWEEP_TAG") or _run_tag()
DATA = ROOT / "shared" / "data"
sys.path.insert(0, str(ROOT / "shared"))

DATASETS = ("math", "amc", "mbpp", "drop", "mmlu_pro")

# Which repo directory each method's artefacts live under.
REPO = {
    "gdesigner": "third_party/gdesigner",
    "card": "third_party/card",
    "gdesigner_authordefault": "third_party/gdesigner",
    "card_authordefault": "third_party/card",
    "maas": "third_party/maas",
    "daao": "third_party/daao",
    "masrouter": "third_party/masrouter",
    "aflow": "third_party/aflow",
    "flowbank": "third_party/flowbank/DiverseFlow",
}

SHARED_KEY = {d: f"SHARED_{d.upper().replace('_', '')}" for d in DATASETS}


def has_isolated_artifacts(job: Path) -> bool:
    """Whether this job was launched after run-scoped artifacts were added."""
    path = job / "protocol.json"
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("run_tag"))
    except (OSError, json.JSONDecodeError):
        return False


def eval_uids(dataset: str) -> set[str]:
    """The uid of every item in the evaluation split.

    Matching on uid, not on question text. Text matching failed twice over:

    * the G-Designer family stores `Question` as the whole dataset row -- a dict
      of task/ref_text/uid -- so its serialisation begins `{"task": ...` and never
      equals the plain question; all 1256 records of card/drop matched nothing;
    * truncating the text to a prefix collapses distinct items. DROP asks several
      questions about one passage, and `question_text` for DROP is the passage, so
      1000 evaluation items produced only 405 distinct keys.

    uid is carried by every row of all five splits (`drop/c8d2fefa-...`,
    `mmlu_pro/2823`, ...) and is what make_train_then_eval.py already keys on.
    """
    import bench as shared_bench

    return {str(row["uid"]) for row in shared_bench.load(dataset)}


def record_uid(record: dict) -> str | None:
    """The uid of one per-item record, wherever the repo happened to put it."""
    question = record.get("Question")
    if isinstance(question, dict):
        uid = question.get("uid")
        if uid:
            return str(uid)
        # Some rows nest the original item one level down.
        for value in question.values():
            if isinstance(value, dict) and value.get("uid"):
                return str(value["uid"])
    for key in ("uid", "id", "task_id"):
        if record.get(key):
            return str(record[key])
    return None


_MBPP_ROWS: dict[str, dict] | None = None


def _grading_row(dataset: str, gold: str) -> dict | None:
    """A row the CURRENT scorer can actually grade against, from a stored gold.

    The old stub {ref_text, answer, code} silently under-specified two datasets:
    mmlu_pro's scorer needs the option list to bound the letter space (KeyError:
    'options') and mbpp's needs the real test harness (KeyError: 'test') -- a
    gold CODE alone cannot be executed against anything. In regrade() the crash
    was swallowed by the fallback and quietly returned the STORED verdict while
    the note still said "recomputed"; in from_maas_csv it surfaced the moment
    the first mbpp cell completed (2026-08-24, daao/mbpp). mmlu_pro gets the
    full letter space (the gold is the correct letter; same construction the
    audits use), mbpp reconnects the dataset row by its reference solution,
    which is unique within the split.
    """
    global _MBPP_ROWS
    import bench as shared_bench

    if dataset == "mbpp":
        if _MBPP_ROWS is None:
            rows = list(shared_bench.load("mbpp"))
            # Validation golds live in the SEARCH split, and regrade_rounds
            # feeds those csvs too; an unmatched gold silently skips the row.
            # Found 2026-08-26: the mbpp round regrade matched 1/128 items,
            # every round re-graded 0.0000, and the "regraded winner"
            # degenerated to round_1 (validation 0.63 vs round_4's 0.68).
            search_path = ROOT / "shared" / "data" / "mbpp_search.jsonl"
            if search_path.exists():
                rows += [json.loads(line) for line
                         in search_path.read_text(encoding="utf-8").splitlines()
                         if line.strip()]
            _MBPP_ROWS = {str(r.get("code", "")).strip(): r for r in rows}
        return _MBPP_ROWS.get(gold.strip())
    if dataset == "mmlu_pro":
        return {"answer": gold, "options": list("ABCDEFGHIJ")}
    return {"ref_text": gold, "answer": gold, "code": gold}


def regrade(dataset: str, record: dict) -> float:
    """Grade one stored record with the CURRENT scorer, not the one it was written with.

    Reading `Solved` looked like recomputation and was not: that field holds the
    verdict the run reached at the time, so a scorer fix applied afterwards changed
    nothing in this table. It hid a real error -- the DROP extractor truncated
    "Answer: 87.9" to "87", and 201 of G-Designer's 1,392 DROP records had a decimal
    gold, every one of them mis-graded -- while the collector cheerfully reported
    the note "recomputed".

    Grading is a pure function of (reply, gold) and both must be in every current
    record. Missing inputs or a grading exception are collection failures; using
    the stored verdict would silently mix scorer versions.
    """
    import bench as shared_bench

    reply = record.get("Response")
    gold = record.get("Answer")
    if reply is None or gold is None:
        raise ValueError("stored item lacks Response or Answer; cannot re-grade")
    # The G-Designer family stores the reply as the repr of a one-element list,
    # e.g. "['Answer: Yangshao']". Left as-is the extractor drags the closing
    # bracket and quote into the span.
    text = reply
    if isinstance(text, str) and text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list) and parsed:
                text = str(parsed[0])
        except (ValueError, SyntaxError):
            pass
    elif isinstance(text, list):
        text = str(text[0]) if text else ""
    row = _grading_row(dataset, str(gold))
    if row is None:
        raise ValueError("stored MBPP gold matches no frozen dataset row")
    value, _extracted = shared_bench.score(dataset, row, str(text))
    return float(value)


def protocol_mismatch(job: Path, method: str, dataset: str) -> str | None:
    """Whether this job ran under the complete protocol now implemented.

    Different prompts, search-time grading, data, sampling, or method adapters
    can all change the selected artifact. Returning a reason turns an otherwise
    invisible mixed-protocol average into a visible refusal.
    """
    import bench as shared_bench

    stamp = job / "protocol.json"
    from vllm_proxy import sampling_protocol

    current = {
        **shared_bench.protocol_fingerprint(),
        "data": shared_bench.data_fingerprint(dataset),
        "sampling": sampling_protocol(),
        "method": method,
        "dataset": dataset,
        "run_tag": RUN_TAG,
        "repeat": 1,
    }
    if not stamp.exists():
        return "no protocol.json: predates protocol stamping, cannot be shown to match"
    try:
        recorded = json.loads(stamp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "protocol.json unreadable"
    mismatches = [key for key, value in current.items()
                  if recorded.get(key) != value]
    if mismatches:
        detail = ", ".join(
            f"{key}={recorded.get(key)!r} (current {current[key]!r})"
            for key in mismatches
        )
        return (f"protocol mismatch: {detail}. Search rewards or model inputs may "
                "differ; re-grading final replies cannot prove equivalence")
    return None


def _find_result_file(method: str, dataset: str) -> Path | None:
    """The per-job file if the run had the --result_file fix, else the legacy name.

    Jobs launched before that fix still write
    `result/<domain>/<domain>_<llm>_<timestamp>.json`, whose stamp has one-second
    resolution -- so two concurrent jobs of the same domain can share one file.
    The record-count check below is what makes that detectable rather than silent.
    """
    per_job = ROOT / REPO[method] / "result" / RUN_TAG / f"{method}_{dataset}_r1.json"
    if per_job.exists():
        return per_job
    legacy = sorted((ROOT / REPO[method] / "result" / dataset).glob(f"{dataset}_*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    return legacy[0] if legacy else None


def from_item_records(job: Path, method: str, dataset: str) -> dict:
    """G-Designer / CARD: re-score the per-item records over evaluation items."""
    result_file = _find_result_file(method, dataset)
    if result_file is None:
        return {"score": None, "note": "no result file yet"}
    try:
        records = json.loads(result_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"score": None, "note": "result file not valid JSON (run may be mid-write)"}
    if not isinstance(records, list) or not records:
        return {"score": None, "note": "result file empty"}

    wanted = eval_uids(dataset)
    solved = total = 0
    unidentified = 0
    # Last write wins per uid: a retried item appears twice, and the retry is the
    # outcome the run actually ended with.
    seen: dict[str, float] = {}
    for record in records:
        uid = record_uid(record)
        if uid is None:
            unidentified += 1
            continue
        if uid not in wanted:
            continue
        seen[uid] = regrade(dataset, record)
    total = len(seen)
    solved = sum(seen.values())
    if not total:
        return {"score": None,
                "note": f"{len(records)} records, none carried an evaluation uid "
                        f"({unidentified} had no uid at all)"}

    # Two jobs writing one file is undetectable from the scores alone: records from
    # the other configuration are also evaluation items, so they pass the text
    # filter and silently blend into the accuracy. The item count is the giveaway --
    # the evaluation split has a known size, and no single run can exceed it.
    expected = len(wanted)
    # Deduplicating by uid makes the old "two jobs in one file" check impossible to
    # trip by record count alone, so the guard is now about *coverage*: a run that
    # only reached part of the split must not be reported as if it had finished.
    if total > expected:
        return {"score": None,
                "note": f"REFUSING: {total} distinct eval uids but the split has "
                        f"{expected}; {result_file.name} is not from one run"}
    coverage = total / expected
    finished = (job / "status").exists() and (job / "status").read_text().strip() == "ok"
    if finished and RUN_TAG != "smoke" and total != expected:
        raise ValueError(f"finished result covers {total}/{expected} evaluation items")
    if finished and RUN_TAG != "smoke" and unidentified:
        raise ValueError(f"finished result contains {unidentified} item(s) without uid")
    note = (f"recomputed over {total}/{expected} eval items "
            f"({len(records)} records total, {result_file.name})")
    if coverage < 0.98:
        note = f"PARTIAL {coverage:.0%} -- " + note
    if unidentified:
        note += f"; {unidentified} record(s) had no uid"
    return {"score": solved / total, "n": total, "coverage": round(coverage, 4), "note": note}


def from_maas_csv(method: str, dataset: str, job: Path | None = None) -> dict | None:
    """Re-grade MaAS / DAAO from the per-item CSV their test phase writes.

    Preferred over the average printed in test.log for the same reason the
    G-Designer family is re-graded rather than read: the printed figure was
    computed by whichever scorer was live during the run, so a scorer fix applied
    afterwards never reaches it. Measured on daao/drop, the printed average was
    0.6975 and the re-graded one 0.8110 -- the gap was entirely decimal answers the
    old extractor truncated.

    Returns None when no per-item file exists yet, so the caller can fall back to
    the log and still distinguish "not started" from "running".
    """
    import bench as shared_bench

    key = SHARED_KEY[dataset]
    base = ROOT / REPO[method]
    package = base.name
    files = sorted((base / package / "ext" / "maas" / "scripts" / "optimized" / key /
                    "test").glob("round_*/0.*.csv"))
    # Only files this job wrote.
    #
    # These workspaces are re-seeded from the authors' own directories, which carry
    # per-item CSVs from the authors' 2025 runs, and archiving the previous sweep
    # does not remove what re-seeding puts back. Taking the newest file matched a
    # 2025-07-10 artefact for daao/math and reported 26 items as this run's result.
    # The job's own test.cmd is written immediately before the phase starts, so its
    # mtime is the earliest a genuine result file can have.
    floor = 0.0
    if job is not None:
        for marker in ("test.cmd", "search.cmd"):
            stamp = job / marker
            if stamp.exists():
                floor = stamp.stat().st_mtime
                break
    files = [f for f in files if f.stat().st_mtime >= floor]
    if not files:
        return None
    path = max(files, key=lambda p: p.stat().st_mtime)
    total = 0.0
    items = 0
    unmatched = 0
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            expected = (row.get("expected_output") or "").strip()
            if not expected:
                continue
            grading = _grading_row(dataset, expected)
            if grading is None:
                unmatched += 1
                continue
            value, _ = shared_bench.score(dataset, grading,
                                          row.get("prediction") or "")
            total += value
            items += 1
    if not items:
        return None
    expected_items = len(eval_uids(dataset))
    coverage = items / expected_items if expected_items else 1.0
    note = f"re-graded over {items}/{expected_items} eval items ({path.name})"
    if unmatched:
        raise ValueError(f"{unmatched} stored row(s) matched no frozen dataset row")
    finished = job is not None and (job / "status").exists() \
        and (job / "status").read_text().strip() == "ok"
    if finished and RUN_TAG != "smoke" and items != expected_items:
        raise ValueError(f"finished result covers {items}/{expected_items} evaluation items")
    if coverage < 0.98:
        note = f"PARTIAL {coverage:.0%} -- " + note
    return {"score": total / items, "n": items, "coverage": round(coverage, 4),
            "note": note}


def from_log_average(job: Path, method: str, dataset: str) -> dict:
    """MaAS / DAAO: re-grade per-item output; logs are progress only."""
    regraded = from_maas_csv(method, dataset, job)
    if regraded is not None:
        return regraded
    if (job / "status").exists() and (job / "status").read_text().strip() == "ok":
        raise ValueError("finished job has no re-gradeable per-item CSV")
    log = job / "test.log"
    if not log.exists():
        return {"score": None, "note": "test phase has not run"}
    text = log.read_text(encoding="utf-8", errors="replace")
    hits = re.findall(r"Average score on \S+ dataset:\s*([0-9.]+)", text)
    if not hits:
        # "Not started" and "half finished" mean different things to whoever reads
        # this table, and the average-score line only appears at the very end -- so
        # report the progress the tqdm bar has already printed instead of one
        # undifferentiated "pending".
        seen = re.findall(r"(\d+)/(\d+) \[", text)
        if seen:
            done, total = seen[-1]
            return {"score": None,
                    "note": f"test phase running: {done}/{total} items evaluated"}
        return {"score": None, "note": "test phase started, no items evaluated yet"}
    return {"score": float(hits[-1]), "note": f"from test.log ({len(hits)} average line(s))"}


def from_masrouter_items(job: Path, dataset: str) -> dict | None:
    """Re-grade MasRouter from logs/scored_items_<dataset>.jsonl, when present.

    The dump was added 2026-08-24 (see the shim's shared_score): before it,
    masrouter's number could only be read from its log's running mean, i.e.
    whatever scorer was live at run time -- the mmlu_pro letter-extractor fix
    would never have reached it. Evaluation items are the uid namespace
    "<dataset>/..."; "<dataset>_search/..." lines are training and are skipped.
    Runs append to one file, so only lines written after this job's own start
    count.
    """
    import bench as shared_bench

    path = ROOT / "third_party/masrouter/logs" / f"scored_items_{dataset}.jsonl"
    if not path.exists():
        return None
    floor = 0.0
    stamp = job / "search.cmd"
    if stamp.exists():
        floor = stamp.stat().st_mtime
    prefix = f"{dataset}/"
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("ts", 0) < floor or not str(rec.get("uid", "")).startswith(prefix):
            continue
        records.append(rec)
    # The FlowBank-split shims name eval uids "<ds>/fb-test-*" and the
    # periodic TRAINING validations "<ds>/fb-valid-*" -- both inside the
    # "<ds>/" namespace, so the bare-prefix filter mixed 800 validation
    # records into amc's "eval" mean (found 2026-08-25 as 1056/648 coverage).
    # When a test channel exists, count only it; dedupe keeps the last record
    # per uid either way.
    test_only = [r for r in records
                 if str(r.get("uid", "")).startswith(f"{dataset}/fb-test")]
    if test_only:
        records = test_only
    by_uid = {str(r.get("uid", "")): r for r in records}
    total = 0.0
    items = 0
    for rec in by_uid.values():
        value, _ = shared_bench.score(dataset, rec.get("row") or {},
                                      rec.get("prediction") or "")
        total += value
        items += 1
    if not items:
        return None
    expected = len(eval_uids(dataset))
    coverage = items / expected if expected else 1.0
    finished = (job / "status").exists() and (job / "status").read_text().strip() == "ok"
    if finished and RUN_TAG != "smoke" and items != expected:
        raise ValueError(f"finished result covers {items}/{expected} evaluation items")
    note = f"re-graded over {items}/{expected} eval items (scored_items dump)"
    if coverage < 0.98:
        note = f"PARTIAL {coverage:.0%} -- " + note
    return {"score": total / items, "n": items, "coverage": round(coverage, 4),
            "note": note}


def from_masrouter(job: Path, method: str, dataset: str) -> dict:
    """MasRouter: re-grade per-item output; logs are progress only."""
    regraded = from_masrouter_items(job, dataset)
    if regraded is not None:
        return regraded
    if (job / "status").exists() and (job / "status").read_text().strip() == "ok":
        raise ValueError("finished job has no re-gradeable per-item dump")
    log = job / "search.log"
    if not log.exists():
        return {"score": None, "note": "job has not started"}
    text = log.read_text(encoding="utf-8", errors="replace")
    marker = text.find("Start testing")
    if marker == -1:
        return {"score": None, "note": "still training (no 'Start testing' yet)"}
    tail = text[marker:]
    hits = re.findall(r"Accuracy:\s*([0-9.]+)", tail)
    if not hits:
        return {"score": None, "note": "test phase started, no accuracy yet"}

    # The printed Accuracy is a *running* mean, not the batch's own -- verified
    # against the `utilities:` list logged beside it: batch 1's 16 utilities average
    # to exactly the first Accuracy, and the first two batches' 32 utilities average
    # to exactly the second. So the last line is the mean over everything evaluated
    # *so far*, which is the right number only once everything has been evaluated.
    #
    # Counting the utilities is what makes that checkable: MasRouter never prints a
    # total, so without this the running mean over 720 of 1000 items reads exactly
    # like a finished result.
    evaluated = sum(len(re.findall(r"[0-9.]+", block))
                    for block in re.findall(r"utilities:\[([^\]]*)\]", tail))
    import bench as shared_bench

    expected = len(shared_bench.load(dataset))
    coverage = evaluated / expected if expected else 0.0
    # "Finish testing" is the runner's own completion banner. The shared runner
    # uses ceiling division for both loops, so a finished current-protocol run
    # must retain the final partial batch and reach the full evaluation split.
    # Older runs that predate that fix remain visibly partial.
    finished = "Finish testing" in tail
    state = "FINAL" if finished else "running"
    note = f"{state} mean over {evaluated}/{expected} items ({len(hits)} batches)"
    if coverage < 0.98:
        note = f"PARTIAL {coverage:.0%} -- " + note
    return {"score": float(hits[-1]), "n": evaluated,
            "coverage": round(coverage, 4), "note": note}


def from_aflow_test(job: Path, method: str, dataset: str) -> dict:
    """AFlow: read this job's held-out result before consulting the workspace.

    The workspace path is fixed by upstream AFlow and can be replaced by a later
    run. The test log belongs to the run directory, so it is the authoritative,
    isolated result for new runs.
    """
    log_path = job / "test.log"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        scores = re.findall(r"HELD-OUT\s+\S+:\s+score=([0-9]+(?:\.[0-9]+)?)", text)
        if scores:
            return {"score": float(scores[-1]), "note": "from this job's test.log"}

    if has_isolated_artifacts(job):
        return {"score": None, "note": "held-out evaluation not complete in this run"}

    path = ROOT / REPO[method] / "workspace" / SHARED_KEY[dataset] / "workflows_test" / "results.json"
    if not path.exists():
        return {"score": None, "note": "held-out evaluation not run yet (aflow_test.py)"}
    data = json.loads(path.read_text(encoding="utf-8"))
    scored = [r for r in data if isinstance(r.get("score"), (int, float))]
    if not scored:
        return {"score": None, "note": "workflows_test/results.json has no score"}
    return {"score": float(scored[-1]["score"]), "note": "from workflows_test/results.json"}


def from_flowbank(job: Path, method: str, dataset: str) -> dict:
    """FlowBank: the selector's achieved test score, from stage 3e's terminal
    test_predictions.json.

    Two corrections baked in (2026-08-26): the products live one run-name level
    below the benchmark dir (lr*_emb*_cw*/), which the old path missed; and the
    old code took max() of a per-epoch test column -- pick-best-epoch-on-test
    would be a mild form of test peeking. In reality query_matching logs only
    train-side columns per epoch and evaluates the test set exactly once after
    training, and summary.result_predict is that single honest number.
    """
    repeat = int(job.name.removeprefix("repeat"))
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", RUN_TAG)
    benchmark = f"{SHARED_KEY[dataset].lower()}_{safe_tag}_r{repeat}"
    base = ROOT / "third_party" / "flowbank" / "experiments" / benchmark
    runs = sorted(base.glob("*/test_predictions.json")) if base.exists() else []
    # Compatibility for tables produced before pipeline outputs were namespaced.
    if not runs and not has_isolated_artifacts(job):
        legacy = ROOT / "third_party" / "flowbank" / "experiments" / f"{SHARED_KEY[dataset].lower()}_full"
        runs = sorted(legacy.glob("*/test_predictions.json")) if legacy.exists() else []
    if not runs:
        return {"score": None, "note": "stages 2-3 not run yet (flowbank_pipeline.py)"}
    summary = json.loads(runs[-1].read_text(encoding="utf-8"))["summary"]
    n = int(summary.get("num_queries") or 0)
    expected = len(eval_uids(dataset))
    if n != expected:
        raise ValueError(f"FlowBank selector covers {n}/{expected} evaluation items")
    oracle = summary.get("result_golden")
    single = summary.get("best_performed_workflow_score")
    extras = []
    if isinstance(oracle, (int, float)):
        extras.append(f"oracle {oracle:.4f}")
    if isinstance(single, (int, float)):
        extras.append(f"best single {single:.4f}")
    note = f"selector result_predict over {n} queries ({', '.join(extras)})"
    return {"score": float(summary["result_predict"]), "n": n, "note": note}


EXTRACTORS = {
    "gdesigner": from_item_records,
    "card": from_item_records,
    "gdesigner_authordefault": from_item_records,
    "card_authordefault": from_item_records,
    "maas": from_log_average,
    "daao": from_log_average,
    "masrouter": from_masrouter,
    "aflow": from_aflow_test,
    "flowbank": from_flowbank,
}

TOKEN_PATTERNS = (
    re.compile(r"PromptTokens\s+([0-9.]+)"),          # G-Designer / CARD
    re.compile(r"prompt[_ ]tokens[^0-9]{0,12}([0-9]+)", re.I),
    re.compile(r"Token usage:\s*([0-9]+) input", re.I),  # DiverseFlow per call
)


def budget(job: Path) -> dict:
    """Cumulative prompt/completion tokens from the job's own stdout."""
    totals = {"prompt": None, "completion": None}
    for name in ("search.log", "test.log"):
        log = job / name
        if not log.exists():
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        prompt = re.findall(r"PromptTokens\s+([0-9.]+)", text)
        completion = re.findall(r"CompletionTokens\s+([0-9.]+)", text)
        if prompt:
            totals["prompt"] = max(totals["prompt"] or 0, int(float(prompt[-1])))
        if completion:
            totals["completion"] = max(totals["completion"] or 0, int(float(completion[-1])))
        usage = re.findall(r"Token usage:\s*([0-9]+) input \+ ([0-9]+) output", text)
        if usage:
            totals["prompt"] = (totals["prompt"] or 0) + sum(int(a) for a, _ in usage)
            totals["completion"] = (totals["completion"] or 0) + sum(int(b) for _, b in usage)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", help="also write the table here")
    parser.add_argument("--ignore-protocol", action="store_true",
                        help="report jobs whose protocol stamp does not match the current "
                             "source/data/sampling protocol. Only for inspecting an older run in isolation -- "
                             "such numbers must not be placed in the same table as current ones.")
    args = parser.parse_args()

    table: dict[str, dict] = {}
    fingerprint = __import__("bench").protocol_fingerprint()
    print(f"  tag={RUN_TAG}  runs={RUNS}  prompt={fingerprint['prompt']} "
          f"scorer={fingerprint['scorer']} adapter={fingerprint['adapter']}")
    print(f"  {'method':<26}{'dataset':<10}{'held-out':>10}{'n':>7}{'tokens':>12}   note")
    for method in EXTRACTORS:
        for dataset in DATASETS:
            job = RUNS / method / dataset / "repeat1"
            if not job.exists():
                continue
            status = (job / "status").read_text(encoding="utf-8").strip() if (job / "status").exists() else "running"
            stale = protocol_mismatch(job, method, dataset)
            if stale and not args.ignore_protocol:
                found = {"score": None, "note": f"STALE PROTOCOL: {stale}"}
            else:
                try:
                    found = EXTRACTORS[method](job, method, dataset)
                except Exception as exc:  # noqa: BLE001 - one bad job must not hide the rest
                    found = {"score": None, "note": f"extractor failed: {type(exc).__name__}: {exc}"}
                if stale:
                    found["note"] = f"[protocol check bypassed] {found.get('note', '')}"
            tokens = budget(job)
            total_tokens = None
            if tokens["prompt"] is not None or tokens["completion"] is not None:
                total_tokens = (tokens["prompt"] or 0) + (tokens["completion"] or 0)
            score = found.get("score")
            table[f"{method}/{dataset}"] = {**found, "tokens": total_tokens, "status": status}
            print(f"  {method:<26}{dataset:<10}"
                  f"{(f'{score:.4f}' if score is not None else 'pending'):>10}"
                  f"{str(found.get('n', '')):>7}"
                  f"{(f'{total_tokens:,}' if total_tokens else '-'):>12}   {found.get('note', '')}")

    done = [v for v in table.values() if v.get("score") is not None]
    print(f"\n  {len(done)} of {len(table)} started job(s) have a held-out score")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  written to {args.json}")


if __name__ == "__main__":
    main()
