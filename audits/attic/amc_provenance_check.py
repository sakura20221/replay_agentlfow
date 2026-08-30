#!/usr/bin/env python3
"""One-shot: where do FlowBank's amc_validate/amc_test rows come from?

Checks two hypotheses by content, not by guessing:
  1. subset of hendrycks MATH (train/test, all 7 subjects, cache-only load)
  2. overlap with our AI-MO amc-83 (2022-2023 AMC12 papers)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parents[2]


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


flowbank = []
for name in ("amc_validate", "amc_test"):
    for line in (ROOT / f"third_party/flowbank/datasets/{name}.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        row["_file"] = name
        flowbank.append(row)
fb_norm = {norm(r["problem"]): r for r in flowbank}
print(f"flowbank amc rows: {len(flowbank)} ({len(fb_norm)} unique by content)")

try:
    from datasets import load_dataset
    hendrycks = set()
    for config in ("algebra", "counting_and_probability", "geometry",
                   "intermediate_algebra", "number_theory", "prealgebra", "precalculus"):
        for split in ("train", "test"):
            try:
                ds = load_dataset("EleutherAI/hendrycks_math", config, split=split)
            except Exception as exc:  # noqa: BLE001
                print(f"  hendrycks {config}/{split}: not in cache ({type(exc).__name__})")
                continue
            for row in ds:
                hendrycks.add(norm(row["problem"]))
    if hendrycks:
        hit = sum(1 for key in fb_norm if key in hendrycks)
        print(f"in hendrycks MATH (n={len(hendrycks)}): {hit}/{len(fb_norm)} flowbank amc rows")
except ImportError:
    print("datasets lib unavailable in this env")

ours = [json.loads(l) for l in (ROOT / "shared/data/amc.jsonl").open(encoding="utf-8")]
ours_norm = {norm(r["problem"]): r["uid"] for r in ours}
exact = set(ours_norm) & set(fb_norm)
prefix_hits = sum(1 for key in ours_norm
                  if any(key[:100] and key[:100] in fk for fk in fb_norm))
print(f"overlap with our amc-83: exact={len(exact)}, 100-char-prefix containment={prefix_hits}")
