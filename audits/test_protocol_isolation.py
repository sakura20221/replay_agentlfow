#!/usr/bin/env python3
"""Regression tests for result-protocol stamping and stale-run rejection."""

from __future__ import annotations

import json
import inspect
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "shared"))

import bench  # noqa: E402
import collect  # noqa: E402
import sweep  # noqa: E402
from vllm_proxy import sampling_protocol  # noqa: E402


def main() -> None:
    fingerprints = bench.protocol_fingerprint()
    if set(fingerprints) != {"prompt", "scorer", "adapter"}:
        raise SystemExit(f"incomplete source fingerprints: {fingerprints}")
    data_hashes = {name: bench.data_fingerprint(name) for name in bench.DATASETS}
    if len(set(data_hashes.values())) != len(data_hashes):
        raise SystemExit(f"dataset fingerprints collide: {data_hashes}")

    current = {
        **fingerprints,
        "data": data_hashes["math"],
        "sampling": sampling_protocol(),
        "method": "maas",
        "dataset": "math",
        "run_tag": collect.RUN_TAG,
        "repeat": 1,
    }
    with tempfile.TemporaryDirectory(prefix="protocol-isolation-") as tmp:
        job = Path(tmp)
        stamp = job / "protocol.json"
        stamp.write_text(json.dumps(current), encoding="utf-8")
        if collect.protocol_mismatch(job, "maas", "math") is not None:
            raise SystemExit("matching protocol was rejected")

        for key in ("prompt", "scorer", "adapter", "data", "sampling",
                    "method", "dataset", "run_tag", "repeat"):
            changed = dict(current)
            changed[key] = "different"
            stamp.write_text(json.dumps(changed), encoding="utf-8")
            if collect.protocol_mismatch(job, "maas", "math") is None:
                raise SystemExit(f"{key} mismatch was accepted")

    sweep_protocol = sweep.current_job_protocol("maas", "math", 1)
    if sweep.protocol_differences(sweep_protocol, sweep_protocol):
        raise SystemExit("sweep rejected its own protocol")
    changed = dict(sweep_protocol, adapter="different")
    if sweep.protocol_differences(changed, sweep_protocol) != ["adapter"]:
        raise SystemExit("sweep resume guard missed adapter mismatch")
    run_source = inspect.getsource(sweep.run_job)
    guard_position = run_source.index("if protocol_path.exists():")
    skip_position = run_source.index('status_path.read_text(encoding="utf-8").strip() == "ok"')
    if guard_position >= skip_position:
        raise SystemExit("completed jobs skip before protocol verification")

    print("protocol isolation OK: source, data, sampling and identity mismatches rejected")


if __name__ == "__main__":
    main()
