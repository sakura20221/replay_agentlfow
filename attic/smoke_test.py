#!/usr/bin/env python3
"""Pre-experiment gate for the local Qwen3-8B serving stack.

Every check here guards a failure mode that would silently invalidate a whole
experiment batch rather than crash it:

  1. thinking actually disabled          -> otherwise parsers break and tokens explode
  2. greedy determinism                  -> otherwise the frozen protocol is a fiction
  3. G-Designer legacy request shape     -> otherwise that repo 500s mid-run
  4. per-namespace accounting            -> otherwise search and test cost are mixed
  5. both upstreams reachable            -> otherwise half the throughput is silently gone
  6. observed concurrency throughput     -> sets the client-side concurrency for real runs

Run with the tools venv: envs/tools/bin/python smoke_test.py
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request

PROXY = "http://127.0.0.1:18080"
UPSTREAMS = ["http://127.0.0.1:8001", "http://127.0.0.1:8002"]
MODEL = "Qwen/Qwen3-8B"

failures: list[str] = []
notes: list[str] = []


def post(path: str, payload: dict, timeout: float = 300.0) -> dict:
    request = urllib.request.Request(
        PROXY + path,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(f"{name}: {detail}")


# ---------------------------------------------------------------- 1. upstreams
print("=== upstream reachability ===")
for base in UPSTREAMS:
    try:
        models = get(f"{base}/v1/models")
        ids = [entry["id"] for entry in models.get("data", [])]
        check(f"{base} serving", bool(ids), f"models={ids}")
    except Exception as exc:  # noqa: BLE001
        check(f"{base} serving", False, f"{type(exc).__name__}: {exc}")

try:
    health = get(f"{PROXY}/health")
    check("proxy /health", health.get("status") == "ok", f"upstreams={len(health.get('upstreams', []))}")
except Exception as exc:  # noqa: BLE001
    check("proxy /health", False, f"{type(exc).__name__}: {exc}")
    print("\nProxy unreachable; aborting.")
    sys.exit(1)

# --------------------------------------------------- 2. thinking is really off
print("\n=== thinking disabled ===")
# A prompt that a hybrid-reasoning model is most tempted to think about.
probe = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "A shop sells pens at 7 yuan and books at 23 yuan. "
                                             "I bought 3 pens and 2 books. Total cost? Answer with the number only."}],
    "max_tokens": 512,
}
try:
    response = post("/test/smoke/v1/chat/completions", probe)
    message = response["choices"][0]["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    completion_tokens = (response.get("usage") or {}).get("completion_tokens", -1)

    check("no <think> tag in content", "<think>" not in content.lower(), repr(content[:120]))
    check("no reasoning_content field", not reasoning, repr(reasoning[:80]))
    # 21 + 46 = 67. Not a correctness gate, but a wrong answer here alongside a
    # huge completion usually means thinking leaked and got truncated.
    check("answer contains 67", "67" in content, repr(content[:120]))
    check("completion is short (thinking would inflate it)", 0 < completion_tokens < 200,
          f"completion_tokens={completion_tokens}")
    notes.append(f"probe completion_tokens={completion_tokens}")
except Exception as exc:  # noqa: BLE001
    check("thinking probe", False, f"{type(exc).__name__}: {exc}")

# ------------------------------------------------------- 3. greedy determinism
print("\n=== greedy determinism ===")
det_payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "List exactly three prime numbers above 40, comma separated."}],
    "max_tokens": 64,
}
try:
    first = post("/test/smoke/v1/chat/completions", det_payload)["choices"][0]["message"]["content"]
    second = post("/test/smoke/v1/chat/completions", det_payload)["choices"][0]["message"]["content"]
    check("same prompt -> identical output", first == second, f"{first[:60]!r} vs {second[:60]!r}")
except Exception as exc:  # noqa: BLE001
    check("determinism", False, f"{type(exc).__name__}: {exc}")

# ------------------------------------------------- 4. temperature override
print("\n=== protocol enforcement ===")
try:
    # Caller tries to set a sampling temperature; the proxy must override it.
    hot = {"model": MODEL, "temperature": 1.5, "top_p": 0.5,
           "messages": [{"role": "user", "content": "Say the single word: apple"}], "max_tokens": 16}
    a = post("/test/smoke/v1/chat/completions", hot)["choices"][0]["message"]["content"]
    b = post("/test/smoke/v1/chat/completions", hot)["choices"][0]["message"]["content"]
    check("caller temperature is overridden to 0", a == b, f"{a[:40]!r} vs {b[:40]!r}")
except Exception as exc:  # noqa: BLE001
    check("temperature override", False, f"{type(exc).__name__}: {exc}")

# --------------------------------------------------- 5. G-Designer legacy shape
print("\n=== G-Designer legacy request shape ===")
try:
    legacy = {
        "name": "gpt-3.5-turbo",
        "inputs": {"msg": repr([{"role": "user", "content": "Reply with the word ok only."}])},
    }
    legacy_response = post("/train/smoke/legacy/v1/chat/completions", legacy)
    check("legacy returns data field", "data" in legacy_response, str(legacy_response)[:120])
except Exception as exc:  # noqa: BLE001
    check("legacy shape", False, f"{type(exc).__name__}: {exc}")

# ------------------------------------------------------- 6. namespace accounting
print("\n=== namespace accounting ===")
try:
    stats = get(f"{PROXY}/stats")
    by_ns = stats.get("by_namespace", {})
    check("train and test namespaces tracked separately",
          any(k.startswith("train") for k in by_ns) and any(k.startswith("test") for k in by_ns),
          f"namespaces={list(by_ns)}")
    totals = stats.get("totals", {})
    check("no thinking leaks recorded", totals.get("thinking_leak", 0) == 0,
          f"thinking_leak={totals.get('thinking_leak', 0)}")
    check("no truncations recorded", totals.get("truncated", 0) == 0,
          f"truncated={totals.get('truncated', 0)}")
except Exception as exc:  # noqa: BLE001
    check("namespace accounting", False, f"{type(exc).__name__}: {exc}")

# ----------------------------------------------------------- 7. throughput probe
print("\n=== concurrency throughput ===")
CONC = 24
prompts = [f"Compute {i} * 37 + 11. Answer with the number only." for i in range(CONC)]


def one(text: str) -> int:
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": 96}
    result = post("/test/smoke/v1/chat/completions", body)
    return (result.get("usage") or {}).get("completion_tokens", 0)


try:
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as pool:
        token_counts = list(pool.map(one, prompts))
    elapsed = time.time() - started
    ok_count = sum(1 for value in token_counts if value > 0)
    check(f"{CONC} concurrent requests all succeeded", ok_count == CONC, f"{ok_count}/{CONC}")
    print(f"       wall={elapsed:.1f}s  throughput={CONC / elapsed:.2f} req/s  "
          f"completion_tokens={sum(token_counts)}")
    notes.append(f"concurrency {CONC}: {elapsed:.1f}s, {CONC / elapsed:.2f} req/s")
except Exception as exc:  # noqa: BLE001
    check("throughput probe", False, f"{type(exc).__name__}: {exc}")

# ------------------------------------------------------------------ verdict
print("\n" + "=" * 60)
for note in notes:
    print("note:", note)
if failures:
    print(f"\nSMOKE TEST FAILED ({len(failures)} checks)")
    for item in failures:
        print("  -", item)
    sys.exit(1)
print("\nSMOKE TEST PASSED - stack is safe for formal runs")
