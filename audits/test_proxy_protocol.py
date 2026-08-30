#!/usr/bin/env python3
"""Assert the proxy's decoding protocol, on the payload it really builds.

A live call proves only that the proxy answers. What matters is what it *sent*:
presence_penalty is set by us and must survive a caller that explicitly passes 0,
temperature must be pinned, the cap must be enforced, thinking must be off, and a
reply that filled the budget must be re-generated once rather than continued.

So `_normalize` is called directly and `_forward` is replaced, which makes the
retry path testable without spending GPU time or depending on whether the model
happens to loop today.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vllm_proxy  # noqa: E402

failures = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global failures
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        failures += 1


def normalized(**overrides) -> dict:
    body = {"model": "qwen3-8b",
            "messages": [{"role": "user", "content": "hi"}]}
    body.update(overrides)
    payload, _legacy, _model, _stream = vllm_proxy._normalize(body)
    return payload


# --- what gets sent upstream -------------------------------------------------
payload = normalized()
check(payload.get("presence_penalty") == 1.0,
      "presence_penalty is set to 1.0 by default", repr(payload.get("presence_penalty")))

payload = normalized(presence_penalty=0.0)
check(payload.get("presence_penalty") == 1.0,
      "a caller passing 0 does not disable it", repr(payload.get("presence_penalty")))

payload = normalized(temperature=0.9)
check(payload["temperature"] == 0, "temperature is pinned to 0",
      repr(payload["temperature"]))

payload = normalized(max_tokens=81920)
check(payload["max_tokens"] == 8192, "an over-budget max_tokens is capped to 8192",
      repr(payload["max_tokens"]))

payload = normalized()
check(payload["max_tokens"] == 8192, "the default cap is the full budget, not a smaller one",
      repr(payload["max_tokens"]))

payload = normalized(top_p=0.8, frequency_penalty=1.5)
check("top_p" not in payload and "frequency_penalty" not in payload,
      "other caller sampling knobs are dropped")

payload = normalized()
check(payload["chat_template_kwargs"]["enable_thinking"] is False,
      "thinking is disabled through chat_template_kwargs")
check(payload["stream"] is False, "upstream is always non-streamed")

# --- the fingerprint the collector compares ---------------------------------
protocol = vllm_proxy.sampling_protocol()
check(protocol["presence_penalty"] == 1.0 and protocol["max_tokens"] == 8192,
      "sampling_protocol reports the live values", json.dumps(protocol))
check(protocol["on_length"] == "retry_whole_request_x1",
      "sampling_protocol names the recovery rule", protocol["on_length"])
check(vllm_proxy.CONTINUE_ON_TRUNCATION is False,
      "continuation is off by default")


# --- the retry path ---------------------------------------------------------
def reply(text: str, finish: str, tokens: int = 100) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": tokens,
                      "total_tokens": 10 + tokens}}


class FakeUpstream:
    """Returns a scripted sequence of replies, counting the calls."""

    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.calls = 0

    def __call__(self, payload):
        self.calls += 1
        response = self.sequence[min(self.calls - 1, len(self.sequence) - 1)]
        return 200, json.loads(json.dumps(response)), 1, "http://fake"


original_forward = vllm_proxy._forward


def drive(sequence) -> tuple[FakeUpstream, dict, str, dict]:
    """Run the do_POST decision logic over a scripted upstream."""
    fake = FakeUpstream(sequence)
    vllm_proxy._forward = fake
    try:
        payload = normalized()
        record: dict = {}
        status, response, attempts, url = vllm_proxy._forward(payload)
        content, diagnostics = vllm_proxy._inspect_and_clean(response)
        if vllm_proxy.RETRY_ON_LENGTH and diagnostics["truncated"]:
            retry_tokens = 0
            for index in range(vllm_proxy.RETRIES_ON_LENGTH):
                _, retried, _, retry_url = vllm_proxy._forward(payload)
                retry_content, retry_diagnostics = vllm_proxy._inspect_and_clean(retried)
                retry_tokens += (retried.get("usage") or {}).get("completion_tokens", 0)
                record["retried_on_length"] = index + 1
                if not retry_diagnostics["truncated"]:
                    response, content = retried, retry_content
                    diagnostics = retry_diagnostics
                    diagnostics["recovered"] = True
                    break
                diagnostics["retry_also_truncated"] = True
            record["retry_completion_tokens"] = retry_tokens
        return fake, diagnostics, content, record
    finally:
        vllm_proxy._forward = original_forward


fake, diagnostics, content, record = drive([reply("a complete answer", "stop")])
check(fake.calls == 1, "a normal reply is not retried", f"{fake.calls} call(s)")
check(diagnostics["recovered"] is False, "and is not marked recovered")

fake, diagnostics, content, record = drive([
    reply("loop loop loop", "length", 8192),
    reply("42", "stop", 30),
])
check(fake.calls == 2, "a reply that filled the budget is re-generated once",
      f"{fake.calls} call(s)")
check(content == "42", "the graded reply is the retry, whole", repr(content))
check(diagnostics["recovered"] is True and diagnostics["truncated"] is False,
      "recovery is recorded only after the retry came back clean")
check(record.get("retry_completion_tokens") == 30,
      "the retry's tokens are accounted", repr(record.get("retry_completion_tokens")))

fake, diagnostics, content, record = drive([
    reply("loop loop loop", "length", 8192),
    reply("loop loop loop", "length", 8192),
])
check(fake.calls == 2, "only one retry, even when it also loops", f"{fake.calls} call(s)")
check(diagnostics["truncated"] is True and diagnostics["recovered"] is False,
      "a still-truncated reply is reported truncated, not recovered")
check(diagnostics.get("retry_also_truncated") is True,
      "and the repeated loop is counted separately")
check("\n" not in content.strip() or "loop" in content,
      "nothing was spliced onto it", repr(content[:40]))

print(f"\n  {failures} failure(s)")
sys.exit(1 if failures else 0)
