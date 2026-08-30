#!/usr/bin/env python3
"""Round-robin OpenAI-compatible proxy in front of local vLLM instances.

Differences from the DashScope-era api_proxy.py:

* Fans out over several local vLLM endpoints instead of one remote account.
* Disables Qwen3 thinking the way vLLM actually accepts it, via
  ``chat_template_kwargs.enable_thinking``. The old top-level
  ``enable_thinking`` key is silently ignored by vLLM, which would have left
  thinking ON for every call without raising an error.
* Records thinking leakage and length truncation per call so the experiment can
  report a parse-health column next to accuracy.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


HOST = os.getenv("PROXY_HOST", "127.0.0.1")
PORT = int(os.getenv("PROXY_PORT", "18080"))
UPSTREAMS = [
    url.strip()
    for url in os.getenv(
        "VLLM_UPSTREAMS",
        "http://127.0.0.1:8001/v1/chat/completions,http://127.0.0.1:8002/v1/chat/completions",
    ).split(",")
    if url.strip()
]
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "Qwen/Qwen3-8B")
# Advertise a few aliases so author repos that hardcode a model name still work.
# Names that legitimately mean "the one model this proxy serves". Anything else
# is refused rather than answered with a different model: silently rewriting the
# name is how a method can appear to route over a pool of five distinct LLMs that
# are secretly all Qwen3-8B -- every request succeeds, the numbers look
# plausible, and nothing anywhere says the pool was fictitious. Extend via
# PROXY_MODEL_ALIASES only for a repo whose hardcoded placeholder really does
# mean the local model.
MODEL_ALIASES = tuple(
    ["Qwen/Qwen3-8B", "qwen3-8b"]
    + [n.strip() for n in os.getenv("PROXY_MODEL_ALIASES", "").split(",") if n.strip()]
)
# 64, raised from 32 after measuring where the ceiling actually was: with the cap
# at 32 one instance sat pegged at exactly "Running: 32 reqs" while vLLM reported
# "Waiting: 0" and only ~50% KV cache use. The server was never the constraint --
# this semaphore was. KV headroom (113,104 tokens, half free at 32 in flight) is
# what makes the higher cap safe.
PER_UPSTREAM_CONCURRENCY = int(os.getenv("PROXY_PER_UPSTREAM_CONCURRENCY", "64"))
MAX_RETRIES = int(os.getenv("PROXY_MAX_RETRIES", "4"))
REQUEST_TIMEOUT = float(os.getenv("PROXY_REQUEST_TIMEOUT", "600"))
DEFAULT_MAX_TOKENS = int(os.getenv("PROXY_DEFAULT_MAX_TOKENS", "8192"))
# Must match the served --max-model-len. vLLM rejects prompt+max_tokens above it
# with a hard 400 instead of clamping, and a multi-agent debate that accumulates
# history will cross it. Left unhandled that failure lands hardest on the methods
# with the richest communication, so the ceiling is known here and respected.
MAX_MODEL_LEN = int(os.getenv("PROXY_MAX_MODEL_LEN", "32768"))
# 256, not 64: the margin has to absorb an "at least N" underestimate, and
# the observed gap between consecutive reports was 65 tokens.
CONTEXT_MARGIN = 256
# The smallest completion worth asking for. Below this a reply cannot carry even a
# formatted answer, so the request is genuinely unservable rather than merely tight.
MIN_COMPLETION_TOKENS = int(os.getenv("PROXY_MIN_COMPLETION_TOKENS", "16"))
LOG_PATH = Path(os.getenv("PROXY_LOG_PATH", "logs/api_calls.jsonl"))

# Repetition, and what to do when a reply fills the whole budget.
#
# A reply cut off at max_tokens scores zero even when the model was on track, and
# methods do not truncate at equal rates, so the noise would land unevenly across
# the comparison. Two earlier answers to this were tried and both were wrong:
#
#   * asking the model to continue. Measured 88.8% recovery, but a continuation is
#     a second, differently-conditioned generation spliced onto a truncated one --
#     the reply that gets graded is not a reply the method produced.
#   * treating it as an accuracy problem at all. Measured on the recorded runs,
#     92.9% of the replies that hit the 8192-token cap were *degenerate* -- the
#     same line or phrase repeated -- and only 5.2% were genuinely long answers.
#     So the cap is mostly a symptom of a decoding loop, not of a long task.
#
# What is done instead, from the Qwen3 model card's own guidance ("do not use
# greedy decoding"; presence_penalty 0-2 reduces endless repetition) plus
# measurement:
#
#   presence_penalty=1.0                 loops on 7% of the calls that used to loop
#   + one retry of the whole request     loops on 2%
#   accuracy on the affected items       0.355 -> 0.654
#
# The retry is a fresh generation of the identical request, not a continuation, so
# what gets graded is always one uninterrupted reply. It works because vLLM's
# continuous batching makes sampling non-bitwise-reproducible even at temperature
# 0: the same request re-sent lands in a different batch and takes a different
# path. One retry only -- the second retry's marginal recovery was measured at
# under a point, and every retry is a doubled cost.
PRESENCE_PENALTY = float(os.getenv("PROXY_PRESENCE_PENALTY", "1.0"))
RETRY_ON_LENGTH = os.getenv("PROXY_RETRY_ON_LENGTH", "1") not in {"0", "false", "False"}
RETRIES_ON_LENGTH = int(os.getenv("PROXY_RETRIES_ON_LENGTH", "1"))
# Kept, defaulting off, so the earlier protocol can be reproduced on demand.
CONTINUE_ON_TRUNCATION = os.getenv("PROXY_CONTINUE_ON_TRUNCATION", "0") not in {"0", "false", "False"}
CONTINUATION_MAX_TOKENS = int(os.getenv("PROXY_CONTINUATION_MAX_TOKENS", "256"))
CONTINUATION_PROMPT = (
    "Your previous reply was cut off before it finished. Do not start over and do not "
    "add new reasoning: state only the final answer, in exactly the format the task asked for."
)


def sampling_protocol() -> dict[str, Any]:
    """Everything about decoding that could move a score.

    Written into every job's protocol.json by sweep.py: a run made with a
    different penalty, cap or recovery rule is answering under different
    conditions, and its numbers must not share a table with these.
    """
    return {
        "temperature": 0,
        "presence_penalty": PRESENCE_PENALTY,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "enable_thinking": False,
        "on_length": (f"retry_whole_request_x{RETRIES_ON_LENGTH}" if RETRY_ON_LENGTH
                      else ("continue" if CONTINUE_ON_TRUNCATION else "keep_truncated")),
    }

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>", re.IGNORECASE)

_log_lock = threading.Lock()
_rr_lock = threading.Lock()
_stat_lock = threading.Lock()
_rr = itertools.cycle(range(len(UPSTREAMS)))
_slots = [threading.BoundedSemaphore(PER_UPSTREAM_CONCURRENCY) for _ in UPSTREAMS]
# In-flight count per upstream, kept explicitly rather than read off the
# semaphore's private counter, so the dispatch order can prefer the idler side.
_inflight = [0 for _ in UPSTREAMS]
_inflight_lock = threading.Lock()
_started_at = time.time()
_counters: dict[str, int] = defaultdict(int)
_ns_counters: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
_model_counters: dict[str, int] = defaultdict(int)
_capped_requests: dict[str, int] = defaultdict(int)
_leak_warned = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _namespace_from_path(path: str) -> str:
    """Map /train/gsm8k/v1/chat/completions -> "train/gsm8k" for accounting."""
    parts = [part for part in path.split("?", 1)[0].split("/") if part]
    ignored = {"v1", "chat", "completions", "legacy"}
    namespace = [part for part in parts if part not in ignored]
    return "/".join(namespace) if namespace else "unspecified"


def _parse_legacy_messages(value: Any) -> list[dict[str, str]]:
    """G-Designer posts {name, inputs: {msg: repr(messages)}}."""
    if isinstance(value, list):
        return [dict(item) for item in value]
    if not isinstance(value, str):
        raise ValueError("legacy inputs.msg must be a string or list")
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError("legacy messages must be a list")
    messages: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("unsupported legacy message representation")
        messages.append({"role": str(item.get("role", "user")), "content": str(item.get("content", ""))})
    return messages


def _normalize(body: dict[str, Any]) -> tuple[dict[str, Any], bool, str, bool]:
    legacy = "inputs" in body and "name" in body
    if legacy:
        requested_model = str(body.get("name", "unknown"))
        payload: dict[str, Any] = {"messages": _parse_legacy_messages(body.get("inputs", {}).get("msg"))}
    else:
        requested_model = str(body.get("model", "unknown"))
        payload = dict(body)

    if requested_model not in MODEL_ALIASES:
        raise ValueError(
            f"model {requested_model!r} is not served here: this proxy serves only "
            f"{UPSTREAM_MODEL}, under the aliases {list(MODEL_ALIASES)}. Refusing "
            "rather than answering with a different model. If the caller means the "
            "local model, add the name to PROXY_MODEL_ALIASES; if it means a "
            "genuinely different model, the experiment config is wrong."
        )

    payload["model"] = UPSTREAM_MODEL
    payload["temperature"] = 0

    # Streaming is answered, not refused or silently downgraded.
    #
    # This proxy needs the whole body to strip leaked thinking, recover truncation
    # and account tokens, so upstream is always non-streamed. But a client that
    # asked for a stream and receives a plain JSON body iterates SSE chunks over
    # it and reads *nothing*: every reply arrives as an empty string. That is how
    # MaAS (llm_config.stream defaults to True) produced empty outputs, pydantic
    # "Missing fields" errors and zero-gradient batches with no error to point at.
    #
    # Refusing with a 400 was the first fix, but a repo's config plumbing may not
    # honour `stream: false` -- MaAS still asked to stream on 5006 of 5949 calls --
    # and wasted work is not much better than wrong work. So the request is served
    # non-streamed upstream and the result is emitted as a single SSE event, which
    # is a valid stream from the client's point of view.
    wants_stream = bool(payload.pop("stream", False))
    payload["stream"] = False
    # One output budget for every method, so the bake-off does not become a
    # comparison of max_tokens settings. MasRouter asks for 81920, which is above
    # the served max_model_len and which vLLM rejects outright with HTTP 400.
    #
    # Capping is deliberate protocol enforcement, not a silent fallback: each cap
    # is counted in /stats so it cannot pass unnoticed the way the earlier
    # stream=True rewrite did.
    requested_max_tokens = payload.get("max_tokens")
    if requested_max_tokens is None:
        payload["max_tokens"] = DEFAULT_MAX_TOKENS
    elif int(requested_max_tokens) > DEFAULT_MAX_TOKENS:
        payload["max_tokens"] = DEFAULT_MAX_TOKENS
        with _stat_lock:
            _counters["max_tokens_capped"] += 1
            _capped_requests[f"{requested_max_tokens}->{DEFAULT_MAX_TOKENS}"] += 1
    payload.pop("n", None)
    # Caller sampling knobs are dropped so the protocol is the same for everyone,
    # then presence_penalty is set by us rather than left unset.
    #
    # Unset means 0, and at 0 this model gets stuck: 92.9% of the replies that
    # filled the 8192-token budget were repeating themselves. The Qwen3 model card
    # names presence_penalty 0-2 as the remedy and warns that high values cause
    # language mixing, so 1.0 is the middle of the useful range; measured, it
    # clears 93% of the loops. It applies to every method identically, which is
    # what keeps it a property of the protocol rather than of a method.
    for key in ("top_p", "top_k", "presence_penalty", "frequency_penalty"):
        payload.pop(key, None)
    if PRESENCE_PENALTY:
        payload["presence_penalty"] = PRESENCE_PENALTY

    # THE critical line: vLLM only honours enable_thinking when it is passed
    # through to the chat template. A top-level "enable_thinking" is ignored.
    template_kwargs = dict(payload.get("chat_template_kwargs") or {})
    template_kwargs["enable_thinking"] = False
    payload["chat_template_kwargs"] = template_kwargs
    payload.pop("enable_thinking", None)

    return payload, legacy, requested_model, wants_stream


# Full prompt/completion capture. Off by default because it is large -- roughly
# 2-6 KB per call against ~300k calls per sweep -- but the one place where every
# method's every stage passes through, so nothing else can give per-stage I/O for
# all seven without touching seven repos.
TRANSCRIPT_PATH = Path(os.getenv("PROXY_TRANSCRIPT_PATH", str(Path(__file__).resolve().parent
                                                        / "logs" / "transcripts.jsonl")))
TRANSCRIPT_ENABLED = os.getenv("PROXY_TRANSCRIPT_ENABLED", "1") not in {"0", "false", "False"}
TRANSCRIPT_MAX_BYTES = int(os.getenv("PROXY_TRANSCRIPT_MAX_BYTES", str(20 * 1024 ** 3)))
_transcript_lock = threading.Lock()
_transcript_bytes = 0
_transcript_full_warned = False


def _write_transcript(record: dict[str, Any], payload: dict[str, Any], content: str) -> None:
    """Append the exact messages sent and the text returned.

    Stored alongside the request_id in api_calls.jsonl so a transcript line can be
    joined back to its latency, token counts, truncation flags and namespace.
    """
    global _transcript_bytes, _transcript_full_warned
    if not TRANSCRIPT_ENABLED:
        return
    entry = {
        "request_id": record.get("request_id"),
        "timestamp": record.get("timestamp"),
        "namespace": record.get("namespace"),
        "requested_model": record.get("requested_model"),
        "messages": payload.get("messages"),
        "completion": content,
        # Present on every line so a reader can tell a captured failure from an
        # empty answer without joining against api_calls.jsonl.
        "failed": bool(record.get("error")),
        "error": record.get("error"),
        "finish_reason": record.get("finish_reason"),
        "prompt_tokens": record.get("prompt_tokens"),
        "completion_tokens": record.get("completion_tokens"),
    }
    line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
    encoded = line.encode("utf-8")
    with _transcript_lock:
        if _transcript_bytes + len(encoded) > TRANSCRIPT_MAX_BYTES:
            if not _transcript_full_warned:
                print(f"[proxy] transcript cap {TRANSCRIPT_MAX_BYTES} bytes reached; "
                      f"stopping capture (scores are unaffected)", flush=True)
                _transcript_full_warned = True
            return
        TRANSCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRANSCRIPT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line)
        _transcript_bytes += len(encoded)


def _write_log(record: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with _log_lock:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _next_upstream_order() -> list[int]:
    """Least-loaded first, with round-robin breaking ties.

    Round-robin alone was not enough: _forward blocks on the chosen upstream's
    semaphore rather than trying the other one, so a request handed to a saturated
    instance waited there while the second sat idle -- measured as 32 requests
    running on one and 7-10 on the other, with neither queueing server-side.
    """
    with _rr_lock:
        start = next(_rr)
    order = [(start + offset) % len(UPSTREAMS) for offset in range(len(UPSTREAMS))]
    with _inflight_lock:
        snapshot = list(_inflight)
    order.sort(key=lambda index: snapshot[index])
    return order


_CONTEXT_TOKENS_RE = re.compile(r"prompt contains at least (\d+) input tokens|value=(\d+)")


def _clamped_payload(payload: dict[str, Any], detail: str) -> dict[str, Any] | None:
    """Shrink max_tokens to whatever the context window still allows.

    vLLM's 400 states the measured prompt length, so the room left can be derived
    without a tokenizer on this side. Returns None when nothing is left to give,
    in which case the request genuinely cannot be served.
    """
    if "maximum context length" not in detail:
        return None
    match = _CONTEXT_TOKENS_RE.search(detail)
    if not match:
        return None
    # Only the reported bound is used. A character-based estimate was tried as an
    # extra floor to converge faster, but at ~4.4 chars/token it overshot badly
    # on repetitive text and rejected requests that were in fact servable. With
    # a 32k window this clamp path is rare, so a false rejection costs more than
    # an extra round trip; the 0.8 shrink below guarantees progress regardless.
    prompt_tokens = int(match.group(1) or match.group(2))
    allowed = MAX_MODEL_LEN - prompt_tokens - CONTEXT_MARGIN
    current = int(payload.get("max_tokens") or DEFAULT_MAX_TOKENS)
    # Strictly decrease even when the reported bound has not moved, so repeated
    # clamps always make progress instead of resending the same request.
    # Every clamp halves max_tokens *and* respects the reported bound.
    #
    # Trusting the bound alone made this loop oscillate: vLLM reports "at least N"
    # -- a lower bound -- so the room computed from it can still be one token too
    # much -- and the underestimate can be severe: the same AFlow optimizer prompt
    # was reported as "at least 25032" and later as "at least 32095", a 7000-token
    # gap. So the bound cannot drive convergence on its own; halving does, in at
    # most 5 steps from the 8192 budget.
    # The room that is left even with no margin at all. When the margin makes
    # `allowed` negative, this is still positive and still servable: a prompt
    # measured at 32,706 in a 32,768 window leaves 61 usable tokens, and 61 tokens
    # is enough for "Answer: (C)" or a boxed number.
    #
    # Ignoring it is what turned a recoverable request into a hard failure. The old
    # code fell back to halving `current`, walked 8192 -> 4096 -> ... -> 32, and
    # then returned None because 32 was still over the real limit of 61-by-way-of-63
    # -- overshooting by a single token. Measured cost: 329 requests across four
    # methods, all at exactly 32,706 prompt tokens, and the concentration of 210 of
    # them in one job exhausted MasRouter's retry budget and killed masrouter/math.
    hard_room = MAX_MODEL_LEN - prompt_tokens - 1
    if hard_room < MIN_COMPLETION_TOKENS:
        # Genuinely unservable: the prompt alone fills the window.
        return None
    ceiling = allowed if allowed > 0 else hard_room
    target = min(ceiling, hard_room, max(int(current * 0.5), MIN_COMPLETION_TOKENS))
    if target < MIN_COMPLETION_TOKENS or target >= current:
        # No progress possible; a further attempt would resend the same request.
        return None
    clamped = dict(payload)
    clamped["max_tokens"] = target
    return clamped


def _forward(payload: dict[str, Any]) -> tuple[int, dict[str, Any], int, str]:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    attempts = 0
    # vLLM reports the prompt length as "at least N", i.e. a lower bound, so one
    # clamp can still land over the limit. Each rejection carries a tighter
    # bound, so a few passes converge.
    clamps_left = 12

    for attempt in range(1, MAX_RETRIES + 1):
        for index in _next_upstream_order():
            url = UPSTREAMS[index]
            attempts += 1
            request = urllib.request.Request(
                url,
                data=encoded,
                method="POST",
                headers={"Authorization": "Bearer local", "Content-Type": "application/json"},
            )
            with _slots[index]:
                with _inflight_lock:
                    _inflight[index] += 1
                try:
                    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                        parsed = json.loads(response.read().decode("utf-8"))
                        return response.status, parsed, attempts, url
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")[:1000]
                    last_error = RuntimeError(f"{url} HTTP {exc.code}: {detail}")
                    if exc.code == 400 and clamps_left > 0:
                        # Clamping loops here, against this same upstream, instead
                        # of falling through to the next one. Previously it did
                        # `continue`, so the number of clamps actually attempted was
                        # bounded by MAX_RETRIES x len(UPSTREAMS) and clamps_left was
                        # never the binding constraint -- a prompt needing ~9 halvings
                        # ran out of requests first and the 400 surfaced as a 502.
                        while clamps_left > 0:
                            clamped = _clamped_payload(payload, detail)
                            if clamped is None:
                                break
                            payload = clamped
                            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                            clamps_left -= 1
                            attempts += 1
                            retry_request = urllib.request.Request(
                                url,
                                data=encoded,
                                method="POST",
                                headers={"Authorization": "Bearer local",
                                         "Content-Type": "application/json"},
                            )
                            try:
                                with urllib.request.urlopen(retry_request, timeout=REQUEST_TIMEOUT) as response:
                                    parsed = json.loads(response.read().decode("utf-8"))
                                    with _stat_lock:
                                        _counters["context_clamped"] += 1
                                    return response.status, parsed, attempts, url
                            except urllib.error.HTTPError as retry_exc:
                                detail = retry_exc.read().decode("utf-8", errors="replace")[:1000]
                                last_error = RuntimeError(
                                    f"{url} HTTP {retry_exc.code}: {detail}")
                                if retry_exc.code != 400:
                                    break
                            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as retry_exc:
                                last_error = RuntimeError(
                                    f"{url}: {type(retry_exc).__name__}: {retry_exc}")
                                break
                    if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                        raise last_error
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                    last_error = RuntimeError(f"{url}: {type(exc).__name__}: {exc}")
                finally:
                    with _inflight_lock:
                        _inflight[index] -= 1
        if attempt < MAX_RETRIES:
            time.sleep(min(20.0, 1.5 * (2 ** (attempt - 1))))

    raise RuntimeError(str(last_error or "all upstreams failed"))


def _inspect_and_clean(response: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return (content, diagnostics). Flags thinking leakage and truncation."""
    global _leak_warned
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    finish_reason = choice.get("finish_reason")

    leaked = bool(reasoning) or bool(_OPEN_THINK_RE.search(content))
    if leaked:
        content = _THINK_RE.sub("", content)
        content = _OPEN_THINK_RE.sub("", content)
        if not _leak_warned:
            _leak_warned = True
            print(
                json.dumps(
                    {
                        "event": "thinking_leak_detected",
                        "hint": "enable_thinking did not take effect upstream; "
                        "verify chat_template_kwargs reaches vLLM before trusting results",
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
        message["content"] = content
        message.pop("reasoning_content", None)

    diagnostics = {
        "thinking_leak": leaked,
        "truncated": finish_reason == "length",
        "finish_reason": finish_reason,
        "empty_content": not content.strip(),
        "recovered": False,
    }
    return content, diagnostics


def _recover_truncated(payload: dict[str, Any], response: dict[str, Any], content: str) -> tuple[dict, str, dict]:
    """Ask once for just the final answer and splice it onto the cut-off reply.

    The continuation is a separate short call rather than a larger max_tokens so
    the frozen per-call budget stays intact and the extra cost is visible in the
    logs instead of hidden inside a bigger cap.
    """
    follow_up = dict(payload)
    follow_up["messages"] = list(payload.get("messages", [])) + [
        {"role": "assistant", "content": content},
        {"role": "user", "content": CONTINUATION_PROMPT},
    ]
    follow_up["max_tokens"] = CONTINUATION_MAX_TOKENS

    status, second, attempts, url = _forward(follow_up)
    tail = ((second.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    merged = content + ("\n" + tail if tail.strip() else "")

    # Fold the continuation into the original response so downstream parsers see
    # one complete reply, and add both calls' usage together.
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    message["content"] = merged
    choice["finish_reason"] = (second.get("choices") or [{}])[0].get("finish_reason") or "stop"
    base_usage = response.get("usage") or {}
    extra_usage = second.get("usage") or {}
    response["usage"] = {
        key: (base_usage.get(key) or 0) + (extra_usage.get(key) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    return response, merged, {"continuation_attempts": attempts, "continuation_upstream": url}


SSE_DONE = "data: [DONE]\n\n"


def _as_sse_chunk(response: dict[str, Any], content: str) -> str:
    """Wrap a completed response as one chat.completion.chunk SSE event."""
    choice = (response.get("choices") or [{}])[0]
    chunk = {
        "id": response.get("id", "chatcmpl-shim"),
        "object": "chat.completion.chunk",
        "created": response.get("created", int(time.time())),
        "model": response.get("model", UPSTREAM_MODEL),
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": content},
            "finish_reason": choice.get("finish_reason") or "stop",
        }],
    }
    return "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"


class Handler(BaseHTTPRequestHandler):
    server_version = "VllmRoundRobinProxy/2.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, response: dict[str, Any], content: str) -> None:
        body = (_as_sse_chunk(response, content) + SSE_DONE).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path.endswith("/v1/models"):
            self._send(200, {"object": "list", "data": [{"id": name, "object": "model"} for name in MODEL_ALIASES]})
            return
        if path in {"/health", "/stats"}:
            with _stat_lock:
                body = {
                    "status": "ok",
                    "upstreams": UPSTREAMS,
                    "upstream_model": UPSTREAM_MODEL,
                    "per_upstream_concurrency": PER_UPSTREAM_CONCURRENCY,
                    "inflight_per_upstream": list(_inflight),
                    "uptime_seconds": round(time.time() - _started_at, 3),
                    "totals": dict(_counters),
                    "by_namespace": {k: dict(v) for k, v in _ns_counters.items()},
                    "by_requested_model": dict(_model_counters),
                    "max_tokens_caps": dict(_capped_requests),
                }
            self._send(200, body)
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        request_id = str(uuid.uuid4())
        started = time.monotonic()
        namespace = _namespace_from_path(self.path)
        record: dict[str, Any] = {
            "timestamp": _utc_now(),
            "request_id": request_id,
            "namespace": namespace,
            "path": self.path.split("?", 1)[0],
            "upstream_model": UPSTREAM_MODEL,
        }
        attempts = 0
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            payload, legacy, requested_model, wants_stream = _normalize(body)
            prompt_material = json.dumps(payload.get("messages", []), ensure_ascii=False, sort_keys=True)
            record["prompt_sha256"] = hashlib.sha256(prompt_material.encode("utf-8")).hexdigest()
            record["requested_model"] = requested_model
            record["legacy"] = legacy

            status, response, attempts, url = _forward(payload)
            content, diagnostics = _inspect_and_clean(response)

            # Filling the whole budget is treated as a decoding loop and answered
            # by re-generating the identical request, not by continuing it. No
            # detector is needed: hitting the cap is itself the signal, which is
            # both cheaper and less arguable than measuring repetition.
            if RETRY_ON_LENGTH and diagnostics["truncated"]:
                retry_tokens = 0
                for attempt_index in range(RETRIES_ON_LENGTH):
                    try:
                        _, retried, retry_attempts, retry_url = _forward(payload)
                    except Exception as exc:  # noqa: BLE001 - keep the first reply
                        record["retry_error"] = f"{type(exc).__name__}: {exc}"
                        break
                    retry_content, retry_diagnostics = _inspect_and_clean(retried)
                    retry_tokens += (retried.get("usage") or {}).get("completion_tokens", 0)
                    attempts += retry_attempts
                    record["retried_on_length"] = attempt_index + 1
                    record["retry_upstream"] = retry_url
                    if not retry_diagnostics["truncated"]:
                        # Only here is anything actually recovered. The old
                        # continuation path set this flag before the second call
                        # returned, which is why it reported 100% recovery against
                        # a true rate of 88.8%.
                        response, content = retried, retry_content
                        diagnostics = retry_diagnostics
                        diagnostics["recovered"] = True
                        break
                    diagnostics["retry_also_truncated"] = True
                # The retry's tokens are real spend and are accounted separately
                # rather than folded into the reply's usage, so cost per method
                # stays attributable and a high retry rate is visible.
                record["retry_completion_tokens"] = retry_tokens
            elif CONTINUE_ON_TRUNCATION and diagnostics["truncated"]:
                try:
                    response, content, extra = _recover_truncated(payload, response, content)
                    diagnostics["finish_reason"] = "stop_after_continuation"
                    diagnostics["empty_content"] = not content.strip()
                    # Set only once the continuation has actually come back.
                    diagnostics["recovered"] = True
                    record.update(extra)
                except Exception as exc:  # noqa: BLE001 - keep the truncated reply
                    record["continuation_error"] = f"{type(exc).__name__}: {exc}"

            usage = response.get("usage") or {}
            record.update(
                {
                    "status": status,
                    "attempts": attempts,
                    "upstream": url,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "latency_ms": round((time.monotonic() - started) * 1000, 3),
                    **diagnostics,
                }
            )
            with _stat_lock:
                _counters["requests"] += 1
                _counters["successes"] += 1
                _ns_counters[namespace]["requests"] += 1
                # Per-name tallies make a fictitious model pool visible in the
                # accounting instead of only in the code that built it.
                _model_counters[requested_model] += 1
                _ns_counters[namespace]["prompt_tokens"] += usage.get("prompt_tokens", 0)
                _ns_counters[namespace]["completion_tokens"] += usage.get("completion_tokens", 0)
                # "truncated" now means *still* truncated after the retry, so the
                # two are counted separately: hit_length_once is how often the cap
                # was reached at all, and truncated is how often it survived.
                for flag in ("thinking_leak", "truncated", "empty_content", "recovered",
                             "retry_also_truncated"):
                    if diagnostics.get(flag):
                        _counters[flag] += 1
                        _ns_counters[namespace][flag] += 1
                if record.get("retried_on_length"):
                    _counters["hit_length_once"] += 1
                    _ns_counters[namespace]["hit_length_once"] += 1
                    _ns_counters[namespace]["retry_completion_tokens"] += \
                        record.get("retry_completion_tokens", 0)

            # Before the reply, and outside the branches: legacy, streaming and
            # plain callers all get captured, and one transcript line exists per
            # successful call regardless of which shape the caller asked for.
            _write_transcript(record, payload, content)

            if legacy:
                self._send(200, {"data": response["choices"][0]["message"]["content"], "usage": usage,
                                 "request_id": request_id})
            elif wants_stream:
                self._send_sse(response, content)
            else:
                self._send(status, response)
        except Exception as exc:  # noqa: BLE001 - proxy must not die on one call
            # A prompt that cannot fit the window is answered, not refused.
            #
            # The clamp ladder shrinks max_tokens until the request fits; when the
            # PROMPT alone exceeds max_model_len there is nothing left to shrink and
            # vLLM keeps returning 400. Surfacing that as a 502 makes the client
            # retry, and tenacity's budget then runs out and takes the whole job
            # with it: masrouter/mmlu_pro died that way after 326 minutes, on 30
            # requests out of 2,675 (1.12% of its calls, 0.015% of the sweep).
            #
            # Returning a well-formed empty completion instead lets the method
            # record that one sample as unanswered -- which is the truth, the
            # request really is unservable within the frozen 32k window -- and keep
            # going. Counted per namespace so the rate stays reportable.
            detail = str(exc)
            if "maximum context length" in detail:
                with _stat_lock:
                    _counters["requests"] += 1
                    _counters["context_overflow"] += 1
                    _ns_counters[namespace]["context_overflow"] += 1
                # Capture the prompt on THIS failure path too. The transcript is
                # written only for successful calls, so when 30 requests overflowed
                # the window there was no way to replay them and check whether a
                # wider window would serve them -- the one question worth asking.
                try:
                    _write_transcript(record, locals().get("payload") or {}, "")
                except Exception:  # noqa: BLE001
                    pass
                record.update({
                    "status": 200,
                    "attempts": attempts,
                    "context_overflow": True,
                    "latency_ms": round((time.monotonic() - started) * 1000, 3),
                    "error": detail[:400],
                })
                empty = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": UPSTREAM_MODEL,
                    "choices": [{"index": 0, "finish_reason": "length",
                                 "message": {"role": "assistant", "content": ""}}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                              "total_tokens": 0},
                }
                try:
                    if "legacy" in record and record["legacy"]:
                        self._send(200, {"data": "", "usage": empty["usage"],
                                         "request_id": request_id})
                    else:
                        self._send(200, empty)
                except Exception:  # noqa: BLE001 - client may already be gone
                    pass
                _write_log(record)
                return
            with _stat_lock:
                _counters["requests"] += 1
                _counters["failures"] += 1
                _ns_counters[namespace]["failures"] += 1
            # The prompt is captured for failures too, not only for successes.
            #
            # It used to be written only on the success path, which is exactly
            # backwards: when 30 requests overflowed the context window there was no
            # record of what they contained, so the one question worth asking --
            # would a wider window have served them -- could not be answered without
            # re-running. A failure is when the input matters most.
            try:
                _write_transcript(record, locals().get("payload") or {}, "")
            except Exception:  # noqa: BLE001 - diagnostics must not mask the error
                pass
            record.update(
                {
                    "status": 502,
                    "attempts": attempts,
                    "latency_ms": round((time.monotonic() - started) * 1000, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            self._send(502, {"error": {"message": str(exc), "type": "proxy_error"}, "request_id": request_id})
        finally:
            _write_log(record)


class Server(ThreadingHTTPServer):
    """ThreadingHTTPServer defaults to a listen backlog of 5, which drops
    connections with a TCP reset whenever the author repos open a burst of
    parallel calls. Those losses look like random upstream failures in the
    experiment logs, so the queue is sized well above our concurrency ceiling."""

    request_queue_size = 512
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    if not UPSTREAMS:
        raise SystemExit("VLLM_UPSTREAMS is empty")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    server = Server((HOST, PORT), Handler)
    print(
        json.dumps(
            {
                "event": "proxy_started",
                "host": HOST,
                "port": PORT,
                "upstreams": UPSTREAMS,
                "upstream_model": UPSTREAM_MODEL,
                "per_upstream_concurrency": PER_UPSTREAM_CONCURRENCY,
                "log_path": str(LOG_PATH),
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
