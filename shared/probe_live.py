"""One live call per dataset, through the proxy, graded by the shared scorer.

Checks the pieces the unit tests cannot: that the format instruction and its
example actually reach the model, that the reply comes back, and that the
transcript records both sides.
"""
import json
import sys
import urllib.request

sys.path.insert(0, ".")
import bench

for dataset in ("math", "amc", "mbpp", "drop", "mmlu_pro"):
    row = bench.load(dataset)[0]
    question = bench.question_text(dataset, row)
    assert "Format example" in question, f"{dataset}: the format example is missing"
    request = urllib.request.Request(
        f"http://127.0.0.1:18080/probe_live/{dataset}/v1/chat/completions",
        data=json.dumps({"model": "qwen3-8b", "max_tokens": 3072,
                         "messages": [{"role": "user", "content": question}]}).encode(),
        headers={"Content-Type": "application/json"})
    reply = json.loads(urllib.request.urlopen(request, timeout=600).read())
    text = reply["choices"][0]["message"]["content"]
    value, extracted = bench.score(dataset, row, text)
    last = text.strip().splitlines()[-1][:58] if text.strip() else "(empty)"
    print(f"  {dataset:<9} score={value:<5} extracted={extracted[:32]!r:<34} last_line={last!r}")

print()
print("  extraction tiers:", json.dumps({k: dict(v) for k, v in bench.extraction_stats().items()},
                                        ensure_ascii=False))
