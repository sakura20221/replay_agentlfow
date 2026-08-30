"""Does max_tokens change what the model writes, or only where it is cut off?

If the model adapted to the budget, a small cap would produce a short *complete*
answer. If the cap is a guillotine, a small cap produces the *same* text stopped
earlier. At temperature 0 the two are trivially distinguishable: identical
prefixes mean the model never knew the limit existed.
"""
import json
import urllib.request

URL = "http://127.0.0.1:8001/v1/chat/completions"
QUESTION = ("A train travels 120 km at 60 km/h, then 180 km at 90 km/h. "
            "Explain your reasoning step by step, then give the average speed "
            "for the whole journey.")


def ask(max_tokens: int) -> tuple[str, str, int]:
    payload = {"model": "Qwen/Qwen3-8B",
               "messages": [{"role": "user", "content": QUESTION}],
               "temperature": 0, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST")
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.loads(response.read())
    choice = body["choices"][0]
    return (choice["message"]["content"] or "",
            choice.get("finish_reason"),
            body["usage"]["completion_tokens"])


results = {}
for cap in (32, 128, 512, 4096):
    text, finish, tokens = ask(cap)
    results[cap] = text
    print(f"  max_tokens={cap:<5} finish={finish:<7} produced {tokens} tokens")
    print(f"      ends with: {text[-70:]!r}")

print("\n  is each shorter reply a prefix of the longest one?")
longest = results[4096]
for cap in (32, 128, 512):
    shorter = results[cap]
    print(f"    max_tokens={cap:<5} prefix of the 4096 reply: {longest.startswith(shorter)}")
