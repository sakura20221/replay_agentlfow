"""Of the replies that hit max_tokens, how many are degenerate repetition?

The distinction decides what to do about them. A reply that was genuinely long and
got cut off is a budget problem -- a bigger cap finishes it. A reply that had
collapsed into repetition is a decoding problem -- it runs to any cap, and only a
penalty (or a non-greedy temperature) stops it.

Two independent measures, because either alone can mislead:

* the share of the reply taken by its single most repeated line -- catches verbatim
  loops, but a reply with few lines can trip it by accident;
* the fraction of distinct 20-token windows -- catches near-repetition that varies
  slightly each cycle, and is stable for long text.

A reply is called degenerate when either measure is extreme.
"""
import collections
import json
from pathlib import Path

CAP = 8192


def line_repetition(text: str) -> float:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 4:
        return 0.0
    return collections.Counter(lines).most_common(1)[0][1] / len(lines)


def distinct_ngram_ratio(text: str, n: int = 20) -> float:
    words = text.split()
    if len(words) < n * 3:
        return 1.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


rows = []
for line in Path("logs/transcripts.jsonl").read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        continue
    tokens = entry.get("completion_tokens") or 0
    text = entry.get("completion") or ""
    rows.append((entry.get("namespace", "?"), tokens, text,
                 entry.get("finish_reason")))

truncated = [r for r in rows if r[1] >= CAP - 200 or r[3] == "length"]
print(f"  exchanges captured: {len(rows):,}")
print(f"  of those, hit the {CAP:,}-token cap: {len(truncated):,} "
      f"({len(truncated) / max(len(rows), 1):.2%})")

buckets = collections.Counter()
per_method = collections.defaultdict(lambda: collections.Counter())
for namespace, tokens, text, _ in truncated:
    rep = line_repetition(text)
    distinct = distinct_ngram_ratio(text)
    if rep >= 0.30 or distinct <= 0.55:
        label = "degenerate repetition"
    elif rep >= 0.15 or distinct <= 0.80:
        label = "partly repetitive"
    else:
        label = "genuinely long"
    buckets[label] += 1
    per_method[namespace][label] += 1

total = max(len(truncated), 1)
print("\n  cause of hitting the cap:")
for label in ("degenerate repetition", "partly repetitive", "genuinely long"):
    count = buckets[label]
    print(f"    {label:<24}{count:>7,}  ({count / total:>5.1%})")

print(f"\n  {'method':<20}{'capped':>9}{'degenerate':>13}{'partly':>10}{'long':>8}")
for namespace, counts in sorted(per_method.items(), key=lambda kv: -sum(kv[1].values())):
    n = sum(counts.values())
    print(f"  {namespace:<20}{n:>9,}{counts['degenerate repetition']:>13,}"
          f"{counts['partly repetitive']:>10,}{counts['genuinely long']:>8,}")

# What share of all calls does each cause represent -- the number that matters for
# how much the final table can move.
print(f"\n  as a share of all {len(rows):,} captured calls:")
for label in ("degenerate repetition", "partly repetitive", "genuinely long"):
    print(f"    {label:<24}{buckets[label] / max(len(rows), 1):>7.3%}")
