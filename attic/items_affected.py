"""How many dataset *items* did the repetition loops touch, not how many calls.

A call-level rate understates the damage: one item drives several calls, so a 0.46%
call rate means a higher item rate, by a factor that differs per method -- which is
the same unequal-budget asymmetry that makes every other per-call statistic
misleading on its own.

Each looping call is matched back to an item by looking for that item's question
text inside the prompt. The question is what the shared layer puts in every prompt,
so it is present whatever scaffolding a method wraps around it.
"""
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "shared")
import bench  # noqa: E402

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


# A distinctive slice of each item's question, long enough not to collide.
signatures = {}
counts_per_dataset = {}
for dataset in ("math", "amc", "mbpp", "drop", "mmlu_pro"):
    rows = list(bench.load(dataset))
    counts_per_dataset[dataset] = len(rows)
    for row in rows:
        text = re.sub(r"\s+", " ", bench.question_text(dataset, row)).strip()
        if len(text) < 80:
            continue
        # Middle slice: the head is often a shared preamble ("Passage:"), the tail is
        # the answer-format instruction, which every item shares.
        middle = text[len(text) // 3: len(text) // 3 + 60]
        if len(middle) == 60:
            signatures.setdefault(middle, (dataset, str(row["uid"])))

print(f"  indexed {len(signatures):,} distinctive question slices "
      f"from {sum(counts_per_dataset.values()):,} evaluation items")

looping = []
for line in Path("logs/transcripts.jsonl").read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        continue
    if (entry.get("completion_tokens") or 0) < CAP - 200:
        continue
    text = entry.get("completion") or ""
    if line_repetition(text) >= 0.30 or distinct_ngram_ratio(text) <= 0.55:
        looping.append(entry)

print(f"  looping calls: {len(looping):,}")

hit_items = collections.defaultdict(set)
unmatched = 0
for entry in looping:
    prompt = " ".join(str(m.get("content", "")) for m in (entry.get("messages") or []))
    prompt = re.sub(r"\s+", " ", prompt)
    found = None
    for slice_, (dataset, uid) in signatures.items():
        if slice_ in prompt:
            found = (dataset, uid)
            break
    if found:
        hit_items[found[0]].add(found[1])
    else:
        unmatched += 1

print(f"  matched to an evaluation item: {len(looping) - unmatched:,}   unmatched: {unmatched:,}")
print(f"    (unmatched are search-split items or optimiser calls, which carry no"
      f" evaluation item)")
print(f"\n  {'dataset':<12}{'items touched':>15}{'split size':>12}{'share':>9}")
for dataset, uids in sorted(hit_items.items()):
    total = counts_per_dataset[dataset]
    print(f"  {dataset:<12}{len(uids):>15,}{total:>12,}{len(uids) / total:>8.2%}")
if not hit_items:
    print("  no looping call matched an evaluation item")
