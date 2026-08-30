"""Rate of rounds whose graph references an undefined prompt_custom variable.

This is the quantity the template fix is supposed to move, so it is measured
rather than assumed. Compares the current run against the archived one.
"""
import re
from pathlib import Path


def rate(workflows: Path) -> tuple[int, int, list[str]]:
    total = bad = 0
    names = []
    for round_dir in sorted(workflows.glob("round_*")):
        graph, prompt = round_dir / "graph.py", round_dir / "prompt.py"
        if not graph.exists():
            continue
        total += 1
        used = set(re.findall(r"prompt_custom\.([A-Z_][A-Z0-9_]*)",
                              graph.read_text(encoding="utf-8")))
        if not used:
            continue
        defined = set(re.findall(r"^([A-Z_][A-Z0-9_]*)\s*=",
                                 prompt.read_text(encoding="utf-8") if prompt.exists() else "",
                                 re.MULTILINE))
        missing = used - defined
        if missing:
            bad += 1
            names.extend(sorted(missing))
    return total, bad, names


for label, root in (("current", Path(".")), ("archived (pre-fix)", Path("archive/pre_icl"))):
    print(f"\n### {label}")
    for workflows in sorted(root.glob("third_party/*/workspace/SHARED_*/workflows")) + \
                     sorted(root.glob("third_party/*/*/workspace/SHARED_*/workflows")):
        total, bad, names = rate(workflows)
        if not total:
            continue
        repo = workflows.parts[len(root.parts) + 1]
        key = [p for p in workflows.parts if p.startswith("SHARED_")][0]
        share = f"{100 * bad / total:.0f}%" if total else "-"
        print(f"  {repo:<10} {key:<16} {bad:>2}/{total:<3} rounds broken ({share:>4})"
              f"{'  missing: ' + ', '.join(sorted(set(names))[:4]) if names else ''}")
