"""Second opinion over every equivalence-awarded case: exact symbolic equality.

Flags any full-credit pair that is NOT exactly equal (these were awarded by
abs-tolerance or another lenient path and are candidate false positives).
"""
import json, glob, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
import sympy
from sympy.parsing.latex import parse_latex


def clean(s):
    s = str(s).strip()
    s = re.sub(r"\\[,;!]", "", s)          # latex spacing incl. ,\! thousands
    s = s.replace(",\\!", "").replace("\\!", "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # 9,900 -> 9900
    s = s.replace("\\%", "").replace("%", "").replace("\\$", "").replace("$", "")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"^\{+", "", s); s = re.sub(r"\}+$", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    return s.strip()


def parse_any(s):
    s = clean(s)
    try:
        return parse_latex(s)
    except Exception:
        pass
    try:
        return sympy.sympify(s.replace("\\", ""))
    except Exception:
        return None


def exactly_equal(g, e):
    a, b = parse_any(g), parse_any(e)
    if a is None or b is None:
        return None  # cannot judge symbolically
    try:
        return bool(sympy.simplify(a - b) == 0)
    except Exception:
        return None


def main():
    base = ROOT / "audits" / "top3_award_audit"
    flagged, unjudged, ok = [], [], 0
    for f in sorted(base.glob("*_equiv.json")):
        cell = f.stem.replace("_equiv", "")
        seen = set()
        for x in json.load(f.open()):
            key = (x["gold"], x["extracted"])
            if key in seen:
                continue
            seen.add(key)
            gold, got = x["gold"], x["extracted"]
            # multi-answer / list golds: skip symbolic, note as unjudged-by-design
            if "|" in gold or ("," in gold and "(" not in gold and "frac" not in gold):
                unjudged.append((cell, gold, got, "list/alt gold"))
                continue
            r = exactly_equal(gold, got)
            if r is True:
                ok += 1
            elif r is False:
                flagged.append((cell, gold, got))
            else:
                unjudged.append((cell, gold, got, "unparseable"))
    print(f"exact-equal confirmed: {ok}")
    print(f"FLAGGED not-exactly-equal: {len(flagged)}")
    for c, g, e in flagged:
        print(f"  [{c}] GOLD {g!r}  GOT {e!r}")
    print(f"unjudged (manual): {len(unjudged)}")
    for c, g, e, why in unjudged[:40]:
        print(f"  ? [{c}] {why}: GOLD {g[:60]!r} GOT {e[:60]!r}")


if __name__ == "__main__":
    main()
