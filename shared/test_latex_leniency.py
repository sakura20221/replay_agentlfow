"""Does the LaTeX-form retry recover typographic misses without inventing points?"""
import sys
sys.path.insert(0, ".")
import bench

RECOVER = [
    (r"\left( 3, \frac{\pi}{2} \right)", r"\boxed{(3, \frac{\pi}{2})}", "left/right coords"),
    (r"\dfrac{1}{2}",                    r"\boxed{\frac{1}{2}}",        "dfrac vs frac"),
    (r"2\text{ cm}",                     r"\boxed{2\mbox{ cm}}",        "text vs mbox"),
    (r"\left[ 0, \infty \right)",        r"\boxed{[0, \infty)}",        "interval"),
    (r"\frac{1}{2}\!",                   r"\boxed{\frac{1}{2}}",        "spacing macro"),
    (r"(3,\frac{\pi}{2})",  r"\boxed{\left(3, \frac{\pi}{2}\right)}",   "reverse direction"),
    (r"\text{Evelyn}",                   r"\boxed{Evelyn}",             "word answer"),
]
# Wrong answers that a careless normalisation would start accepting.
REJECT = [
    (r"\frac{1}{2}",      r"\boxed{\frac{1}{3}}", "different denominator"),
    (r"\left(3,4\right)", r"\boxed{(4,3)}",       "coords swapped"),
    (r"5",                r"\boxed{-5}",          "sign flipped"),
    (r"2\text{ cm}",      r"\boxed{2\text{ m}}",  "different unit"),
    (r"\frac{1}{2}",      r"\boxed{}",            "empty answer"),
    (r"\left[0,1\right)", r"\boxed{[0,1]}",       "open vs closed interval"),
    (r"\text{Evelyn}",    r"\boxed{Brenda}",      "wrong name"),
    (r"\frac{1}{2}",      r"\boxed{\frac{2}{1}}", "fraction inverted"),
]


def run(gold, prediction):
    try:
        return bench.score("math", {"problem": "x", "answer": gold}, prediction)
    except Exception as exc:  # noqa: BLE001 - a raise is a failure, not a skip
        return (f"RAISED {type(exc).__name__}: {exc}", "")


def main() -> int:
    print("  ### typographic differences that should be recovered ###")
    recovered = 0
    for gold, prediction, why in RECOVER:
        value, extracted = run(gold, prediction)
        hit = value == 1.0
        recovered += hit
        print(f"    {'OK  ' if hit else 'MISS'} {why:<22} score={value} extracted={extracted!r}")
    print(f"    -> {recovered}/{len(RECOVER)} recovered")

    print("  ### wrong answers that must stay wrong ###")
    false_positives = 0
    for gold, prediction, why in REJECT:
        value, extracted = run(gold, prediction)
        bad = value != 0.0
        false_positives += bad
        print(f"    {'BAD ' if bad else 'OK  '} {why:<22} score={value} extracted={extracted!r}")
    print(f"    -> {false_positives} false positive(s) of {len(REJECT)}")

    print("  counters:", dict(bench.extraction_stats()["math"]))
    if false_positives:
        print("  FAIL: leniency is accepting wrong answers")
        return 1
    if recovered < len(RECOVER):
        print(f"  PARTIAL: {len(RECOVER) - recovered} typographic case(s) still missed")
        return 2
    print("  PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
