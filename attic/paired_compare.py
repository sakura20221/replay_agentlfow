"""Compare two methods on exactly the items both have finished.

Two runs at the same coverage are not automatically comparable: they are only
comparable on the intersection of the uids each actually evaluated. On that
intersection the comparison is paired, so the right test is McNemar's on the
discordant pairs -- the unpaired standard error would overstate the uncertainty
by counting variance the pairing removes.
"""
import json
import sys
from math import sqrt
from pathlib import Path

sys.path.insert(0, "shared")
sys.path.insert(0, ".")
from collect import RUN_TAG, record_uid  # noqa: E402


def outcomes(repo: str, method: str, dataset: str) -> dict[str, float]:
    path = Path(f"third_party/{repo}/result/{RUN_TAG}/{method}_{dataset}_r1.json")
    if not path.exists():
        return {}
    import bench
    wanted = {str(r["uid"]) for r in bench.load(dataset)}
    out = {}
    for record in json.loads(path.read_text(encoding="utf-8")):
        uid = record_uid(record)
        if uid in wanted:
            out[uid] = float(record.get("Solved") or 0.0)
    return out


def compare(a_label, a, b_label, b) -> None:
    shared = sorted(set(a) & set(b))
    print(f"\n### {a_label}  vs  {b_label}")
    print(f"    {a_label}: {len(a)} items   {b_label}: {len(b)} items   "
          f"shared: {len(shared)}")
    if not shared:
        print("    no shared items yet -- not comparable")
        return
    a_score = sum(a[u] for u in shared) / len(shared)
    b_score = sum(b[u] for u in shared) / len(shared)
    # Discordant pairs: only these carry information about the difference.
    a_only = sum(1 for u in shared if a[u] > b[u])
    b_only = sum(1 for u in shared if b[u] > a[u])
    print(f"    on the shared {len(shared)}: {a_label}={a_score:.4f}  {b_label}={b_score:.4f}"
          f"  diff={a_score - b_score:+.4f}")
    print(f"    discordant: {a_label}-only correct {a_only}, {b_label}-only correct {b_only}")
    n = a_only + b_only
    if n == 0:
        print("    identical on every shared item")
        return
    # McNemar without continuity correction; exact enough for a progress read.
    chi2 = (a_only - b_only) ** 2 / n
    se = sqrt(n) / len(shared)
    print(f"    McNemar chi2={chi2:.2f} (needs ~3.84 for p<0.05)  "
          f"se of the difference ~{se:.4f} ({100 * se:.1f}pp)")
    print(f"    -> {'a real difference' if chi2 > 3.84 else 'NOT distinguishable yet'}")


gd = outcomes("gdesigner", "gdesigner", "drop")
cd = outcomes("card", "card", "drop")
compare("gdesigner/drop", gd, "card/drop", cd)

gda = outcomes("gdesigner", "gdesigner_authordefault", "mmlu_pro")
cda = outcomes("card", "card_authordefault", "mmlu_pro")
compare("gdesigner_ad/mmlu_pro", gda, "card_ad/mmlu_pro", cda)
