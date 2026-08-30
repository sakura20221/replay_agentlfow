import json, re, sys
sys.path.insert(0, 'shared')
import bench

def norm(t): return re.sub(r'\s+', ' ', str(t)).strip()[:300]

def load_theirs(name):
    rows = []
    for split in ('validate', 'test'):
        p = f'third_party/flowbank/datasets/{name}_{split}.jsonl'
        try:
            part = [json.loads(l) for l in open(p) if l.strip()]
        except FileNotFoundError:
            part = []
        rows.append(part)
    return rows

KEYS = {  # 各自的题面字段
    'math': ('problem', 'problem'),
    'amc': ('problem', 'problem'),
    'drop': ('context', 'task'),
    'mmlu_pro': ('question', 'question'),
    'mbpp': ('prompt', 'prompt'),
}
for ds in ('math', 'amc', 'drop', 'mmlu_pro', 'mbpp'):
    tv, tt = load_theirs(ds)
    ours = bench.load(ds)
    tk, ok_ = KEYS[ds]
    if tv or tt:
        sample = (tt or tv)[0]
        tkey = tk if tk in sample else next((k for k in ('problem','question','context','prompt','task') if k in sample), None)
    else:
        tkey = None
    their_texts = {norm(r.get(tkey)) for r in tv + tt} if tkey else set()
    our_texts = set()
    for r in ours:
        for k in (ok_, 'problem', 'question', 'task', 'context', 'prompt'):
            if r.get(k):
                our_texts.add(norm(r[k])); break
    inter = len(their_texts & our_texts)
    print(f'  {ds:<10} 官方 {len(tv):>4}+{len(tt):>4}  我们 {len(ours):>5}(测试)  重叠 {inter:>4}'
          f'  字段={tkey}')
    if tt:
        extra = {k: str(v)[:40] for k, v in tt[0].items() if k not in (tkey,)}
        print(f'             官方字段: {list(tt[0].keys())}')
