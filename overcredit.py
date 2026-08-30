"""daao/math 0.838 是否虚高:抽查所有"非精确匹配却得满分"的题。"""
import csv, sys, re
sys.path.insert(0, 'shared')
import bench

path = 'third_party/daao/daao/ext/maas/scripts/optimized/SHARED_MATH/test/round_1/0.83800_20260823_233805.csv'
rows = list(csv.DictReader(open(path, newline='', errors='replace')))
exact = loose = 0
samples = []
for r in rows:
    if float(r.get('score') or 0) < 0.999:
        continue
    gold = (r.get('expected_output') or '').strip()
    v, extracted = bench.score('math', {'answer': gold}, r.get('prediction') or '')
    e = extracted.strip()
    # 与 gold 逐字符一致(去空格)的算精确;其余是等价判定给分的,需要人眼
    if re.sub(r'\s+', '', e) == re.sub(r'\s+', '', gold):
        exact += 1
    else:
        loose += 1
        if len(samples) < 14:
            samples.append((gold, e))
n1 = exact + loose
print(f'  满分题 {n1}/{len(rows)}: 精确匹配 {exact}, 等价判定 {loose} ({loose/max(n1,1):.1%})')
print(f'  --- 等价判定的样本(gold vs 模型答案,人眼核对是否真等价)---')
for g, e in samples:
    print(f'    gold={g[:34]!r:<38} model={e[:44]!r}')
