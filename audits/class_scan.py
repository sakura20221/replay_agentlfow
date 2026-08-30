import csv,glob,re,sys
csv.field_size_limit(sys.maxsize)
tot=cls=0; hits=[]
for f in glob.glob("third_party/*/workspace/SHARED_MBPP/**/*.csv",recursive=True)+glob.glob("runs_v5/*/mbpp/**/*.csv",recursive=True):
    try:
        for row in csv.DictReader(open(f)):
            pred=(row.get("prediction") or row.get("output") or row.get("answer") or "")
            if not pred: continue
            tot+=1
            if re.search(r"^\s*class\s+\w+",pred,re.M):
                cls+=1
                if len(hits)<5: hits.append(f)
    except Exception: pass
print(f"mbpp作答总数={tot} class型作答={cls}")
for h in hits: print("  hit:",h)
