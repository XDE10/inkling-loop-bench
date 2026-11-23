import math, random
from collections import defaultdict
def summarize(rows):
    n=len(rows); correct=sum(bool(r["correct"]) for r in rows)
    times=sorted(float(r["e2e_ms"]) for r in rows if r.get("e2e_ms") is not None)
    def pct(p):
        if not times:return None
        x=(len(times)-1)*p;lo=math.floor(x);hi=math.ceil(x)
        return times[lo]*(hi-x)+times[hi]*(x-lo) if hi!=lo else times[lo]
    by={}
    for key in ("modality","task_type"):
        vals=defaultdict(list)
        for row in rows: vals[row[key]].append(bool(row["correct"]))
        by[key]={k:sum(v)/len(v) for k,v in vals.items()}
    ci=cluster_bootstrap_accuracy_ci(rows)
    return {"n":n,"accuracy":correct/n if n else None,"correct":correct,
            "e2e_p50_ms":pct(.5),"e2e_p90_ms":pct(.9),
            "parse_failures":sum(r["parse_status"]=="parse_failure" for r in rows),
            "truncated":sum(r["finish_reason"]=="length" for r in rows),
            "by_modality":by["modality"],"by_task_type":by["task_type"],
            "accuracy_cluster_bootstrap_95ci":ci,
            "mock":all(r.get("is_mock",False) for r in rows)}
def cluster_bootstrap_accuracy_ci(rows,n_boot=1000,seed=20261123):
    groups=defaultdict(list)
    for row in rows: groups[row["image_id"]].append(bool(row["correct"]))
    if not groups:return [None,None]
    clusters=list(groups.values());rng=random.Random(seed);samples=[]
    for _ in range(n_boot):
        chosen=[clusters[rng.randrange(len(clusters))] for _ in clusters]
        flat=[v for cluster in chosen for v in cluster]
        samples.append(sum(flat)/len(flat))
    samples.sort()
    return [samples[int(.025*(n_boot-1))],samples[int(.975*(n_boot-1))]]
