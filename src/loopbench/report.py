import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .metrics import summarize
def build_report(jsonl_path,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    summary=summarize(rows);summary["label"]="MOCK / TEST DATA — NOT REAL MODEL PERFORMANCE"
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with open(out/"summary.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f);w.writerow(["metric","value"])
        for k,v in summary.items():w.writerow([k,json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v])
    (out/"report.md").write_text("# LoopBench mock report\n\n> MOCK / TEST DATA — NOT REAL MODEL PERFORMANCE\n\n"+
        f"- Questions: {summary['n']}\n- Accuracy: {summary['accuracy']}\n- Cluster bootstrap 95% CI: {summary['accuracy_cluster_bootstrap_95ci']}\n- E2E P50: {summary['e2e_p50_ms']} ms\n- E2E P90: {summary['e2e_p90_ms']} ms\n",encoding="utf-8")
    fig,ax=plt.subplots(figsize=(6,4));ax.scatter([summary["e2e_p50_ms"] or 0],[summary["accuracy"] or 0])
    ax.set_xlabel("E2E P50 (ms)");ax.set_ylabel("Accuracy");ax.set_title("MOCK / TEST DATA — NOT REAL MODEL PERFORMANCE")
    fig.tight_layout();fig.savefig(out/"pareto_mock.png",metadata={"Software":"LoopBench"});plt.close(fig)
    return summary
