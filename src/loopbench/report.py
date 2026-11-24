import csv
import io
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .metrics import summarize
from .storage import atomic_write_text, load_rows

MOCK_STAMP = "MOCK / TEST DATA — NOT REAL MODEL PERFORMANCE"


def build_report(jsonl_path, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(jsonl_path)
    summary = summarize(rows)
    is_mock = bool(summary["mock"])
    summary["label"] = MOCK_STAMP if is_mock else "REAL MODEL EVALUATION"

    atomic_write_text(out / "summary.json", json.dumps(summary, indent=2) + "\n")

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf, lineterminator="\n")
    writer.writerow(["metric", "value"])
    for k, v in summary.items():
        writer.writerow([k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v])
    atomic_write_text(out / "summary.csv", csv_buf.getvalue(), encoding="utf-8-sig")

    banner = f"> {MOCK_STAMP}\n\n" if is_mock else ""
    title = "# LoopBench mock report" if is_mock else "# LoopBench evaluation report"
    md_lines = [
        title,
        "",
        banner.rstrip() if banner else "> Real model evaluation output",
        "",
        "## Summary metrics",
        "",
        f"- Questions: {summary['n']}",
        f"- Accuracy: {summary['accuracy']}",
        f"- Cluster bootstrap 95% CI: {summary['accuracy_cluster_bootstrap_95ci']}",
        f"- Correct: {summary['correct']}",
        f"- Wrong answers: {summary['wrong_answers']}",
        f"- Parse failures: {summary['parse_failures']}",
        f"- Truncated (`finish_reason=length`): {summary['truncated']}",
        f"- Backend errors: {summary['backend_errors']}",
        f"- E2E P50: {summary['e2e_p50_ms']} ms",
        f"- E2E P90: {summary['e2e_p90_ms']} ms",
        "",
        "## Breakdown",
        "",
        f"- By modality: `{json.dumps(summary['by_modality'])}`",
        f"- By task type: `{json.dumps(summary['by_task_type'])}`",
        "",
    ]
    atomic_write_text(out / "report.md", "\n".join(md_lines))

    plot_name = "pareto_mock.png" if is_mock else "pareto.png"
    with matplotlib.rc_context({"font.family": "DejaVu Sans"}):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=128)
        ax.scatter([summary["e2e_p50_ms"] or 0.0], [summary["accuracy"] or 0.0], s=60, color="#1F4E79")
        ax.set_xlabel("E2E P50 (ms)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title(MOCK_STAMP if is_mock else "LoopBench Accuracy vs Latency")
        fig.tight_layout()
        fig.savefig(out / plot_name, metadata={"Software": "LoopBench"})
        plt.close(fig)

    return summary
