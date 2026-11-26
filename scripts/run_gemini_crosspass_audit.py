"""Single-shot cached qualitative cross-pass CoT failure & repair audit using Google AI Studio Gemini API.

Design guarantees:
- Reads API keys from `--keys-file`, `GEMINI_STUDIO_KEYS`, or `GEMINI_STUDIO_KEYS_FILE`
  without ever printing or persisting raw keys into logs or git-tracked files.
- Uses a persistent local SQLite WAL cache (`~/.cache/loopbench/gemini_audit_cache.sqlite3`)
  so re-runs incur zero API quota usage.
- Packs all cross-pass discordant / truncated / failure cases from Phase 4 (`synth-240`)
  and Phase 5 (`effort48` & `chartqa80`) into ONE concise structured JSON request
  (`models/gemini-4-hydrogen` with `thinkingLevel="HIGH"` or `"XHIGH"`, falling back to
  `models/gemini-3.8-flash` with `thinkingLevel="HIGH"`) well within the 5 req / 5h & 250K limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loopbench.storage import atomic_write_text, load_rows

API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def load_studio_keys(keys_file: str | Path | None = None) -> list[str]:
    env_raw = os.environ.get("GEMINI_STUDIO_KEYS", "").strip()
    if env_raw:
        keys = [k.strip() for k in env_raw.replace(",", "\n").splitlines() if k.strip()]
        if keys:
            return keys
    candidates: list[Path] = []
    if keys_file:
        candidates.append(Path(keys_file))
    env_p = os.environ.get("GEMINI_STUDIO_KEYS_FILE", "").strip()
    if env_p:
        candidates.append(Path(env_p))
    for p in candidates:
        if p.is_file():
            keys = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")]
            if keys:
                return keys
    raise FileNotFoundError(
        "No Gemini Studio API keys found. Provide `--keys-file`, `GEMINI_STUDIO_KEYS`, or `GEMINI_STUDIO_KEYS_FILE`."
    )


def _cache_db_path() -> Path:
    p = Path.home() / ".cache" / "loopbench" / "gemini_audit_cache.sqlite3"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _cache_get(key: str) -> dict | None:
    conn = sqlite3.connect(str(_cache_db_path()), timeout=20.0)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS responses (k TEXT PRIMARY KEY, created REAL, payload TEXT)")
        row = conn.execute("SELECT payload FROM responses WHERE k = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None
    finally:
        conn.close()


def _cache_put(key: str, payload: dict) -> None:
    conn = sqlite3.connect(str(_cache_db_path()), timeout=20.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("CREATE TABLE IF NOT EXISTS responses (k TEXT PRIMARY KEY, created REAL, payload TEXT)")
        conn.execute("INSERT OR REPLACE INTO responses (k, created, payload) VALUES (?, ?, ?)", (key, time.time(), json.dumps(payload, ensure_ascii=False)))
        conn.commit()
    finally:
        conn.close()


def _call_gemini(prompt_text: str, keys: list[str], model_candidates: list[tuple[str, str]]) -> dict:
    body_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    cache_key = hashlib.sha256(json.dumps({"models": model_candidates, "h": body_hash}, sort_keys=True).encode("utf-8")).hexdigest()
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "from_local_cache": True}

    last_err = None
    attempts_log: list[str] = []
    for model_name, thinking_lvl in model_candidates:
        for k_idx, api_key in enumerate(keys):
            for retry in range(2):
                url = f"{API_BASE}/models/{model_name}:generateContent?key={api_key}"
                req_body = {
                    "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
                    "generationConfig": {
                        "thinkingConfig": {"thinkingLevel": thinking_lvl},
                        "responseMimeType": "application/json",
                        "maxOutputTokens": 8192,
                    },
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(req_body, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                t0 = time.perf_counter()
                try:
                    with urllib.request.urlopen(req, timeout=180) as resp:
                        raw = json.loads(resp.read().decode("utf-8", errors="replace"))
                    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                    cands = raw.get("candidates") or [{}]
                    parts = ((cands[0].get("content") or {}).get("parts")) or []
                    txt = "".join(p.get("text", "") for p in parts if not p.get("thought") and p.get("text")).strip()
                    out = {
                        "model": model_name,
                        "thinking_level": thinking_lvl,
                        "key_slot": f"key_{k_idx}",
                        "elapsed_ms": elapsed_ms,
                        "usage": raw.get("usageMetadata") or {},
                        "text": txt,
                        "from_local_cache": False,
                    }
                    _cache_put(cache_key, out)
                    return out
                except urllib.error.HTTPError as exc:
                    err_txt = exc.read().decode("utf-8", errors="replace")[:260]
                    msg = f"HTTP {exc.code} on {model_name} ({thinking_lvl}, key_{k_idx}, try={retry + 1}): {err_txt}"
                    attempts_log.append(msg)
                    print("[gemini-audit]", msg.replace("\n", " "), flush=True)
                    last_err = msg
                    if exc.code == 400 and "thinkingLevel" in err_txt and thinking_lvl == "XHIGH":
                        try:
                            req_body["generationConfig"]["thinkingConfig"]["thinkingLevel"] = "HIGH"
                            req2 = urllib.request.Request(
                                url,
                                data=json.dumps(req_body, ensure_ascii=False).encode("utf-8"),
                                headers={"Content-Type": "application/json"},
                                method="POST",
                            )
                            with urllib.request.urlopen(req2, timeout=180) as resp2:
                                raw2 = json.loads(resp2.read().decode("utf-8", errors="replace"))
                            parts2 = (((raw2.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
                            txt2 = "".join(p.get("text", "") for p in parts2 if not p.get("thought") and p.get("text")).strip()
                            out2 = {
                                "model": model_name,
                                "thinking_level": "HIGH",
                                "key_slot": f"key_{k_idx}",
                                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                                "usage": raw2.get("usageMetadata") or {},
                                "text": txt2,
                                "from_local_cache": False,
                            }
                            _cache_put(cache_key, out2)
                            return out2
                        except Exception as exc2:
                            last_err = repr(exc2)
                            attempts_log.append(f"fallback-HIGH err on {model_name} (key_{k_idx}): {exc2!r}")
                            break
                    if exc.code in (503, 500) and retry == 0:
                        time.sleep(5)
                        continue
                    break
                except Exception as exc:
                    msg = f"ERR on {model_name} (key_{k_idx}, try={retry + 1}): {exc!r}"
                    attempts_log.append(msg)
                    print("[gemini-audit]", msg, flush=True)
                    last_err = msg
                    break
    raise RuntimeError(f"All Gemini API attempts failed ({len(attempts_log)} tries): {last_err}")


def _clip(s: str | None, limit: int = 420) -> str:
    txt = (s or "").strip().replace("\n", " ")
    return txt if len(txt) <= limit else (txt[:limit] + " ...[clipped]")


def build_audit_dossier(phase4_dir: Path, phase5_dir: Path) -> dict:
    # 1. Phase 4 synth-240 cross-pass discordant / failure cases
    p4_by_p: dict[int, dict[str, dict]] = {}
    for p in (1, 2, 3, 5):
        cand = phase4_dir / f"synth240_loop{p}.jsonl"
        if cand.is_file():
            p4_by_p[p] = {r["question_id"]: r for r in load_rows(cand)}

    p4_cases = []
    if len(p4_by_p) == 4:
        all_qids = sorted(set.intersection(*(set(m) for m in p4_by_p.values())))
        for qid in all_qids:
            outcomes = {str(p): p4_by_p[p][qid]["outcome"] for p in (1, 2, 3, 5)}
            if all(v == "correct" for v in outcomes.values()):
                continue
            r1 = p4_by_p[1][qid]
            per_pass = {}
            for p in (1, 2, 3, 5):
                rp = p4_by_p[p][qid]
                per_pass[f"loop{p}"] = {
                    "outcome": rp["outcome"],
                    "pred": rp.get("predicted"),
                    "r_tok": rp.get("reasoning_tokens"),
                    "a_tok": rp.get("answer_tokens"),
                    "answer_snippet": _clip(rp.get("text"), 180),
                    "cot_tail": _clip((rp.get("reasoning_text") or "")[-360:], 360),
                }
            p4_cases.append({
                "question_id": qid,
                "template_id": r1.get("template_id"),
                "difficulty": r1.get("difficulty"),
                "task_type": r1.get("task_type"),
                "expected": r1.get("expected"),
                "passes": per_pass,
            })

    # 2. Phase 5 effort48 key contrasts: low_loop3 vs low_loop1 / none_loop1 / high_loop3
    p5_cells = {}
    for name in ("none_loop1", "low_loop1", "low_loop3", "medium_loop3", "high_loop3", "high_loop5"):
        cand = phase5_dir / f"effort48_{name}.jsonl"
        if cand.is_file():
            p5_cells[name] = {r["question_id"]: r for r in load_rows(cand)}

    p5_cases = []
    if p5_cells:
        qids5 = sorted(set.intersection(*(set(m) for m in p5_cells.values())))
        for qid in qids5:
            outs = {k: p5_cells[k][qid]["outcome"] for k in p5_cells}
            if all(v == "correct" for v in outs.values()):
                continue
            base = next(iter(p5_cells.values()))[qid]
            p5_cases.append({
                "question_id": qid,
                "template_id": base.get("template_id"),
                "difficulty": base.get("difficulty"),
                "task_type": base.get("task_type"),
                "expected": base.get("expected"),
                "cells": {
                    k: {
                        "outcome": p5_cells[k][qid]["outcome"],
                        "pred": p5_cells[k][qid].get("predicted"),
                        "r_tok": p5_cells[k][qid].get("reasoning_tokens"),
                        "answer_snippet": _clip(p5_cells[k][qid].get("text"), 140),
                    }
                    for k in p5_cells
                },
            })

    return {
        "phase4_non_unanimous_count": len(p4_cases),
        "phase4_cases": p4_cases[:40],
        "phase5_non_unanimous_count": len(p5_cases),
        "phase5_cases": p5_cases[:25],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase4-dir", default=str(ROOT / "artifacts/phase4"))
    ap.add_argument("--phase5-dir", default=str(ROOT / "artifacts/phase5"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/phase6/gemini_crosspass_audit.json"))
    ap.add_argument("--keys-file", default=None)
    args = ap.parse_args()

    keys = load_studio_keys(args.keys_file)
    dossier = build_audit_dossier(Path(args.phase4_dir), Path(args.phase5_dir))

    prompt = (
        "You are an expert multimodal LLM evaluation & inference-systems researcher auditing "
        "`thinkingmachines/TML-d24-varloop-research` across Phase 4 (`synth-240`, loop=1/2/3/5 at effort=medium) "
        "and Phase 5 (`effort48`, effort=none/low/medium/high × loop=1/3/5).\n\n"
        "Analyze the structured non-unanimous failure/repair dossier below and return ONLY valid JSON with schema:\n"
        "{\n"
        '  "phase4_taxonomy_counts": {"target_swap_question_mark": <int>, "reasoning_loop_truncation": <int>, "visual_perception_slip": <int>, "arithmetic_or_deduction_slip": <int>},\n'
        '  "phase4_cross_pass_mechanisms": ["<concise mechanism 1>", "<mechanism 2>", "<mechanism 3>"],\n'
        '  "phase5_low_loop3_synergy_explanation": "<concise explanation of why low+loop3 outperforms high+loop3 and none+loop1>",\n'
        '  "exemplar_case_studies": [\n'
        '    {"question_id": "<id>", "pattern": "deep_loop_repair | overthinking_regression | high_effort_truncation", "evidence": "<short quantitative evidence>"}\n'
        '  ],\n'
        '  "actionable_serving_takeaways": ["<takeaway 1>", "<takeaway 2>", "<takeaway 3>"]\n'
        "}\n\n"
        "DOSSIER:\n"
        + json.dumps(dossier, ensure_ascii=False)
    )

    resp = _call_gemini(
        prompt,
        keys=keys,
        model_candidates=[
            ("gemini-4-hydrogen", "XHIGH"),
            ("gemini-3.8-flash", "HIGH"),
        ],
    )
    raw_txt = resp["text"].strip()
    if raw_txt.startswith("```"):
        lines = raw_txt.splitlines()
        raw_txt = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    parsed = json.loads(raw_txt)

    final_payload = {
        "audit_provider": "Google AI Studio Gemini API (cached SQLite WAL)",
        "model": resp["model"],
        "thinking_level": resp["thinking_level"],
        "key_slot": resp["key_slot"],
        "from_local_cache": resp["from_local_cache"],
        "elapsed_ms": resp["elapsed_ms"],
        "usage": resp["usage"],
        "dossier_counts": {
            "phase4_non_unanimous": dossier["phase4_non_unanimous_count"],
            "phase5_non_unanimous": dossier["phase5_non_unanimous_count"],
        },
        "audit": parsed,
    }
    atomic_write_text(Path(args.out), json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n")
    print("GEMINI_CROSSPASS_AUDIT_OK:", json.dumps({
        "model": resp["model"],
        "thinking_level": resp["thinking_level"],
        "from_local_cache": resp["from_local_cache"],
        "phase4_taxonomy": parsed.get("phase4_taxonomy_counts"),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
