"""Multi-key OpenCode Go `gemini-4-argon` client with two-layer caching.

Caching layers:
1. Local persistent SQLite WAL cache (content-addressed by SHA-256 of canonical
   request + image bytes SHA-256) so repeated/resumed queries incur zero network
   calls and zero quota usage.
2. Upstream implicit prefix caching on `https://opencode.ai/inference/go/google/v1beta`:
   places a byte-identical >=4,200-token canonical specification prefix in
   `contents[0]` + `contents[1]` and pins each worker shard to a sticky
   `(api_key, x-opencode-session)` pair so subsequent calls within the TTL
   window hit `cachedContentTokenCount` (`cache_read` at 95% discount).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
import urllib.error
import urllib.request
import uuid

DEFAULT_BASE_URL = "https://opencode.ai/inference/go/google/v1beta"
DEFAULT_MODEL_ID = "gemini-4-argon"

# Deterministic >=6,500-token domain specification block placed in contents[0]
# so Google v1beta implicit prefix caching activates reliably (>4,096-token cached block).
CANONICAL_PREFIX_TEXT = (
    "LOOPBENCH MULTIMODAL VERIFICATION & AUDIT SPECIFICATION (FROZEN PREFIX v2)\n"
    "Role: Exact mathematical, geometric, and quantitative chart reasoning auditor.\n"
    "Output discipline: Follow user JSON schema strictly; verify every visual mark, "
    "axis gridline, legend series, tick mark, right-angle square, parallel arrow, "
    "and vertex label directly from pixels before computing numeric results.\n\n"
    + "\n".join(
        f"[INVARIANT-{i:03d}] For multimodal benchmark verification across chart and "
        f"Euclidean geometry pairs (direct lookup vs multi-step compositional deduction), "
        f"confirm pixel-level legibility, absence of standalone numeric leakage in question "
        f"stems, exact symbolic recomputation consistency, and unambiguous target binding "
        f"even when auxiliary unknown markers ('?') or multi-series distractors are present. "
        f"Cross-check every coordinate, bar height, line marker, stacked segment, signed delta, "
        f"and polygon interior/exterior angle sum before emitting final JSON fields."
        for i in range(1, 125)
    )
)
CANONICAL_PREFIX_ACK = "ACK_LOOPBENCH_SPEC_V1: Ready for deterministic multimodal verification."


def load_opencode_keys(keys_file: str | Path | None = None) -> list[str]:
    env_raw = os.environ.get("OPENCODE_GO_KEYS", "").strip()
    if env_raw:
        keys = [k.strip() for k in env_raw.replace(",", "\n").splitlines() if k.strip()]
        if keys:
            return keys

    candidates: list[Path] = []
    if keys_file:
        candidates.append(Path(keys_file))
    env_file = os.environ.get("OPENCODE_GO_KEYS_FILE", "").strip()
    if env_file:
        candidates.append(Path(env_file))
    candidates.extend([
        Path(r"D:\Users\Mr.Z\Desktop\openapi.txt"),
        Path("/mnt/d/Users/Mr.Z/Desktop/openapi.txt"),
    ])

    for path in candidates:
        if path.is_file():
            keys = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            if keys:
                return keys
    raise FileNotFoundError(
        "No OpenCode Go API keys found. Provide `keys_file` or set `OPENCODE_GO_KEYS_FILE`."
    )


def _default_cache_db_path() -> Path:
    env_p = os.environ.get("LOOPBENCH_ARGON_CACHE_DB", "").strip()
    if env_p:
        return Path(env_p)
    return Path.home() / ".cache" / "loopbench" / "argon_cache.sqlite3"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ArgonSQLiteCache:
    """Thread- and process-safe persistent SQLite response cache."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else _default_cache_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS argon_responses (
                        cache_key TEXT PRIMARY KEY,
                        model_id TEXT NOT NULL,
                        created_utc REAL NOT NULL,
                        response_json TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT response_json FROM argon_responses WHERE cache_key = ?",
                    (cache_key,),
                )
                row = cur.fetchone()
                return json.loads(row[0]) if row else None
            finally:
                conn.close()

    def put(self, cache_key: str, model_id: str, payload: dict[str, Any]) -> None:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO argon_responses
                    (cache_key, model_id, created_utc, response_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (cache_key, model_id, time.time(), serialized),
                )
                conn.commit()
            finally:
                conn.close()

    def count(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("SELECT COUNT(*) FROM argon_responses")
                row = cur.fetchone()
                return int(row[0]) if row else 0
            finally:
                conn.close()


class ArgonClient:
    """Sticky-session, multi-key `gemini-4-argon` client with disk & prefix caching."""

    def __init__(
        self,
        keys: list[str] | None = None,
        keys_file: str | Path | None = None,
        cache_db_path: str | Path | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model_id: str = DEFAULT_MODEL_ID,
        enable_prefix_cache: bool = True,
        session_namespace: str | None = None,
        urlopen_fn=None,
    ):
        self.keys = list(keys) if keys is not None else load_opencode_keys(keys_file)
        if not self.keys:
            raise ValueError("At least one OpenCode Go API key is required")
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.enable_prefix_cache = enable_prefix_cache
        self.cache = ArgonSQLiteCache(cache_db_path)
        self._urlopen = urlopen_fn or urllib.request.urlopen
        ns = session_namespace or uuid.uuid4().hex[:10]
        self._sessions = [f"ses_loopbench_{ns}_k{i}" for i in range(len(self.keys))]
        self._stats_lock = threading.Lock()
        self._rr_counter = 0
        self._cooldown_until: dict[int, float] = {}
        self._stats: dict[str, Any] = {
            "local_cache_hits": 0,
            "network_calls": 0,
            "prompt_tokens": 0,
            "cache_read_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "allowance_usd": 0.0,
            "list_cost_usd": 0.0,
            "per_key_calls": {f"key_{i}": 0 for i in range(len(self.keys))},
            "per_key_cache_read_tokens": {f"key_{i}": 0 for i in range(len(self.keys))},
        }

    def get_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            snap = json.loads(json.dumps(self._stats))
        prompt_t = snap["prompt_tokens"]
        cache_t = snap["cache_read_tokens"]
        snap["upstream_cache_hit_ratio"] = round(cache_t / prompt_t, 4) if prompt_t > 0 else 0.0
        snap["sqlite_cached_entries"] = self.cache.count()
        return snap

    def _build_user_parts(
        self,
        items: list[str | Path | tuple[str, bytes] | dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        wire_parts: list[dict[str, Any]] = []
        hash_parts: list[dict[str, Any]] = []

        for item in items:
            if isinstance(item, str):
                wire_parts.append({"text": item})
                hash_parts.append({"text": item})
            elif isinstance(item, Path):
                raw = item.read_bytes()
                mime = "image/png" if item.suffix.lower() == ".png" else "image/jpeg"
                wire_parts.append({
                    "inlineData": {
                        "mimeType": mime,
                        "data": base64.b64encode(raw).decode("ascii"),
                    }
                })
                hash_parts.append({"image_sha256": _sha256_bytes(raw), "mime": mime})
            elif isinstance(item, tuple) and len(item) == 2:
                mime, raw = item
                wire_parts.append({
                    "inlineData": {
                        "mimeType": mime,
                        "data": base64.b64encode(raw).decode("ascii"),
                    }
                })
                hash_parts.append({"image_sha256": _sha256_bytes(raw), "mime": mime})
            elif isinstance(item, dict):
                wire_parts.append(item)
                hash_parts.append(item)
            else:
                raise TypeError(f"Unsupported content part: {type(item)!r}")

        return wire_parts, hash_parts

    def generate(
        self,
        parts: list[str | Path | tuple[str, bytes] | dict[str, Any]],
        *,
        thinking_level: str = "LOW",
        max_output_tokens: int = 4096,
        response_mime_type: str | None = "application/json",
        shard_id: int | None = None,
        use_prefix_cache: bool | None = None,
        bypass_local_cache: bool = False,
        timeout: int = 90,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        wire_user_parts, hash_user_parts = self._build_user_parts(parts)
        use_prefix = self.enable_prefix_cache if use_prefix_cache is None else bool(use_prefix_cache)

        gen_cfg: dict[str, Any] = {
            "thinkingConfig": {"thinkingLevel": thinking_level.upper()},
            "maxOutputTokens": int(max_output_tokens),
        }
        if response_mime_type:
            gen_cfg["responseMimeType"] = response_mime_type

        contents: list[dict[str, Any]] = []
        if use_prefix:
            contents.append({"role": "user", "parts": [{"text": CANONICAL_PREFIX_TEXT}]})
            contents.append({"role": "model", "parts": [{"text": CANONICAL_PREFIX_ACK}]})
        contents.append({"role": "user", "parts": wire_user_parts})

        canonical_key_obj = {
            "v": 1,
            "model_id": self.model_id,
            "prefix_enabled": use_prefix,
            "gen_cfg": gen_cfg,
            "user_parts": hash_user_parts,
        }
        cache_key = hashlib.sha256(
            json.dumps(canonical_key_obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

        if not bypass_local_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                with self._stats_lock:
                    self._stats["local_cache_hits"] += 1
                return {**cached, "from_local_cache": True, "cache_key": cache_key}

        if shard_id is None:
            with self._stats_lock:
                primary_idx = self._rr_counter % len(self.keys)
                self._rr_counter += 1
        else:
            primary_idx = int(shard_id) % len(self.keys)

        now = time.time()
        candidate_indices = [
            (primary_idx + offset) % len(self.keys)
            for offset in range(len(self.keys))
            if self._cooldown_until.get((primary_idx + offset) % len(self.keys), 0.0) <= now
        ]
        if not candidate_indices:
            candidate_indices = [(primary_idx + offset) % len(self.keys) for offset in range(len(self.keys))]

        eff_retries = max_retries if max_retries is not None else max(len(candidate_indices), 4)
        url = f"{self.base_url}/models/{self.model_id}:generateContent"
        body_bytes = json.dumps({"contents": contents, "generationConfig": gen_cfg}, ensure_ascii=False).encode("utf-8")

        last_err: Exception | None = None
        for attempt in range(eff_retries):
            key_idx = candidate_indices[attempt % len(candidate_indices)]
            api_key = self.keys[key_idx]
            session_id = self._sessions[key_idx]

            req = urllib.request.Request(
                url,
                data=body_bytes,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "x-opencode-session": session_id,
                    "User-Agent": "opencode/2.2.7",
                },
                method="POST",
            )
            t0 = time.perf_counter()
            try:
                with self._urlopen(req, timeout=timeout) as resp:
                    raw_resp = json.loads(resp.read().decode("utf-8", errors="replace"))
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                parsed = self._extract_response(raw_resp, elapsed_ms, key_idx)
                self.cache.put(cache_key, self.model_id, parsed)
                return {**parsed, "from_local_cache": False, "cache_key": cache_key}
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace")[:500]
                last_err = RuntimeError(f"HTTP {exc.code} on key_{key_idx}: {err_body}")
                if exc.code == 429 or "GoUsageLimitError" in err_body:
                    self._cooldown_until[key_idx] = time.time() + 600.0
                    continue
                if exc.code in (400, 404) and "rate" not in err_body.lower() and "quota" not in err_body.lower():
                    raise last_err from exc
            except Exception as exc:
                last_err = exc

            if attempt + 1 < eff_retries:
                time.sleep(1.2)

        raise RuntimeError(f"ArgonClient failed after {eff_retries} attempts: {last_err!r}")

    def _extract_response(self, raw_resp: dict[str, Any], elapsed_ms: float, key_idx: int) -> dict[str, Any]:
        cands = raw_resp.get("candidates") or [{}]
        cand0 = cands[0] if cands else {}
        parts = (cand0.get("content") or {}).get("parts") or []
        text_chunks = [p.get("text", "") for p in parts if not p.get("thought") and p.get("text")]
        thought_chunks = [p.get("text", "") for p in parts if p.get("thought") and p.get("text")]
        text = "".join(text_chunks).strip()
        finish_reason = cand0.get("finishReason", "STOP")

        usage = raw_resp.get("usageMetadata") or {}
        oc = usage.get("opencode") or {}
        oc_toks = oc.get("tokens") or {}

        prompt_toks = int(usage.get("promptTokenCount") or 0)
        cache_read = int(oc_toks.get("cache_read") or usage.get("cachedContentTokenCount") or 0)
        out_toks = int(oc_toks.get("output") or usage.get("candidatesTokenCount") or 0)
        reason_toks = int(oc_toks.get("reasoning") or usage.get("thoughtsTokenCount") or 0)
        allowance = float(oc.get("allowance") or 0.0)
        list_cost = float(oc.get("list_cost") or 0.0)

        with self._stats_lock:
            self._stats["network_calls"] += 1
            self._stats["prompt_tokens"] += prompt_toks
            self._stats["cache_read_tokens"] += cache_read
            self._stats["output_tokens"] += out_toks
            self._stats["reasoning_tokens"] += reason_toks
            self._stats["allowance_usd"] += allowance
            self._stats["list_cost_usd"] += list_cost
            self._stats["per_key_calls"][f"key_{key_idx}"] += 1
            self._stats["per_key_cache_read_tokens"][f"key_{key_idx}"] += cache_read

        return {
            "text": text,
            "thought_text": "".join(thought_chunks).strip(),
            "finish_reason": finish_reason,
            "elapsed_ms": round(elapsed_ms, 2),
            "key_index": key_idx,
            "usage": {
                "prompt_tokens": prompt_toks,
                "cache_read_tokens": cache_read,
                "output_tokens": out_toks,
                "reasoning_tokens": reason_toks,
                "allowance_usd": allowance,
                "list_cost_usd": list_cost,
            },
        }
