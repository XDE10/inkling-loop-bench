import json
import os
from pathlib import Path


def atomic_write_text(path, content: str, encoding: str = "utf-8") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / f".{p.name}.tmp.{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
    try:
        payload = content.encode(encoding)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, p)


def _is_valid_record_bytes(line_bytes: bytes) -> bool:
    if not line_bytes or b"\x00" in line_bytes:
        return False
    try:
        obj = json.loads(line_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("config_id"), str)
        and isinstance(obj.get("question_id"), str)
    )


def _repair_trailing_partial(path) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    repaired = False
    with open(p, "rb+") as f:
        raw = f.read()
        if not raw:
            return False
        if not raw.endswith(b"\n"):
            cut = raw.rfind(b"\n") + 1
            f.seek(cut)
            f.truncate()
            f.flush()
            os.fsync(f.fileno())
            raw = raw[:cut]
            repaired = True
        if not raw:
            return repaired
        # Check if the final terminated line is an interrupted/zero-filled tail record.
        body = raw[:-1]
        prev_nl = body.rfind(b"\n")
        last_line = body[prev_nl + 1 :]
        if last_line.strip() and not _is_valid_record_bytes(last_line):
            cut = prev_nl + 1
            f.seek(cut)
            f.truncate()
            f.flush()
            os.fsync(f.fileno())
            repaired = True
    return repaired


def load_rows(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    _repair_trailing_partial(p)
    rows = []
    raw_lines = p.read_bytes().splitlines()
    for number, line_bytes in enumerate(raw_lines, 1):
        if not line_bytes.strip():
            continue
        if b"\x00" in line_bytes:
            raise ValueError(f"Corrupt JSONL at line {number}: NUL byte detected")
        try:
            row = json.loads(line_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Corrupt JSONL at line {number}: {e}") from e
        if not isinstance(row, dict) or "config_id" not in row or "question_id" not in row:
            raise ValueError(f"Corrupt JSONL at line {number}: missing config_id or question_id")
        rows.append(row)
    return rows


def load_completed(path) -> set[tuple[str, str]]:
    return {(r["config_id"], r["question_id"]) for r in load_rows(path)}


def append_jsonl(path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
