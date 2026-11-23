import json, os
from pathlib import Path
def _repair_trailing_partial(path):
    p=Path(path)
    if not p.exists(): return
    with open(p,"rb+") as f:
        raw=f.read()
        if raw and not raw.endswith(b"\n"):
            cut=raw.rfind(b"\n")+1
            f.seek(cut);f.truncate()
def load_completed(path):
    p=Path(path)
    if not p.exists(): return set()
    _repair_trailing_partial(p)
    done=set()
    for number,line in enumerate(p.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        try: row=json.loads(line)
        except json.JSONDecodeError as e: raise ValueError(f"Corrupt JSONL at line {number}: {e}") from e
        done.add((row["config_id"],row["question_id"]))
    return done
def append_jsonl(path,row):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(row,ensure_ascii=False)+"\n").encode("utf-8")
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o666)
    try:
        view=memoryview(payload)
        while view:
            written=os.write(fd,view);view=view[written:]
        os.fsync(fd)
    finally: os.close(fd)
