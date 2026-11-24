import io
import json
from pathlib import Path
from PIL import Image

from loopbench.argon_client import ArgonClient


class _FakeHTTPResp:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_argon_client_sqlite_and_prefix_cache_accounting(tmp_path):
    calls = []

    def _fake_urlopen(req, timeout=90):
        hdrs = {k.lower(): v for k, v in req.header_items()}
        body = json.loads(req.data.decode("utf-8"))
        calls.append({"headers": hdrs, "body": body})
        call_no = len(calls)
        cache_read = 4340 if call_no > 1 else 0
        return _FakeHTTPResp({
            "candidates": [
                {
                    "content": {"parts": [{"text": json.dumps({"call": call_no, "ok": True})}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5200,
                "cachedContentTokenCount": cache_read,
                "candidatesTokenCount": 14,
                "opencode": {
                    "list_cost": 0.002 if cache_read else 0.010,
                    "allowance": 0.004 if cache_read else 0.020,
                    "tokens": {
                        "input": 5200 - cache_read,
                        "output": 14,
                        "reasoning": 22,
                        "cache_read": cache_read,
                    },
                },
            },
        })

    img_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(img_path)
    db_path = tmp_path / "cache.sqlite3"

    client = ArgonClient(
        keys=["oc_sk_test_0", "oc_sk_test_1"],
        cache_db_path=db_path,
        session_namespace="unit",
        urlopen_fn=_fake_urlopen,
    )

    # 1. Cold call on shard 1
    r1 = client.generate([img_path, "Verify question 1"], shard_id=1)
    assert r1["from_local_cache"] is False
    assert json.loads(r1["text"])["ok"] is True
    assert len(calls) == 1
    assert calls[0]["headers"]["authorization"] == "Bearer oc_sk_test_1"
    assert calls[0]["headers"]["x-opencode-session"] == "ses_loopbench_unit_k1"
    # Verify canonical prefix was prepended in contents[0..1]
    assert len(calls[0]["body"]["contents"]) == 3

    # 2. Identical call -> hits SQLite local cache with ZERO network traffic!
    r1_repeat = client.generate([img_path, "Verify question 1"], shard_id=1)
    assert r1_repeat["from_local_cache"] is True
    assert r1_repeat["text"] == r1["text"]
    assert len(calls) == 1

    # 3. Different question on same shard -> hits sticky key + session and records upstream cache_read
    r2 = client.generate([img_path, "Verify question 2"], shard_id=1)
    assert r2["from_local_cache"] is False
    assert len(calls) == 2
    assert calls[1]["headers"]["x-opencode-session"] == "ses_loopbench_unit_k1"

    stats = client.get_stats()
    assert stats["local_cache_hits"] == 1
    assert stats["network_calls"] == 2
    assert stats["cache_read_tokens"] == 4340
    assert stats["sqlite_cached_entries"] == 2
    assert stats["per_key_calls"]["key_1"] == 2
