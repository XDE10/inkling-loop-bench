import json
from loopbench.data.generate import generate_dataset
from loopbench.runner import run
from loopbench.storage import load_completed,append_jsonl
def test_resume(tmp_path):
    d=tmp_path/"data";generate_dataset(d,seed=7)
    cfg={"model_id":"mock","model_revision":"v1","backend":"mock","loop_num_passes":3,
         "reasoning_effort":"medium","temperature":0.0,"top_p":1.0,"repetition_penalty":1.0,
         "max_new_tokens":128,"stop":[],"image_size":[768,768],"color_mode":"RGB",
         "render_version":"test","concurrency":1,"seed":7}
    out=tmp_path/"results.jsonl"
    assert run(cfg,d,out)["new_rows"]==48
    assert run(cfg,d,out)["new_rows"]==0
    assert len(out.read_text().splitlines())==48
def test_trailing_partial_line_recovery(tmp_path):
    p=tmp_path/"r.jsonl"
    append_jsonl(p,{"config_id":"c","question_id":"q1"})
    with open(p,"ab") as f:f.write(b'{"config_id":"c","question_id":')
    assert load_completed(p)=={("c","q1")}
    assert p.read_bytes().endswith(b"\n")
