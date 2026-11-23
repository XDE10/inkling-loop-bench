import json
from loopbench.data.generate import generate_dataset
from loopbench.data.validate import validate_dataset
def test_smoke48_and_determinism(tmp_path):
    a=tmp_path/"a";b=tmp_path/"b"
    generate_dataset(a,seed=42);generate_dataset(b,seed=42)
    assert validate_dataset(a)["questions"]==48
    ja=json.loads((a/"questions.json").read_text());jb=json.loads((b/"questions.json").read_text())
    for x,y in zip(ja["questions"],jb["questions"]):
        assert {k:v for k,v in x.items() if k!="image_path"} == {k:v for k,v in y.items() if k!="image_path"}
    assert len({q["image_id"] for q in ja["questions"]})==24
