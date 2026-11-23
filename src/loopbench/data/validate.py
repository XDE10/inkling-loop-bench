import hashlib, json
from pathlib import Path
from PIL import Image
import numpy as np

def _pixel_sha(path):
    with Image.open(path) as im: arr=np.asarray(im.convert("RGB"))
    return hashlib.sha256(arr.tobytes()).hexdigest()

def validate_dataset(dataset_dir):
    root=Path(dataset_dir); data=json.loads((root/"questions.json").read_text(encoding="utf-8"))
    qs=data["questions"]; errors=[]
    if len(qs)!=48: errors.append(f"expected 48 questions, got {len(qs)}")
    groups={}
    for q in qs: groups.setdefault(q["image_id"],[]).append(q)
    if len(groups)!=24: errors.append(f"expected 24 images, got {len(groups)}")
    for image_id,items in groups.items():
        if len(items)!=2 or {x["task_type"] for x in items}!={"direct","multi_step"}:
            errors.append(f"{image_id}: expected one direct and one multi_step question")
    for q in qs:
        p=Path(q["image_path"])
        if not p.is_file(): errors.append(f"missing image: {p}"); continue
        try:
            with Image.open(p) as im:
                if im.size!=(768,768): errors.append(f"{p}: expected 768x768, got {im.size}")
                im.verify()
            if _pixel_sha(p)!=q["metadata"]["pixel_sha256"]: errors.append(f"{p}: pixel hash mismatch")
        except Exception as e: errors.append(f"invalid image {p}: {e}")
        if q["modality"]=="chart":
            v=q["metadata"]["values"]
            expected=v["South"] if q["task_type"]=="direct" else v["North"]+v["West"]
            if not (v["South"]>v["North"] and v["South"]>v["West"]): errors.append(f"{q['image_id']}: chart highest-value invariant failed")
        else:
            a=q["metadata"]["angle_A"]; c=q["metadata"]["angle_C"]
            expected=a if q["task_type"]=="direct" else 180-a-c
            if a<=0 or c<=0 or expected<=0: errors.append(f"{q['image_id']}: invalid triangle angles")
        if abs(float(q["answer"])-float(expected))>1e-9: errors.append(f"{q['question_id']}: answer recomputation mismatch")
    if errors: raise ValueError("\n".join(errors))
    return {"valid":True,"images":len(groups),"questions":len(qs),
            "modalities":{"chart":sum(q["modality"]=="chart" for q in qs),"geometry":sum(q["modality"]=="geometry" for q in qs)}}
