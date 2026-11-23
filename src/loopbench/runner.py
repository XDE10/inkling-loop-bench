import hashlib, json, time
from pathlib import Path
from .backends.mock import MockBackend
from .grading import grade
from .storage import append_jsonl,load_completed

OPTIONAL={"reasoning_tokens":None,"answer_tokens":None,"visual_tokens":None,"image_grid_thw":None,
          "ttft_ms":None,"tpot_ms":None,"e2e_ms":None,"peak_vram_mb":None,
          "speculative_decoding":None,"loop_share_cache":None}
def run(config,dataset_dir,output_path):
    data=json.loads((Path(dataset_dir)/"questions.json").read_text(encoding="utf-8"))
    config_id=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    done=load_completed(output_path);backend=MockBackend();count=0
    for q in data["questions"]:
        key=(config_id,q["question_id"])
        if key in done: continue
        start=time.perf_counter()
        try:
            response=backend.generate(q,config)
            scored=grade(response["text"],q["answer"])
            row={**OPTIONAL,"config_id":config_id,"question_id":q["question_id"],
                 "image_id":q["image_id"],"modality":q["modality"],"task_type":q["task_type"],
                 "expected":q["answer"],**response,**scored,"error":None}
        except Exception as e:
            row={**OPTIONAL,"config_id":config_id,"question_id":q["question_id"],
                 "image_id":q["image_id"],"modality":q["modality"],"task_type":q["task_type"],
                 "expected":q["answer"],"text":"","finish_reason":"error","correct":False,
                 "parse_status":"backend_error","error":repr(e),"is_mock":True}
        row["wall_ms"]=(time.perf_counter()-start)*1000
        append_jsonl(output_path,row);count+=1
    return {"config_id":config_id,"new_rows":count,"total_questions":len(data["questions"])}
