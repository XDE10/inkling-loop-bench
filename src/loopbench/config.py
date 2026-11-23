import yaml
REQUIRED=("model_id","model_revision","backend","loop_num_passes","reasoning_effort",
          "temperature","top_p","repetition_penalty","max_new_tokens","stop",
          "image_size","color_mode","render_version","concurrency","seed")
def load_config(path):
    data=yaml.safe_load(open(path,encoding="utf-8"))
    missing=[k for k in REQUIRED if k not in data]
    if missing: raise ValueError("Missing explicit config fields: "+", ".join(missing))
    if data["loop_num_passes"] not in (1,2,3,5): raise ValueError("loop_num_passes must be 1,2,3,5")
    if data["backend"] not in ("mock","transformers","vllm","sglang"): raise ValueError("Unsupported backend")
    if data["max_new_tokens"]<=0 or data["concurrency"]<=0: raise ValueError("max_new_tokens and concurrency must be positive")
    if data["image_size"] != [768,768]: raise ValueError("Phase 1 requires fixed 768x768 images")
    if not isinstance(data["stop"],list): raise ValueError("stop must be an explicit list")
    return data
