import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from loopbench.config import load_config
from loopbench.data.generate import generate_dataset
from loopbench.data.validate import validate_dataset
from loopbench.runner import run
from loopbench.report import build_report

p=argparse.ArgumentParser()
p.add_argument("--config",default=str(ROOT/"configs/mock.yaml"))
p.add_argument("--out",default=str(ROOT/"artifacts/mock"))
p.add_argument("--seed",type=int,default=20261123)
args=p.parse_args()
out=Path(args.out);dataset=out/"dataset";dataset.mkdir(parents=True,exist_ok=True)
generate_dataset(dataset,args.seed);print("DATA:",validate_dataset(dataset))
cfg=load_config(args.config)
result=run(cfg,dataset,out/"results.jsonl");print("RUN:",result)
print("REPORT:",build_report(out/"results.jsonl",out/"report"))
