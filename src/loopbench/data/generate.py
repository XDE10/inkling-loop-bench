import hashlib, json, random, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .schema import Question

def _pixel_sha(path):
    from PIL import Image
    import numpy as np
    with Image.open(path) as im: arr=np.asarray(im.convert("RGB"))
    return hashlib.sha256(arr.tobytes()).hexdigest()

def generate_dataset(out_dir, seed=20261123, n_per_type=12):
    out=Path(out_dir); images=out/"images"; images.mkdir(parents=True,exist_ok=True)
    rng=random.Random(seed); rows=[]
    for modality in ("chart","geometry"):
        for i in range(n_per_type):
            image_id=f"{modality}-{i:03d}"; template=f"{modality}-v1-{i%6}"
            if modality=="chart":
                a=rng.randint(12,45); b=rng.randint(50,95); c=rng.randint(20,49)
                fig,ax=plt.subplots(figsize=(6,6),dpi=128)
                ax.bar(["North","South","West"],[a,b,c],color=["#4472C4","#ED7D31","#70AD47"])
                ax.set_ylim(0,110); ax.set_ylabel("Units"); ax.set_title("Regional values")
                ax.grid(axis="y",alpha=.25); fig.tight_layout()
                path=images/f"{image_id}.png"; fig.savefig(path,metadata={"Software":"LoopBench"}); plt.close(fig)
                direct=float(b); multi=float(a+c)
                p1="Which region has the highest value? Answer with its value in units."
                p2="What is the combined value of North and West? Answer with a number in units."
                common={"values":{"North":a,"South":b,"West":c}}
            else:
                alpha=rng.randint(35,65); gamma=rng.randint(25,55)
                beta=180-alpha-gamma
                # AB=1; law of sines determines C. Diagram geometry matches labels.
                length_ac=math.sin(math.radians(beta))/math.sin(math.radians(gamma))
                cx=length_ac*math.cos(math.radians(alpha)); cy=length_ac*math.sin(math.radians(alpha))
                fig,ax=plt.subplots(figsize=(6,6),dpi=128)
                ax.set_aspect("equal"); ax.axis("off")
                ax.plot([0,1,cx,0],[0,0,cy,0],linewidth=2)
                ax.text(-.08,-.08,"A",fontsize=13); ax.text(1.04,-.08,"B",fontsize=13); ax.text(cx+.04,cy+.02,"C",fontsize=13)
                ax.text(.22,-.18,f"{alpha}°",fontsize=14)
                ax.text(cx-.18,cy-.18,f"{gamma}°",fontsize=14)
                ax.set_xlim(-.35,max(1,cx)+.5); ax.set_ylim(-.4,cy+.5); fig.tight_layout()
                path=images/f"{image_id}.png"; fig.savefig(path,metadata={"Software":"LoopBench"}); plt.close(fig)
                direct=float(alpha); multi=float(beta)
                p1="What is the marked interior angle at vertex A, in degrees?"
                p2="Given the marked interior angles A and C, calculate the remaining interior angle B of the triangle."
                common={"angle_A":alpha,"angle_C":gamma,"angle_B":beta}
            pix=_pixel_sha(path)
            for task,question,answer in (("direct",p1,direct),("multi_step",p2,multi)):
                rows.append(Question(f"{image_id}-{task}",image_id,template,modality,task,question,answer,
                    "units" if modality=="chart" else "degrees",str(path),{"seed":seed,**common,"pixel_sha256":pix}))
    out.mkdir(parents=True,exist_ok=True)
    manifest={"schema_version":1,"seed":seed,"image_count":n_per_type*2,"question_count":len(rows),
              "questions":[q.to_dict() for q in rows]}
    (out/"questions.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest
