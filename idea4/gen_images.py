"""Part B: generate the proposer's specification as an image with Qwen-Image-2.1.

  CUDA_VISIBLE_DEVICES=1 python gen_images.py --shard 0 --nshards 2   # renders/gen/<pid>.png, resumable
  python gen_images.py --specs proposer/human_specs.jsonl --out renders/gen_human   # André's step-2 specs

The prompt is built only from the specification and the document type. Fixed seed per item.
"""
import argparse
import json
import time
from pathlib import Path

from common import IDEA4

PDIR = IDEA4 / "proposer"
OUTD = IDEA4 / "renders" / "gen"
MODEL = "Qwen/Qwen-Image-2.1"
WIDTH, HEIGHT, STEPS, SEED = 1024, 768, 30, 777


def q(s):
    return f'"{s}"' if s else ""


def lst(v):
    return ", ".join(map(str, v))


def prompt_for(spec, doc_type):
    t = spec["element"]
    if t == "chart":
        p = (f"A clean {spec.get('subtype') or 'bar'} chart as printed in a {doc_type} PDF. "
             f"Title at the top: {q(spec.get('title'))}. ")
        if spec.get("x_label"):
            p += f"X-axis label: {q(spec['x_label'])}. "
        if spec.get("y_label"):
            p += f"Y-axis label: {q(spec['y_label'])}. "
        if spec.get("categories"):
            p += f"Category labels: {lst(spec['categories'])}. "
        if spec.get("series"):
            p += f"Legend: {lst(spec['series'])}. "
        if spec.get("values"):
            p += f"Data labels: {spec['values']}. "
        p += f"{spec.get('style', '')}. Flat 2D, white background, sharp legible text."
    elif t == "table":
        p = (f"A data table as printed in a {doc_type} PDF. Caption above the table: {q(spec.get('caption'))}. "
             f"Column headers: {lst(spec.get('columns', []))}. Row labels: {lst(spec.get('rows', []))}. "
             "Numbers in the cells, thin grid lines, black text on white, sharp legible text.")
    elif t == "text":
        p = (f"A page excerpt from a {doc_type} PDF. Heading: {q(spec.get('heading'))}. "
             f"Paragraph text: {q(spec.get('passage'))}. Black text on a white page, sharp legible text.")
    else:  # figure, layout
        p = f"{spec.get('description', '')} As it appears on a page of a {doc_type} PDF."
        if spec.get("caption"):
            p += f" Caption below: {q(spec['caption'])}."
    return " ".join(p.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--specs", default=str(PDIR / "specs.jsonl"))
    ap.add_argument("--out", default=str(OUTD))
    args = ap.parse_args()
    outd = Path(args.out)

    doc_type = {}
    for f in sorted(PDIR.glob("inputs_*.json")):
        for x in json.loads(f.read_text()):
            doc_type[x["id"]] = x["document_type"]
    specs = [json.loads(l) for l in open(args.specs)]
    outd.mkdir(parents=True, exist_ok=True)
    todo = [s for i, s in enumerate(specs) if i % args.nshards == args.shard and not (outd / f"{s['pid']}.png").exists()]
    print(f"{len(todo)} to generate", flush=True)
    if not todo:
        return

    import diffusers.utils.import_utils as _iu
    _iu._xformers_available = False
    import torch
    from diffusers import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")

    with (outd / f"prompts_shard{args.shard}.jsonl").open("a") as log:
        for s in todo:
            p = prompt_for(s["spec"], doc_type[s["pid"]])
            t0 = time.time()
            img = pipe(prompt=p, width=WIDTH, height=HEIGHT, num_inference_steps=STEPS, true_cfg_scale=1.0,
                       generator=torch.Generator("cuda").manual_seed(SEED)).images[0]
            img.save(outd / f"{s['pid']}.png")
            log.write(json.dumps({"pid": s["pid"], "prompt": p, "seconds": round(time.time() - t0, 1)}) + "\n")
            log.flush()
            print(s["pid"], f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
