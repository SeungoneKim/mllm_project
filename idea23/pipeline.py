"""The whole pipeline of one version (versions.py), as SLURM jobs with dependencies.

  python pipeline.py --version v1 --plan        # the steps of v1 and which of them are done
  python pipeline.py --version v1 --submit      # submit what is not done; the CPU steps run after their inputs
  python pipeline.py --version v2 --local select   # run one CPU step in this shell
  python pipeline.py --version v2 --submit --steps aggregate,units --partition l3lab

Steps whose outputs are complete are skipped (every model step is resumable), except selection and analysis, which
are cheap and always rerun. The shared steps (decomposition, embeddings, segmentation, need forms) are run once and
used by every version; selections, metrics and analysis go to out/<version>/, answers to the shared out/reader/.
"""
import argparse
import os
import shlex
import subprocess

from common import BAD_DOCS, CACHE, ELEMENTS, OUT, doc_stem, load_qa, read_jsonl, test_questions
from versions import VERSIONS

COLQWEN = "vidore/colqwen2.5-v0.2 vidore/colqwen2.5-base"
READER_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
SHARDS = {"segment": 4, "embed_elements": 4, "aggregate": 8, "reader": 6}


def docs():
    return sorted(set(test_questions(load_qa()).doc_id) - BAD_DOCS)


def all_files(folder, suffix):
    return lambda v: all((folder / f"{doc_stem(d)}{suffix}").exists() for d in docs())


def agg_model(version):
    return os.environ.get("AGG_MODEL") or VERSIONS[version].get("agg_model", "")


def aggregated(version):
    agg = OUT / version / "aggregate" / agg_model(version).split("/")[-1]
    sets = VERSIONS[version].get("agg_sets", "multi,control").split(",")
    want = {r["qid"] for r in read_jsonl(OUT / version / "candidates.jsonl") if r["set"] in sets}
    got = {r["qid"] for p in agg.glob("facts_*.jsonl") for r in read_jsonl(p)}
    return bool(want) and want <= got


def shard(name, cmd):
    n = SHARDS[name]
    return f"{cmd} --shard $SLURM_ARRAY_TASK_ID --nshards {n}", n


# name: (gpu, command, models, inputs, done(version)); a command may be (command, array size)
STEPS = {
    "decompose": (True, "$PY -u decompose.py", "Qwen/Qwen3-8B", [], lambda v: (OUT / "decomp.jsonl").exists()),
    "embed_pages": (True, "$PY -u embed.py pages", COLQWEN, [], all_files(CACHE / "colqwen", ".pt")),
    "score_pages": (True, "$PY -u embed.py score", COLQWEN, ["decompose", "embed_pages"],
                    lambda v: (OUT / "scores.pkl").exists()),
    "segment": (True, shard("segment", "$PY -u segment.py"),
                "PaddlePaddle/PP-DocLayoutV3_safetensors docling-project/DocumentFigureClassifier-v2.5", [],
                all_files(ELEMENTS, ".json")),
    "need_forms": (True, "$PY -u need_forms.py", "Qwen/Qwen3-8B", ["decompose"],
                   lambda v: (OUT / "need_forms.jsonl").exists()),
    "embed_elements": (True, ("$PY -u embed.py elements $SLURM_ARRAY_TASK_ID 4", 4), COLQWEN, ["segment"],
                       all_files(CACHE / "colqwen_elements", ".pt")),
    "score_elements": (True, "$PY -u embed.py score_elements", COLQWEN, ["decompose", "embed_elements"],
                       lambda v: (OUT / "element_scores.pkl").exists()),
    "candidates": (False, "$PY -u selection.py --candidates", "", ["score_pages", "need_forms", "score_elements"],
                   lambda v: (OUT / v / "candidates.jsonl").exists()),
    "aggregate": (True, shard("aggregate", "$PY -u aggregate.py"), None, ["candidates"], aggregated),
    "units": (True, "$PY -u units.py", "Qwen/Qwen3-Embedding-4B", ["aggregate"],
              lambda v: (OUT / v / "unit_scores.pkl").exists()),
    "select": (False, "$PY -u selection.py", "", None, lambda v: False),
    "reader": (True, shard("reader", "$PY -u reader.py"), READER_MODEL, ["select"], lambda v: False),
    "analyze": (False, "$PY -u analyze.py", "", ["reader"], lambda v: False),
}
SELECT_INPUTS = {"v0": ["score_pages"], "v1": ["score_pages", "segment", "need_forms"],
                 "v2": ["score_pages", "segment", "need_forms", "score_elements", "units"]}
SELECT_INPUTS["v2_32b"] = SELECT_INPUTS["v2_page"] = SELECT_INPUTS["v2_page_eq"] = SELECT_INPUTS["v2"]


def inputs(step, version):
    return SELECT_INPUTS[version] if step == "select" else STEPS[step][3]


def plan(version, only=None):
    """Steps of a version in dependency order."""
    order, seen = [], set()

    def visit(s):
        if s in seen:
            return
        seen.add(s)
        for d in inputs(s, version):
            visit(d)
        order.append(s)

    visit("analyze")
    return [s for s in order if only is None or s in only]


def command(step, version):
    gpu, cmd, models, _, _ = STEPS[step]
    cmd, n = cmd if isinstance(cmd, tuple) else (cmd, 1)
    if models is None:
        models = agg_model(version)
    if step == "reader":
        cmd += " --methods " + ",".join(VERSIONS[version]["reader"])
    return gpu, cmd, models, n


def submit(version, steps, partition, dry):
    ids = {}
    for s in steps:
        if STEPS[s][4](version):
            print(f"  {s:15s} done")
            continue
        gpu, cmd, models, n = command(s, version)
        deps = [ids[d] for d in inputs(s, version) if d in ids]
        args = ["sbatch", "--parsable", "-J", f"{version}_{s}" if s in ("select", "reader", "analyze") else s]
        if n > 1:
            args += [f"--array=0-{n - 1}"]
        if deps:
            args += [f"--dependency=afterok:{':'.join(deps)}", "--kill-on-invalid-dep=yes"]
        if partition == "l3lab":
            args += ["--partition=l3lab", "--qos=l3lab_qos"]
        if s == "aggregate" and "32B" in agg_model(version):  # 66 GB of weights: two 48 GB GPUs, or AGG_GRES=gpu:A100_80GB:1
            args += [f"--gres={os.environ.get('AGG_GRES', 'gpu:2')}", "--mem=128G"]
        if not gpu:  # general accepts no job without a GPU and preempt's CPU queue is days long: hold one briefly
            args += ["--cpus-per-task=4", "--mem=48G", "--time=02:00:00"]
        args += ["jobs/gpu.sbatch", "bash", "-c", cmd]
        env = {**os.environ, "IDEA23_VERSION": version, "MODELS": models}
        print(f"  {s:15s} {'(dry run) ' if dry else ''}{' '.join(shlex.quote(a) for a in args)}")
        if not dry:
            ids[s] = subprocess.run(args, env=env, check=True, capture_output=True, text=True).stdout.strip()
            print(f"  {'':15s} job {ids[s]}")
        else:
            ids[s] = f"<{s}>"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, choices=sorted(VERSIONS))
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry", action="store_true", help="with --submit: print the sbatch commands only")
    ap.add_argument("--local", default=None, help="run this CPU step here")
    ap.add_argument("--steps", default=None, help="comma-separated subset of the steps")
    ap.add_argument("--partition", default="general", choices=["general", "l3lab"])
    a = ap.parse_args()
    steps = plan(a.version, a.steps.split(",") if a.steps else None)
    print(f"version {a.version} ({VERSIONS[a.version]['name']}): {' -> '.join(steps)}")
    if a.local:
        gpu, cmd, _, _ = command(a.local, a.version)
        assert not gpu, f"{a.local} needs a GPU; use --submit"
        py = os.environ.get("PY", "python")
        subprocess.run(cmd.replace("$PY", py), shell=True, check=True, env={**os.environ, "IDEA23_VERSION": a.version})
    elif a.submit:
        submit(a.version, steps, a.partition, a.dry)
    else:
        for s in steps:
            print(f"  {s:15s} {'done' if STEPS[s][4](a.version) else 'to run'}")


if __name__ == "__main__":
    main()
