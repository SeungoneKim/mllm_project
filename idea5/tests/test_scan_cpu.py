"""CPU test of scan.py's scoring path with random embeddings in place of ColQwen (no model, no GPU).

  IDEA5_DEVICE=cpu IDEA5_OUT=tests/out_SYNTHETIC_SCAN python tests/test_scan_cpu.py

1. maxsim_all matches idea4/retrieve.py's per-page loop, and page_sim matches a direct double loop.
2. scan.score() runs end to end on random page embeddings for the documents on disk (the model and the text
   encoder are replaced by random vectors), then attribute.py runs on its output.
Values are random, not model outputs. Refuses to write anywhere whose path does not contain SYNTHETIC.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scan  # noqa: E402
from common import CACHE, OUT, SCAN, doc_stem, eligible, emb_path, load_qa  # noqa: E402

assert scan.DEVICE == "cpu", "set IDEA5_DEVICE=cpu"
if "SYNTHETIC" not in str(OUT) or "SYNTHETIC" not in str(CACHE):
    raise SystemExit("refusing: IDEA5_OUT and IDEA5_CACHE must both contain SYNTHETIC")
g = torch.Generator().manual_seed(0)


def unit(n, d=128):
    x = torch.randn(n, d, generator=g)
    return (x / x.norm(dim=1, keepdim=True)).to(torch.float16)


# 1. the math
pages = [unit(int(t)) for t in torch.randint(40, 90, (7,), generator=g)]
q = unit(12)
P, mask = scan.pad(pages)
ours = scan.maxsim_all(q, P, mask)
andre = [float((q.float() @ p.float().T).max(dim=1).values.sum()) for p in pages]
assert np.allclose(ours, andre, atol=1e-4), (ours, andre)
sim = scan.page_sim(P, mask, chunk=3)
direct = np.array([[float((a.float() @ b.float().T).max(dim=1).values.mean()) for b in pages] for a in pages])
assert np.allclose(sim, (direct + direct.T) / 2, atol=1e-5)
print("maxsim_all and page_sim match the direct loops")

# 2. score() end to end
qa = eligible(load_qa())
for d, n in qa.groupby("doc_id")["n_pages"].first().items():
    path = emb_path(d)
    if "SYNTHETIC" not in str(path):
        raise SystemExit(f"refusing to write {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save([unit(int(t)) for t in torch.randint(60, 120, (int(n),), generator=g)], path)
scan.load_model = lambda: (None, None)
scan.embed_texts = lambda model, proc, texts: [unit(10 + len(t) % 7) for t in texts]
if (SCAN / "scores.jsonl").exists():
    (SCAN / "scores.jsonl").unlink()
scan.score(0)
scan.score(0)  # resumable: the second call must add nothing
rows = [json.loads(l) for l in (SCAN / "scores.jsonl").open()]
assert len(rows) == len(qa) and len({r["qid"] for r in rows}) == len(qa), (len(rows), len(qa))
for d, n in qa.groupby("doc_id")["n_pages"].first().items():
    s = np.load(SCAN / "pagesim" / f"{doc_stem(d)}.npy")
    assert s.shape == (n, n) and np.allclose(s, s.T)
print(f"score(): {len(rows)} questions, {qa['doc_id'].nunique()} page-similarity matrices")

import attribute  # noqa: E402

attribute.main()
