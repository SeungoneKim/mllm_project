"""Input-level selectors: which K visual tokens of a page the reader gets, decided before the first LLM layer.

The visual tokens of a page form a grid over 28 x 28 pixel blocks (row-major, the order Qwen2.5-VL feeds them to the
LLM). Each selector returns sorted indices into that grid. tokinfo comes from build_test.py:
  ink  share of the block's pixels that are drawn (darker than 200 gray)
  txt  share of the block covered by text-layer word boxes

  rand   K blocks uniformly at random (seeded per question and budget)
  ink    the K blocks with the most ink, ignoring the text layer
  comp   the K blocks with the most ink that the text layer does not cover: score = ink * (1 - txt).
         The untrained form of Idea 5's premise that the visual tokens should carry what the text lacks; it is the
         token-level version of the analysis's painted-out image. Ties are broken by ink.

The in-model selectors (FastV, SparseVLM) need the model's attention and live in reader.py.
"""
import numpy as np

from shared import SEED

def select_input(method, k, ink, txt, qid):
    ink, txt = np.asarray(ink, np.float32).ravel(), np.asarray(txt, np.float32).ravel()
    n = ink.size
    if k >= n:
        return np.arange(n)
    if method == "rand":
        return np.sort(np.random.default_rng([SEED, qid, k]).choice(n, size=k, replace=False))
    if method == "ink":
        score = ink
    elif method == "comp":
        score = ink * (1.0 - txt)
    else:
        raise ValueError(method)
    order = np.lexsort((np.arange(n), -ink, -score))  # score, then ink, then grid order (deterministic)
    return np.sort(order[:k])
