"""Paths and helpers for the Idea 5 midterm baseline.

Data loading, page rendering and the answer scorer are reused unchanged from ../analysis (the second assignment), so
the midterm numbers are computed exactly as the analysis numbers were. Paths can be overridden with environment
variables (see README.md).
"""
import importlib.util
import os
import sys
from pathlib import Path

MIDTERM = Path(__file__).resolve().parent
ANALYSIS = MIDTERM.parent / "analysis"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_common = _load("idea5_analysis_common", "common.py")
_score = _load("idea5_analysis_score", "score.py")

load_qa, eligible = _common.load_qa, _common.eligible
render, text_layer, fit, n_words = _common.render, _common.text_layer, _common.fit, _common.n_words
DOC_DIR, TYPES, SEED = _common.DOC_DIR, _common.TYPES, _common.SEED
PATCH, TOKENS_HIGH, MIN_TEXT_WORDS = _common.PATCH, _common.TOKENS_HIGH, _common.MIN_TEXT_WORDS
score = _score.score

OUT = Path(os.environ.get("IDEA5_MT_OUT", MIDTERM / "out"))
CACHE = Path(os.environ.get("IDEA5_MT_CACHE", MIDTERM / "cache"))
TEST = OUT / "test"            # items.jsonl, tokinfo.npz
PAGES = CACHE / "pages"        # page images at the full budget (1280 visual tokens)

BUDGETS = (64, 128, 256)       # visual tokens per page kept by every selector
INK_GRAY = 200                 # a pixel darker than this (0-255 grayscale) counts as ink
