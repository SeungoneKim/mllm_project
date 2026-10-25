"""The input conditions of the midterm experiment (no torch, so evaluate.py can import it). See reader.py."""
from shared import BUDGETS

INPUT_METHODS = ("rand", "ink", "comp")          # chosen before the LLM (token_select.py)
MODEL_METHODS = ("fastv", "svlm", "svlmnr")      # chosen inside the LLM from its attention (reader.py)
SELECTORS = ("down",) + INPUT_METHODS + MODEL_METHODS
LABELS = {"text": "text layer only", "full": "text + full image", "full_imgfirst": "full, image first",
          "down": "downscaled page", "rand": "random tokens", "ink": "most ink",
          "comp": "ink not in text layer (ours, untrained)", "fastv": "FastV", "svlm": "SparseVLM",
          "svlmnr": "SparseVLM, no recycling"}


def conditions(budgets=BUDGETS):
    return (["text", "full", "full_imgfirst"] + [f"down_{k}" for k in budgets]
            + [f"{m}_{k}" for m in INPUT_METHODS + MODEL_METHODS for k in budgets])


def parse(cond):
    """-> (prompt variant, method or None, K or None). Variants: text, tf (text first), if (image first), down_K."""
    if cond == "text":
        return "text", None, None
    if cond == "full":
        return "tf", None, None
    if cond == "full_imgfirst":
        return "if", None, None
    method, k = cond.rsplit("_", 1)
    if method == "down":
        return f"down_{k}", None, int(k)
    return "tf", method, int(k)
