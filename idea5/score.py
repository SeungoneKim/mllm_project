"""Answer scoring, following the rule-based part of MMLongBench-Doc's eval_score.py.

The benchmark first asks GPT-4o to extract a short answer from a free-form response. Here the reader is told to give
only the short answer in the expected format, and a light rule-based extraction (first number for Int and Float, a
Python list for List, and for strings: an answer that contains the reference as whole words counts as exact) takes
the place of the GPT-4o step. Scores are in [0, 1] (ANLS for strings); a question counts as correct when its score is
above 0.
"""
import ast
import math
import re


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(gt, pred, threshold=0.5):
    length = max(len(gt), len(pred))
    value = 1.0 - (levenshtein(gt, pred) / length if length else 0.0)
    return 0.0 if value <= threshold else value


def clean(s):
    s = str(s).lower().strip()
    for suffix in ["miles", "mile", "million"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    s = re.sub(r"\s*\([^)]*\)", "", s).strip()
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.lstrip("$").strip().rstrip("%").strip()
    return s


def is_exact_match(s):
    """Strings the benchmark compares exactly (URLs, files, page refs, phone numbers, times, dates, emails)."""
    return ("https://" in s or s.endswith(".py") or s.endswith("ipynb") or s.startswith("page")
            or re.fullmatch(r"\b\d+(-\d+|\s\d+)?\b", s) is not None or "a.m." in s or "p.m." in s
            or re.fullmatch(r"\b\d{4}[-\s]\d{2}[-\s]\d{2}\b", s) is not None
            or re.fullmatch(r"\b\d{4}[-\s]\d{2}\b", s) is not None
            or re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", s) is not None)


def is_float_equal(ref, pred):
    def precision(x):
        return len(str(x).split(".")[-1]) if "." in str(x) else 3
    for item in [ref / 100, ref, ref * 100]:  # include_percentage=True, as in the benchmark
        if math.isclose(item, pred, rel_tol=0.01):
            return True
        p = max(min(precision(pred), precision(item)), 2)
        if round(pred, p) == round(item, p):
            return True
    return False


def first_number(s):
    m = re.search(r"-?\d+(?:\.\d+)?", str(s).replace(",", ""))
    return float(m.group()) if m else None


def as_list(s):
    if isinstance(s, list):
        return s
    s = str(s).strip()
    if s.startswith("["):
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else [v]
        except (ValueError, SyntaxError):
            s = s.strip("[]")
    return [x.strip() for x in s.split(",") if x.strip()]


def score(gt, pred, fmt):
    pred = (pred or "").strip()
    if fmt == "Int":
        p = first_number(pred)
        try:
            return float(p is not None and int(float(gt)) == int(p))
        except ValueError:
            return 0.0
    if fmt == "Float":
        p = first_number(pred)
        try:
            return float(p is not None and is_float_equal(float(clean(gt)), p))
        except ValueError:
            return 0.0
    if fmt in ("Str", "None"):
        g, p = clean(gt), clean(pred)
        # Stand-in for the benchmark's GPT-4o answer extraction: "Ages 18-34" contains the answer "18-34".
        if g and re.search(rf"(?<!\w){re.escape(g)}(?!\w)", p):
            return 1.0
        return float(g == p) if is_exact_match(g) else anls(g, p)
    # List
    g = sorted(clean(x) for x in as_list(gt))
    p = sorted(clean(x) for x in as_list(pred))
    if len(g) != len(p) or not g:
        return 0.0
    if first_number(g[0]) is not None and re.fullmatch(r"-?[\d.]+", g[0]) or is_exact_match(g[0]):
        return float("-".join(g) == "-".join(p))
    return min(anls(a, b) for a, b in zip(g, p))


if __name__ == "__main__":
    assert score("6", "6", "Int") == 1.0 and score("6", "There are 6 charts.", "Int") == 1.0
    assert score("Poor Financial Condition", "Poor financial condition", "Str") == 1.0
    assert score("25.5%", "25.5", "Float") == 1.0 and score("0.255", "25.5%", "Float") == 1.0
    assert score("['a', 'b']", "['B', 'A']", "List") == 1.0 and score("['a', 'b']", "['a']", "List") == 0.0
    assert score("Apple", "Banana", "Str") == 0.0
    assert score("18-34", "Ages 18-34", "Str") == 1.0 and score("25mm Sheeting", "25mm Sheeting is appropriate.", "Str") == 1.0
    assert score("cat", "category", "Str") == 0.0
    print("score.py checks passed")
