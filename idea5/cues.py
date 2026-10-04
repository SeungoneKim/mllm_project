"""Question-side cues used to attribute retrieval misses: page references and content words."""
import re

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
            "eighth": 8, "ninth": 9, "tenth": 10}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10}
UNIT = r"(?:page|pages|slide|slides|pg\.?|p\.)"
# "page 9", "pages 3 and 4", "slide 12", "p. 5", "page 0286"
RE_NUM = re.compile(rf"\b{UNIT}\s*(?:no\.?\s*|number\s*)?#?(\d{{1,4}})(?:\s*(?:and|&|,|-|to)\s*(\d{{1,4}}))?", re.I)
# "the second page", "first slide", "last page"
RE_ORD = re.compile(rf"\b({'|'.join(ORDINALS)}|last|final)\s+{UNIT}", re.I)
# "page two", "pages three and four"
_W = "|".join(NUMBER_WORDS)
RE_WORD = re.compile(rf"\b{UNIT}\s+({_W})\b(?:\s*(?:and|&|,|to)\s*({_W})\b)?", re.I)
# "Figure 3", "Table 2", "Fig. 4"
RE_ELEMENT = re.compile(r"\b(fig(?:ure)?\.?|table|chart|exhibit)\s*(\d{1,3})\b", re.I)


def page_refs(question, n_pages):
    """Page numbers the question cites, as written (printed labels or positions). 'last' maps to n_pages."""
    refs = []
    for m in RE_NUM.finditer(question):
        a = int(m.group(1))
        refs.append(a)
        if m.group(2):
            b = int(m.group(2))
            refs += list(range(a + 1, b + 1)) if 0 < b - a <= 10 else [b]
    for m in RE_ORD.finditer(question):
        w = m.group(1).lower()
        refs.append(n_pages if w in ("last", "final") else ORDINALS[w])
    for m in RE_WORD.finditer(question):
        refs += [NUMBER_WORDS[g.lower()] for g in m.groups() if g]
    return sorted(set(refs))


def element_refs(question):
    """Numbered elements the question cites, such as 'Figure 3'."""
    return [f"{m.group(1).lower().rstrip('.')[:3]} {int(m.group(2))}" for m in RE_ELEMENT.finditer(question)]


STOP = set("""a an the of in on at to for from by with and or but not no is are was were be been being do does did
has have had it its this that these those there their they them he she his her we our you your i me my what which who
whom whose when where why how many much more most less least than then as if into over under about after before
between during per each any all both either neither some such same other another only own so too very can could
should would will shall may might must also just according document report page pages slide slides shown show shows
mentioned mention mentions given give list answer answers format formats round decimal decimals places place
integer float string terms term percentage percent please write e.g eg i.e ie etc number numbers value values
total figure table chart image images picture section text based provide provided state states stated year years
following above below among within""".split())


def content_words(text):
    """Lower-cased word and number tokens of 3+ characters (numbers of any length), minus stop words."""
    toks = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", (text or "").lower())
    return [t for t in toks if (len(t) >= 3 or t.isdigit()) and t not in STOP]


def coverage(question_words, page_text):
    """Share of the question's distinct content words that occur in the page text."""
    q = set(question_words)
    if not q:
        return float("nan")
    page = set(content_words(page_text))
    return len(q & page) / len(q)
