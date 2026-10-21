"""LiteLLM, OpenAI-compatible"""
import os, json, base64, time, pathlib, hashlib, io, re

def _load_env():
    p = pathlib.Path.home() / ".env"
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
_load_env()

API_BASE = os.environ["API_BASE"].rstrip("/")
API_KEY = os.environ["LITELLM_API_KEY"]
CACHE = pathlib.Path(__file__).resolve().parent.parent / "cache" / "llm"
CACHE.mkdir(parents=True, exist_ok=True)

VLM = "Qwen3-VL-235B" #write model names here
STRONG_VLM = "Gemini-3.1-Pro" #write model names here

MAX_SIDE = 1500          # px; keeps request bodies under provider limits
MAX_BYTES = 3_500_000


def shrink(b, max_side=MAX_SIDE, max_bytes=MAX_BYTES):
    """Downscale/recompress so a page image fits the provider's body limit."""
    from PIL import Image
    im = Image.open(io.BytesIO(b))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    q = 88
    for _ in range(6):
        w, h = im.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=q, optimize=True)
        out = buf.getvalue()
        if len(out) <= max_bytes:
            return out, "image/jpeg"
        max_side = int(max_side * 0.8); q = max(55, q - 8)
    return out, "image/jpeg"


def b64img(path_or_bytes, mime="image/png"):
    b = path_or_bytes if isinstance(path_or_bytes, bytes) else pathlib.Path(path_or_bytes).read_bytes()
    try:
        b, mime = shrink(b)
    except Exception:
        pass
    return f"data:{mime};base64," + base64.b64encode(b).decode()

def chat(messages, model=VLM, temperature=0.0, max_tokens=4096, use_cache=True, retries=4):
    import requests
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:40]
    f = CACHE / f"{key}.json"
    if use_cache and f.exists():
        return json.loads(f.read_text())["content"]
    last = None
    for i in range(retries):
        try:
            r = requests.post(f"{API_BASE}/v1/chat/completions",
                              headers={"Authorization": f"Bearer {API_KEY}"},
                              json=payload, timeout=300)
            if r.status_code == 200:
                c = r.json()["choices"][0]["message"]["content"]
                f.write_text(json.dumps({"content": c}))
                return c
            last = f"{r.status_code}: {r.text[:400]}"
        except Exception as e:
            last = repr(e)
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"chat failed after {retries}: {last}")

def ask(prompt, images=(), **kw):
    """images: list of (bytes|path). Returns text."""
    content = [{"type": "text", "text": prompt}]
    for im in images:
        content.append({"type": "image_url", "image_url": {"url": b64img(im)}})
    return chat([{"role": "user", "content": content}], **kw)

def _extract_json(t):
    """Pull the first complete JSON object out of a model reply.

    Brace counting has to respect string state: model output routinely contains
    braces and quotes inside the values, and a naive first-brace/last-brace cut
    produced malformed slices on long replies.
    """
    t = t.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    i = t.find("{")
    if i < 0:
        raise ValueError("no JSON object in reply")
    depth, in_str, esc = 0, False, False
    for j in range(i, len(t)):
        c = t[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return t[i:j + 1]
    return t[i:]                      # truncated reply: let the repair pass try


def _loads(t):
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r",\s*([}\]])", r"\1", t)          # trailing commas
    fixed = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return json.loads(fixed.replace("\n", "\\n"))   # raw newlines in strings


def ask_json(prompt, images=(), **kw):
    note = ("\n\nRespond with ONLY a single valid JSON object. No prose, no markdown "
            "fences. Escape every newline and double quote inside string values.")
    t = ask(prompt + note, images, **kw)
    try:
        return _loads(_extract_json(t))
    except Exception:
        pass
    # one repair pass: hand the broken reply back and ask for valid JSON only
    fix = ask("The following was meant to be a single JSON object but does not parse. "
              "Return the same content as ONE valid JSON object, escaping newlines and "
              "quotes inside strings. Output only the JSON.\n\n" + t[:12000],
              model=kw.get("model", VLM), use_cache=kw.get("use_cache", True))
    return _loads(_extract_json(fix))
