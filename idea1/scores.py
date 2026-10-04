import re, sys
import pandas as pd, pypdfium2 as pdfium, torch
from rank_bm25 import BM25Okapi
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# data isnt in the repo, you have to download it (see README) and pass the folder: python scores.py data
D = sys.argv[1] if len(sys.argv) > 1 else "data"
df = pd.read_parquet(f"{D}/data/train-00000-of-00001.parquet")
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"(?<=\d),(?=\d{3})", "", s.lower())).strip()  # 1,358 -> 1358
words = lambda s: re.findall(r"\w+", norm(s))

text = {d: [norm(p.get_textpage().get_text_range()) for p in pdfium.PdfDocument(f"{D}/documents/{d}")]
        for d in df.doc_id.unique()}
bm25 = {d: BM25Okapi([words(t) or ["_"] for t in ts]) for d, ts in text.items()}
txt = [bm25[d].get_scores(words(q)) for d, q in zip(df.doc_id, df.question)]

model = ColQwen2_5.from_pretrained("vidore/colqwen2.5-v0.2", dtype=torch.bfloat16, device_map="cuda").eval()
proc = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")
img = {}
with torch.no_grad():
    for d, g in df.groupby("doc_id"):
        pdf, pages = pdfium.PdfDocument(f"{D}/documents/{d}"), []
        for i in range(0, len(pdf), 8):  # 8 pages at a time, 108 dpi
            b = proc.process_images([pdf[j].render(scale=1.5).to_pil() for j in range(i, min(i + 8, len(pdf)))]).to("cuda")
            pages += [e[m.bool()].cpu() for e, m in zip(model(**b), b["attention_mask"])]
        qs = model(**proc.process_queries(g.question.tolist()).to("cuda"))
        img.update(zip(g.index, proc.score_multi_vector(list(qs.cpu()), pages).float().numpy()))

pd.to_pickle({"text": text, "bm25": txt, "colqwen": [img[i] for i in df.index]}, "scores.pkl")
