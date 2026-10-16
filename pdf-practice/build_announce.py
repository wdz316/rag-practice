# 公告进库：表整存（公告表小）+ 文500字滑窗；doc_id按文件区分来源
import json
import pdfplumber

FILES = {
    "announce_fenhong2024": ("./announce/fenhong2024.pdf", "2024年年度权益分派实施公告", "2025-06-20"),
    "announce_huigou2026": ("./announce/huigou2026.pdf", "回购股份实施进展公告", "2026-04-03"),
    "announce_gudonghui2024": ("./announce/gudonghui2024.pdf", "2024年度股东大会决议公告", "2025-05-20"),
}
chunks = []
for doc_id, (pdf_path, title, date) in FILES.items():
    with pdfplumber.open(pdf_path) as pdf:
        for pi, page in enumerate(pdf.pages, start=1):
            for ti, tb in enumerate(page.extract_tables({"text_x_tolerance": 3, "text_y_tolerance": 3}) or []):
                rows = [" | ".join((c or "").replace("\n", "") for c in r) for r in tb if any(r)]
                if len(rows) < 2:
                    continue
                chunks.append({"doc_id": doc_id, "chunk_id": f"{doc_id}#T{pi}_{ti}",
                               "chunk_type": "table", "text": "\n".join(rows),
                               "source": f"茅台{title}第{pi}页表{ti+1}",
                               "version": "announce-v1", "updated_at": date})
        full = "\n".join([(p.extract_text() or "") for p in pdf.pages])
    n = 0
    for s in range(0, len(full), 400):
        part = full[s:s + 500].strip()
        if len(part) < 80:
            continue
        n += 1
        chunks.append({"doc_id": doc_id, "chunk_id": f"{doc_id}#P{n}", "chunk_type": "text",
                       "text": part, "source": f"茅台{title}正文块{n}",
                       "version": "announce-v1", "updated_at": date})

with open("./announce_chunks.jsonl", "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
from collections import Counter
print(Counter(c["doc_id"] + "/" + c["chunk_type"] for c in chunks), "合计", len(chunks))
