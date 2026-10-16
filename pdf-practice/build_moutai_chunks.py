# 茅台年报进库：全书表格（pdfplumber还原）+ 正文（500字滑窗100重叠）
# doc_id=moutai2024，表块 moutai2024#T{页}_{表序号}，文块 moutai2024#P{序号}
import json
import pdfplumber

DOC = "moutai2024"
VERSION = "moutai-2024-annual"
chunks = []

with pdfplumber.open("./moutai2024.pdf") as pdf:
    for pi, page in enumerate(pdf.pages, start=1):
        tables = page.extract_tables({"text_x_tolerance": 3, "text_y_tolerance": 3}) or []
        for ti, tb in enumerate(tables):
            rows = [" | ".join((c or "").replace("\n", "") for c in r) for r in tb if any(r)]
            if len(rows) < 2:
                continue
            chunks.append({
                "doc_id": DOC,
                "chunk_id": f"{DOC}#T{pi}_{ti}",
                "chunk_type": "table",
                "text": "\n".join(rows),
                "rows": len(rows),
                "source": f"贵州茅台2024年报第{pi}页表{ti+1}",
                "version": VERSION,
                "updated_at": "2025-04-03",
            })

with open("./moutai2024.txt", encoding="utf-8") as f:
    full = f.read()
STEP, OVERLAP = 500, 100
n = 0
for s in range(0, len(full), STEP - OVERLAP):
    part = full[s:s + STEP].strip()
    if len(part) < 100:
        continue
    n += 1
    chunks.append({
        "doc_id": DOC,
        "chunk_id": f"{DOC}#P{n}",
        "chunk_type": "text",
        "text": part,
        "source": f"贵州茅台2024年报正文块{n}",
        "version": VERSION,
        "updated_at": "2025-04-03",
    })

with open("./moutai_chunks.jsonl", "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
nt = sum(1 for c in chunks if c["chunk_type"] == "table")
print(f"表块 {nt}，文块 {len(chunks)-nt}，合计 {len(chunks)}")
