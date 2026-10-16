# 茅台 v2：超10行表按行拆（首行当表头，行行重复贴），其余不动
# 行块 moutai2024#T{页}_{表}row{行号}
import json
import pdfplumber

DOC = "moutai2024"
VERSION = "moutai-2024-annual-v2"
ROW_SPLIT = 10
chunks, nrowsplit = [], 0

with pdfplumber.open("./moutai2024.pdf") as pdf:
    for pi, page in enumerate(pdf.pages, start=1):
        tables = page.extract_tables({"text_x_tolerance": 3, "text_y_tolerance": 3}) or []
        for ti, tb in enumerate(tables):
            rows = [" | ".join((c or "").replace("\n", "") for c in r) for r in tb if any(r)]
            if len(rows) < 2:
                continue
            base = {"doc_id": DOC, "source": f"贵州茅台2024年报第{pi}页表{ti+1}",
                    "version": VERSION, "updated_at": "2025-04-03"}
            if len(rows) > ROW_SPLIT:
                header = rows[0]
                for ri, row in enumerate(rows[1:], start=1):
                    chunks.append({**base, "chunk_id": f"{DOC}#T{pi}_{ti}row{ri}",
                                   "chunk_type": "table_row", "text": header + "\n" + row})
                    nrowsplit += 1
            else:
                chunks.append({**base, "chunk_id": f"{DOC}#T{pi}_{ti}",
                               "chunk_type": "table", "text": "\n".join(rows), "rows": len(rows)})

with open("./moutai2024.txt", encoding="utf-8") as f:
    full = f.read()
STEP, OVERLAP, n = 500, 100, 0
for s in range(0, len(full), STEP - OVERLAP):
    part = full[s:s + STEP].strip()
    if len(part) < 100:
        continue
    n += 1
    chunks.append({"doc_id": DOC, "chunk_id": f"{DOC}#P{n}", "chunk_type": "text",
                   "text": part, "source": f"贵州茅台2024年报正文块{n}",
                   "version": VERSION, "updated_at": "2025-04-03"})

with open("./moutai_chunks_v2.jsonl", "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
nt = sum(1 for c in chunks if c["chunk_type"] in ("table", "table_row"))
print(f"表/行块 {nt}（其中行拆 {nrowsplit} 行），文块 {len(chunks)-nt}，合计 {len(chunks)}")
# 抽查：应收票据行
hit = [c for c in chunks if "应收票据" in c["text"] and c["chunk_type"] == "table_row"]
print("应收票据行块数:", len(hit))
if hit:
    print(hit[0]["chunk_id"], "|", hit[0]["text"][:160].replace("\n", " / "))
