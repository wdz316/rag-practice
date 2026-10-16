# 公告 v2：按节切 + 标题复读 + 表整存
# 节正则 ^[一二三四五六七八九十]+、 ；<100字段并下节；节内>800字段内按句滑窗600/100
# 每块前缀【标题】；ID doc#S{节号} / 表 doc#T{页}_{表}
import json
import re
import pdfplumber

FILES = {
    "announce_fenhong2024": ("./announce/fenhong2024.pdf", "贵州茅台2024年年度权益分派实施公告"),
    "announce_huigou2026": ("./announce/huigou2026.pdf", "贵州茅台回购股份实施进展公告"),
    "announce_gudonghui2024": ("./announce/gudonghui2024.pdf", "贵州茅台2024年度股东大会决议公告"),
}
SEC = re.compile(r"^([一二三四五六七八九十]+、)")
SHORT, LONG, WIN, OVERLAP = 100, 800, 600, 100
chunks = []

for doc_id, (pdf_path, title) in FILES.items():
    prefix = f"【{title}】\n"
    with pdfplumber.open(pdf_path) as pdf:
        for pi, page in enumerate(pdf.pages, start=1):
            for ti, tb in enumerate(page.extract_tables({"text_x_tolerance": 3, "text_y_tolerance": 3}) or []):
                rows = [" | ".join((c or "").replace("\n", "") for c in r) for r in tb if any(r)]
                if len(rows) < 2:
                    continue
                chunks.append({"doc_id": doc_id, "chunk_id": f"{doc_id}#T{pi}_{ti}",
                               "chunk_type": "table", "text": prefix + "\n".join(rows),
                               "source": f"茅台{title}第{pi}页表{ti+1}",
                               "version": "announce-v2", "updated_at": "2025-01-01"})
        full = "\n".join([(p.extract_text() or "") for p in pdf.pages])
    # 按节切
    secs, cur_no, cur_buf = [], None, []
    for line in full.split("\n"):
        m = SEC.match(line.strip())
        if m:
            if cur_no is not None:
                secs.append((cur_no, "\n".join(cur_buf)))
            cur_no, cur_buf = m.group(1), [line]
        elif cur_no is not None:
            cur_buf.append(line)
    if cur_no is not None:
        secs.append((cur_no, "\n".join(cur_buf)))
    # 短节合并 + 长节滑窗
    i = 0
    while i < len(secs):
        no, text = secs[i]
        if len(text) < SHORT and i + 1 < len(secs):
            text = text + "\n" + secs[i + 1][1]
            no = no + secs[i + 1][0]
            i += 2
        else:
            i += 1
        parts = [text]
        if len(text) > LONG:
            sents = re.split(r"(?<=[。！？])", text)
            parts, cur = [], ""
            for s in sents:
                if len(cur + s) > WIN and cur:
                    parts.append(cur)
                    cur = cur[-OVERLAP:]
                cur = (cur + s).strip()
            if cur:
                parts.append(cur)
        for j, part in enumerate(parts):
            chunks.append({"doc_id": doc_id, "chunk_id": f"{doc_id}#S{no}" + (chr(97 + j) if len(parts) > 1 else ""),
                           "chunk_type": "text", "text": prefix + part.strip(),
                           "source": f"茅台{title}第{no}节",
                           "version": "announce-v2", "updated_at": "2025-01-01"})

with open("./announce_chunks_v2.jsonl", "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
from collections import Counter
print(Counter(c["doc_id"] + "/" + c["chunk_type"] for c in chunks), "合计", len(chunks))
# 验收自检：核心数同块？
core = [c for c in chunks if "34,671" in c["text"] and "27.673" in c["text"]]
print("同块含346亿+27.673的块数:", len(core), [c["chunk_id"] for c in core])
