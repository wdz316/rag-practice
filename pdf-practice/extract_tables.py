# 解药：用 pdfplumber 按线框/字符位置还原表格（对比 extract.cjs 的乱文本）
import pdfplumber

with pdfplumber.open("./moutai2024.pdf") as pdf:
    print("总页数：", len(pdf.pages))
    # 先定位“主要会计数据”在哪几页
    hits = []
    for i, page in enumerate(pdf.pages):
        t = page.extract_text() or ""
        if "主要会计数据" in t:
            hits.append(i)
    print("含'主要会计数据'的页：", [h + 1 for h in hits])

    # 取第一页，抽表格（lattice 线框优先，失败回退 text 策略）
    page = pdf.pages[hits[0]]
    tables = page.extract_tables({"text_x_tolerance": 3, "text_y_tolerance": 3})
    print(f"第{hits[0]+1}页抽到 {len(tables)} 张表")
    for ti, tb in enumerate(tables[:3]):
        print(f"--- 表{ti+1}（{len(tb)}行 x {len(tb[0])}列）前6行 ---")
        for row in tb[:6]:
            print(" | ".join((c or "").replace("\n", "") for c in row))
