// TAT-QA v2 切块：标题合并 + 长段滑窗拆 + 大表按行贴头
// 规则：段<50字并入下一段；段>800字按句打包≤600字、重叠约100字；表markdown>1000字按行拆，前2行当表头每行重复贴
// 用法：node make_v2.cjs  → 输出 tatqa_dev_chunks_v2.jsonl / tatqa_train200_chunks_v2.jsonl
const fs = require('fs');
const path = require('path');

const SHORT = 50, LONG = 800, WIN = 600, OVERLAP = 100, BIGTABLE = 1000;
const stats = { merged: 0, split: 0, rowsplit: 0 };

function tableToMarkdown(rows) {
  return rows.map((r) => '| ' + r.join(' | ') + ' |').join('\n');
}

// 长段按句打包
function splitLong(text) {
  const sents = text.split(/(?<=[.!?])\s+/);
  const parts = [];
  let cur = '';
  for (const s of sents) {
    if ((cur + ' ' + s).length > WIN && cur) {
      parts.push(cur);
      cur = cur.slice(-OVERLAP); // 带100字重叠
    }
    cur = (cur + ' ' + s).trim();
  }
  if (cur) parts.push(cur);
  return parts.length > 1 ? parts : [text];
}

function chunkDoc(x) {
  const docId = x.table.uid;
  const out = [];
  // --- 表 ---
  const rowKeys = Object.keys(x.table.table);
  const md = tableToMarkdown(rowKeys.map((k) => x.table.table[k]));
  if (md.length > BIGTABLE) {
    const header = tableToMarkdown([x.table.table[0], x.table.table[1]]);
    rowKeys.slice(2).forEach((k, i) => {
      out.push({ doc_id: docId, chunk_id: docId + '#Trow' + k, chunk_type: 'table_row', text: header + '\n' + tableToMarkdown([x.table.table[k]]) });
      stats.rowsplit++;
    });
  } else {
    out.push({ doc_id: docId, chunk_id: docId + '#T', chunk_type: 'table', text: md });
  }
  // --- 段：短标题焊下段，长段滑窗 ---
  let pendingTitles = [];
  const flush = (p, isLast) => {
    let text = [...pendingTitles, p.text].join('\n');
    if (pendingTitles.length) stats.merged += pendingTitles.length;
    pendingTitles = [];
    if (text.length > LONG) {
      splitLong(text).forEach((part, i, arr) => {
        out.push({ doc_id: docId, chunk_id: docId + '#P' + p.order + (arr.length > 1 ? String.fromCharCode(97 + i) : ''), chunk_type: 'text', text: part });
      });
      if (splitLong(text).length > 1) stats.split++;
    } else {
      out.push({ doc_id: docId, chunk_id: docId + '#P' + p.order, chunk_type: 'text', text });
    }
  };
  x.paragraphs.forEach((p, i) => {
    const last = i === x.paragraphs.length - 1;
    if (p.text.length < SHORT && !last) pendingTitles.push(p.text);
    else if (p.text.length < SHORT && last && out.length) out[out.length - 1].text += '\n' + p.text; // 文末短段挂前一块
    else flush(p, last);
  });
  return out;
}

function build(srcFile, nDocs, outFile) {
  const items = JSON.parse(fs.readFileSync(path.join(__dirname, srcFile), 'utf-8')).slice(0, nDocs);
  const chunks = items.flatMap(chunkDoc);
  chunks.forEach((c) => {
    c.source = 'TAT-QA-v2::' + c.doc_id;
    c.version = 'tatqa-raw-v1';
    c.updated_at = 'static';
  });
  fs.writeFileSync(path.join(__dirname, outFile), chunks.map((c) => JSON.stringify(c)).join('\n'));
  console.log(outFile + ': ' + items.length + ' 篇 → ' + chunks.length + ' 块');
}

build('tatqa_dev.json', 278, 'tatqa_dev_chunks_v2.jsonl');
build('tatqa_train.json', 200, 'tatqa_train200_chunks_v2.jsonl');
console.log('规则命中：标题合并' + stats.merged + '次，长段拆分' + stats.split + '次，大表按行' + stats.rowsplit + '行');
