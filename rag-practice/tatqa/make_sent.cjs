// 句切裸版实验：前30篇dev，各做两套块
// A套 dev30_chunks.jsonl：v1原样（整表+整段）
// B套 sent30_chunks.jsonl：表整存，段落按句裸切（无指代还原，看无头块效应）
// 用法：node make_sent.cjs
const fs = require('fs');
const path = require('path');

function tableToMarkdown(t) {
  return Object.keys(t).map((k) => '| ' + t[k].join(' | ') + ' |').join('\n');
}
const meta = (c, docId) => {
  c.source = 'TAT-QA-sent-exp::' + docId;
  c.version = 'tatqa-raw-v1';
  c.updated_at = 'static';
  return c;
};

const dev = JSON.parse(fs.readFileSync(path.join(__dirname, 'tatqa_dev.json'), 'utf-8')).slice(0, 30);
const setA = [], setB = [];
for (const x of dev) {
  const docId = x.table.uid;
  const md = tableToMarkdown(x.table.table);
  setA.push(meta({ doc_id: docId, chunk_id: docId + '#T', chunk_type: 'table', text: md }, docId));
  setB.push(meta({ doc_id: docId, chunk_id: docId + '#T', chunk_type: 'table', text: md }, docId));
  for (const p of x.paragraphs) {
    setA.push(meta({ doc_id: docId, chunk_id: docId + '#P' + p.order, chunk_type: 'text', text: p.text }, docId));
    // 裸切：按英文句号切，不管缩写小数点（分句错误本身也是实验现象）
    p.text.split(/(?<=[.!?])\s+/).filter((s) => s.trim().length > 0).forEach((s, i) => {
      setB.push(meta({ doc_id: docId, chunk_id: docId + '#S' + p.order + '_' + i, chunk_type: 'sentence', text: s.trim() }, docId));
    });
  }
}
fs.writeFileSync(path.join(__dirname, 'dev30_chunks.jsonl'), setA.map((c) => JSON.stringify(c)).join('\n'));
fs.writeFileSync(path.join(__dirname, 'sent30_chunks.jsonl'), setB.map((c) => JSON.stringify(c)).join('\n'));
console.log(`A套原样：${setA.length} 块，B套句切：${setB.length} 块`);
const tiny = setB.filter((c) => c.chunk_type === 'sentence' && c.text.length < 50).length;
console.log(`B套中短于50字的句子块：${tiny}（无头块候选）`);
