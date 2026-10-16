// TAT-QA 切块 + ID/元数据方案
// doc_id  = 数据自带的 table.uid（稳定UUID，更新对账靠它）
// chunk_id = doc_id + '#' + 层级位：整表记 T，段落记 P{order}
// 表整存（平均10x4，不拆，防行列关系断裂），转 markdown 文本进库
// 用法：node make_tatqa_chunks.cjs
const fs = require('fs');
const path = require('path');

const VERSION = 'tatqa-raw-v1'; // 公开数据集无版本，生产里换成客户文档版本号
const UPDATED_AT = 'static';    // 公开数据集无更新时间，生产里换成文档更新日期

function tableToMarkdown(t) {
  return Object.keys(t)
    .map((k) => '| ' + t[k].join(' | ') + ' |')
    .join('\n');
}

function buildChunks(items, split) {
  const out = [];
  for (const x of items) {
    const docId = x.table.uid;
    const md = tableToMarkdown(x.table.table);
    out.push({
      doc_id: docId,
      chunk_id: docId + '#T',
      chunk_type: 'table',
      text: md,
      rows: Object.keys(x.table.table).length,
      cols: x.table.table[0].length,
      source: 'TAT-QA-' + split + '::' + docId,
      version: VERSION,
      updated_at: UPDATED_AT,
    });
    for (const p of x.paragraphs) {
      out.push({
        doc_id: docId,
        chunk_id: docId + '#P' + p.order,
        chunk_type: 'text',
        text: p.text,
        order: p.order,
        source: 'TAT-QA-' + split + '::' + docId + '#para' + p.order,
        version: VERSION,
        updated_at: UPDATED_AT,
      });
    }
  }
  return out;
}

for (const split of ['dev', 'train']) {
  const items = JSON.parse(
    fs.readFileSync(path.join(__dirname, 'tatqa_' + split + '.json'), 'utf-8')
  );
  const chunks = buildChunks(items, split);
  fs.writeFileSync(
    path.join(__dirname, 'tatqa_' + split + '_chunks.jsonl'),
    chunks.map((c) => JSON.stringify(c)).join('\n')
  );
  console.log(split + ': ' + items.length + ' 篇 → ' + chunks.length + ' 块');
}
