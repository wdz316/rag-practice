// train 前200篇向量化（断点续算），与 dev 组成混合池做检索评测
// 用法：node embed_train200.cjs
const fs = require('fs');
const path = require('path');
const { pipeline } = require('@xenova/transformers');

const MODEL = 'Xenova/bge-small-en-v1.5';
const N_DOCS = 200;
const CHUNKS_FILE = path.join(__dirname, 'tatqa_train200_chunks.jsonl');
const CACHE = path.join(__dirname, 'tatqa_train200_embeddings.json');

function tableToMarkdown(t) {
  return Object.keys(t).map((k) => '| ' + t[k].join(' | ') + ' |').join('\n');
}

async function main() {
  const items = JSON.parse(
    fs.readFileSync(path.join(__dirname, 'tatqa_train.json'), 'utf-8')
  ).slice(0, N_DOCS);

  let chunks = [];
  if (fs.existsSync(CHUNKS_FILE)) {
    chunks = fs.readFileSync(CHUNKS_FILE, 'utf-8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
  } else {
    for (const x of items) {
      const docId = x.table.uid;
      chunks.push({ doc_id: docId, chunk_id: docId + '#T', chunk_type: 'table', text: tableToMarkdown(x.table.table) });
      for (const p of x.paragraphs) {
        chunks.push({ doc_id: docId, chunk_id: docId + '#P' + p.order, chunk_type: 'text', text: p.text });
      }
    }
    fs.writeFileSync(CHUNKS_FILE, chunks.map((c) => JSON.stringify(c)).join('\n'));
  }
  console.log('train200 块数：' + chunks.length);

  let vectors = [];
  if (fs.existsSync(CACHE)) {
    vectors = JSON.parse(fs.readFileSync(CACHE, 'utf-8'));
    console.log('断点续算：已有 ' + vectors.length + '/' + chunks.length);
  }
  const extractor = await pipeline('feature-extraction', MODEL);
  for (let i = vectors.length; i < chunks.length; i++) {
    const out = await extractor(chunks[i].text.slice(0, 2000), { pooling: 'mean', normalize: true });
    vectors.push(Array.from(out.data));
    if ((i + 1) % 50 === 0 || i + 1 === chunks.length) {
      fs.writeFileSync(CACHE, JSON.stringify(vectors));
      console.log('  已完成 ' + (i + 1) + '/' + chunks.length);
    }
  }
  console.log('完成，维度 ' + vectors[0].length);
}
main().catch((e) => { console.error('运行失败：' + e.message); process.exit(1); });
