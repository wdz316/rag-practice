// 向量检索版 RAG 演示（本地 BGE embedding，无须 API）
// 用法：node vector_demo.cjs "你的问题"
// 首次运行会从 HuggingFace 下载模型（约100MB），向量算好后缓存到 embeddings.json
const fs = require('fs');
const path = require('path');
const { pipeline } = require('@xenova/transformers');

const MODEL = 'Xenova/bge-small-zh-v1.5';
const CACHE = path.join(__dirname, 'embeddings.json');

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

async function main() {
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'cmrc2018_trial.json'), 'utf-8'));
  const chunks = [];
  for (const doc of data.data) {
    for (const p of doc.paragraphs) chunks.push({ title: doc.title, context: p.context });
  }

  console.log('加载本地 embedding 模型：' + MODEL + '（首次需下载）...');
  const extractor = await pipeline('feature-extraction', MODEL);

  let vectors;
  if (fs.existsSync(CACHE)) {
    console.log('命中向量缓存，直接载入 ' + chunks.length + ' 条');
    vectors = JSON.parse(fs.readFileSync(CACHE, 'utf-8'));
  } else {
    console.log('逐条向量化 ' + chunks.length + ' 个片段（CPU，约需几分钟，仅此一次）...');
    vectors = [];
    for (let i = 0; i < chunks.length; i++) {
      const out = await extractor(chunks[i].context, { pooling: 'mean', normalize: true });
      vectors.push(Array.from(out.data));
      if ((i + 1) % 50 === 0) console.log('  已完成 ' + (i + 1) + '/' + chunks.length);
    }
    fs.writeFileSync(CACHE, JSON.stringify(vectors));
    console.log('向量已缓存到 embeddings.json，下次秒开');
  }

  const query = process.argv.slice(2).join(' ') || '游戏中离多久不能配对？';
  const qOut = await extractor(query, { pooling: 'mean', normalize: true });
  const qVec = Array.from(qOut.data);

  const ranked = chunks
    .map((c, i) => ({ ...c, s: cosine(qVec, vectors[i]) }))
    .sort((a, b) => b.s - a.s)
    .slice(0, 3);

  console.log('\n问题：' + query + '\n');
  ranked.forEach((h, i) => {
    console.log(`【向量 Top${i + 1}】《${h.title}》(余弦 ${h.s.toFixed(4)})`);
    console.log('片段：' + h.context.slice(0, 120) + '...\n');
  });
  console.log('共检索库：' + chunks.length + ' 个片段（向量维度 ' + qVec.length + '）');
}

main().catch((e) => { console.error('运行失败：' + e.message); process.exit(1); });
