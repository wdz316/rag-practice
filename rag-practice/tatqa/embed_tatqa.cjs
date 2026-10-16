// TAT-QA dev 向量化（英文 BGE-small，断点续算）
// 用法：node embed_tatqa.cjs
// 首次下载模型约130MB；每算50条落盘一次，中断重跑自动续上
const fs = require('fs');
const path = require('path');
const { pipeline } = require('@xenova/transformers');

const MODEL = 'Xenova/bge-small-en-v1.5';
const CACHE = path.join(__dirname, 'tatqa_dev_embeddings.json');

async function main() {
  const chunks = fs
    .readFileSync(path.join(__dirname, 'tatqa_dev_chunks.jsonl'), 'utf-8')
    .split('\n')
    .filter(Boolean)
    .map((l) => JSON.parse(l));

  let vectors = [];
  if (fs.existsSync(CACHE)) {
    vectors = JSON.parse(fs.readFileSync(CACHE, 'utf-8'));
    console.log('断点续算：已有 ' + vectors.length + '/' + chunks.length);
  }

  console.log('加载模型：' + MODEL + '（首次需下载）...');
  const extractor = await pipeline('feature-extraction', MODEL);

  for (let i = vectors.length; i < chunks.length; i++) {
    const out = await extractor(chunks[i].text.slice(0, 2000), { pooling: 'mean', normalize: true });
    vectors.push(Array.from(out.data));
    if ((i + 1) % 50 === 0 || i + 1 === chunks.length) {
      fs.writeFileSync(CACHE, JSON.stringify(vectors));
      console.log('  已完成 ' + (i + 1) + '/' + chunks.length);
    }
  }
  console.log('全部完成，向量维度 ' + vectors[0].length);
}
main().catch((e) => { console.error('运行失败：' + e.message); process.exit(1); });
