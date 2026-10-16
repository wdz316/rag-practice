// v2 向量化通用脚本（断点续算）
// 用法：node embed_v2.cjs <chunks.jsonl> <out_embeddings.json>
const fs = require('fs');
const path = require('path');
const { pipeline } = require('@xenova/transformers');

async function main() {
  const [chunksFile, cacheFile] = process.argv.slice(2);
  const chunks = fs.readFileSync(path.join(__dirname, chunksFile), 'utf-8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
  let vectors = [];
  if (fs.existsSync(path.join(__dirname, cacheFile))) {
    vectors = JSON.parse(fs.readFileSync(path.join(__dirname, cacheFile), 'utf-8'));
    console.log('断点续算：已有 ' + vectors.length + '/' + chunks.length);
  }
  const extractor = await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5');
  const t0 = Date.now(), n0 = vectors.length;
  for (let i = vectors.length; i < chunks.length; i++) {
    const out = await extractor(chunks[i].text.slice(0, 2000), { pooling: 'mean', normalize: true });
    vectors.push(Array.from(out.data));
    if ((i + 1) % 50 === 0 || i + 1 === chunks.length) {
      fs.writeFileSync(path.join(__dirname, cacheFile), JSON.stringify(vectors));
      const el = (Date.now() - t0) / 1000, speed = (i + 1 - n0) / Math.max(el, 0.1);
      const eta = Math.round((chunks.length - (i + 1)) / speed);
      console.log(`  已完成 ${i + 1}/${chunks.length} 已用${Math.round(el)}s，预计还剩${Math.round(eta / 60)}分${eta % 60}秒`);
    }
  }
  console.log(chunksFile + ' 完成，维度 ' + vectors[0].length);
}
main().catch((e) => { console.error('运行失败：' + e.message); process.exit(1); });
