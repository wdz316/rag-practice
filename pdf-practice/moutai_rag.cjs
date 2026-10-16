// 茅台库向量化（中文BGE，断点续算）+ 问答检索（含混合权重）
// 用法：node moutai_rag.cjs embed [--db v2]              → 向量化并缓存
// 用法：node moutai_rag.cjs "问题" [--db v2] [--alpha 0.5]
const fs = require('fs');
const path = require('path');
const { pipeline } = require('@xenova/transformers');

const DB = process.argv.includes('--db') && process.argv[process.argv.indexOf('--db') + 1] === 'v2' ? 'v2' : 'v1';
const CHUNKS_FILE = DB === 'v2' ? 'moutai_chunks_v2.jsonl' : 'moutai_chunks.jsonl';
const CACHE_FILE = DB === 'v2' ? 'moutai_embeddings_v2.json' : 'moutai_embeddings.json';
const MODEL = 'Xenova/bge-small-zh-v1.5';
const CACHE = path.join(__dirname, CACHE_FILE);
const cos = (a, b) => {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / (Math.sqrt(na) * Math.sqrt(nb));
};
// 中文关键词分：单字+二元组字面命中（专治应收/应付、2024/2023这种向量分不清的）
function tokenize(s) {
  s = s.replace(/\s+/g, '');
  const set = new Set();
  for (let i = 0; i < s.length; i++) set.add(s[i]);
  for (let i = 0; i < s.length - 1; i++) set.add(s.slice(i, i + 2));
  return set;
}
function kwScore(qt, context) {
  const ctx = tokenize(context);
  let hit = 0;
  for (const t of qt) if (ctx.has(t)) hit += t.length > 1 ? 2 : 1;
  return hit / Math.sqrt(context.length);
}

async function main() {
  const chunks = fs.readFileSync(path.join(__dirname, CHUNKS_FILE), 'utf-8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
  const extractor = await pipeline('feature-extraction', MODEL);

  if (process.argv[2] === 'embed') {
    let vecs = [];
    if (fs.existsSync(CACHE)) vecs = JSON.parse(fs.readFileSync(CACHE, 'utf-8'));
    console.log('断点续算：已有 ' + vecs.length + '/' + chunks.length);
    const t0 = Date.now(), n0 = vecs.length;
    for (let i = vecs.length; i < chunks.length; i++) {
      const out = await extractor(chunks[i].text.slice(0, 1500), { pooling: 'mean', normalize: true });
      vecs.push(Array.from(out.data));
      if ((i + 1) % 50 === 0 || i + 1 === chunks.length) {
        fs.writeFileSync(CACHE, JSON.stringify(vecs));
        const el = (Date.now() - t0) / 1000, speed = (i + 1 - n0) / Math.max(el, 0.1);
        const eta = Math.round((chunks.length - (i + 1)) / speed);
        console.log(`  已完成 ${i + 1}/${chunks.length} 已用${Math.round(el)}s，预计还剩${Math.round(eta / 60)}分${eta % 60}秒`);
      }
    }
    console.log('完成，维度 ' + vecs[0].length);
    return;
  }

  const qargs = process.argv.slice(2).filter((a) => !a.startsWith('--') && process.argv[process.argv.indexOf(a) - 1] !== '--db');
  const query = qargs.join(' ') || '茅台2024年营业收入多少';
  const ALPHA = process.argv.includes('--alpha')
    ? parseFloat(process.argv[process.argv.indexOf('--alpha') + 1]) : 1.0;
  const vecs = JSON.parse(fs.readFileSync(CACHE, 'utf-8'));
  const qv = Array.from((await extractor(query, { pooling: 'mean', normalize: true })).data);
  const qt = tokenize(query);
  const ks = chunks.map((c) => kwScore(qt, c.text));
  const vs = chunks.map((v, i) => cos(qv, vecs[i]));
  const norm = (arr) => {
    const mn = Math.min(...arr), mx = Math.max(...arr);
    return arr.map((v) => (mx === mn ? 0 : (v - mn) / (mx - mn)));
  };
  const nk = norm(ks), nv = norm(vs);
  console.log(`问题：${query}（库${DB}，alpha=${ALPHA}）\n`);
  chunks.map((c, i) => ({ c, s: ALPHA * nv[i] + (1 - ALPHA) * nk[i] }))
    .sort((a, b) => b.s - a.s).slice(0, 3)
    .forEach((h, i) => {
      console.log(`Top${i + 1} [${h.c.chunk_type}] ${h.c.chunk_id} (${h.s.toFixed(4)})出处：${h.c.source}`);
      console.log(h.c.text.slice(0, 220).replace(/\n/g, ' / ') + '\n');
    });
}
main().catch((e) => { console.error('运行失败：' + e.message); process.exit(1); });
