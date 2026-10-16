// 跨文档检索：年报v2池 + 公告池同池比
// 用法：node cross_rag.cjs "问题" [--topK 6] [--alpha 0.5] [--pool v1|v2]
// 先跑：node cross_rag.cjs embed [--pool v2] （v1公告31块，v2公告30块）
// 注意：公告v1/v2向量各存各的缓存（announce_embeddings.json / _v2.json）
const fs = require('fs');
const path = require('path');
const { pipeline } = require('@xenova/transformers');

const MODEL = 'Xenova/bge-small-zh-v1.5';
const opt = (n, d) => {
  const i = process.argv.indexOf(n);
  return i >= 0 && process.argv[i + 1] !== undefined ? process.argv[i + 1] : d;
};
const POOL = opt('--pool', 'v1');
const AFILE = POOL === 'v2' ? 'announce_chunks_v2.jsonl' : 'announce_chunks.jsonl';
const ACACHE = path.join(__dirname, POOL === 'v2' ? 'announce_embeddings_v2.json' : 'announce_embeddings.json');
const cos = (a, b) => {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / (Math.sqrt(na) * Math.sqrt(nb));
};
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
  const rd = (f) => fs.readFileSync(path.join(__dirname, f), 'utf-8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
  const chunks = [...rd('moutai_chunks_v2.jsonl'), ...rd(AFILE)];
  const extractor = await pipeline('feature-extraction', MODEL);

  if (process.argv[2] === 'embed') {
    const aChunks = rd(AFILE);
    let vecs = [];
    if (fs.existsSync(ACACHE)) vecs = JSON.parse(fs.readFileSync(ACACHE, 'utf-8'));
    for (let i = vecs.length; i < aChunks.length; i++) {
      const out = await extractor(aChunks[i].text.slice(0, 1500), { pooling: 'mean', normalize: true });
      vecs.push(Array.from(out.data));
      if ((i + 1) % 10 === 0 || i + 1 === aChunks.length) console.log('  公告' + POOL + ' ' + (i + 1) + '/' + aChunks.length);
    }
    fs.writeFileSync(ACACHE, JSON.stringify(vecs));
    console.log('公告向量完成，共' + vecs.length);
    return;
  }

  const TOPK = parseInt(opt('--topK', '6'), 10);
  const ALPHA = parseFloat(opt('--alpha', '0.5'));
  const query = process.argv.slice(2).filter((a) => !a.startsWith('--') && !['--topK', '--alpha', '--pool'].includes(process.argv[process.argv.indexOf(a) - 1])).join(' ');
  const vecs = [...JSON.parse(fs.readFileSync(path.join(__dirname, 'moutai_embeddings_v2.json'), 'utf-8')),
                ...JSON.parse(fs.readFileSync(ACACHE, 'utf-8'))];
  const qv = Array.from((await extractor(query, { pooling: 'mean', normalize: true })).data);
  const qt = tokenize(query);
  const ks = chunks.map((c) => kwScore(qt, c.text));
  const vs = chunks.map((v, i) => cos(qv, vecs[i]));
  const norm = (arr) => {
    const mn = Math.min(...arr), mx = Math.max(...arr);
    return arr.map((v) => (mx === mn ? 0 : (v - mn) / (mx - mn)));
  };
  const nk = norm(ks), nv = norm(vs);
  console.log(`问题：${query}（跨文档池${chunks.length}块，公告${POOL}，topK=${TOPK}，alpha=${ALPHA}）\n`);
  chunks.map((c, i) => ({ c, s: ALPHA * nv[i] + (1 - ALPHA) * nk[i] }))
    .sort((a, b) => b.s - a.s).slice(0, TOPK)
    .forEach((h, i) => {
      console.log(`Top${i + 1} [${h.c.chunk_type}] ${h.c.chunk_id} (${h.s.toFixed(4)})｜${h.c.source}`);
      console.log(h.c.text.slice(0, 200).replace(/\n/g, ' / ') + '\n');
    });
}
main().catch((e) => { console.error('运行失败：' + e.message); process.exit(1); });
