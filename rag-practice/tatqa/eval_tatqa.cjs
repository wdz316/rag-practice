// TAT-QA 检索评测：混合池(dev1634 + train200约1196) + dev抽样考题
// 指标：金答案doc(chunk的doc_id)是否在TopK内 → 命中率
// 用法：node eval_tatqa.cjs [--n 60] [--topK 5] [--alpha 1.0] [--pool v1|v2|dev30pool|sentpool]
const fs = require('fs');
const path = require('path');
const { pipeline } = require('@xenova/transformers');

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : def;
};
const NQ = parseInt(opt('--n', '60'), 10);
const TOPK = parseInt(opt('--topK', '5'), 10);
const ALPHA = parseFloat(opt('--alpha', '1.0'));

// 英文词重叠关键词分
function words(s) {
  return (s.toLowerCase().match(/[a-z0-9%$.\-]+/g) || []).filter((w) => w.length > 1);
}
function kwScore(qw, text) {
  const set = new Set(words(text));
  let hit = 0;
  for (const w of qw) if (set.has(w)) hit += w.length >= 5 ? 2 : 1;
  return hit / Math.sqrt(text.length);
}
function cosSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

async function main() {
  const rd = (f) => fs.readFileSync(path.join(__dirname, f), 'utf-8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
  const pool = opt('--pool', 'v1');
  const poolFiles = {
    v1: ['tatqa_dev_chunks.jsonl', 'tatqa_dev_embeddings.json', 'tatqa_train200_chunks.jsonl', 'tatqa_train200_embeddings.json'],
    v2: ['tatqa_dev_chunks_v2.jsonl', 'tatqa_dev_embeddings_v2.json', 'tatqa_train200_chunks_v2.jsonl', 'tatqa_train200_embeddings_v2.json'],
    dev30pool: ['dev30_chunks.jsonl', 'dev30_embeddings.json', 'tatqa_train200_chunks.jsonl', 'tatqa_train200_embeddings.json'],
    sentpool: ['sent30_chunks.jsonl', 'sent30_embeddings.json', 'tatqa_train200_chunks.jsonl', 'tatqa_train200_embeddings.json'],
  }[pool] || [];
  const chunks = [...rd(poolFiles[0]), ...rd(poolFiles[2])];
  const vecs = [...JSON.parse(fs.readFileSync(path.join(__dirname, poolFiles[1]), 'utf-8')),
                ...JSON.parse(fs.readFileSync(path.join(__dirname, poolFiles[3]), 'utf-8'))];
  console.log(`混合池(${pool})：${chunks.length} 块`);

  // 抽样：每篇取第1题，凑 NQ 道（分散到不同文档，防一表多题刷分）
  const dev = JSON.parse(fs.readFileSync(path.join(__dirname, 'tatqa_dev.json'), 'utf-8'));
  const qs = [];
  for (const x of dev) {
    if (qs.length >= NQ) break;
    qs.push({ q: x.questions[0].question, gold: x.table.uid });
  }

  const extractor = await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5');
  let hit = 0;
  const misses = [];
  for (const { q, gold } of qs) {
    const qv = Array.from((await extractor(q, { pooling: 'mean', normalize: true })).data);
    const qw = words(q);
    const ks = chunks.map((c) => kwScore(qw, c.text));
    const vs = chunks.map((v, i) => cosSim(qv, vecs[i]));
    const norm = (arr) => {
      const mn = Math.min(...arr), mx = Math.max(...arr);
      return arr.map((v) => (mx === mn ? 0 : (v - mn) / (mx - mn)));
    };
    const nk = norm(ks), nv = norm(vs);
    const top = chunks
      .map((c, i) => ({ c, s: ALPHA * nv[i] + (1 - ALPHA) * nk[i] }))
      .sort((a, b) => b.s - a.s)
      .slice(0, TOPK);
    if (top.some((t) => t.c.doc_id === gold)) hit++;
    else if (misses.length < 3) misses.push({ q, top1: top[0].c.chunk_id + ' [' + top[0].c.chunk_type + ']' });
  }
  console.log(`考题 ${qs.length} 道，topK=${TOPK}，alpha=${ALPHA}`);
  console.log(`命中率：${hit}/${qs.length} = ${(100 * hit / qs.length).toFixed(1)}%`);
  misses.forEach((m, i) => console.log(`脱靶${i + 1} Q:${m.q.slice(0, 70)} → Top1 ${m.top1}`));
}
main().catch((e) => { console.error('运行失败：' + e.message); process.exit(1); });
