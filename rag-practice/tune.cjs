// 可调参检索实验台：TopK / 混合权重 / 距离算法 / 元数据过滤
// 用法：node tune.cjs "问题" [--topK N] [--alpha 0~1] [--metric cosine|euclid] [--filter 标题关键字]
// alpha=1 纯向量，alpha=0 纯关键词，默认 0.7（生产常用起点）
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const query = args.find((a) => !a.startsWith('--')) || '逃赛要被禁赛几分钟？';
const opt = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : def;
};
const TOPK = parseInt(opt('--topK', '3'), 10);
const ALPHA = parseFloat(opt('--alpha', '0.7'));
const METRIC = opt('--metric', 'cosine');
const FILTER = opt('--filter', '');

// --- 关键词打分（照搬 demo.cjs） ---
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
// --- 向量相似度 ---
function cosSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
function euclidSim(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
  return 1 - Math.sqrt(s) / 2; // 归一向量下距离∈[0,2]，转成相似度
}
const { pipeline } = require('@xenova/transformers');

async function main() {
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'cmrc2018_trial.json'), 'utf-8'));
  let chunks = [];
  for (const doc of data.data) {
    for (const p of doc.paragraphs) chunks.push({ title: doc.title, context: p.context });
  }
  if (FILTER) {
    const before = chunks.length;
    chunks = chunks.filter((c) => c.title.includes(FILTER));
    console.log(`元数据过滤 title含"${FILTER}"：${before} → ${chunks.length} 个片段`);
    if (!chunks.length) { console.log('过滤后库空了，直接返回（生产里这就是权限配错的效果）'); return; }
  }

  const allVecs = JSON.parse(fs.readFileSync(path.join(__dirname, 'embeddings.json'), 'utf-8'));
  // embeddings.json 与 chunks 一一对应；过滤后按原下标取回向量
  const data2 = JSON.parse(fs.readFileSync(path.join(__dirname, 'cmrc2018_trial.json'), 'utf-8'));
  const flat = [];
  data2.data.forEach((doc) => doc.paragraphs.forEach((p) => flat.push(doc.title + '\u0000' + p.context)));
  const idxMap = new Map(flat.map((k, i) => [k, i]));
  const vectors = chunks.map((c) => allVecs[idxMap.get(c.title + '\u0000' + c.context)]);

  const extractor = await pipeline('feature-extraction', 'Xenova/bge-small-zh-v1.5');
  const qVec = Array.from((await extractor(query, { pooling: 'mean', normalize: true })).data);
  const simFn = METRIC === 'euclid' ? euclidSim : cosSim;

  const qt = tokenize(query);
  const kws = chunks.map((c) => kwScore(qt, c.context));
  const vecs = chunks.map((v, i) => simFn(qVec, vectors[i]));

  // min-max 归一到 [0,1] 再加权（不归一的话两路量纲不同，加权无意义）
  const norm = (arr) => {
    const mn = Math.min(...arr), mx = Math.max(...arr);
    return arr.map((v) => (mx === mn ? 0 : (v - mn) / (mx - mn)));
  };
  const nk = norm(kws), nv = norm(vecs);
  const ranked = chunks
    .map((c, i) => ({ ...c, s: ALPHA * nv[i] + (1 - ALPHA) * nk[i], sv: vecs[i], sk: kws[i] }))
    .sort((a, b) => b.s - a.s)
    .slice(0, TOPK);

  console.log(`问题：${query}\n参数：topK=${TOPK} alpha=${ALPHA}(向量权重) metric=${METRIC}${FILTER ? ' filter=' + FILTER : ''}\n`);
  ranked.forEach((h, i) => {
    console.log(`Top${i + 1}《${h.title}》混合分${h.s.toFixed(3)}（向量${h.sv.toFixed(3)}｜关键词${h.sk.toFixed(3)}）`);
  });
}
main().catch((e) => { console.error('运行失败：' + e.message); process.exit(1); });
