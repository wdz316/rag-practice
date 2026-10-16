// trial 6切分方法切块 + 向量化（断点续算，按切分方法缓存）
// 用法：node methods_trial.cjs [fixed500|para|recur|punct|struct|child]（不带参数全跑）
// fixed500: 500字一刀重叠50 | para: 原生段落 | recur: 段→句→分号逐级 | punct: 句打包300-600
// struct: 标题面包屑前缀+段落 | child: 子200字（父段凭ID取回，检索子给父）
const fs = require('fs');
const path = require('path');
const { pipeline } = require('@xenova/transformers');

const MODEL = 'Xenova/bge-small-zh-v1.5';
const raw = JSON.parse(fs.readFileSync(path.join(__dirname, 'cmrc2018_trial.json'), 'utf-8'));
const docs = raw.data.map((d) => ({ title: d.title, paras: d.paragraphs.map((p) => p.context) }));

const splitSent = (t) => t.split(/(?<=[。！？])/).map((s) => s.trim()).filter(Boolean);
function pack(sentParts, maxLen, minLen = 0) {
  const out = [];
  let cur = '';
  for (const s of sentParts) {
    if ((cur + s).length > maxLen && cur) { out.push(cur); cur = ''; }
    cur += s;
  }
  if (cur) out.push(cur);
  if (minLen <= 0) return out;
  const merged = [];
  for (const c of out) {
    if (c.length < minLen && merged.length) merged[merged.length - 1] += c;
    else merged.push(c);
  }
  return merged;
}

const METHODS = {
  fixed500: { file: 'emb_trial_fixed500.json', build() {
    const out = [];
    docs.forEach((d) => d.paras.forEach((p, pi) => {
      for (let s = 0; s < p.length; s += 450) out.push({ title: `${d.title}#F${pi}_${s}`, context: p.slice(s, s + 500) });
    }));
    return out;
  } },
  para: { file: 'emb_trial_para.json', build() {
    const out = [];
    docs.forEach((d) => d.paras.forEach((p, pi) => out.push({ title: `${d.title}#P${pi}`, context: p })));
    return out;
  } },
  recur: { file: 'emb_trial_recur.json', build() {
    const out = [];
    docs.forEach((d) => d.paras.forEach((p, pi) => {
      splitSent(p).forEach((s) => {
        if (s.length <= 500) out.push({ title: `${d.title}#R${pi}`, context: s, _m: true });
        else s.split(/(?<=[；，、])/).forEach((ss, k) => out.push({ title: `${d.title}#R${pi}_${k}`, context: ss }));
      });
    }));
    // 同段短句合并（段→句退让后回填）
    const merged = [];
    let cur = null;
    for (const c of out) {
      if (c._m && cur && cur.title.split('#')[0] === c.title.split('#')[0] && (cur.context + c.context).length <= 500) cur.context += c.context;
      else { if (cur) { delete cur._m; merged.push(cur); } cur = { ...c }; }
    }
    if (cur) { delete cur._m; merged.push(cur); }
    return merged;
  } },
  punct: { file: 'emb_trial_punct.json', build() {
    const out = [];
    docs.forEach((d) => d.paras.forEach((p, pi) => {
      pack(splitSent(p), 600).forEach((part, k, arr) => {
        if (part.length < 300 && out.length && arr.length > 1) out[out.length - 1].context += part;
        else out.push({ title: `${d.title}#S${pi}_${k}`, context: part });
      });
    }));
    return out;
  } },
  struct: { file: 'emb_trial_struct.json', build() {
    const out = [];
    docs.forEach((d) => d.paras.forEach((p, pi) => out.push({ title: `${d.title}#H${pi}`, context: `${d.title}\n${p}` })));
    return out;
  } },
  child: { file: 'emb_trial_child.json', meta: 'parents_trial_child.json', build() {
    const out = [], parents = {};
    docs.forEach((d) => d.paras.forEach((p, pi) => {
      const pid = `${d.title}#D${pi}`;
      parents[pid] = { title: d.title, context: p };
      for (let s = 0, k = 0; s < p.length; s += 200, k++) {
        out.push({ title: `${d.title}#C${pi}_${k}`, context: p.slice(s, s + 200), parent: pid });
      }
    }));
    fs.writeFileSync(path.join(__dirname, 'parents_trial_child.json'), JSON.stringify(parents));
    return out;
  } },
};

async function main() {
  const only = process.argv[2];
  const ex = await pipeline('feature-extraction', MODEL);
  for (const [name, m] of Object.entries(METHODS)) {
    if (only && only !== name) continue;
    const chunks = m.build();
    fs.writeFileSync(path.join(__dirname, `chunks_trial_${name}.json`), JSON.stringify(chunks));
    const cacheFile = path.join(__dirname, m.file);
    let vecs = fs.existsSync(cacheFile) ? JSON.parse(fs.readFileSync(cacheFile, 'utf-8')) : [];
    console.log(`${name}: ${chunks.length}块，已有${vecs.length}`);
    const t0 = Date.now(), n0 = vecs.length; // 倒计时：用已耗时间推算剩余
    for (let i = vecs.length; i < chunks.length; i++) {
      const o = await ex(chunks[i].context.slice(0, 1500), { pooling: 'mean', normalize: true });
      vecs.push(Array.from(o.data));
      if ((i + 1) % 100 === 0 || i + 1 === chunks.length) {
        fs.writeFileSync(cacheFile, JSON.stringify(vecs));
        const el = (Date.now() - t0) / 1000, speed = (i + 1 - n0) / Math.max(el, 0.1);
        const eta = Math.round((chunks.length - (i + 1)) / speed);
        console.log(`  ${name} ${i + 1}/${chunks.length} 已用${Math.round(el)}s，预计还剩${Math.round(eta / 60)}分${eta % 60}秒`);
      }
    }
  }
  console.log('done');
}
main().catch((e) => { console.error(e.message); process.exit(1); });
