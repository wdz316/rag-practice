// 茅台正文6切分方法 + 向量化（断点续算，按切分方法缓存，带倒计时）
// 表线不动：269张整表从v1直接切片共享（零重算），只算正文向量
// 用法：node methods_moutai.cjs [fixed500|para|recur|punct|struct|child]（不带参数全跑）
const fs = require('fs');
const path = require('path');
const { pipeline } = require('@xenova/transformers');

const MODEL = 'Xenova/bge-small-zh-v1.5';
const full = fs.readFileSync(path.join(__dirname, 'moutai2024.txt'), 'utf-8');
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
// 结构法标题： 第X节 / 一、 / （一） / 1. / 附表
const HEAD = /^(第[一二三四五六七八九十]+节|^[一二三四五六七八九十]+、|^[（(][一二三四五六七八九十]+[)）]|^\d+\.|附表)/;

const METHODS = {
  fixed500: { file: 'emb_moutai_fixed500.json', build() {
    const out = [];
    for (let s = 0, k = 0; s < full.length; s += 450, k++) {
      const t = full.slice(s, s + 500).trim();
      if (t.length >= 100) out.push({ title: `moutai2024#MF${k}`, context: t });
    }
    return out;
  } },
  para: { file: 'emb_moutai_para.json', build() {
    return full.split(/\n\s*\n/).map((p) => p.trim()).filter((p) => p.length >= 50)
      .map((p, k) => ({ title: `moutai2024#MP${k}`, context: p }));
  } },
  recur: { file: 'emb_moutai_recur.json', build() {
    const out = [];
    full.split(/\n\s*\n/).forEach((p) => {
      p = p.trim();
      if (!p) return;
      splitSent(p).forEach((s) => {
        if (s.length <= 500) out.push({ title: `moutai2024#MR${out.length}`, context: s, _m: true });
        else s.split(/(?<=[；，、])/).forEach((ss) => out.push({ title: `moutai2024#MR${out.length}`, context: ss.trim() }));
      });
    });
    const merged = [];
    let cur = null;
    for (const c of out) {
      if (c._m && cur && (cur.context + c.context).length <= 500) cur.context += c.context;
      else { if (cur) { delete cur._m; merged.push(cur); } cur = { ...c }; }
    }
    if (cur) { delete cur._m; merged.push(cur); }
    return merged;
  } },
  punct: { file: 'emb_moutai_punct.json', build() {
    const out = [];
    full.split(/\n\s*\n/).forEach((p) => {
      p = p.trim();
      if (!p) return;
      pack(splitSent(p), 600).forEach((part) => {
        if (part.length < 300 && out.length) out[out.length - 1].context += part;
        else if (part.length >= 50) out.push({ title: `moutai2024#MS${out.length}`, context: part });
      });
    });
    return out;
  } },
  struct: { file: 'emb_moutai_struct.json', build() {
    // 按标题切节，每块前缀章节路径（面包屑）
    const secs = [];
    let curPath = ['全文'], curBuf = [];
    for (const line of full.split('\n')) {
      const t = line.trim();
      if (HEAD.test(t) && t.length < 60) {
        if (curBuf.join('\n').trim()) secs.push({ path: [...curPath], text: curBuf.join('\n').trim() });
        if (/^第/.test(t)) curPath = [t];
        else curPath = [curPath[0], t];
        curBuf = [line];
      } else curBuf.push(line);
    }
    if (curBuf.join('\n').trim()) secs.push({ path: [...curPath], text: curBuf.join('\n').trim() });
    const out = [];
    secs.forEach((s, k) => {
      const head = s.path.join(' > ') + '\n';
      if (s.text.length <= 1500) out.push({ title: `moutai2024#MH${k}`, context: head + s.text });
      else pack(splitSent(s.text), 600).forEach((part, j) => out.push({ title: `moutai2024#MH${k}_${j}`, context: head + part }));
    });
    return out.filter((c) => c.context.length >= 100);
  } },
  child: { file: 'emb_moutai_child.json', meta: 'parents_moutai_child.json', build() {
    const out = [], parents = {};
    let pn = 0;
    for (let s = 0; s < full.length; s += 400) {
      const p = full.slice(s, s + 500).trim();
      if (p.length < 100) continue;
      const pid = `moutai2024#MD${pn++}`;
      parents[pid] = { title: '茅台年报', context: p };
      for (let o = 0, k = 0; o < p.length; o += 200, k++) {
        const sub = p.slice(o, o + 200).trim();
        if (sub.length >= 30) out.push({ title: `${pid.replace('#MD', '#MC')}_${k}`, context: sub, parent: pid });
      }
    }
    fs.writeFileSync(path.join(__dirname, 'parents_moutai_child.json'), JSON.stringify(parents));
    return out;
  } },
};

async function main() {
  const only = process.argv[2];
  const ex = await pipeline('feature-extraction', MODEL);
  for (const [name, m] of Object.entries(METHODS)) {
    if (only && only !== name) continue;
    const chunks = m.build();
    fs.writeFileSync(path.join(__dirname, `chunks_moutai_${name}.json`), JSON.stringify(chunks));
    const cacheFile = path.join(__dirname, m.file);
    let vecs = fs.existsSync(cacheFile) ? JSON.parse(fs.readFileSync(cacheFile, 'utf-8')) : [];
    console.log(`${name}: ${chunks.length}块，已有${vecs.length}`);
    const t0 = Date.now(), n0 = vecs.length;
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
