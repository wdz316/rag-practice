// RAG仪表盘后端 v2：三池切换 + 空查拦截 + token计数 + 每次查询自动落表 + 切分方法切换 + /api/apply
// 用法：node server.cjs → 打开 http://localhost:8080/rag-dash.html
// 切块是建库时定的：仪表盘只能切换"已切好的库"，不能运行时改切块（切块锁定原则）
const fs = require('fs');
const path = require('path');
const http = require('http');
const { pipeline } = require('@xenova/transformers');

const PORT = 8080;
const ROOT = __dirname;
const PDFD = path.join(__dirname, '..', 'pdf-practice');

const opt = (args, n, d) => {
  const i = args.indexOf(n);
  return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : d;
};

const METHODS = {
  fixed500: '固定500字：按固定字数切分，好比豆腐刀，快但可能从句中切断',
  para: '按自然段：按作者原有的段落切分，一段一块',
  recur: '递归切分：按段落→句子→分号逐级切分，大块拆小直到达标，不断句',
  punct: '标点语义边界：按句号断句，块长度保持在300–600字之间',
  struct: '按文档结构：标题与段落组合切分，每块前面带上标题（好比每页盖公章）',
  child: '父子块：用200字小块检索，命中后返回它所属的整段（小块定位、大块给料）',
};
const POOLS = {
  trial: {
    name: 'trial·中文百科', lang: 'zh', model: 'Xenova/bge-small-zh-v1.5',
    chunkRule: '6切分方法可切（见切分方法下拉），基线256块',
    methods: true,
    load(method = 'para') {
      const chunks = JSON.parse(fs.readFileSync(path.join(ROOT, `chunks_trial_${method}.json`), 'utf-8'))
        .map((c) => ({ title: c.title.split('#')[0], context: c.context, _id: c.title, parent: c.parent }));
      let parents = null;
      if (method === 'child') parents = JSON.parse(fs.readFileSync(path.join(ROOT, 'parents_trial_child.json'), 'utf-8'));
      return { chunks, vecs: JSON.parse(fs.readFileSync(path.join(ROOT, `emb_trial_${method}.json`), 'utf-8')), parents };
    },
  },
  moutai: {
    name: 'moutai·茅台年报', lang: 'zh', model: 'Xenova/bge-small-zh-v1.5',
    chunkRule: '7切分方法可切（正文）+ 269张表共享（行拆旧池保留为tablerow）',
    methods: { ...METHODS, tablerow: '行拆旧池：表>10行按行拆+首行贴头+文500滑窗（保留对照，不用现切）' },
    load(method = 'tablerow') {
      if (method === 'tablerow') {
        // 旧池原样保留：行拆+滑窗，不用现切
        const chunks = fs.readFileSync(path.join(PDFD, 'moutai_chunks_v2.jsonl'), 'utf-8').split('\n').filter(Boolean).map((l) => {
          const c = JSON.parse(l);
          return { title: c.chunk_id, context: c.text };
        });
        return { chunks, vecs: JSON.parse(fs.readFileSync(path.join(PDFD, 'moutai_embeddings_v2.json'), 'utf-8')) };
      }
      // 6新法：正文现切 + 269张整表共享（从v1切片，零重算）
      const texts = JSON.parse(fs.readFileSync(path.join(PDFD, `chunks_moutai_${method}.json`), 'utf-8'))
        .map((c) => ({ title: '茅台年报', context: c.context, _id: c.title, parent: c.parent }));
      const tvecs = JSON.parse(fs.readFileSync(path.join(PDFD, `emb_moutai_${method}.json`), 'utf-8'));
      const v1chunks = fs.readFileSync(path.join(PDFD, 'moutai_chunks.jsonl'), 'utf-8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
      const v1vecs = JSON.parse(fs.readFileSync(path.join(PDFD, 'moutai_embeddings.json'), 'utf-8'));
      const tables = [], tvec = [];
      v1chunks.forEach((c, i) => {
        if (c.chunk_type === 'table') { tables.push({ title: c.chunk_id, context: c.text }); tvec.push(v1vecs[i]); }
      });
      let parents = null;
      if (method === 'child') parents = JSON.parse(fs.readFileSync(path.join(PDFD, 'parents_moutai_child.json'), 'utf-8'));
      return { chunks: [...texts, ...tables], vecs: [...tvecs, ...tvec], parents };
    },
  },
  tatqa: {
    name: 'tatqa1634·英文财报表文', lang: 'en', model: 'Xenova/bge-small-en-v1.5',
    chunkRule: '表整存转markdown，段落原生，共1634块（dev）',
    load() {
      const chunks = fs.readFileSync(path.join(ROOT, 'tatqa', 'tatqa_dev_chunks.jsonl'), 'utf-8').split('\n').filter(Boolean).map((l) => {
        const c = JSON.parse(l);
        return { title: c.chunk_id, context: c.text };
      });
      return { chunks, vecs: JSON.parse(fs.readFileSync(path.join(ROOT, 'tatqa', 'tatqa_dev_embeddings.json'), 'utf-8')) };
    },
  },
};
const cache = {};
const jobs = {}; // 后台现切任务：id -> {status, done, total, pool, method, error, result}
// 自定义池注册表（本地目录导入，持久化，重启不丢）
const REG = path.join(ROOT, 'custom_pools.json');
function loadRegistry() {
  try {
    if (fs.existsSync(REG)) {
      for (const r of JSON.parse(fs.readFileSync(REG, 'utf-8'))) {
        POOLS[r.id] = {
          name: r.name, lang: 'zh', model: 'Xenova/bge-small-zh-v1.5',
          chunkRule: r.chunkRule, custom: true,
          load() {
            const chunks = JSON.parse(fs.readFileSync(path.join(ROOT, r.chunksFile), 'utf-8'));
            return { chunks, vecs: JSON.parse(fs.readFileSync(path.join(ROOT, r.vecFile), 'utf-8')) };
          },
        };
      }
    }
  } catch (e) { console.error('注册表损坏，已跳过：' + e.message); }
}
loadRegistry();
function getPool(id, method) {
  const key = id + ':' + (method || '');
  if (!cache[key]) {
    const P = POOLS[id];
    const { chunks, vecs, parents } = P.load(method);
    if (vecs.length !== chunks.length) throw new Error(id + '向量与块数对不上');
    cache[key] = { ...P, chunks, vecs, parents: parents || null };
  }
  return cache[key];
}
const models = {};
async function getExtractor(model) {
  if (!models[model]) models[model] = await pipeline('feature-extraction', model);
  return models[model];
}

// 中文按字、英文按词的字面命中分（关键词那条腿，不经过向量模型）
function kwScoreZh(qt, context) {
  const s = context.replace(/\s+/g, '');
  const set = new Set();
  for (let i = 0; i < s.length; i++) set.add(s[i]);
  for (let i = 0; i < s.length - 1; i++) set.add(s.slice(i, i + 2));
  let hit = 0;
  for (const t of qt) if (set.has(t)) hit += t.length > 1 ? 2 : 1;
  return hit / Math.sqrt(context.length);
}
function kwScoreEn(qw, context) {
  const set = new Set((context.toLowerCase().match(/[a-z0-9%$.\-]+/g) || []));
  let hit = 0;
  for (const w of qw) if (set.has(w)) hit += w.length >= 5 ? 2 : 1;
  return hit / Math.sqrt(context.length);
}
function cosSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
function euclidSim(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
  return 1 - Math.sqrt(s) / 2;
}
// token预估：中文字符×0.9，英文单词×1.3（量级参考，非计费口径）
function estTokens(text, lang) {
  if (lang === 'en') return Math.round(((text.match(/[a-zA-Z0-9%$.\-]+/g) || []).length) * 1.3);
  return Math.round(text.replace(/\s+/g, '').length * 0.9);
}

const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8' };
const HIST = path.join(ROOT, 'experiments.jsonl');
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://x');
  try {
    if (url.pathname === '/api/pools' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(Object.entries(POOLS).map(([id, p]) => ({ id, name: p.name, chunkRule: p.chunkRule, methods: p.methods === true ? METHODS : (p.methods || null) }))));
      return;
    }
    if (req.method === 'POST' && url.pathname === '/api/apply') {
      // 应用切分：向量就绪直接返回；缺失则起后台任务现切现算，前端轮询/api/job看进度条
      let body = '';
      for await (const ch of req) body += ch;
      const { pool = 'trial', method = 'para' } = JSON.parse(body || '{}');
      if (!POOLS[pool]) throw new Error('未知库');
      const needBuild = pool === 'moutai' && method !== 'tablerow' &&
        (!fs.existsSync(path.join(PDFD, `chunks_moutai_${method}.json`)) ||
         !fs.existsSync(path.join(PDFD, `emb_moutai_${method}.json`)));
      if (needBuild) {
        const id = 'job_' + Date.now().toString(36);
        jobs[id] = { status: 'running', done: 0, total: 0, pool, method, error: null };
        const { spawn } = require('child_process');
        const child = spawn(process.execPath, ['methods_moutai.cjs', method], { cwd: PDFD });
        let buf = '';
        child.stdout.on('data', (d) => {
          buf += d.toString();
          // 进度行如 "  recur 300/703 ..." 或首行 "recur: 703块"
          let m = buf.match(/(\d+)\/(\d+)\s*(已用|，|$)/) || buf.match(/(\d+)\/(\d+)/);
          const total = buf.match(/(\d+)块，已有\d*/);
          const j = jobs[id];
          if (!j) return;
          if (total) j.total = +total[1];
          if (m) { j.done = +m[1]; if (!j.total) j.total = +m[2]; }
          buf = buf.slice(-200);
        });
        child.on('error', (e) => { if (jobs[id]) { jobs[id].status = 'error'; jobs[id].error = e.message; } });
        child.on('close', (code) => {
          const j = jobs[id];
          if (!j) return;
          if (code !== 0) { j.status = 'error'; j.error = '现切进程退出码' + code; return; }
          try {
            delete cache[pool + ':' + method];
            const P = getPool(pool, method);
            const avg = Math.round(P.chunks.reduce((a, c) => a + c.context.length, 0) / P.chunks.length);
            j.status = 'done'; j.done = j.total || P.chunks.length; j.total = P.chunks.length;
            j.result = { ok: true, pool, method, chunks: P.chunks.length, avgChars: avg };
          } catch (e) { j.status = 'error'; j.error = e.message; }
        });
        res.writeHead(202, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ job: id }));
        return;
      }
      const P = getPool(pool, POOLS[pool].methods ? method : undefined);
      const avg = Math.round(P.chunks.reduce((a, c) => a + c.context.length, 0) / P.chunks.length);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, pool, method: POOLS[pool].methods ? method : '-', chunks: P.chunks.length, avgChars: avg }));
      return;
    }
    if (url.pathname === '/api/job' && req.method === 'GET') {
      const id = url.searchParams.get('id') || '';
      const j = jobs[id];
      if (!j) { res.writeHead(404, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ error: '任务不存在' })); return; }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(j));
      return;
    }
    if (req.method === 'POST' && url.pathname === '/api/search') {
      let body = '';
      for await (const ch of req) body += ch;
      const { query = '', topK = 5, alpha = 0.7, metric = 'cosine', pool = 'trial', method = 'para' } = JSON.parse(body || '{}');
      if (!query.trim()) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: '请先输入问题' }));
        return;
      }
      if (!POOLS[pool]) throw new Error('未知库');
      const P = getPool(pool, POOLS[pool].methods ? method : undefined);
      const t0 = Date.now();
      const ex = await getExtractor(P.model);
      const qv = Array.from((await ex(query, { pooling: 'mean', normalize: true })).data);
      const simFn = metric === 'euclid' ? euclidSim : cosSim;
      const kwFn = P.lang === 'en' ? kwScoreEn : kwScoreZh;
      const qt = P.lang === 'en'
        ? (query.toLowerCase().match(/[a-z0-9%$.\-]+/g) || []).filter((w) => w.length > 1)
        : (() => { const s = query.replace(/\s+/g, ''); const set = new Set(); for (let i = 0; i < s.length; i++) set.add(s[i]); for (let i = 0; i < s.length - 1; i++) set.add(s.slice(i, i + 2)); return set; })();
      const ks = P.chunks.map((c) => kwFn(qt, c.context));
      const vs = P.chunks.map((v, i) => simFn(qv, P.vecs[i]));
      const norm = (arr) => {
        const mn = Math.min(...arr), mx = Math.max(...arr);
        return arr.map((v) => (mx === mn ? 0 : (v - mn) / (mx - mn)));
      };
      const nk = norm(ks), nv = norm(vs);
      const k = Math.max(1, Math.min(20, topK));
      const top = P.chunks.map((c, i) => {
        // 父子块：检索命中的是子块，展示与计费按父段（检得准、给得全）
        const shown = (method === 'child' && P.methods && P.parents && P.parents[c.parent])
          ? { title: P.parents[c.parent].title, context: P.parents[c.parent].context, via: c._id }
          : { title: c.title, context: c.context, via: null };
        const chars = shown.context.length;
        // 标题后缀块ID：同名块（如茅台年报#MP3）靠它区分，否则Top1列全一个样
        const tag = c._id && c._id.includes('#') ? '#' + c._id.split('#').pop() : '';
        const fullTitle = shown.title + tag;
        return {
          title: fullTitle.length > 40 ? fullTitle.slice(0, 40) + '…' : fullTitle,
          score: +(alpha * nv[i] + (1 - alpha) * nk[i]).toFixed(4),
          sv: +vs[i].toFixed(4), sk: +ks[i].toFixed(4),
          chars, tokens: estTokens(shown.context, P.lang),
          snippet: shown.context.slice(0, 120),
          via: shown.via,
        };
      }).sort((a, b) => b.score - a.score).slice(0, k);
      const totalTokens = top.reduce((a, h) => a + h.tokens, 0);
      const ms = Date.now() - t0;
      // top1sv记原始向量分：混合分是归一化逼出来的，第一名恒为1.0，无信息量
      const rec = { time: new Date().toISOString(), pool, method: POOLS[pool].methods ? method : '-', q: query.slice(0, 30), topK: k, alpha, metric, top1: top[0].title.slice(0, 20), top1s: top[0].score, top1sv: top[0].sv, ms, tokens: totalTokens };
      fs.appendFileSync(HIST, JSON.stringify(rec) + '\n'); // 每次查询自动落表
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ query, topK: k, alpha, metric, pool, method: POOLS[pool].methods ? method : '-', poolName: P.name, ms, poolSize: P.chunks.length, totalTokens, top }));
      return;
    }
    if (url.pathname === '/api/runs' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(fs.existsSync(HIST) ? fs.readFileSync(HIST, 'utf-8').split('\n').filter(Boolean).map((l) => JSON.parse(l)) : []));
      return;
    }
    if (url.pathname === '/api/runs' && req.method === 'DELETE') {
      // 清空历史记录（experiments.jsonl 清零，文件保留）
      fs.writeFileSync(HIST, '');
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true }));
      return;
    }
    if (req.method === 'POST' && (url.pathname === '/api/import' || url.pathname === '/api/import_texts')) {
      // 本地导入两种姿势：①填本机目录（服务端读文件）；②页面点选目录（浏览器读内容POST过来，跨机器也行）
      // 统一：txt/md，500字滑窗100重叠，<50字丢弃；限制50文件/合计5MB
      let body = '';
      for await (const ch of req) body += ch;
      const parsed = JSON.parse(body || '{}');
      let files = [];
      if (url.pathname === '/api/import_texts') {
        if (!Array.isArray(parsed.files) || !parsed.files.length) throw new Error('没收到文件，请重新选择目录');
        files = parsed.files
          .filter((f) => /\.(txt|md)$/i.test(f.name || ''))
          .slice(0, 50)
          .map((f) => ({ name: (f.name || '未命名').split('/').pop(), text: String(f.text || '') }));
        if (!files.length) throw new Error('所选目录里没有 .txt/.md 文件（PDF请先转成文本再导）');
      } else {
        const { dir = '' } = parsed;
        if (!dir.trim() || !fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) throw new Error('目录不存在，请填本机文件夹完整路径');
        files = fs.readdirSync(dir).filter((f) => /\.(txt|md)$/i.test(f)).slice(0, 50)
          .map((f) => ({ name: f, text: fs.readFileSync(path.join(dir, f), 'utf-8') }));
        if (!files.length) throw new Error('目录里没有 .txt/.md 文件（PDF请先转成文本再导）');
      }
      const total = files.reduce((a, f) => a + f.text.length, 0);
      if (total > 5 * 1024 * 1024) throw new Error('文件合计超5MB，先拆小再导');
      const chunks = [];
      for (const { name, text } of files) {
        for (let s = 0, k = 0; s < text.length; s += 400, k++) {
          const part = text.slice(s, s + 500).trim();
          if (part.length < 50) continue;
          chunks.push({ title: `${name}#${k}`, context: part });
        }
      }
      const crypto = require('crypto');
      const idSeed = url.pathname === '/api/import_texts'
        ? files.map((f) => f.name + ':' + f.text.length).join('|')
        : parsed.dir;
      const id = 'custom_' + crypto.createHash('md5').update(idSeed).digest('hex').slice(0, 8);
      const chunksFile = `chunks_${id}.json`, vecFile = `emb_${id}.json`;
      fs.writeFileSync(path.join(ROOT, chunksFile), JSON.stringify(chunks));
      const ex = await getExtractor('Xenova/bge-small-zh-v1.5');
      const vecs = [];
      for (let i = 0; i < chunks.length; i++) {
        const o = await ex(chunks[i].context.slice(0, 1500), { pooling: 'mean', normalize: true });
        vecs.push(Array.from(o.data));
      }
      fs.writeFileSync(path.join(ROOT, vecFile), JSON.stringify(vecs));
      const name = url.pathname === '/api/import_texts'
        ? '自建池·页面勾选(' + files.length + '个文件)'
        : '自建池·' + path.basename(parsed.dir);
      POOLS[id] = {
        name, lang: 'zh', model: 'Xenova/bge-small-zh-v1.5',
        chunkRule: `本地目录导入（${files.length}个文件，500字滑窗100重叠），共${chunks.length}块`,
        custom: true,
        load() {
          return {
            chunks: JSON.parse(fs.readFileSync(path.join(ROOT, chunksFile), 'utf-8')),
            vecs: JSON.parse(fs.readFileSync(path.join(ROOT, vecFile), 'utf-8')),
          };
        },
      };
      const reg = fs.existsSync(REG) ? JSON.parse(fs.readFileSync(REG, 'utf-8')) : [];
      fs.writeFileSync(REG, JSON.stringify([...reg.filter((r) => r.id !== id), { id, name, chunkRule: POOLS[id].chunkRule, chunksFile, vecFile }]));
      delete cache[id + ':']; // 索引更新必须清内存缓存，否则查到的是旧向量（刚现抓一条）
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, id, name, files: files.length, chunks: chunks.length }));
      return;
    }
    const allow = new Set(['rag-dash.html', 'chart.min.js']);
    const name = decodeURIComponent(url.pathname.replace(/^\//, '')) || 'rag-dash.html';
    if (!allow.has(name)) { res.writeHead(404); res.end('not found'); return; }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(name)] || 'text/plain', 'Cache-Control': 'no-store' });
    res.end(fs.readFileSync(path.join(ROOT, name)));
  } catch (e) {
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: e.message }));
  }
});
server.listen(PORT, () => console.log('仪表盘 http://localhost:' + PORT + '/rag-dash.html'));
