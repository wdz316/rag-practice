// 最小 RAG 演示：检索版（无须大模型 API）
// 用法：node demo.cjs "你的问题"
// 原理：关键词重叠度检索 trial 数据 -> 展示 Top3 文档片段 + 标准答案
const fs = require('fs');
const path = require('path');

const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'cmrc2018_trial.json'), 'utf-8'));

// 把所有段落拍平成待检索的块
const chunks = [];
for (const doc of data.data) {
  for (const p of doc.paragraphs) {
    chunks.push({ title: doc.title, context: p.context, qas: p.qas });
  }
}

// 极简中文分词：按字符二元组 + 原始问句词做重叠打分
function tokenize(s) {
  s = s.replace(/\s+/g, '');
  const set = new Set();
  for (let i = 0; i < s.length; i++) set.add(s[i]);
  for (let i = 0; i < s.length - 1; i++) set.add(s.slice(i, i + 2));
  return set;
}

function score(queryTokens, context) {
  const ctxTokens = tokenize(context);
  let hit = 0;
  for (const t of queryTokens) if (ctxTokens.has(t)) hit += t.length > 1 ? 2 : 1;
  return hit / Math.sqrt(context.length); // 除以长度做简单归一，避免长文档占便宜
}

function retrieve(query, topK = 3) {
  const qt = tokenize(query);
  return chunks
    .map((c) => ({ ...c, s: score(qt, c.context) }))
    .sort((a, b) => b.s - a.s)
    .slice(0, topK);
}

const query = process.argv.slice(2).join(' ') || '生命数耗完即算为什么？';
const hits = retrieve(query, 3);

console.log('问题：' + query + '\n');
hits.forEach((h, i) => {
  console.log(`【检索 Top${i + 1}】《${h.title}》(分数 ${h.s.toFixed(2)})`);
  console.log('片段：' + h.context.slice(0, 120) + '...\n');
});

// 如果检索到的块里正好有 gold 问答，顺带展示，模拟“生成”阶段的依据
const gold = hits[0].qas.find((q) => q.question.includes(query.slice(0, 6)) || query.includes(q.question.slice(0, 6)));
if (gold) {
  console.log('参考答案（该片段自带标准答案，完整 RAG 会换成大模型组织语言）：' + gold.answers[0].text);
} else {
  console.log('(注：Top1 片段的标准问题与你问的不完全一致，完整 RAG 会把上面 3 个片段一起喂给大模型来组织答案)');
}
console.log('\n共检索库：' + chunks.length + ' 个片段');
