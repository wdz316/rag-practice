// PDF 文本抽取（pdf-parse v2 API）
const fs = require('fs');
const { PDFParse } = require('pdf-parse');

(async () => {
  const data = fs.readFileSync('./moutai2024.pdf');
  const parser = new PDFParse({ data });
  const result = await parser.getText();
  console.log('页数：' + result.pages.length);
  console.log('总字符：' + result.text.length);
  fs.writeFileSync('./moutai2024.txt', result.text, 'utf-8');
  console.log('已存 moutai2024.txt');
  const i = result.text.indexOf('主要会计数据');
  console.log('---“主要会计数据”附近500字（表格破坏现场）---');
  console.log(result.text.slice(i, i + 500));
})();
