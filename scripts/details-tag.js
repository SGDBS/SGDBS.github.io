'use strict';

// 直接调用 kramed 本体，完全绕开 Hexo 的渲染管道（escapeAllSwigTags / restoreAllSwigTags），
// 避免 tag 函数内嵌套 renderSync 导致 stored[] 缓存索引错乱的 ERR_ASSERTION 崩溃。
const kramed = require('kramed');

hexo.extend.tag.register('details', function (args, content) {
  const summary = args.join(' ');
  const rendered = kramed(content);
  return `<details><summary>${summary}</summary>${rendered}</details>`;
}, { ends: true });
