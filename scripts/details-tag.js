'use strict';

hexo.extend.tag.register('details', function (args, content) {
  const summary = args.join(' ');
  const rendered = hexo.render.renderSync({ text: content, engine: 'markdown' });
  return `<details><summary>${summary}</summary>${rendered}</details>`;
}, { ends: true });
