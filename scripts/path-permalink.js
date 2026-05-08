'use strict';

// Permalink 派生自 _posts/ 下的相对路径,让生成的站点目录结构 1:1 映射源文件层级。
//   _posts/AI_Infra/DDP.md                                  ->  /AI_Infra/DDP/
//   _posts/ACM-ICPC/Competition_Records/2018-09-12-x.md     ->  /ACM-ICPC/Competition_Records/2018-09-12-x/
//
// 配合 _config.yml 中 `permalink: :dirpath/` 使用。
hexo.extend.filter.register('before_post_render', function (data) {
  if (!data.source) return data;
  const normalized = data.source.replace(/\\/g, '/');
  const match = /^_posts\/(.+)\.md$/i.exec(normalized);
  if (match) {
    data.dirpath = match[1];
  }
  return data;
});
