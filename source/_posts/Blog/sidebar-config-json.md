---
title: 用 config.json 自定义侧边栏分组名称
categories: 博客搞建
date: 2026-04-20 19:00:00
tags:
    - Hexo
    - 教程
    - EJS
    - JavaScript
---

本文记录了为分类页侧边栏增加「自定义分组名称」功能的完整实现。只需在子文件夹中放一个 `config.json`，侧边栏就会用其中定义的 `name` 字段替换原始文件夹名，无需修改任何模板。

---

## 1. 需求背景

之前实现的侧边栏已经能按文件夹层级自动分组，但分组标签直接取自文件夹名——例如 `Dynamic_Programming`、`Competition_Records`。这些英文名对读者不够友好，但又不想重命名文件夹（会影响 URL 和 git 历史）。

目标：在不改变文件夹名的前提下，为每个子文件夹配置一个可读的中文显示名。

---

## 2. 设计方案

在每个子文件夹内放一个 `config.json`：

```json
{ "name": "动态规划" }
```

侧边栏渲染时，若该文件夹有 `config.json` 且其中有 `name` 字段，则用该值作为分组标题；否则回退到文件夹名。

这样文件系统仍是唯一事实来源，`config.json` 只是附加的显示层，不影响 URL 或任何其他逻辑。

---

## 3. 实现

### 3.1 为什么不能在 EJS 里直接读文件

Hexo 的 EJS 模板运行在沙箱环境中，`require` 不可用。如果在 `category.ejs` 里写：

```js
const fs = require('fs');  // ReferenceError: require is not defined
```

会直接崩溃。所有文件 I/O 必须在 `scripts/` 目录下的脚本中完成，通过 Hexo 提供的 API 将结果注入到模板可访问的上下文里。

### 3.2 scripts/sidebar-folder-config.js

在 `scripts/` 下新建 `sidebar-folder-config.js`：

```javascript
'use strict';
const fs   = require('fs');
const path = require('path');

hexo.locals.set('sidebarFolderConfigs', function () {
    const postsDir = path.join(hexo.source_dir, '_posts');
    const configs  = {};

    function walk(dir) {
        if (!fs.existsSync(dir)) return;
        for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
            const full = path.join(dir, entry.name);
            if (entry.isDirectory()) {
                walk(full);
            } else if (entry.name === 'config.json') {
                try {
                    const cfg = JSON.parse(fs.readFileSync(full, 'utf8'));
                    const rel = path.relative(postsDir, dir).replace(/\\/g, '/');
                    configs[rel] = cfg;
                } catch (e) { /* 忽略格式损坏的 config.json */ }
            }
        }
    }

    walk(postsDir);
    return configs;
});
```

**关键点：**

- `hexo.locals.set(key, fn)` 注册一个**懒加载 getter**。Hexo 在每次构建时调用 `fn()`，结果以 `site.sidebarFolderConfigs` 的形式暴露给所有 EJS 模板。
- `walk` 递归遍历整个 `_posts/` 目录树，找到所有 `config.json`。
- 键为相对 `_posts/` 的路径（如 `ACM-ICPC/Dynamic_Programming`），值为解析后的 JSON 对象。
- `replace(/\\/g, '/')` 处理 Windows 下 `path.relative` 返回反斜杠的问题。

### 3.3 在 category.ejs 中使用

在侧边栏分组数据构建阶段，取出这份配置表，并在生成 `label` 时优先使用 `cfg.name`：

```ejs
<%
var folderConfigs = site.sidebarFolderConfigs || {};
var folderRelPaths = {};   /* folderKey → 相对 _posts 的路径 */

/* ... 遍历 allPosts 建立 folderGroups、folderRelPaths ... */

folderOrder.forEach(function(k) {
    var cfg   = folderConfigs[folderRelPaths[k]] || {};
    var label = cfg.name || k;           /* 有 name 用 name，否则用文件夹名 */
    sidebarGroups.push({ label: label, posts: folderGroups[k] });
});
%>
```

`folderRelPaths[k]` 存的是该分组对应的相对路径（建立 `folderGroups` 时同步记录），与 `folderConfigs` 的键格式完全匹配，直接查表即可。

---

## 4. config.json 规范

文件放在对应子文件夹的**直接层级**下：

```
source/_posts/
└── ACM-ICPC/
    ├── Dynamic_Programming/
    │   ├── config.json          ← {"name": "动态规划"}
    │   ├── post-a.md
    │   └── post-b.md
    └── Graph_Theory/
        ├── config.json          ← {"name": "图论"}
        └── post-c.md
```

目前只用到 `name` 字段，格式极简，未来可以扩展（如排序权重、图标等）而无需修改脚本的核心逻辑。

**注意：** 纯图片文件夹（即帖子同名资源文件夹，内部只有图片，没有 `.md`）不需要也不应该放 `config.json`——它们不会产生任何 post，永远不会出现在侧边栏里。

---

## 5. 踩过的坑

### 5.1 模板里 require 不可用

第一反应是在 `category.ejs` 里直接 `require('fs')` 读文件，报 `ReferenceError`。Hexo EJS 运行在沙箱里，Node.js 的全局 `require` 不在作用域内。解决方案是把所有文件操作移到 `scripts/` 脚本，再通过 `hexo.locals.set` 传给模板。

### 5.2 Windows 路径分隔符

`path.relative()` 在 Windows 下返回 `ACM-ICPC\Dynamic_Programming`（反斜杠），而 `post.source` 经过 `.replace(/\\/g, '/')` 后是正斜杠。两者格式不一致导致查表永远命中不了。在 `sidebar-folder-config.js` 里加一行 `.replace(/\\/g, '/')` 统一格式即可。

---

## 6. 最终效果

各子文件夹 `config.json` 配置如下：

| 文件夹路径 | config.json name |
|:--|:--|
| `ACM-ICPC/Competition_Records` | 竞赛题解 |
| `ACM-ICPC/Dynamic_Programming` | 动态规划 |
| `ACM-ICPC/Graph_Theory` | 图论 |
| `ACM-ICPC/Data_Structure` | 数据结构 |
| `ACM-ICPC/Math` | 数学 |
| `ACM-ICPC/Greedy` | 贪心算法 |
| `ACM-ICPC/Game_Theory` | 博弈论 |
| `Recommendation/Study_Notes` | 学习笔记 |
| `Recommendation/Paper_Reading` | 论文阅读 |
| `Diary/2026` | 2026年 |

侧边栏分组标题全部显示为中文，文件夹名保持英文不变，URL 和 git 历史完全不受影响。
