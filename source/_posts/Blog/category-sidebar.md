---
title: 为分类页面实现固定侧边栏：全量文章导航与文件夹层级分组
categories: 博客搞建
date: 2026-04-20 18:00:00
tags:
    - Hexo
    - 教程
    - EJS
    - CSS
---

本文记录了为博客分类页面（`/categories/xxx/`）新增固定侧边栏的完整实现过程。侧边栏固定在视口左侧，显示当前分类下的**全部文章**（不受分页限制），按文件夹层级自动分组（支持两级嵌套），支持折叠展开，分组默认收起。

---

## 1. 需求背景

博客的 ACM-ICPC 分类下有 100+ 篇文章，分 10 余页展示。在某一页翻看题解时，完全不知道其他文章的全貌，跳转也很不方便。

目标是：在分类页面左侧常驻一个导航栏，列出该分类下所有文章的标题，点击可直接跳转，且不占用主内容区域的宽度。对于有两级子文件夹的分类（如 `Recommendation/Study_Notes/Chapter1/`），侧边栏应呈现对应的两级嵌套结构。

---

## 2. 整体设计

| 要素 | 方案 |
|------|------|
| 定位 | `position: fixed; left: 0` 固定在视口左边缘 |
| 高度 | `height: calc(100vh - 64px)`，撑满导航栏以下的全部空间 |
| 文章来源 | `site.categories.findOne({name: page.category}).posts`，取全量而非当前页 |
| 分组逻辑 | 解析 `post.source` 路径，构建两级树：L1 = 第一级子文件夹，L2 = 第二级子文件夹 |
| 显示名称 | 子文件夹内有 `config.json` 则用其 `name` 字段，否则用文件夹名 |
| 折叠交互 | 纯 JS，点击组标题切换 `collapsed` class，默认全部收起 |

关键决策：侧边栏必须放在 `<main>` **之外**。若放在 `<main>` 内部，父元素的 `overflow` 或 CSS 变换可能导致 `position: fixed` 失效，表现为侧边栏跟随页面滚动或位置偏移。

---

## 3. 数据构建：全量文章 + 两级树

### 3.1 获取全量文章

`page.posts` 只包含当前页的文章（受分页限制）。要在侧边栏显示该分类下的**所有**文章，需要通过 `site.categories` 查询：

```ejs
<%
var catObj = site.categories.findOne({name: page.category});
var allPosts = catObj
    ? catObj.posts.sort('date').reverse().toArray()
    : page.posts.toArray();
%>
```

这样无论用户在第几页，侧边栏始终显示完整文章列表。

### 3.2 按路径构建两级树

每个 post 对象的 `source` 属性包含其源文件路径，例如：

```
_posts/Recommendation/Study_Notes/Chapter2/Chapter2.3.md
```

解析路径后去掉文件名，剩余各段的含义：

```
['_posts', 'Recommendation', 'Study_Notes', 'Chapter2']
  index 0      index 1         index 2 = L1   index 3 = L2
```

以 `dirParts[2]` 作为一级分组键（L1），`dirParts[3]`（若存在）作为二级分组键（L2）：

```javascript
allPosts.forEach(function(p) {
    var src      = (p.source || '').replace(/\\/g, '/');
    var parts    = src.split('/');
    var dirParts = parts.slice(0, -1);

    var L1key = dirParts.length >= 3 ? dirParts[2] : '一般';
    var L1rel = dirParts.length >= 3 ? dirParts.slice(1, 3).join('/') : '';

    if (!tree[L1key]) {
        tree[L1key] = { relPath: L1rel, subgroups: {}, sgOrder: [], posts: [] };
        treeOrder.push(L1key);
    }

    if (dirParts.length >= 4) {
        var L2key = dirParts[3];
        var L2rel = dirParts.slice(1, 4).join('/');
        if (!tree[L1key].subgroups[L2key]) {
            tree[L1key].subgroups[L2key] = { relPath: L2rel, posts: [] };
            tree[L1key].sgOrder.push(L2key);
        }
        tree[L1key].subgroups[L2key].posts.push(p);
    } else {
        tree[L1key].posts.push(p);
    }
});
```

- 只有单级子文件夹（如 `ACM-ICPC/Dynamic_Programming/post.md`）时，所有文章归入 L1 的 `posts`，不产生 L2。
- 有两级子文件夹时，文章归入对应的 L2 `posts`。

### 3.3 排序

章节类文件夹（名称匹配 `Chapter N`）按章节号升序；其余按文章数降序：

```javascript
var chapterRe = /Chapter\s*(\d+)[.\-]?(\d*)/i;
function chapterCmp(a, b, sizeFn) {
    var ma = a.match(chapterRe), mb = b.match(chapterRe);
    if (ma && mb) {
        var d = parseInt(ma[1]) - parseInt(mb[1]);
        return d !== 0 ? d : parseInt(ma[2] || 0) - parseInt(mb[2] || 0);
    }
    return sizeFn(b) - sizeFn(a);
}
```

### 3.4 读取 config.json 显示名

`site.sidebarFolderConfigs`（由 `scripts/sidebar-folder-config.js` 注入）以相对 `_posts` 的路径为键，存放各文件夹的 `config.json` 内容。L1 和 L2 分别查表取 `name` 字段：

```javascript
var L1cfg   = folderConfigs[node.relPath] || {};
var L1label = L1cfg.name || L1key;          // 有则用中文名，无则用文件夹名

var L2cfg   = folderConfigs[sg.relPath] || {};
var L2label = L2cfg.name || L2key;
```

---

## 4. HTML 结构

侧边栏的 `<nav>` 放在 `<main class="content">` **之前**，渲染两级嵌套：

```ejs
<nav class="category-sidebar">
    <% sidebarTree.forEach(function(L1, i) { %>
    <div class="sidebar-group collapsed" id="sg-<%= i %>">
        <div class="sidebar-group-title" onclick="toggleGroup('sg-<%= i %>')">
            <span class="sidebar-arrow">▼</span>
            <%= L1.label %>
            <span style="margin-left:auto;opacity:0.5;font-weight:400"><%= L1.total %></span>
        </div>
        <div class="sidebar-items">
            <% L1.subgroups.forEach(function(sg, j) { %>
            <div class="sidebar-subgroup collapsed" id="sg-<%= i %>-<%= j %>">
                <div class="sidebar-subgroup-title" onclick="toggleGroup('sg-<%= i %>-<%= j %>')">
                    <span class="sidebar-arrow">▼</span>
                    <%= sg.label %>
                    <span style="margin-left:auto;opacity:0.5;font-weight:400"><%= sg.posts.length %></span>
                </div>
                <div class="sidebar-subitems">
                    <% sg.posts.forEach(function(p) { %>
                    <a class="sidebar-item sidebar-item--l2" href="<%- url_for(p.path) %>" title="<%= p.title %>">
                        <%= p.title %>
                    </a>
                    <% }); %>
                </div>
            </div>
            <% }); %>
            <% L1.posts.forEach(function(p) { %>
            <a class="sidebar-item" href="<%- url_for(p.path) %>" title="<%= p.title %>">
                <%= p.title %>
            </a>
            <% }); %>
        </div>
    </div>
    <% }); %>
</nav>

<main class="content">
    <!-- 文章卡片 -->
</main>
```

初始状态下 `sidebar-group` 和 `sidebar-subgroup` 均带有 `collapsed` class，默认全部收起。

---

## 5. CSS 样式

```css
.category-sidebar {
    position: fixed;
    left: 0;
    top: 64px;
    width: 240px;
    height: calc(100vh - 64px);
    overflow-y: auto;
    background: #fff;
    border-radius: 0 8px 8px 0;
    box-shadow: 2px 4px 16px rgba(0,0,0,0.15);
    padding: 14px 0;
    font-size: 0.88rem;
    z-index: 990;
}

/* 一级分组标题 */
.sidebar-group-title {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    font-weight: 700;
    color: #5b8dee;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    cursor: pointer;
    user-select: none;
}

/* 二级分组标题（缩进、字号略小、颜色更浅） */
.sidebar-subgroup-title {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px 4px 28px;
    font-weight: 600;
    color: #7a9ee8;
    font-size: 0.78rem;
    cursor: pointer;
    user-select: none;
}

/* 折叠箭头动画 */
.sidebar-group-title .sidebar-arrow,
.sidebar-subgroup-title .sidebar-arrow {
    font-size: 0.6em;
    transition: transform 0.2s;
    display: inline-block;
}
.sidebar-group.collapsed    .sidebar-arrow  { transform: rotate(-90deg); }
.sidebar-group.collapsed    .sidebar-items  { display: none; }
.sidebar-subgroup.collapsed .sidebar-arrow  { transform: rotate(-90deg); }
.sidebar-subgroup.collapsed .sidebar-subitems { display: none; }

/* 文章条目 */
.sidebar-item {
    display: block;
    padding: 5px 16px 5px 26px;
    color: #444;
    line-height: 1.4;
    text-decoration: none;
    border-left: 2px solid transparent;
    transition: border-color 0.15s, background 0.15s;
    word-break: break-all;
}

/* 二级条目额外缩进 */
.sidebar-item--l2 { padding-left: 40px; }

.sidebar-item:hover {
    background: #f0f4ff;
    border-left-color: #5b8dee;
    color: #333;
}

/* 小屏幕隐藏侧边栏，避免遮挡内容 */
@media (max-width: 1200px) {
    .category-sidebar { display: none; }
}
```

---

## 6. 折叠交互

统一用一个函数处理所有层级的折叠，ID 由 EJS 渲染时注入：

```javascript
function toggleGroup(id) {
    var el = document.getElementById(id);
    if (el) el.classList.toggle('collapsed');
}
```

一级分组 ID 格式为 `sg-{i}`，二级分组为 `sg-{i}-{j}`，互不干扰。

---

## 7. 踩过的坑

### 7.1 `position: fixed` 失效

**现象**：侧边栏设置了 `position: fixed; left: 0`，但渲染后不在页面左边，而是跟着内容偏移。

**原因**：`position: fixed` 的参考系是**最近的包含块（containing block）**。若任意祖先元素存在 `transform`、`filter` 或 `perspective` CSS 属性，固定定位会相对于该祖先而非视口。

**解决**：将侧边栏的 `<nav>` 移出 `<main class="content">`，放在 `<body>` 的顶层，确保没有任何带变换属性的祖先元素干扰。

### 7.2 侧边栏只显示当前页文章

**现象**：第 2 页之后，侧边栏的文章列表变少，只显示当前页的文章。

**原因**：直接使用 `page.posts` —— 这是 Hexo 分页后的结果，每页只有若干篇。

**解决**：改用 `site.categories.findOne({name: page.category}).posts`，这是该分类在数据库中的完整集合，不受分页影响。

### 7.3 卡片宽度变大

**现象**：引入侧边栏后，文章卡片比首页的明显更宽。

**原因**：首页的文章列表容器带有 Materialize 的 `.container` class（提供最大宽度约束），而分类页模板里的 `<article>` 缺少这个 class。

**解决**：给分类页的文章容器加上 `container` class：

```ejs
<article id="articles" class="container articles">
```

### 7.4 EJS 模板里无法读取文件

**现象**：在 `category.ejs` 里直接 `require('fs')` 读取 `config.json`，报 `ReferenceError: require is not defined`。

**原因**：Hexo 的 EJS 模板运行在沙箱环境中，`require` 不在作用域内。

**解决**：将文件 I/O 移到 `scripts/sidebar-folder-config.js`，通过 `hexo.locals.set('sidebarFolderConfigs', fn)` 注入，模板通过 `site.sidebarFolderConfigs` 访问。

---

## 8. 文件夹与侧边栏联动

完成以上实现后，侧边栏分组完全由文件夹结构决定。维护规则非常简单：

- **新建一级子文件夹并放入 `.md` 文件** → 侧边栏自动出现对应一级分组
- **在一级文件夹下再建二级子文件夹** → 侧边栏自动呈现嵌套结构
- **移动文章到不同子文件夹** → 侧边栏分组自动更新
- **纯图片文件夹**（无 `.md`）→ 不会出现在 `post.source` 中，自动忽略
- **添加 `config.json`** → 侧边栏立即使用其中的 `name` 字段替换文件夹名

无需修改任何配置或模板，文件系统即是配置。
