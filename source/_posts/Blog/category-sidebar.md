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

本文记录了为博客分类页面（`/categories/xxx/`）新增固定侧边栏的完整实现过程。侧边栏固定在视口左侧，显示当前分类下的**全部文章**（不受分页限制），并按文件夹层级自动分组、支持折叠展开。

---

## 1. 需求背景

博客的 ACM-ICPC 分类下有 100+ 篇文章，分 10 余页展示。在某一页翻看题解时，完全不知道其他文章的全貌，跳转也很不方便。

目标是：在分类页面左侧常驻一个导航栏，列出该分类下所有文章的标题，点击可直接跳转，且不占用主内容区域的宽度。

---

## 2. 整体设计

| 要素 | 方案 |
|------|------|
| 定位 | `position: fixed; left: 0` 固定在视口左边缘 |
| 高度 | `height: calc(100vh - 64px)`，撑满导航栏以下的全部空间 |
| 文章来源 | `site.categories.findOne({name: page.category}).posts`，取全量而非当前页 |
| 分组逻辑 | 读取 `post.source` 中的子文件夹名，与实际文件夹结构保持一致 |
| 折叠交互 | 纯 JS，点击组标题切换 `collapsed` class |

关键决策：侧边栏必须放在 `<main>` **之外**。若放在 `<main>` 内部，父元素的 `overflow` 或 CSS 变换可能导致 `position: fixed` 失效，表现为侧边栏跟随页面滚动或位置偏移。

---

## 3. HTML 结构

修改 `layout/category.ejs`，将侧边栏的 `<nav>` 提升到 `<main class="content">` 之前：

```ejs
<%- partial('_partial/bg-cover') %>

<% /* 数据计算逻辑 */ %>

<!-- 侧边栏：独立于 main，position:fixed 相对视口定位 -->
<nav class="category-sidebar">
    <% sidebarGroups.forEach(function(group, gi) { %>
    <div class="sidebar-group" id="sg-<%= gi %>">
        <div class="sidebar-group-title" onclick="toggleSidebarGroup(<%= gi %>)">
            <span class="sidebar-arrow">▼</span>
            <%= group.label %>
            <span style="margin-left:auto;opacity:0.5;font-weight:400">
                <%= group.posts.length %>
            </span>
        </div>
        <div class="sidebar-items">
            <% group.posts.forEach(function(p) { %>
            <a class="sidebar-item" href="<%- url_for(p.path) %>" title="<%= p.title %>">
                <%= p.title %>
            </a>
            <% }); %>
        </div>
        <% if (gi < sidebarGroups.length - 1) { %>
        <div class="sidebar-divider"></div>
        <% } %>
    </div>
    <% }); %>
</nav>

<main class="content">
    <!-- 文章卡片 -->
</main>
```

---

## 4. 数据构建：全量文章 + 文件夹分组

### 4.1 获取全量文章

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

### 4.2 按文件夹层级分组

每个 post 对象的 `source` 属性包含其源文件路径，例如：

```
_posts/ACM-ICPC/Dynamic_Programming/2022-02-05-122788273.md
```

从路径中提取**第二层文件夹名**（即直属子文件夹）作为分组标签：

```javascript
allPosts.forEach(function(p) {
    var src = (p.source || '').replace(/\\/g, '/').replace(/^_posts\//, '');
    var parts = src.split('/');
    // parts = ['ACM-ICPC', 'Dynamic_Programming', 'filename.md']
    var label = parts.length > 2 ? parts[1] : parts[0];

    if (!folderGroups[label]) { folderGroups[label] = []; folderOrder.push(label); }
    folderGroups[label].push(p);
});
```

这样，只要调整文件所在的子文件夹，侧边栏的分组就会自动更新，无需手动维护任何配置。

---

## 5. CSS 样式

```css
.category-sidebar {
    position: fixed;
    left: 0;
    top: 64px;              /* Materialize 导航栏高度 */
    width: 240px;
    height: calc(100vh - 64px);  /* 撑满导航栏以下所有空间 */
    overflow-y: auto;
    background: #fff;
    border-radius: 0 8px 8px 0;
    box-shadow: 2px 4px 16px rgba(0,0,0,0.15);
    padding: 14px 0;
    font-size: 0.88rem;
    z-index: 990;           /* 低于导航栏(997)，高于普通内容 */
}

/* 分组标题 */
.sidebar-group-title {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    font-weight: 700;
    color: #5b8dee;
    font-size: 0.82rem;
    text-transform: uppercase;
    cursor: pointer;
    user-select: none;
}

/* 折叠箭头动画 */
.sidebar-group-title .sidebar-arrow {
    font-size: 0.6em;
    transition: transform 0.2s;
}
.sidebar-group.collapsed .sidebar-arrow { transform: rotate(-90deg); }
.sidebar-group.collapsed .sidebar-items { display: none; }

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

在 `category.ejs` 末尾添加简单的 JS：

```javascript
function toggleSidebarGroup(idx) {
    var el = document.getElementById('sg-' + idx);
    if (el) el.classList.toggle('collapsed');
}
```

点击分组标题时切换 `collapsed` class，CSS 负责控制箭头旋转和列表显示/隐藏。

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

---

## 8. 文件夹与侧边栏联动

完成以上实现后，侧边栏分组完全由文件夹结构决定。维护规则非常简单：

- **新建子文件夹并放入 `.md` 文件** → 侧边栏自动出现对应分组
- **移动文章到不同子文件夹** → 侧边栏分组自动更新
- **纯图片文件夹**（无 `.md`）→ 不会出现在 `post.source` 中，自动忽略

无需修改任何配置或模板，文件系统即是配置。
