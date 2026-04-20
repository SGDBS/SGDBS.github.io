---
title: 修复 details 折叠标签并将样式迁移至主题 CSS
categories: 博客搞建
date: 2026-04-11 14:00:00
tags:
    - Hexo
    - 调试
    - CSS
---

本文记录了排查 `hexo g` 报错、安装 `details` 标签插件、修复文章中孤立标签，以及将内联样式迁移到主题 CSS 文件的完整过程。

---

## 1. 问题起因

执行 `hexo g` 时报致命错误：

```
Nunjucks Error: _posts/AI_Model/Transformer.md [Line 168, Column 4]
unknown block tag: enddetails
```

Nunjucks 不认识 `enddetails`，说明 `details` 块标签没有被注册。

---

## 2. 安装插件与排查

### 一、 安装 hexo-tag-details

```bash
npm install hexo-tag-details --save
```

加 `--debug` 运行后确认插件已加载：

```
DEBUG Plugin loaded: hexo-tag-details
DEBUG Script loaded: scripts\details-tag.js
```

两个来源都成功加载，但错误依旧。

### 二、 真正的根本原因

仔细核对 `Transformer.md` 中所有 `{% details %}` / `{% enddetails %}` 的配对关系，发现**第 384 行附近缺少一个开始标签**：

| 行号 | 内容 |
| :--- | :--- |
| 292 | `{% details 愉快的代码时间 %}` |
| 382 | `{% enddetails %}` |
| 384 | 残差连接原理正文（**无开始标签**）|
| 436 | `{% enddetails %}` ← 孤立 |

第 384 行到第 436 行是关于残差连接原理的深入讲解，本应被一个 `{% details %}` 块包裹，但开始标签丢失了。Nunjucks 在解析到孤立的 `{% enddetails %}` 时，因为没有配对的块上下文，直接报 unknown block tag。

### 三、 修复

在第 384 行前补上：

```
{% details 残差连接，为什么有效？ %}
```

重新运行 `hexo g`，报错消失。

---

## 3. 内联样式迁移

### 一、 问题

之前为了快速调整 `details` 的视觉样式，将 `<style>` 标签直接内嵌在 `scripts/details-tag.js` 的输出 HTML 里。这导致页面上**每个折叠块都携带一份完整的 CSS**，产生大量冗余。

### 二、 迁移方案

正确做法是将样式集中写入主题的 `my.css`，JS 只输出纯 HTML 结构：

**`scripts/details-tag.js`（修改后）：**

```js
hexo.extend.tag.register('details', function (args, content) {
  const summary = args.join(' ');
  const rendered = kramed(content);
  return `<details><summary>${summary}</summary><div class="details-content">${rendered}</div></details>`;
}, { ends: true });
```

**`themes/hexo-theme-matery/source/css/my.css`（新增样式）：**

```css
details {
  margin: 1.2em 0;
  border: 1px solid #d0d7de;
  border-left: 4px solid #5b8dee;
  border-radius: 8px;
  overflow: hidden;
}

details summary {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  background: #f6f8fa;
  cursor: pointer;
  font-weight: 600;
  color: #24292f;
  list-style: none;
}

details summary::before {
  content: '▶';
  font-size: 0.65em;
  color: #5b8dee;
  transition: transform 0.2s ease;
  display: inline-block;
}

details[open] summary::before {
  transform: rotate(90deg);
}

details[open] summary {
  border-bottom: 1px solid #d0d7de;
}

.details-content {
  padding: 14px 20px;
  background: #ffffff;
  font-size: 0.95em;
  line-height: 1.8;
  color: #3a3a3a;
}
```

### 三、 设计要点

* **折叠箭头**：用 `::before` 伪元素输出 `▶`，配合 `details[open]` 选择器和 CSS `transform: rotate(90deg)` 实现旋转动画，无需 JavaScript
* **内容区分离**：展开区域用 `<div class="details-content">` 包裹，与 `summary` 之间有分割线，白底与标题栏浅灰背景形成层次
* **左侧蓝色强调线**：`border-left: 4px solid #5b8dee`，与博客整体配色一致
* **CSS 只加载一次**：由主题统一引入，无论页面有多少个折叠块，样式文件只请求一次

---

## 4. 最终结果

`hexo g` 正常通过，折叠块渲染效果：标题栏浅灰背景，箭头点击旋转展开，内容区白底，左侧蓝色强调线。$$