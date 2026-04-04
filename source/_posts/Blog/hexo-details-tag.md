---
title: 为 Hexo 实现可折叠内容块：从功能开发到深挖 ERR_ASSERTION 崩溃
date: 2026-04-04 14:00:00
categories: 博客搭建
tags:
    - Hexo
    - 调试
    - 教程
---

本文记录了为博客新增 `details` 折叠标签的完整过程——从功能实现、样式定制，到排查一个深藏在 Hexo 渲染管道中的 `ERR_ASSERTION` 崩溃，以及最终定位到一个几乎不可能通过猜测发现的根本原因。

> **阅读说明**：本文在描述 Hexo 的 Swig 标签语法时，使用 `{ %` 和 `% }` （含空格）来表示实际的分隔符。这是为了避免 Hexo 自身的解析器在生成本文时将说明文字当作真实标签处理——这恰好也是本文讨论的核心问题之一。

---

## 1. 需求背景

在推荐系统笔记中，有些内容（例如手推的数学示例、大段代码）放在正文里会严重干扰阅读节奏，但又不想删掉。一个可折叠的"展开详情"区块是最合适的解决方案。

HTML 原生支持 `<details>` / `<summary>` 标签实现折叠效果，但在 Hexo 中直接把 Markdown 写进 `<details>` 标签内，内容会原样输出——`hexo-renderer-kramed` 遇到 HTML 块标签后，不再对其内部进行 Markdown 渲染。

---

## 2. 实现：自定义 Hexo 标签

### 一、 工作原理

Hexo 允许在 `scripts/` 目录下放置 `.js` 文件，这些文件会在 Hexo 启动时自动加载。通过 `hexo.extend.tag.register()` 可以注册自定义的块标签，其行为与内置的 `codeblock`、`quote` 等标签完全相同。

### 二、 第一版实现

在 `scripts/details-tag.js` 中注册 `details` 标签：

```js
hexo.extend.tag.register('details', function (args, content) {
  const summary = args.join(' ');
  const rendered = hexo.render.renderSync({ text: content, engine: 'markdown' });
  return `<details><summary>${summary}</summary>${rendered}</details>`;
}, { ends: true });
```

**参数说明：**

* `args`：标签名后紧跟的文字，作为折叠区块的标题（传给 `<summary>`）
* `content`：开闭标签之间的原始 Markdown 内容
* `{ ends: true }`：声明这是一个块标签，需要配套的 `enddetails` 结束标记

**在文章中的写法：**

开头写 `details 折叠标题`，结尾写 `enddetails`，中间放任意 Markdown 内容，格式与 Hexo 所有内置块标签完全相同。

这一版对于**不含代码围栏**的内容可以正常工作，文章中的 iPhone 点击统计示例顺利渲染。

---

## 3. 样式定制

### 一、 为什么要加样式

折叠区块如果和正文没有视觉区别，读者很容易忽略它是可交互的。目标是让它：

1. 有明显的背景色和边框，与正文白底区分
2. 使用不同字体，进一步区隔"内容"与"附录"的层次感
3. 折叠箭头有明确的展开/收起状态反馈

### 二、 样式实现

所有自定义样式写入 `themes/hexo-theme-matery/source/css/my.css`（主题升级不会覆盖此文件）：

```css
/* ── Details 折叠块 ── */
details {
  background-color: #f5f7fa;
  border: 1px solid #dce3ed;
  border-left: 4px solid #5b8dee;
  border-radius: 6px;
  padding: 12px 18px;
  margin: 16px 0;
  font-family: 'Georgia', 'Noto Serif SC', serif;
  font-size: 0.95em;
  color: #3a3a3a;
  line-height: 1.8;
}

details summary {
  cursor: pointer;
  font-weight: bold;
  color: #2c5fbd;
  user-select: none;
}

details summary::before {
  content: '▶ ';
  font-size: 0.75em;
  display: inline-block;
}

details[open] summary::before {
  content: '▼ ';
}
```

**设计决策：**

* **背景**：浅灰蓝 `#f5f7fa` + 左侧蓝色强调线，与正文白底对比明显
* **字体**：衬线体（Georgia），正文为无衬线体，字体风格差异天然形成层次区分
* **箭头**：隐藏浏览器默认的 `▶`，改用自定义字符并通过 `[open]` 属性选择器切换状态

---

## 4. 引入代码块后的崩溃

### 一、 现象

将一段 Python 代码放入 `details` 折叠块后，执行 `hexo g` 时报错：

```
AssertionError [ERR_ASSERTION]: The expression evaluated to a falsy value:

  (cache[index])

    at PostRenderEscape.restoreAllSwigTags (hexo/dist/hexo/post.js:49:30)
    at Object.onRenderEnd (hexo/dist/hexo/post.js:487:45)
```

### 二、 定位根本原因

阅读 `hexo/dist/hexo/post.js` 源码后，理清了 Hexo 的完整渲染管道：

#### Hexo 渲染管道（简化版）

| 步骤 | 操作 | 说明 |
| :--- | :--- | :--- |
| 1 | `escapeAllSwigTags` | 把所有块标签替换为 `<!--swig-n-->` 占位符，存入 `cacheObj.stored[]` |
| 2 | `escapeCodeBlocks` | 把代码块也存入同一个 `stored[]` |
| 3 | Markdown 渲染（kramed） | 处理正常 Markdown 语法 |
| 4 | `restoreAllSwigTags` | 还原占位符，调用注册的标签函数 |
| 5 | `tag.render`（Nunjucks） | 执行标签函数，获得最终 HTML |
| 6 | `restoreCodeBlocks` | 还原代码块占位符 |

#### 冲突的产生

当我们的 `details` 标签函数在**步骤 4/5** 中被调用时，它内部又调用了 `hexo.render.renderSync()`。

`renderSync` 内部会触发 `execFilterSync('after_render:html', ...)`，该 filter 会试图再次从**同一个** `cacheObj.stored[]` 中还原占位符。

此时 `stored[]` 的状态已被外层渲染消耗了一部分（部分索引被设为 `null`）。内层 `renderSync` 对同一缓存的读写导致索引混乱，当外层后续试图访问 `stored[n]` 时，`n` 指向的槽位已是 `null`，断言失败。

> **核心结论**：在 Hexo 标签函数内部调用 `hexo.render.renderSync`，会与外层渲染共享同一个 `PostRenderEscape` 缓存，产生索引冲突。

### 三、 修复

完全绕开 Hexo 的渲染管道，直接调用 `kramed` 渲染器本体：

```js
'use strict';
const kramed = require('kramed');

hexo.extend.tag.register('details', function (args, content) {
  const summary = args.join(' ');
  const rendered = kramed(content);
  return `<details><summary>${summary}</summary>${rendered}</details>`;
}, { ends: true });
```

`kramed` 是 `hexo-renderer-kramed` 的依赖，`require('kramed')` 可直接调用，不经过 Hexo 的任何管道，彻底消除缓存冲突。

---

## 5. 第二个崩溃：博客文档本身的问题

### 一、 现象

修复代码后，`hexo g` 依然报同样的错误。

通过逐一禁用文章文件排查，发现根源是**新写的 `blog-guide.md` 文档本身**，而不是推荐系统笔记。

### 二、 `escapeAllSwigTags` 的盲区

阅读 `post.js` 中 `escapeAllSwigTags` 的状态机实现后，发现一个关键盲区：

> **`escapeAllSwigTags` 是纯字符级状态机，完全不感知 Markdown 语法。**

它不知道自己是否在代码围栏（` ``` `）或 backtick 内联代码（`` ` `` ）里。只要扫描到 `{` 紧跟 `%`，就会无条件地尝试将其解析为 Swig 标签。

#### 具体触发过程

`blog-guide.md` 中有一行解释性文字（简化示意）：

```
代表开头的 `{ %`，代表结尾的 `% }`，写法与内置标签相同。
```

这行里用于演示的 `{ %` 字符组合，被状态机识别为标签开头，随即向后扫描寻找 `% }` 作为标签结束符，一路穿越多行文字，直到找到**后面真实 `details` 标签的闭合符**为止。

结果：说明文字吞掉了真实 `details` 标签的闭合符，整个文档的标签配对关系混乱，同样引发 `stored[n]` 为 `null` 的断言错误。

这个 bug 还有一层讽刺：本文在写作时也遭遇了同样的问题——撰写"如何避免 `{ %` 被解析"的文字，本身就会触发被解析，最终不得不在全文中统一使用 `{ %`（含空格）来规避。

### 三、 修复

将文档中所有不打算执行的 `{ %` 字符组合替换为不会触发解析器的描述方式，只保留真正需要运行的那对 `details` 标签。

---

## 6. 总结与注意事项

{% details 点击查看完整注意事项 %}

#### 在 Hexo 自定义标签中渲染 Markdown

不要使用 `hexo.render.renderSync`，应直接调用对应渲染器的 npm 包：

```js
const kramed = require('kramed');     // hexo-renderer-kramed 项目
// 或
const marked = require('marked');     // hexo-renderer-marked 项目
```

#### 在博客文章中展示 Hexo 标签语法

文章里任何位置的 `{` 紧跟 `%` 字符组合，都会被 `escapeAllSwigTags` 捕获，**包括代码围栏和 backtick 内联代码内部**。

推荐的规避方式：

* 代码围栏内：使用 `raw` / `endraw` 块标签包裹整个代码块（注意：这两个关键字本身也不能直接写在正文里）
* 行内描述：加空格写成 `{ %`，或完全改用文字描述

#### `details` 标签的已知限制

使用 `kramed` 直接渲染时，代码块不会经过 Hexo 的 `highlight.js` 管道，因此**不会有行号和主题的代码高亮样式**。如需高亮，建议将代码块放在折叠区块外部。

{% enddetails %}
