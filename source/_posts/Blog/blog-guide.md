---
title: 博客工程全解析：项目结构与自定义样式指南
date: 2026-04-04 12:00:00
categories: 博客搭建
tags:
    - Hexo
    - 博客
    - 教程
---

本博客基于 **Hexo 8** 静态站点生成器与 **hexo-theme-matery** 主题构建，托管于 GitHub Pages。本文完整梳理项目的目录结构、核心配置文件、写作规范，以及如何通过自定义样式与脚本对博客进行二次开发。

---

## 1. 项目总览

### 一、 目录结构

```
MyBlog/blog/
├── _config.yml              # 站点全局配置
├── package.json             # Node 依赖声明
├── scaffolds/               # 新文章/页面的模板
│   ├── post.md
│   └── page.md
├── scripts/                 # 自定义 Hexo 插件脚本
│   └── details-tag.js
├── source/                  # 所有原始内容
│   ├── _posts/              # 博客文章（按分类建子目录）
│   │   ├── Recommendation/
│   │   ├── Blog/
│   │   └── ...
│   ├── about/               # 关于页面
│   ├── categories/          # 分类页面
│   └── tags/                # 标签页面
├── themes/
│   └── hexo-theme-matery/   # 主题文件
│       ├── _config.yml      # 主题配置
│       ├── layout/          # EJS 模板
│       └── source/
│           ├── css/
│           │   ├── matery.css   # 主题核心样式（勿直接修改）
│           │   └── my.css       # 自定义样式入口
│           └── js/
└── public/                  # Hexo 生成的静态文件（自动产生，勿手动编辑）
```

Hexo 的工作流程：读取 `source/_posts/` 中的 Markdown → 套用 `themes/` 中的 EJS 模板 → 输出到 `public/` → 推送到 GitHub Pages。

---

## 2. 核心配置文件详解

### 一、 站点配置 `_config.yml`

站点根目录下的 `_config.yml` 控制博客的全局行为。

#### 1. 基本信息

```yaml
title: YANG's Blog
subtitle: 'Welcome'
description: 'Cogito, ergo sum'
author: YANG
language: en
```

#### 2. 永久链接格式

```yaml
permalink: :year/:month/:day/:title/
```

* 生成的文章 URL 形如 `https://example.com/2026/04/04/my-post/`。
* 建议文章文件名使用英文，避免 URL 出现中文编码乱码。

#### 3. 渲染器与数学公式

```yaml
mathjax:
  enable: true
```

* 本站使用 `hexo-renderer-kramed` 渲染 Markdown。
* MathJax 在**主题层面**全局启用，但为避免每篇文章都加载 MathJax 脚本（加载较慢），需要在需要公式的文章 Front-matter 中**单独声明** `mathjax: true`。

#### 4. 部署配置

```yaml
deploy:
  type: 'git'
  repo: https://github.com/SGDBS/SGDBS.github.io.git
  branch: main
```

执行 `hexo deploy` 时，Hexo 会将 `public/` 目录强制推送到该仓库的 `main` 分支，GitHub Pages 即自动更新。

---

### 二、 主题配置 `themes/hexo-theme-matery/_config.yml`

主题配置负责控制页面的视觉与功能模块，以下为常用项。

| 配置项 | 作用 | 关键字段 |
| :--- | :--- | :--- |
| `menu` | 顶部导航栏菜单 | `url`、`icon`（Font Awesome） |
| `cover` | 首页封面轮播图 | `autoLoop`、`intervalTime` |
| `toc` | 文章目录 (TOC) | `enable`、`heading`、`collapseDepth` |
| `mathjax` | 数学公式渲染 CDN | `cdn` |
| `reward` | 文末打赏功能 | `wechat`、`alipay` |
| `code` | 代码块功能 | `copy`、`shrink`（可折叠） |
| `music` | 首页背景音乐播放器 | `server`、`id`、`autoplay` |

---

## 3. 写作规范

### 一、 Front-matter 模板

每篇文章顶部的 YAML 区块（Front-matter）控制文章的元数据与渲染行为。

```yaml
---
title: 文章标题
date: 2026-04-04 12:00:00
categories: 分类名称
mathjax: true          # 需要数学公式时才加
tags:
    - 标签A
    - 标签B
---
```

* `categories`：对应 `source/_posts/` 下的子目录分类，建议保持一致。
* `mathjax: true`：按需开启，不需要公式的文章省略此项，可加快页面加载。

### 二、 文章文件存放

将文章存放到对应分类的子目录下：

```
source/_posts/
├── Recommendation/
│   ├── Chapter1.md
│   └── Chapter1/       # 同名资源文件夹（post_asset_folder: true 时自动创建）
├── Blog/
│   └── blog-guide.md
```

> `_config.yml` 中 `post_asset_folder: true` 时，每篇文章会拥有一个同名文件夹，可将该文章的图片等资源放入其中，通过相对路径引用。

### 三、 数学公式写法

行内公式使用单个 `$`，独立公式块使用 `$$`：

```markdown
行内：用户兴趣得分 $S_u = \sum w_i \cdot f(t_i)$

独立公式块：
$$CTR_{smoothed} = \frac{Clicks + \alpha}{Impressions + \alpha + \beta}$$
```

---

## 4. 自定义样式

### 一、 样式文件入口

**不要修改 `matery.css`**（主题升级时会被覆盖）。所有自定义样式统一写入：

```
themes/hexo-theme-matery/source/css/my.css
```

该文件在主题模板中被最后加载，天然具有最高优先级，可以覆盖主题的任何默认样式。

### 二、 常用自定义场景

#### A. 修改文章正文字体

```css
.post-content {
    font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
    font-size: 16px;
    line-height: 1.9;
}
```

#### B. 调整代码块样式

```css
.post-content pre {
    border-radius: 8px;
    font-size: 0.88em;
}
```

#### C. 为特定元素添加背景色

```css
.post-content blockquote {
    background-color: #f0f4ff;
    border-left: 4px solid #5b8dee;
    padding: 10px 16px;
    border-radius: 0 6px 6px 0;
}
```

### 三、 当前已有的自定义样式：折叠块

本博客为 `<details>` 折叠元素定义了统一的视觉风格：浅灰蓝背景、蓝色左侧强调线、衬线字体，与正文无衬线体形成区分。具体实现见 `my.css` 中的 `/* Details 折叠块 */` 区块。

---

## 5. 自定义 Hexo 标签

### 一、 工作原理

`scripts/` 目录下的 `.js` 文件会在 Hexo 启动时自动加载执行。通过 `hexo.extend.tag.register()` 可以注册自定义的 Liquid 风格标签，在 Markdown 文章中直接使用。

### 二、 折叠标签 details

当前注册了一个 `details` 标签，用于创建**内容可折叠区块**，允许在其中书写完整的 Markdown 语法。

**`scripts/details-tag.js` 实现：**

```js
const kramed = require('kramed');

hexo.extend.tag.register('details', function (args, content) {
  const summary = args.join(' ');
  const rendered = kramed(content);
  return `<details><summary>${summary}</summary>${rendered}</details>`;
}, { ends: true });
```

* `args`：标签名后跟着的文字，作为折叠标题。
* `content`：开闭标签之间的内容，由 `kramed` 直接渲染为 HTML。
* **注意**：此处直接调用 `kramed` 而非 `hexo.render.renderSync`，原因是后者会触发 Hexo 内部的 `PostRenderEscape` 缓存机制，与外层渲染流程产生索引冲突，导致 `ERR_ASSERTION` 崩溃。

**使用方式（在文章 Markdown 中这样写）：**

开头写 `details 折叠标题`，结尾写 `enddetails`，格式与 Hexo 所有内置块标签完全相同。中间可以放任意 Markdown 内容。

**渲染效果：**

{% details 点击展开示例 %}

这里可以写 **加粗**、`代码`、表格、数学公式等任意 Markdown 内容。

| 列A | 列B |
| :-- | :-- |
| 数据1 | 数据2 |

{% enddetails %}

### 三、 扩展更多标签

参照同样的模式，可以在 `scripts/` 下注册任意新标签，例如 `note` 警告块：

```js
// scripts/note-tag.js
const kramed = require('kramed');

hexo.extend.tag.register('note', function (args, content) {
  const type = args[0] || 'info'; // info / warning / danger
  const rendered = kramed(content);
  return `<div class="note note-${type}">${rendered}</div>`;
}, { ends: true });
```

---

## 6. 构建与部署

### 一、 常用命令

| 命令 | 作用 |
| :--- | :--- |
| `hexo clean` | 清除 `public/` 缓存（修改配置后必须执行） |
| `hexo generate` | 重新生成静态文件到 `public/` |
| `hexo server` | 本地预览，默认监听 `http://localhost:4000` |
| `hexo deploy` | 推送 `public/` 到 GitHub Pages |

### 二、 标准发布流程

```bash
hexo clean && hexo generate && hexo deploy
```

> 修改了 `_config.yml`、主题配置或 `scripts/` 中的脚本后，必须先执行 `hexo clean` 清除缓存，否则旧的构建结果可能残留。

### 三、 依赖管理

```json
// package.json 中的关键依赖
"hexo-renderer-kramed"    // Markdown 渲染器（支持 LaTeX）
"hexo-generator-search"   // 全文搜索索引生成
"hexo-generator-feed"     // RSS 订阅 atom.xml
"hexo-tag-embed"          // 第三方内容嵌入标签
```

新增 npm 插件后，重新 `hexo clean && hexo generate` 即可生效。
