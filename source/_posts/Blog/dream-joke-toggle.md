---
title: 为博客首页"我的梦想"栏添加每日笑话切换功能
categories: 博客搞建
date: 2026-04-09 14:00:00
tags:
    - Hexo
    - JavaScript
    - 前端
---

本文记录了为博客主页的"我的梦想"小部件新增每日笑话切换功能的完整过程——包括读取原始笑话文件的编码处理、数据提取、前端切换逻辑，以及样式定制。

---

## 1. 需求背景

博客主页顶部有一个"我的梦想"卡片，展示一段固定的励志文字。想在这个卡片里加入一个隐藏彩蛋：点击按钮可切换成"每日笑话"模式，随机展示一条苏东政治笑话，并支持刷新和切回。

笑话原始文件存放于 `source/source/原版苏东政治笑话合集.txt`，共 671 条。

---

## 2. 笑话数据提取

### 一、 编码问题

直接用 `cat` 查看时内容乱码。通过 Python 测试发现文件为 **UTF-16 LE with BOM** 编码，GBK 解码失败，UTF-16 解码正常。

### 二、 格式分析

文件中每条笑话以 `数字＋全角右括号` 开头，例如：

```
1） 
一位公民打电话到苏联台问播音员……

2） 
斯大林的权威
……
```

关键字符是 Unicode `U+FF09`（全角右括号 `）`），通过 $`re.match(r'^\d+\uff09',$ line)` 匹配各条笑话的起始行。

### 三、 提取脚本

```python
import json, re, os

with open('source/source/原版苏东政治笑话合集.txt', 'rb') as f:
    content = f.read()

text = content.decode('utf-16')
lines = text.split('\r\n')

jokes = []
current_lines = []
in_joke = False

for line in lines:
    stripped = line.strip()
    if re.match(r'^\d+\uff09', stripped):
        if current_lines and in_joke:
            jokes.append(' '.join(l.strip() for l in current_lines if l.strip()))
            current_lines = []
        in_joke = True
        content_part = re.sub(r'^\d+\uff09\s*', '', stripped)
        if content_part:
            current_lines.append(content_part)
    elif in_joke and stripped:
        current_lines.append(stripped)

if current_lines and in_joke:
    jokes.append(' '.join(l.strip() for l in current_lines if l.strip()))

os.makedirs('source/js', exist_ok=True)
with open('source/js/jokes.json', 'w', encoding='utf-8') as f:
    json.dump(jokes, f, ensure_ascii=False, indent=2)
```

将提取结果保存至 `source/js/jokes.json`，Hexo 构建时会自动将该文件复制到 `public/js/jokes.json`，前端可通过 `/js/jokes.json` 路径访问。

---

## 3. 前端切换逻辑

### 一、 结构设计

修改 `themes/hexo-theme-matery/layout/_widget/dream.ejs`，在原有梦想内容外层套一个容器，内部分为两个互斥视图：

- `#dream-view`：原始"我的梦想"内容 + 切换到笑话的按钮
- `#joke-view`：笑话文本 + 刷新按钮 + 切回按钮，默认 `display:none`

### 二、 JavaScript 逻辑

采用**懒加载 + 内存缓存**的方案：首次切换到笑话视图时才发起 fetch 请求，成功后将数据缓存在闭包变量 `_jokesCache` 里，后续刷新直接从缓存中随机取值，不再重复请求。

```javascript
(function () {
    var _jokesCache = null;
    var _jokesUrl = '<%- url_for("/js/jokes.json") %>';

    window.dreamToggleToJoke = function () {
        document.getElementById('dream-view').style.display = 'none';
        document.getElementById('joke-view').style.display = '';
        if (_jokesCache) {
            dreamShowRandom();
        } else {
            document.getElementById('joke-text').textContent = '加载中……';
            fetch(_jokesUrl)
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    _jokesCache = data;
                    dreamShowRandom();
                })
                .catch(function () {
                    document.getElementById('joke-text').textContent = '加载失败，请稍后再试。';
                });
        }
    };

    window.dreamToggleToDream = function () {
        document.getElementById('joke-view').style.display = 'none';
        document.getElementById('dream-view').style.display = '';
    };

    window.dreamRefreshJoke = function () {
        if (_jokesCache) { dreamShowRandom(); }
        else { window.dreamToggleToJoke(); }
    };

    function dreamShowRandom() {
        var idx = Math.floor(Math.random() * _jokesCache.length);
        document.getElementById('joke-text').textContent = _jokesCache[idx];
    }
})();
```

**设计决策：**

- 使用 IIFE（立即执行函数）封装，避免 `_jokesCache` 污染全局作用域，但通过 `window.xxx` 暴露按钮点击需要的三个函数。
- fetch URL 通过 EJS 的 `url_for()` 生成，自动适配 Hexo 配置的 `root` 路径，不硬编码。

---

## 4. 样式定制

在 `themes/hexo-theme-matery/source/css/my.css` 中添加两段样式：

```css
/* ── 笑话切换按钮 ── */
.joke-toggle-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  border-radius: 20px !important;
  letter-spacing: 0.05em;
  box-shadow: 0 3px 10px rgba(102, 126, 234, 0.4) !important;
  transition: box-shadow 0.2s, transform 0.15s;
}

.joke-toggle-btn:hover {
  box-shadow: 0 5px 16px rgba(102, 126, 234, 0.6) !important;
  transform: translateY(-1px);
}

.joke-text-box {
  text-align: left;
  line-height: 1.9;
  font-size: 1.05rem;
  background: rgba(102, 126, 234, 0.06);
  border-left: 4px solid #764ba2;
  border-radius: 6px;
  padding: 16px 20px;
  min-height: 60px;
}
```

**设计决策：**

- 按钮使用紫色渐变（`#667eea → #764ba2`），与主题蓝绿色调形成区分，视觉上明确标示这是一个"彩蛋"入口。
- 笑话文本框采用左边框高亮 + 浅色背景，与正文段落形成层次对比，和已有的 `details` 折叠块风格保持一致。

---

## 5. 涉及文件

| 文件 | 操作 | 说明 |
| :--- | :--- | :--- |
| `source/js/jokes.json` | 新建 | 提取后的 671 条笑话，UTF-8 JSON |
| `themes/.../layout/_widget/dream.ejs` | 修改 | 双视图结构 + 切换 JS |
| `themes/.../source/css/my.css` | 追加 | 按钮与笑话文本框样式 |$$