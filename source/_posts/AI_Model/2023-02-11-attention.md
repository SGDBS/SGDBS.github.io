---
layout:     post
title:      Attention机制 学习笔记
subtitle:
date:       2023-02-11
author:     BY 水蓝
header-img: img/sv1.jpg
catalog: true
categories: AI模型
mathjax: true
tags:
    - AI
    - Attention
    - 深度学习
---

学习自https://easyai.tech/ai-definition/attention/

### Attention本质

Attention（注意力）机制如果浅层的理解，跟他的名字非常匹配。他的核心逻辑就是“从关注全部到关注重点”。

比如我们人在看图片时，对图片的不同地方的注意力是不同的。  
![](https://i-blog.csdnimg.cn/blog_migrate/cb184e4b45af227c984991d2d070025d.png)

即,我们的视觉系统就是一种 Attention机制，将有限的注意力集中在重点信息上，从而节省资源，快速获得最有效的信息。

Attention与NLP的联系  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c8ac535049984fd8bd37a84de7c57674.png)

### Attention的优点

* 参数少，与CNN，RNN相比，参数少，复杂度更低
* 速度快，Attention机制每一步计算不依赖于上一步的计算结果，因此可以和CNN一样并行处理。
* 在 Attention 机制引入之前，有一个问题大家一直很苦恼：长距离的信息会被弱化，就好像记忆能力弱的人，记不住过去的事情是一样的。

  Attention 是挑重点，就算文本比较长，也能从中间抓住重点，不丢失重要的信息。下图红色的预期就是被挑出来的重点。  
  ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/bf95dc62795f50295ac99f7f9f44312e.png)

### Attention原理

一个小小的例子,比如我想要更多的了解漫威，那么我就应该多读一读相关的书籍，与之关系不大的书就不用大量地看。  
![](https://i-blog.csdnimg.cn/blog_migrate/2e64f913981f690075ac6af690d419cc.png)

稍微具体化一点就是：图书管（source）里有很多书（value），为了方便查找，我们给书做了编号（key）。当我们想要了解漫威（query）的时候，我们就可以看看那些动漫、电影、甚至二战（美国队长）相关的书籍。不过为了提升效率，动漫、电影的书籍需要多看一下，而二战类的书籍就不需要看那么多了。

### Attention具体流程

* query 和 key 进行相似度计算，得到权值
* 将权值进行归一化，得到直接可用的权重
* 将权重和 value 进行加权求和

![](https://i-blog.csdnimg.cn/blog_migrate/9f6d1dd797858a4d0d75dbd038f6653f.png)