---
layout:     post
title:      Transformer机制学习笔记
categories: AI模型
subtitle:
date:       2023-02-14
author:     BY 水蓝
header-img: img/sv1.jpg
catalog: true
mathjax: true
tags:
    - AI
    - Transformer
    - 深度学习
---

学习自https://www.bilibili.com/video/BV1J441137V6

#### RNN，CNN网络的缺点

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/abfdc283bffb00ba98c726e3841ad37e.png)

难以平行化处理，比如我们要算 $b^4$，我们需要一次将 $a^1$~$a^4$ 依次进行放入网络中进行计算。

于是有人提出用CNN代替RNN  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c498f840ab2dd605c0768c607ab6baa4.png)  
三角形表示输入，$b^1$ 的结果是由 $a^1, a^2$ 产生。  
$a^1$~$a^4$ 可以同时并行输入到CNN中。  
但是，这么做的话可以表示的内容非常有限，解决方法是再往上继续建造。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7edf827572e76cb92b16bc55b623974f.png)  
这样的话，蓝色的输入，就相当于获得了 $a^1$~$a^4$ 的输入。  
**CNN的优点就是可以同时计算，缺点就是需要叠很多层**。

### self-Attention层

self-Attention层要做的就是，既能达到RNN的功能，同时又能像CNN一样平行化。  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a680ee9fee278a276fd1269092979c0c.png)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/94bfc4541113b218d3ce4bc7aa67caab.png)

#### self-attention层运作步骤

* 拿每个q与每个k进行attention运算  
  ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d511b5efbb3520d9e6cbfd64c277a24a.png)  
  $d$ 为 $q,k$ 的维度，这个可以理解为是为了平衡维度带来的影响，因为维度越大，点乘出来的结果就会相应的较大，所以除以维度可以消除一部分影响。
* 然后再统一做一下softmax  
  ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e241ef69d815d6c1acc4cf1056ba725f.png)  
  ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c57759667a218f95354b15cf0c812180.png)
* 随后 $\hat{a}$ 再和 $v$ 相乘  
  ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/98bb9242da3112d5cd4df82e9328889c.png)  
  ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4c1d720d85aa1318b32bace48ed133d8.png)  
  这样，计算 $b^1$ 既可以并行计算，也能获取到 $x^1$~$x^4$ 的全部数据。

> 如何并行化  
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/97628ec3269f4f95e78d293410256ae5.png)  
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/101f706d9f6704e191dba882e8a8c4d8.png)  
> ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a8b894a334d0f4dd687717c14376150e.png)  
> 可以把上一层的内容统统放入到矩阵中，进行一次矩阵乘法即可算出下一层。而矩阵乘法可以用GPU加速。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1ab445d5bf7fe797952a651d67ecc7ff.png)  
$q,k,v$ 也是可以用多层的。

### Position Encoding

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8214b52057f46973bfa89923ae189281.png)  
实际上，$x$ 序列的位置信息是不重要的，因为每个位置都有一个独一无二的 $e^i$ 向量与它相加，依次来表示位置信息。这个 $e^i$ 不是从数据中学到的，而是人为赋值的。

### Sequence To Sequence

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/46de0d6618a9e5a6fae1f64b67b6422b.png)  
在Sequence To Sequence模型中，就可以用self-Attention层来代替RNN或者CNN。

### Transformer

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/90db4a897e52a52b3e768516538e2024.png)
