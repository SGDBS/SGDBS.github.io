---
title: Transformer详解
categories: 学习笔记-AI模型
date: 2026-04-10 12:00:00
mathjax: true
tags:
    - AI
    - Transformer
---

## 1. Introduction

Transformer 是一个完全基于**自注意力机制（Self-Attention）的编解码器**模型，它摒弃了传统的 RNN 和 CNN，通过**多头注意力**捕捉序列的**全局依赖关系**，并利用**前馈神经网络**进行**特征变换**，辅以**残差连接**和**层归一化**来保证深层网络的稳定训练。

### 1.1 Transformer 解决了什么痛点？

* **传统 RNN 的致命弱点：串行计算与遗忘**。 RNN 必须逐字阅读（例如先处理"我"，再处理"爱"，再处理"中"……）。这种串行机制导致两个致命问题：一是**无法并行加速**（GPU 都在闲置等待）；二是**长距离依赖问题**（读到句子末尾时，早把句子开头的细节忘光了）。
* Transformer 彻底抛弃了"从左到右"的顺序读取机制。它如同一个"上帝视角"，**一次性把整个句子吞进去**。无论两个词相隔多远，它们之间的距离永远是 1 步。这种设计不仅让 GPU 可以火力全开（**并行计算**），还完美**解决了长距离信息丢失**的问题。

### 1.2 宏观架构：Encoder-Decoder

原始的 Transformer 被设计用来做机器翻译，因此它采用的是经典的 **Encoder-Decoder** 架构。我们可以把它想象成一个极度高效的翻译团队：

**编码器（Encoder）：负责"理解"**
* **输入**： 源语言（比如一段中文）。
* **任务**： 深入理解这句话的语法、语义、指代关系，将其提炼成一个高度浓缩的、包含所有上下文信息的稠密矩阵。
* **特点**： 它里面的所有词都可以互相看到彼此（这叫双向注意力）。BERT 就是一个纯 Encoder 架构。

**解码器（Decoder）：负责"生成"**
* **输入**： Encoder 提炼的"中间语言" + 已经翻译出来的部分目标端词汇。
* **任务**： 根据这些信息，像挤牙膏一样，一个字一个字地生成目标语言（比如英文）。
* **特点**： 在生成第 $t$ 个目标词时，它只能看到前 $t-1$ 个目标词（不能偷看未来），但可以看到 Encoder 输出的全部源端表示。如今火爆的 GPT 系列，是一个纯 Decoder 架构。

## 2. 输入层

### 2.1 词嵌入（Word Embedding）：给文字分配"坐标"

在 Transformer 真正开始工作之前，必须先过语言转换这一关，因为计算机只认识数字。

* **分词与映射**： 首先将句子切分成一个个 Token（词或字），然后去词表中查表，把它映射成一个高维的稠密向量（在原论文中，这个维度是 $d_{model} = 512$）。

* **核心直觉**： 你可以把这 512 维想象成一个有 512 个坐标轴的超高维空间。在这个空间里，**"语义相近"的词，它们的物理距离就越近**（比如"苹果"和"橘子"靠得很近，但离"汽车"就很远）。

* **乘以 $\sqrt{d_{model}}$**： 原论文的 embedding 输出还要再乘以 $\sqrt{d_{model}}$。原因是 `nn.Embedding` 默认初始化方差约为 $1/d_{model}$ 量级，乘 $\sqrt{d_{model}}$ 是为了把 embedding 的尺度提升到与位置编码（值域 $[-1, 1]$）大致同量级，否则相加时位置编码会被淹没或反之。

### 2.2 位置编码（Positional Encoding）：解决"词序失忆症"

* **原因**： 上一节提到，Transformer 的核心优势是"一次性全局并行处理"。但这也带来了一个致命缺陷：**Self-Attention 本身是没有位置感的（它是置换不变的）**。

* **例子**： 在原始的自注意力机制眼里，"狗咬人"和"人咬狗"算出来的结果是一模一样的，因为它只看词和词之间的相关性，完全不管谁在前谁在后。

* **解法**： 既然模型自己看不出顺序，我们就必须人为地把位置信息"刻"在词向量里，再喂给模型。这就好比给参加会议的每个人不仅发了名牌（词嵌入），还发了座位号（位置编码）。

### 2.3 位置编码的数学原理

**核心公式：**
假设我们要为位置 $pos$ 的词生成一个 512 维的位置向量，它的第 $i$ 个维度（$i$ 从 0 到 255）的值由以下公式计算：
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
其中：
* $pos$ 是词在句子中的绝对位置（比如第 0 个词，第 1 个词）。
* $2i$ 和 $2i+1$ 代表向量中的偶数维度和奇数维度。
* $d_{model}$ 是向量的总维度（如 512）。
* 它能让模型轻松学习到**相对位置信息**。根据三角函数的和差化积公式：
$$\sin(\alpha + \beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$$
这意味着，位置 $pos + k$ 的编码，可以通过位置 $pos$ 的编码进行线性变换得到。这对模型理解"词 A 在词 B 后面 3 个位置"这种相对关系极其重要。

> 想象一个有 512 个指针的钟表：
>    * 低维度（$i$ 很小）： 分母接近 1，正余弦函数的频率很高。就像秒针，位置稍微变一点（$pos$ 加 1），它的值就剧烈变化。这帮模型区分相邻的词。
>    * 高维度（$i$ 很大）： 分母非常大（接近 10000），函数的频率极低。就像时针，走得很慢。这帮模型感知长距离的宏观位置。

### 2.4 常见问题

#### 1. 为什么要用三角函数？直接用 0, 1, 2, 3 作为位置编号不行吗？

* 不行。如果用整数，句子越长，位置数值就越大。这会导致模型在处理长句子时，后面的位置编码数值把词嵌入原本的语义信息"淹没"掉，并且模型难以泛化到比训练集更长的句子。
* 而三角函数的值域永远被限制在 $[-1, 1]$ 之间，非常稳定。

#### 2. 输入层是把词嵌入和位置编码相加（Add），为什么不是拼接（Concat）？

* 拼接会增加模型的维度，导致参数量和计算量翻倍。
* 至于为什么"相加"不会造成信息混乱？因为在一个极高维（如 512 维）的空间中，词嵌入向量和位置编码向量几乎是近似正交的。正交的向量相加，就像在 $x$ 轴的信息上加了 $y$ 轴的信息，彼此其实互不干扰。详细证明见下面折叠块。

{% details 为什么位置编码有效 %}

##### 1. 物理直觉：信号的"叠加"与"解耦"

想象一下你在听一首交响乐。

**词嵌入（Word Embedding）**： 就像是小提琴拉出的旋律（语义）。

**位置编码（Positional Encoding）**： 就像是背后极其规律的架子鼓节拍（位置）。

在空气中传到你耳朵里时，这两个声音的物理声波是**直接相加（叠加）**的。但你的大脑会把它们混淆成一种"非琴非鼓"的怪声吗？不会。你的大脑依然能清晰地分辨出旋律是什么，节拍在哪个位置。

在神经网络中，只要这两个信号的"频率"或"特征"差异足够大，后续的线性层（可以理解为极其高级的滤波器）就能轻松把它们分离开来。

##### 2. 空间几何：高维空间的"正交性"

我们在纸上画图习惯了二维或三维空间，但在 Transformer 中，这是一个 512 维的超高维空间。高维空间有一个非常反直觉的数学特性：**任意两个独立生成的向量，大概率是近似正交（互相垂直）的。**
* **正交意味着互不干扰**： 假设在二维平面上，词向量对应 $X$ 轴的信息 $(x, 0)$，位置向量对应 $Y$ 轴的信息 $(0, y)$。把它们相加得到 $(x, y)$。
* 在这个结果里，$X$ 轴的投影依然是纯粹的语义，$Y$ 轴的投影依然是纯粹的位置。它们在同一个载体里，但彼此透明，互不遮挡。

##### 3. 数学证明

假设我们有两个词，词 $i$ 的输入是 $x_i = E_i + P_i$，词 $j$ 的输入是 $x_j = E_j + P_j$。在 Self-Attention 中，它们会分别乘以权重矩阵变成 $Q$ 和 $K$，然后做点积求相似度。为了简化问题，我们直接看原始输入向量的点积：
$$x_i \cdot x_j = (E_i + P_i) \cdot (E_j + P_j)$$

利用乘法分配律，我们可以把它展开成四项：$$x_i \cdot x_j = \underbrace{E_i \cdot E_j}_{\text{语义相似度}} + \underbrace{E_i \cdot P_j}_{\text{语义对位置}} + \underbrace{P_i \cdot E_j}_{\text{位置对语义}} + \underbrace{P_i \cdot P_j}_{\text{位置相似度}}$$

本来是揉成一团的加法，在经过点积运算后，自动解耦成了四种纯粹的交互：

1. $E_i \cdot E_j$： 纯看这两个词的**语义**搭不搭（比如"苹果"和"吃"分很高）。
2. $P_i \cdot P_j$： 纯看这两个词的**物理距离**近不近（比如位置 3 和位置 4 分很高，位置 3 和位置 100 分很低）。
3. $E_i \cdot P_j$ 与 $P_i \cdot E_j$： 这两项**原始值**因为正交性大致在 0 附近，但模型可以通过学习 $W^Q, W^K$ 主动把它们放大利用——下一段会展开。

结论： 简单的相加，不仅省去了拼接带来的显存翻倍，而且在后续的乘法（Attention）运算中，能够自然而然地被模型"拆解"出各自的价值。模型通过训练 $W^Q$ 和 $W^K$ 矩阵，可以放大有用的项、抑制无用的项。

##### 4. 方向感从哪里来？

真实的注意力得分计算是 $Q_i \cdot K_j = (x_i W^Q) \cdot (x_j W^K)$，把它中关于位置的那部分拆出来：
$$(P_i W^Q) \cdot (P_j W^K) = P_i \big(W^Q (W^K)^T\big) P_j^T$$

只要中间夹着的这个矩阵 $M = W^Q (W^K)^T$ 不是一个对称矩阵（在神经网络通过反向传播随机学习的过程中，它极大概率不是对称的），那么：
$$P_i M P_j^T \neq P_j M P_i^T$$

所以原生的绝对位置编码（Sine/Cosine）**本身确实是"不辨方向"的，只能提供距离感**。**方向感（前后关系）是依靠注意力机制中 $W^Q$ 和 $W^K$ 这两个不对称的权重矩阵，在模型训练过程中"学"出来的**。这也是为什么后来很多更先进的大模型（比如采用 RoPE 旋转位置编码的模型）要改进位置编码——RoPE 直接在数学层面就把"方向性"内嵌进去了，不需要完全依赖模型去"死记硬背"，从而表现得更加优雅和高效。

{% enddetails %}

{% details 代码：Positional Encoding 实现 %}

```py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预先计算 [max_len, d_model] 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 把 1/10000^(2i/d_model) 写成 exp(-log(10000) * 2i / d_model)
        # 数值更稳定，且只需算 d_model/2 次
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维: cos
        pe = pe.unsqueeze(0)                          # [1, max_len, d_model]

        # register_buffer：不参与梯度更新，但会跟着 .to(device) 移动
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

{% enddetails %}


## 3. Attention 机制

### 3.1 Self-Attention：Q、K、V 与四步推导

#### 1. 什么是 $Q$、$K$、$V$

* $Q$ (Query / 查询向量)： 这是你手里拿着的搜索框里的关键词。它代表着："我想寻找什么样的信息来补充我自己？"
* $K$ (Key / 键向量)： 这是图书馆里每本书书脊上的标签/书名。它代表着："我包含了什么样的信息？"
* $V$ (Value / 值向量)： 这是书里面真正的正文内容。它代表着："如果我的标签 ($K$) 匹配了你的搜索 ($Q$)，你可以把我里面的内容 ($V$) 拿走。"

#### 2. Self-Attention 的四个数学步骤

假设我们的输入是一个矩阵 $X$（包含了句子中所有词的词嵌入+位置编码）。

##### 第一步：生成 $Q, K, V$
输入矩阵 $X$ 分别乘以三个不同的权重矩阵 $W^Q$、$W^K$、$W^V$（这三个矩阵是模型在训练过程中要学习的核心参数），映射出三个新矩阵：
$$Q = XW^Q$$
$$K = XW^K$$
$$V = XW^V$$

##### 第二步：计算注意力得分（打分）
词要想知道自己该关注谁，需要拿自己的 $Q$ 去和所有词的 $K$ 计算相似度。在数学上，两个向量越相似，它们的**点积（Dot Product）**就越大。
$$Scores = QK^T$$
这个操作会得到一个 $N \times N$ 的方阵（$N$ 是序列长度）。例如，矩阵第 $i$ 行第 $j$ 列的数值，就代表了"词 $i$ 对词 $j$ 的关注程度"。

##### 第三步：缩放与归一化（把分数变成百分比）
为了防止点积的结果过大（这会导致反向传播时梯度消失），我们需要除以一个缩放因子 $\sqrt{d_k}$（$d_k$ 是 $K$ 向量的维度）。

>  当维度 $d_k$ 很大时，两个独立分布的向量做点积，其结果的方差会随着维度 $d_k$ 变大而变大，导致点积值极大或极小。极大的值送入 $\text{softmax}$ 后，会把概率分布推向绝对的 $0$ 或 $1$，使得进入了 $\text{softmax}$ 的饱和区（平坦区）。此时梯度几乎为 $0$（梯度消失），模型根本无法更新。除以 $\sqrt{d_k}$ 就是为了把点积结果的方差强行拉回 $1$，保证梯度平稳。

然后再套上一个 $\text{softmax}$ 函数，把所有的分数转化成 $0 \sim 1$ 之间的概率分布，并且保证每一行的和为 1。

$$Attention\ Weights = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

##### 第四步：加权求和（提取信息）

现在，我们知道了每一个词应该对其他词分配多少注意力比例（权重）。最后一步，就是用这些权重去乘以它们对应的真正内容 $V$。
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
出来的结果，就是一个**融合了全局上下文信息**的全新矩阵！在这个新矩阵里，"苹果"这个词的向量，不仅包含了它自己的本意，还悄悄融合了"吃"和"红色"的特征。

### 3.2 Multi-Head Attention：为什么要多头？

#### 1. 单头的局限

单头 Self-Attention 在每一层只会输出**一种**注意力分布。这就像让一个人同时关心句子的语法、指代、情感、词性——他会被迫做平均，最终什么都不擅长。

#### 2. 多头的核心思路

把原本 $d_{model}$ 维的 $Q, K, V$ 在通道维度上**切成 $h$ 份**，每一份都在自己的 $d_k = d_{model}/h$ 维子空间里独立做一次 Attention。每个头可以专攻一种"关系视角"，互不干扰。

* 头 1 可能学到"主谓宾的句法依赖"
* 头 2 可能学到"代词的指代消解"
* 头 3 可能学到"形容词与被修饰名词的搭配"
* ……

最后把 $h$ 个头的输出**沿通道维度拼接**回 $d_{model}$ 维，再过一个线性层 $W^O$ 做信息整合。

#### 3. 数学公式

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\,W^O$$

其中 $W_i^Q, W_i^K \in \mathbb{R}^{d_{model} \times d_k}$，$W_i^V \in \mathbb{R}^{d_{model} \times d_v}$，$W^O \in \mathbb{R}^{h d_v \times d_{model}}$。

原论文取 $h=8$，$d_k = d_v = 64$，$d_{model} = 512$，恰好 $h \cdot d_k = d_{model}$。

#### 4. 计算量为什么没翻倍？

直觉上"做 8 次 attention"听起来贵 8 倍，但实际不是：
* 每个头的维度从 $d_{model}=512$ 降到了 $d_k = 64$
* 单头计算量 $\propto N^2 \cdot d_{model}$；多头每头 $\propto N^2 \cdot d_k$，共 $h$ 头，合计 $\propto N^2 \cdot h \cdot d_k = N^2 \cdot d_{model}$ — 和单头同量级
* 工程上还可以**把 $h$ 个头合并到一个大矩阵乘法里**，速度甚至比单头更快（参考 Flash Attention）

#### 5. 类比 CNN 的多通道

可以把多头 Attention 类比成 CNN 的"多卷积核"：
* CNN 一层用 64 个 3×3 卷积核，每个核学一种局部特征（边缘、纹理、颜色块）
* MHA 一层用 8 个 head，每个 head 学一种全局关系（依存、指代、语义）

二者本质都是**用更多独立子空间来增强特征多样性**。

{% details 代码：Multi-Head Attention 实现 %}

```py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MyMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # 计算每个注意力头的维度 (例如 512 // 8 = 64)
        self.d_k = d_model // num_heads

        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        # 定义 Q, K, V 的线性映射层
        # 工业界常把这三个合并成一个 nn.Linear(d_model, 3 * d_model) 来加速计算
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 最终拼接后的输出映射层
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        # Q, K, V 维度: [batch_size, seq_len, d_model]
        # 注意：Self-Attention 时 Q=K=V=x; Cross-Attention 时 Q≠K=V
        batch_size = Q.size(0)

        # ==========================================
        # 步骤 1: 线性映射并拆分多头
        # ==========================================
        # [B, L, d_model] -> [B, L, h, d_k] -> [B, h, L, d_k]
        q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # ==========================================
        # 步骤 2: Scaled Dot-Product Attention
        # ==========================================
        # q: [B, h, Lq, d_k]; k^T: [B, h, d_k, Lk]
        # scores: [B, h, Lq, Lk]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 处理 Mask（Padding Mask 或 Causal Mask）
        if mask is not None:
            # mask 中为 0 (不允许看) 的地方替换为极小负数
            # 经过 softmax 后，这些位置的权重会变成 0
            scores = scores.masked_fill(mask == 0, -1e9)

        # ==========================================
        # 步骤 3: Softmax 提取注意力权重并乘以 V
        # ==========================================
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)  # [B, h, Lq, d_k]

        # ==========================================
        # 步骤 4: 拼接多头并经过最终的线性层
        # ==========================================
        # transpose 会打乱 Tensor 的内存连续性，view 要求连续，所以加 .contiguous()
        context = context.transpose(1, 2).contiguous()              # [B, Lq, h, d_k]
        context = context.view(batch_size, -1, self.d_model)        # [B, Lq, d_model]

        return self.W_o(context)
```

{% enddetails %}

### 3.3 Mask：Padding Mask 与 Causal Mask

#### 1. 为什么需要 Mask

Self-Attention 默认让每个位置都能看到全部位置，但**有些位置我们不希望它被看到**：
* **Padding Mask**：实际句子长度参差不齐，需要补 `<pad>` 凑到固定长度。这些位置没有真实语义，不应该被其他词关注到。
* **Causal Mask（look-ahead mask）**：Decoder 在训练时会一次性看到完整的目标句子（Teacher Forcing），但每个位置只允许看自己之前（含自己）的位置——否则模型就在"作弊"，直接抄答案。

#### 2. Mask 怎么作用到 Attention？

直接在 softmax 前把不允许看的位置加上一个极大的负数 $-10^9$：

$$\text{Scores}_{ij} \leftarrow \begin{cases} \text{Scores}_{ij} & \text{Mask}_{ij}=1 \\ -10^9 & \text{Mask}_{ij}=0 \end{cases}$$

这样 softmax 后这些位置的概率会变成 $\approx 0$，自然不会被加权进 $V$。

#### 3. 两种 Mask 的形状

设 $B$ = batch size，$L$ = 序列长度，$h$ = head 数：

| Mask 类型 | 形状 | 作用位置 |
|---|---|---|
| Padding Mask | $[B, 1, 1, L_k]$ | Encoder Self-Attn / Cross-Attn 的 K 端 |
| Causal Mask | $[1, 1, L_q, L_k]$（下三角） | Decoder Self-Attn |
| Decoder Self-Attn 实际用 | Padding ∧ Causal，$[B, 1, L, L]$ | 同时屏蔽 pad 和未来位置 |

中间那两个 1 是为了和 multi-head 的形状 $[B, h, L_q, L_k]$ 做广播。

#### 4. Causal Mask 长什么样

对 $L=4$ 的序列，Causal Mask 是：
$$\begin{bmatrix} 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 1 \end{bmatrix}$$

第 $t$ 行表示"生成第 $t$ 个词时能看到哪些位置"——只能看到 $\le t$ 的位置。

{% details 代码：Mask 构造 %}

```py
import torch

def make_padding_mask(seq, pad_idx=0):
    """
    seq: [batch_size, seq_len]
    返回: [batch_size, 1, 1, seq_len]，其中 True 表示真实 token，False 表示 pad
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size, device='cpu'):
    """
    返回: [1, 1, size, size] 的下三角矩阵
    True 表示允许 attend，False 表示不允许（屏蔽未来位置）
    """
    mask = torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)


def make_tgt_mask(tgt, pad_idx=0):
    """
    Decoder Self-Attention 用的复合 mask
    同时屏蔽 padding 位置和未来位置
    tgt: [batch_size, tgt_len]
    返回: [batch_size, 1, tgt_len, tgt_len]
    """
    pad_mask = make_padding_mask(tgt, pad_idx)              # [B, 1, 1, L]
    causal_mask = make_causal_mask(tgt.size(1), tgt.device) # [1, 1, L, L]
    # 二者都为 True 才允许 attend
    return pad_mask & causal_mask
```

{% enddetails %}


## 4. Encoder Block

### 4.1 整体结构
<div style="width:50%;margin:0 auto;">{% asset_img transformer.png Transformer架构图 %}</div>

每个 Encoder Block 由两个子层组成，原论文堆叠 $N=6$ 层：

1. **Multi-Head Self-Attention** — 让句子里的词互相交流，提取全局上下文
2. **Position-wise Feed Forward Network** — 对每个词独立做特征变换

每个子层都包裹在 **Add & Norm**（残差连接 + 层归一化）里，公式：

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

### 4.2 Multi-Head Self-Attention 子层

让句子里的词互相交流，提取全局上下文。

#### 1. 浅层 Encoder Block（通常是前 1-2 层）
* **输出含义**：局部句法特征与短语级别的信息。
* **直觉**： 经过浅层的 Self-Attention，词与词之间开始初步交流。"苹果"看到了它前面的动词是"吃"，于是它的向量内部开始发生变化：关于"科技公司"的特征被抑制，关于"水果"的特征被放大。此时的输出代表了局部搭配和基础语法结构。

#### 2. 深层 Encoder Block（中间到顶层）
* **输出含义**：高度抽象的、全局上下文相关的动态语义（Contextualized Representation）。
* **直觉**： 到了 Encoder 的最后一层，每一个词向量都已经不仅仅是它自己了。最后一个 Block 输出的"苹果"，是一个融合了整句话所有背景信息的超级向量。它代表的意思可能是："在一个阳光明媚的下午，张三满怀期待地咬下去的那颗红富士"。

### 4.3 Feed Forward Network（FFN）

这是一个两层的全连接网络（通常中间用 ReLU 或 GELU 激活函数）。它只对**单个词的向量**进行局部的非线性映射：

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

原论文中 $d_{model}=512$，中间隐层维度 $d_{ff}=2048$（4 倍）。

{% details 为什么要用前馈神经网络？ %}

##### 1. 核心直觉：从"团队开会"到"独立思考"

在 Transformer 的一个 Block 里，信息处理是分两步走的：

* 第一步（Self-Attention）： 相当于**"团队开会"**。句子里的每个词都在四处张望，互相交流，从别的词那里吸取上下文信息（比如"苹果"结合"吃"决定自己是水果）。在这个阶段，词与词之间发生了强烈的信息混合（Mixing）。

* 第二步（FFN）： 相当于**"散会后回工位独立思考"**。开完会后，每个词（带着它新吸收的全局信息）回到自己的位置上，断开与其他词的联系，独立地进行内部信息的消化、提炼和特征转换。

注意一个极其重要的细节： FFN 在学术上全称叫 Position-wise Feed-Forward Network（逐位置前馈网络）。这意味着 FFN 是对句子里的每一个词独立且平等地执行完全相同的计算，处理"苹果"的 FFN 权重，和处理"吃"的 FFN 权重是一模一样的。

##### 2. 结构拆解："升维再降维"的特征提取

从数学结构上看，FFN 其实非常简单，就是一个包含两层线性映射和中间激活函数的多层感知机（MLP）：

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

($x$ 是 Attention 层输出并经过残差和归一化后的词向量，如今的大模型常把 ReLU 换成 GELU 或 SwiGLU)

在原版 Transformer 中，输入词向量维度是 512。$W_1$ 会把它强行拉升到 2048 维（通常是 4 倍），经过激活函数后，$W_2$ 再把它压缩回 512 维。

这在机器学习中叫**"流形展开"**。在低维空间中严重纠缠、难以区分的复杂特征（比如一句话里极其微妙的讽刺意味），被投射到 2048 维的超高维空间后，会变得更容易被线性分割和提取。FFN 就像一个放大镜，先用极高的维度把词向量里的所有细微特征全部撑开、激活，然后再提炼出最核心的精华，压缩回 512 维传给下一层。

##### 3. FFN 是模型的"知识记忆库"

《Transformer Feed-Forward Layers Are Key-Value Memories》（Geva et al., 2021）指出，FFN 实际上充当了大型的"键值对（Key-Value）内存库"：

* 第一层 ($W_1$ / Key)： 相当于模式匹配器。高维空间中的每一个神经元（或者说每一个维度）都在寻找特定的语义模式。比如第 886 号神经元可能专门对"历史事件"敏感，第 1024 号神经元对"编程语法"敏感。

* 第二层 ($W_2$ / Value)： 相当于知识输出器。一旦 $W_1$ 中的某个神经元被激活（比如检测到了"法国的首都"），$W_2$ 就会把对应的具体知识（比如"巴黎"的特征向量）附加到这个词向量上。

##### 4. 从数学角度上提供"非线性"

* Attention 层的短板： 虽然 Self-Attention 里面有 Softmax，但它本质上还是对各个 Value 矩阵进行线性加权求和。如果没有 FFN，哪怕你叠 100 层 Attention，模型依然严重缺乏拟合复杂非线性函数的能力。

* FFN 的补足： FFN 中间夹带的那个非线性激活函数（ReLU/GELU），是整个 Transformer Block 中唯一的非线性来源（除了 Softmax）。正是这成千上万次的非线性激活，赋予了 Transformer 强大的表达能力，使其能够拟合极其复杂的人类语言分布。

{% enddetails %}

{% details 代码：FFN 实现 %}

```py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)     # 升维 512 -> 2048
        self.w_2 = nn.Linear(d_ff, d_model)     # 降维 2048 -> 512
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        hidden = F.relu(self.w_1(x))
        hidden = self.dropout(hidden)
        return self.w_2(hidden)
```

{% enddetails %}

### 4.4 Add & Norm

#### 1. Add（残差连接 / Residual Connection）

公式：$\text{Output} = x + \text{Sublayer}(x)$。为了解决网络加深后的梯度消失/退化问题，保证底层信息能直接短路传到高层。

{% details 残差连接，为什么有效？ %}

##### 1. 为什么网络不能无限加深？（退化问题）

在残差连接出现之前，人们发现了一个诡异的现象：当我们不断增加网络层数时，训练误差不仅没有降低，反而升高了。

* 注意：这不仅是过拟合！ 过拟合是训练误差低、测试误差高。而深度网络面临的是退化问题（Degradation）——层数深了之后，模型连训练集都拟合不好了。
* 因为在传统的网络中，让**非线性层去学习一个完美的"恒等映射"（Identity Mapping，即输出等于输入）是极其困难的**。

##### 2. 核心

传统的网络层是在试图直接学习一个完整的映射函数 $H(x)$。

而残差连接改变了思路：它不让网络层直接学 $H(x)$，而是让网络去**学习输入和输出之间的差值（残差）**，即 $\mathcal{F}(x) = H(x) - x$。

最终的输出公式变成了著名的：

$$y = \mathcal{F}(x) + x$$

##### 3. 数学解释一：前向传播（学 $0$ 比学 $1$ 容易）

假设我们要网络在某一层啥也不干，保持输出等于输入。

* 在传统网络 $y = Wx$ 中，网络必须极其精准地把权重矩阵 $W$ 学习成一个**单位矩阵 $I$**（对角线全是 1，其余全是 0）。这在大规模非线性优化中极其困难。

* 在残差网络 $y = \mathcal{F}(x, W) + x$ 中，网络只需要把权重 $W$ 全部更新为 $0$，就能得到 $y = 0 + x = x$。

* 结论： 神经网络的权重初始化通常都在 $0$ 附近，通过 L2 正则化（Weight Decay）也倾向于把权重往 $0$ 压。因此，学习 $\mathcal{F}(x) \rightarrow 0$ 比学习 $\mathcal{F}(x) \rightarrow x$ 要简单无数倍。残差连接给网络提供了一个"极低成本的保底方案"。

##### 4. 数学解释二：反向传播

假设我们要求损失函数 $L$ 对输入 $x$ 的梯度。根据微积分的链式法则，对 $y = \mathcal{F}(x) + x$ 求导：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

$$ = \frac{\partial L}{\partial y} \cdot \left( \frac{\partial \mathcal{F}(x)}{\partial x} + 1 \right)$$

$$= \frac{\partial L}{\partial y} \cdot \frac{\partial \mathcal{F}(x)}{\partial x} + \frac{\partial L}{\partial y}$$

我们来看这个展开后的结果，它完美分为两项：
* 左边 $\left(\frac{\partial L}{\partial y} \cdot \frac{\partial \mathcal{F}(x)}{\partial x}\right)$： 这是经过复杂网络层（卷积、全连接等）传回来的梯度。如果网络很深，这一项大概率会因为连续相乘而变得极小（梯度消失）。

* 右边 $\left(\frac{\partial L}{\partial y}\right)$：那个 "+1" 产生了一项无损的梯度， 它意味着，无论网络有多深，无论左边的复杂梯度是不是变成了 $0$，高层的梯度 $\frac{\partial L}{\partial y}$ 永远能够通过这个 "+1" 形成的高速公路，原封不动地、100% 地直接传回浅层。

##### 5. 前沿分析

###### 视角一：模型集成（Ensemble View）

有论文指出，一个包含残差连接的深层网络，实际上可以看作是**指数级个浅层网络的隐式集成（Ensemble）**。因为每一次残差连接都提供了两条路（走 $\mathcal{F}(x)$ 或是走跳跃连接），信号的流动有无数种组合。这极大地提升了模型的鲁棒性和泛化能力。

###### 视角二：损失地形更加平滑（Loss Landscape Smoothing）

学术界通过 3D 可视化证明了，传统的深层网络，其损失函数的等高线图就像是充满悬崖峭壁的恶劣山脉，梯度下降极容易卡死在局部最优解或鞍点。而**加入了残差连接后，损失地形会变得像一个平滑的大碗**。这种平滑性使得优化器（如 SGD 或 Adam）能够极其顺畅地滑向全局最优点。

{% enddetails %}

#### 2. Norm（层归一化 / Layer Normalization）

把每个词向量沿特征维度做标准化，再用可学习的 $\gamma, \beta$ 仿射回来。

{% details 为啥要用层归一化 %}

##### 1. 为什么要"归一化"？
在深度神经网络中，随着层数的加深，每一层的数据分布都在不断变化（这在学术上叫 **Internal Covariate Shift，内部协变量偏移**）。
* 痛点： 想象一下，上一层传过来的数据，一会儿在 $[0, 1]$ 徘徊，一会儿又飙升到 $[-1000, 1000]$。下一层的神经元就会彻底"懵圈"，为了适应这种剧烈波动的输入，它不得不拼命调整自己的权重。这会导致梯度忽大忽小，模型极难训练，甚至直接崩溃（梯度爆炸/消失）。
* 归一化的作用： 强行把每一层的数据分布拉回到一个稳定的标准状态（通常是均值为 0，方差为 1 的正态分布）。这就像是给神经元戴上了"防抖云台"，让数据平稳地向后传递。

##### 2. LayerNorm 的数学原理

LayerNorm 的核心思想是：**对某一个具体的词向量（Token），在它的所有特征维度上进行归一化。**

假设我们有一个维度为 $H$（比如 512）的词向量 $x = [x_1, x_2, \dots, x_H]$。

LayerNorm 的计算分为三步：

**第一步：计算该词向量内部的均值和方差**
$$\mu = \frac{1}{H} \sum_{i=1}^{H} x_i$$

$$\sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2$$

**第二步：标准化（减去均值，除以标准差）**

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

(注：$\epsilon$ 是一个极小的常数，如 $1e-5$，纯粹是为了防止分母为 0 导致报错。)

**第三步：仿射变换（极其关键！）**

标准化虽然让数据稳定了，但也**强行破坏了网络好不容易学到的特征表达**。为了弥补这一点，LayerNorm 引入了两个可学习的参数：缩放因子 $\gamma$（Gamma）和偏置 $\beta$（Beta）。

$$y_i = \gamma \hat{x}_i + \beta$$

网络在训练过程中，如果发现"强行拉回标准正态分布"效果不好，它可以通过学习 $\gamma$ 和 $\beta$ 把数据分布再变回去。这相当于给网络留了一条退路，保证了模型的表达能力。

##### 3. LayerNorm vs BatchNorm

我们可以用一个极其直观的"考试分数"比喻来区分它们：
假设有一个班级（Batch Size），学生考了多门课（特征维度 $H$）。

* **BatchNorm (BN)**: 像是在算某一门课的年级排名。把全班所有人的"数学成绩"拉出来算均值和方差。
    * CV 最爱： 图片大小固定，像素特征对齐，非常适用。
    * NLP 的噩梦： 句子的长度参差不齐！你的第 3 个词是动词，我的第 3 个词是标点，强制把它们放一起算均值毫无物理意义。而且 Batch Size 太小时，BN 计算的方差极其不准。
* **LayerNorm (LN)**： 像是在算**某一个学生的总成绩水平**。只看张三这一个学生，把他自己的"语数外物化生"所有成绩加起来算均值和方差。
    * NLP 的救星： 它完全不受 Batch Size 大小的影响，也不受句子长度变化的干扰（不管句子多长，每个词都是独立计算自己的归一化）。因此它是 Transformer 的绝配。

##### 4. RMSNorm

RMSNorm（均方根归一化）是 LayerNorm 的一种"青春版"。它认为 LayerNorm 真正起作用的是"除以方差"来缩放尺度，而"减去均值"做平移其实没什么必要，反而拖慢了计算速度。因此，RMSNorm 直接砍掉了减均值的步骤，只保留方差缩放，既保证了效果，又提升了 10%~50% 的计算效率。LLaMA 系列就用的 RMSNorm。

{% enddetails %}

{% details 代码：LayerNorm 实现 %}

```py
import torch
import torch.nn as nn

class MyLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(MyLayerNorm, self).__init__()
        self.eps = eps

        # 可学习的两个参数
        # gamma (缩放) 初始化为 1，beta (平移) 初始化为 0
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # LayerNorm 是在最后一个维度 (d_model) 上求均值和方差
        # keepdim=True 保持 [B, L, 1] 方便后续广播
        mean = x.mean(dim=-1, keepdim=True)
        # PyTorch 默认计算无偏估计 (除以 N-1)，但 LayerNorm 标准公式是除以 N
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta
```

{% enddetails %}

### 4.5 Pre-Norm vs Post-Norm

子层的"包装方式"有两种写法：

**Post-Norm（原版 Transformer）**
$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**Pre-Norm（GPT、LLaMA 等现代 LLM）**
$$\text{output} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

| | Post-Norm | Pre-Norm |
|---|---|---|
| 残差路径 | 经过 LayerNorm | 直接相加（无损） |
| 深层稳定性 | 差，梯度可能爆炸/消失 | 好，可以堆几十上百层 |
| 是否需要 Warmup | 必须 | 可选 |
| 代表模型 | Vanilla Transformer | GPT-2/3、LLaMA、PaLM |

直觉：Pre-Norm 让残差路径"始终保持原始尺度"，深层堆叠时梯度可以无损通过，这就是为什么大模型清一色 Pre-Norm。

### 4.6 完整 EncoderBlock 代码

```py
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MyMultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Pre-Norm 风格（更稳定，现代实现常用）
        # Sub-layer 1: Multi-Head Self-Attention
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, mask=src_mask))

        # Sub-layer 2: FFN
        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return x
```


## 5. Decoder Block

### 5.1 整体结构

每个 Decoder Block 由**三个**子层组成（比 Encoder 多了一个 Cross-Attention），原论文同样堆叠 $N=6$ 层：

1. **Masked Multi-Head Self-Attention** — 在已生成的目标端序列上做 Self-Attention，但加 Causal Mask 防止偷看未来
2. **Cross-Attention** — Encoder 与 Decoder 的握手层，Q 来自 Decoder，K/V 来自 Encoder 输出
3. **Position-wise FFN** — 同 Encoder

每个子层同样包裹 Add & Norm。

### 5.2 Masked Multi-Head Self-Attention

* **输入**：当前已经生成（或训练时 Teacher Forcing 喂进来）的目标端序列。
* **特殊点**：用 Causal Mask 屏蔽未来位置，每个位置 $t$ 只能 attend 到 $\le t$ 的位置。
* **物理意义**：让模型学到"自回归生成"的约束——预测第 $t$ 个词时，只能依赖前 $t-1$ 个词。
* **直觉**： 假设要翻译成 "I eat an apple"，现在已经生成了 "I eat"。Masked Self-Attention 让 "I" 只看到自己、"eat" 看到 "I" 和自己，但不让任何位置看到后面的 "an apple"。

### 5.3 Cross-Attention

这是 Encoder 和 Decoder **唯一的连接点**，是整个 Transformer 最关键的信息流通枢纽。

#### 1. Q、K、V 的来源

| 角色 | 来自 | 含义 |
|---|---|---|
| $Q$ | Decoder 上一子层（Masked Self-Attn）输出 | "我现在生成到这里，需要什么源端信息？" |
| $K$ | Encoder 最终输出 | "源端有这些信息可以提供，用这些标签来匹配你的查询" |
| $V$ | Encoder 最终输出 | "如果匹配上了，这是你能拿走的内容" |

公式：

$$\text{CrossAttn} = \text{softmax}\left(\frac{Q_{dec} K_{enc}^T}{\sqrt{d_k}}\right) V_{enc}$$

#### 2. Mask 是什么？

只用**源端的 Padding Mask**（屏蔽 Encoder 输入里的 `<pad>` 位置），**不用 Causal Mask**——因为 Decoder 当前位置可以自由地看 Encoder 的全部源端内容（源端是完整给定的）。

#### 3. 物理意义

每个 Decoder 位置都在用自己的 Q 主动**"查询" Encoder 的全部源端表示**，把最相关的部分加权汇总进来。这就是机器翻译的核心机制：生成每个目标词时都重新参考源句的全部内容，决定该看源句的哪一部分。

例如翻译 "Je mange une pomme" → "I eat an apple"：
* 生成 "I" 时，Cross-Attention 会重点关注 "Je"
* 生成 "eat" 时，重点关注 "mange"
* 生成 "apple" 时，重点关注 "pomme"

这种**软对齐（soft alignment）** 是 Transformer 能取代统计机器翻译的核心。

### 5.4 Feed Forward Network

完全等同于 Encoder 的 FFN（结构、维度、参数都一样），不再重复。

### 5.5 完整 DecoderBlock 代码

```py
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MyMultiHeadAttention(d_model, num_heads)
        self.cross_attn = MyMultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Sub-layer 1: Masked Self-Attention
        # Q, K, V 都来自 x（目标端），用 tgt_mask（causal + padding）
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, mask=tgt_mask))

        # Sub-layer 2: Cross-Attention
        # Q 来自 Decoder 的 x，K, V 来自 Encoder 的输出
        # 用 src_mask（源端 padding mask）
        h = self.norm2(x)
        x = x + self.dropout(
            self.cross_attn(h, encoder_output, encoder_output, mask=src_mask)
        )

        # Sub-layer 3: FFN
        h = self.norm3(x)
        x = x + self.dropout(self.ffn(h))
        return x
```


## 6. 完整 Transformer 模型组装

### 6.1 Embedding 层细节

#### 1. Embedding × $\sqrt{d_{model}}$
原论文中 token embedding 输出后会乘以 $\sqrt{d_{model}}$，原因是 `nn.Embedding` 的初始化方差约 $1/d_{model}$ 量级，乘 $\sqrt{d_{model}}$ 后输出方差约为 1，与位置编码的尺度匹配，避免相加时一方淹没另一方。

#### 2. 共享权重（Weight Tying）
原论文里有三个相关的权重矩阵：
* 源端 token embedding $E_{src} \in \mathbb{R}^{V_{src} \times d_{model}}$
* 目标端 token embedding $E_{tgt} \in \mathbb{R}^{V_{tgt} \times d_{model}}$
* 输出投影 $W_{out} \in \mathbb{R}^{d_{model} \times V_{tgt}}$（把 Decoder 输出映射到目标词表）

如果源端和目标端**用同一个共享词表**，可以让 $E_{src} = E_{tgt}$；同时把 $W_{out} = E_{tgt}^T$（"weight tying"），既减少参数量，又让 embedding 空间和输出空间对齐。

### 6.2 Encoder / Decoder 堆叠

* Encoder: 6 个 EncoderBlock 串联，输入是源端 embedding + PE，输出是源端的最终上下文表示
* Decoder: 6 个 DecoderBlock 串联，每一层都接收 Encoder 最终输出做 Cross-Attention

### 6.3 Prediction Head

#### 1. 线性映射层（Linear Layer）
* **输入**：Decoder 给出的抽象意图（$d_{model}$ 维向量）
* **输出**：词表空间上的无差别分数（Logits）
* **直觉**：线性层就像一个庞大的匹配器，把那个"概念"和词汇表里所有的词（比如 5 万个单词）逐一进行内积比对。算出一个长度为 50000 的一维数组，里面装满了未经处理的得分（如 apple 得分 15.2，banana 得分 8.1，car 得分 -3.5）。

#### 2. Softmax 层
* **输出**：下一个词在整个词表上的概率分布
* **直觉**：把 Logits 压缩到 $0 \sim 1$ 之间。此时 apple 的概率可能变成了 $0.95$。这代表模型有 95% 的把握认为下一个词应该输出 "apple"。

注意：训练时通常把 softmax 和 cross-entropy 合并成 `log_softmax + NLLLoss`（数值更稳定）。所以模型本身只输出 logits，softmax 留给 loss 计算或推理阶段。

### 6.4 完整 Transformer 类代码

```py
import math
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size, tgt_vocab_size,
                 d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 d_ff=2048, max_len=5000, dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embedding 层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder / Decoder 堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        # Pre-Norm 风格的最后一层归一化
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        # 输出投影
        self.fc_out = nn.Linear(d_model, tgt_vocab_size, bias=False)
        # Weight tying：把输出投影权重和目标端 embedding 共享
        self.fc_out.weight = self.tgt_embedding.weight

    def make_src_mask(self, src):
        # [B, 1, 1, src_len]
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        # 目标端同时屏蔽 padding 和未来位置
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)   # [B,1,1,L]
        L = tgt.size(1)
        causal = torch.tril(
            torch.ones((L, L), device=tgt.device, dtype=torch.bool)
        )                                                             # [L,L]
        return pad_mask & causal                                      # [B,1,L,L]

    def encode(self, src, src_mask):
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_norm(x)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.decoder_norm(x)

    def forward(self, src, tgt):
        """
        src: [B, src_len]  源端 token id
        tgt: [B, tgt_len]  目标端 token id（训练时是 shift right 后的版本）
        返回: [B, tgt_len, tgt_vocab_size] logits
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.fc_out(decoder_output)
```


## 7. 训练

### 7.1 Teacher Forcing

#### 1. 是什么
训练时 Decoder 的输入**不是它上一步预测出来的词**，而是 ground truth 目标序列（右移一位）。
* Decoder 输入：`[<BOS>, y_1, y_2, ..., y_{T-1}]`
* Decoder 目标（ground truth）：`[y_1, y_2, ..., y_T]`
* 在每个位置预测下一个词，所有位置可以**并行**计算

#### 2. 为什么不用上一步的预测做下一步输入

* **训练效率**：用 ground truth 时所有位置可以并行算（一次 forward）；如果用模型自己的预测，必须像 RNN 一样串行 $T$ 步。
* **训练稳定**：模型刚开始预测全是噪声，如果用噪声当输入，相当于在错误的基础上学习，根本收敛不了。

#### 3. 副作用：Exposure Bias

训练时 decoder 永远看到的是"完美历史"（ground truth），但**推理时它只能看到自己生成的（可能错的）历史**——这两种分布不一致就叫 exposure bias。常见缓解方法：scheduled sampling、minimum risk training 等。

### 7.2 Label Smoothing

#### 1. 是什么
把 one-hot 的硬标签 $[0, 0, 1, 0, ..., 0]$ 改成"软"标签：
* 真实类：概率从 1 降到 $1-\varepsilon$
* 其余 $V-1$ 类：每个分到 $\varepsilon / (V-1)$

原论文用 $\varepsilon = 0.1$。

#### 2. 为什么有用

* **防止过度自信**：one-hot 的目标会鼓励模型把正确类的 logit 推向 $+\infty$、其他全推向 $-\infty$，导致输出分布过尖，泛化变差。
* **缓解过拟合**：相当于在标签上加噪声，是一种正则化。
* **校准（calibration）变好**：模型预测的概率值更接近真实置信度。

代价：训练困惑度（perplexity）会变差，但 BLEU/Accuracy 通常变好。

```py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, pad_idx=0, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        # logits: [N, V] (N = batch * seq_len)
        # target: [N]
        log_probs = F.log_softmax(logits, dim=-1)

        # 构造平滑后的"软标签"分布
        # 注意：vocab 里有一个 pad token 不参与，所以分母是 V-2
        true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0  # pad 不分配概率

        # 屏蔽 padding 位置（不计算 loss）
        mask = (target != self.pad_idx).float().unsqueeze(1)
        loss = -(true_dist * log_probs * mask).sum() / mask.sum()
        return loss
```

### 7.3 Warmup + Inverse Sqrt 学习率调度

#### 1. 公式（原论文）

$$\text{lr} = d_{model}^{-0.5} \cdot \min(\text{step}^{-0.5},\ \text{step} \cdot \text{warmup\_steps}^{-1.5})$$

#### 2. 直觉

* **Warmup 阶段（step ≤ warmup_steps）**：lr 从 0 线性增长。原因是模型参数随机初始化，初期梯度方向极其不稳，大 lr 会让训练直接炸掉。
* **Decay 阶段**：lr 按 $\text{step}^{-0.5}$ 衰减，让模型逐渐收敛。

原论文用 `warmup_steps=4000`，配合 Adam ($\beta_1=0.9, \beta_2=0.98, \varepsilon=10^{-9}$)。

```py
class NoamScheduler:
    """原论文的 Noam 调度：warmup + inverse sqrt decay"""
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** -0.5 * min(
            self.step_num ** -0.5,
            self.step_num * self.warmup_steps ** -1.5
        )
        for p in self.optimizer.param_groups:
            p['lr'] = lr
```

### 7.4 完整训练循环

```py
import torch

# 假设已经有了:
# model: Transformer 实例
# train_loader: 每个 batch 返回 (src, tgt)，都是 [B, L] 的 LongTensor

model = Transformer(src_vocab_size=V_src, tgt_vocab_size=V_tgt).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0,
                             betas=(0.9, 0.98), eps=1e-9)
scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)
criterion = LabelSmoothingLoss(vocab_size=V_tgt, pad_idx=0, smoothing=0.1)

model.train()
for epoch in range(num_epochs):
    for src, tgt in train_loader:
        src, tgt = src.cuda(), tgt.cuda()

        # Teacher Forcing：tgt 右移构造输入和目标
        # tgt = [<BOS>, y1, y2, ..., yT, <EOS>, <PAD>...]
        # tgt_input  = [<BOS>, y1, y2, ..., yT]      (去掉最后)
        # tgt_output = [y1, y2, ..., yT, <EOS>]      (去掉首位)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # forward: 一次性算出所有目标位置的 logits（并行）
        logits = model(src, tgt_input)              # [B, L, V_tgt]

        # loss
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),    # [B*L, V_tgt]
            tgt_output.reshape(-1)                  # [B*L]
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        scheduler.step()
        optimizer.step()
```


## 8. 推理

训练时 Decoder 一次 forward 算完所有位置（Teacher Forcing），但**推理时没有 ground truth**——必须自回归地一步步生成。

每一步：
1. 把已生成的目标序列喂给 Decoder
2. 取最后一个位置的 logits
3. 用某种**解码策略**选下一个 token
4. 加到序列末尾，回到第 1 步，直到生成 `<EOS>` 或达到 max_len

不同的"选下一个 token"的方法就是不同的解码策略。

### 8.1 Greedy Search

每一步取概率最大的 token。简单但容易陷入局部最优——某一步选错了后面就崩盘，且生成结果缺乏多样性（每次都一样）。

```py
import torch
import torch.nn.functional as F

@torch.no_grad()
def greedy_decode(model, src, max_len=50, bos_idx=1, eos_idx=2):
    model.eval()
    src_mask = model.make_src_mask(src)
    encoder_output = model.encode(src, src_mask)

    ys = torch.tensor([[bos_idx]], device=src.device)  # [1, 1]
    for _ in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(ys)
        out = model.decode(ys, encoder_output, src_mask, tgt_mask)
        logits = model.fc_out(out[:, -1])              # [1, V]
        next_token = logits.argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        if next_token.item() == eos_idx:
            break
    return ys
```

### 8.2 Beam Search

每一步保留累积 log-prob 最高的 $k$ 条候选路径（beam），把每条路径都扩展词表 $V$ 个候选，再剪到 top-$k$。最后输出累积分数最高的那条。

* $k=1$ 退化成 greedy
* $k$ 越大质量越高，但耗时也越大
* 翻译/摘要等任务常用 $k=4 \sim 10$

```py
@torch.no_grad()
def beam_search(model, src, beam_size=4, max_len=50,
                bos_idx=1, eos_idx=2, length_penalty=0.6):
    model.eval()
    device = src.device
    src_mask = model.make_src_mask(src)
    encoder_output = model.encode(src, src_mask)            # [1, S, d]

    # 把 encoder 输出复制到 beam_size 份
    encoder_output = encoder_output.expand(beam_size, -1, -1).contiguous()
    src_mask = src_mask.expand(beam_size, -1, -1, -1).contiguous()

    # 初始化 beam
    ys = torch.full((beam_size, 1), bos_idx, dtype=torch.long, device=device)
    log_probs = torch.zeros(beam_size, device=device)
    log_probs[1:] = -1e9   # 第一步只让一个 beam 有效，避免 k 个 beam 算一样的东西

    finished = []  # (sequence, score) 列表

    for _ in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(ys)
        out = model.decode(ys, encoder_output, src_mask, tgt_mask)
        logits = model.fc_out(out[:, -1])                   # [k, V]
        log_p = F.log_softmax(logits, dim=-1)               # [k, V]

        # 累加历史 log-prob，得到所有 (beam, 词) 组合的总分
        scores = log_probs.unsqueeze(1) + log_p             # [k, V]

        # flat 后取 top-k
        V = log_p.size(-1)
        top_scores, top_idx = scores.view(-1).topk(beam_size)
        beam_idx = top_idx // V                             # 来自哪个 beam
        token_idx = top_idx % V                             # 选了哪个词

        ys = torch.cat([ys[beam_idx], token_idx.unsqueeze(1)], dim=1)
        log_probs = top_scores

        # 把已经生成 EOS 的 beam 收藏起来，并把它的 log_prob 设为 -inf 防止再被选
        for i in range(beam_size):
            if token_idx[i].item() == eos_idx:
                # 长度惩罚：避免短序列总分天然占便宜
                lp = (ys.size(1) ** length_penalty)
                finished.append((ys[i].clone(), log_probs[i].item() / lp))
                log_probs[i] = -1e9

        if len(finished) >= beam_size:
            break

    if not finished:
        finished = [(ys[i], log_probs[i].item()) for i in range(beam_size)]
    finished.sort(key=lambda x: x[1], reverse=True)
    return finished[0][0]
```

### 8.3 Top-k / Top-p Sampling

生成式任务（写诗、对话）需要**多样性**，所以引入采样：

* **Temperature**：把 logits 除以 $\tau$。$\tau<1$ 让分布更尖锐（接近 greedy）；$\tau>1$ 让分布更平坦（更随机）。
* **Top-k**：只从概率最高的 $k$ 个 token 中按概率采样（其余截断到 0）。
* **Top-p (nucleus)**：按概率从大到小累加，取累计概率刚刚超过 $p$ 的最小集合，从这个集合里采样。比 top-k 更自适应——分布尖锐时只用很少 token，分布平坦时用很多。

```py
@torch.no_grad()
def sample_decode(model, src, max_len=50, bos_idx=1, eos_idx=2,
                  temperature=1.0, top_k=0, top_p=0.0):
    model.eval()
    src_mask = model.make_src_mask(src)
    encoder_output = model.encode(src, src_mask)
    ys = torch.tensor([[bos_idx]], device=src.device)

    for _ in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(ys)
        out = model.decode(ys, encoder_output, src_mask, tgt_mask)
        logits = model.fc_out(out[:, -1]) / temperature      # [1, V]

        # Top-k 截断
        if top_k > 0:
            v, _ = logits.topk(top_k)
            logits[logits < v[..., -1:]] = -float('inf')

        # Top-p (nucleus) 截断
        if top_p > 0:
            sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
            cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_remove = cum_probs > top_p
            # 右移一位：保留累积刚好达到 top_p 的那个 token
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            # 把"sorted 顺序下要删除的位置"映射回原始顺序
            indices_to_remove = sorted_remove.scatter(-1, sorted_idx, sorted_remove)
            logits = logits.masked_fill(indices_to_remove, -float('inf'))

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        ys = torch.cat([ys, next_token], dim=1)
        if next_token.item() == eos_idx:
            break
    return ys
```

### 8.4 解码策略选择指南

| 场景 | 推荐策略 | 备注 |
|---|---|---|
| 机器翻译、摘要 | Beam Search ($k=4 \sim 10$) | 追求"最优解"，确定性输出 |
| 对话、创作、写诗 | Top-p ($p=0.9$) + Temperature ($\tau=0.7 \sim 1.0$) | 追求多样性 |
| 代码生成 | Greedy 或 Top-p ($p=0.95, \tau=0.2$) | 通常希望生成稳定 |
| 调研/分析 | Greedy + 看 log_prob 阈值 | 排除"过于不确定"的回答 |
