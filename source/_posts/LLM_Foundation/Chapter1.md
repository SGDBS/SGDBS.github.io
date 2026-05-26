---
title: Chapter1 数学工具箱：相似度、散度与 LLM 损失基础
categories: 学习笔记-大模型
date: 2026-03-27 12:48:00
mathjax: true
tags:
    - AI
    - AI面试知识
---

> **本章定位**：整套笔记的"数学地基"。后续所有章节（Attention、InfoNCE、PPO KL 惩罚、DPO 闭式解）都建立在本章的相似度/散度公式之上。

---

# §A 数学原理

## 1. 欧氏距离 (Euclidean Distance) —— 绝对位置的度量

欧氏距离是最基础的 $L_2$ 范数，衡量 $n$ 维空间中两点的直线距离。

$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

* **物理意义**：两点间的位移矢量长度。
* **局限性**：对**量级（Scale）**极其敏感。如果特征未归一化，数值大的维度将主导距离。
* **维度灾难（Curse of Dimensionality）**：高维空间中，由于"测度集中（concentration of measure）"现象，任意两点距离趋于一个相近值——最远点距离与最近点距离的比值 $\to 1$。这也是 LLM Embedding（768~4096 维）几乎不直接用欧氏距离做检索的原因。

---

## 2. 余弦相似度 (Cosine Similarity) —— 方向的共鸣

$$S_{cos}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \frac{\sum_{i=1}^{n} x_i y_i}{\sqrt{\sum x_i^2} \sqrt{\sum y_i^2}}$$

* **取值范围**：$[-1, 1]$。$1$ 同向，$0$ 正交，$-1$ 反向。
* **与欧氏距离的关键转换**：当 $\|\mathbf{x}\| = \|\mathbf{y}\| = 1$ 时，
$$\|\mathbf{x} - \mathbf{y}\|^2 = 2 - 2\mathbf{x}^T\mathbf{y} = 2(1 - \cos\theta)$$
即 **L2 归一化后，最小化欧氏距离 ⇔ 最大化余弦相似度**。这是 BYOL（Ch4）和 CLIP（Ch3）损失设计的数学基础。

---

## 3. 点积 / 内积 (Dot Product) —— 注意力机制的心脏

$$\mathbf{x} \cdot \mathbf{y} = \sum_{i=1}^{n} x_i y_i = \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta$$

**与余弦的关系**：点积 = 余弦相似度 × 两向量模长乘积。L2 归一化后两者等价。

### Scaled Dot-Product Attention 的两个 "为什么"

Transformer 的核心打分函数：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**为什么用点积而非余弦？** 点积保留向量幅值信息（强 query / 弱 query），表达力更强；且只需一次矩阵乘法，硬件友好。

**为什么除以 $\sqrt{d_k}$？** 设 $q, k \in \mathbb{R}^{d_k}$ 各分量独立同分布、零均值、单位方差：
$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$
点积方差随维度 **线性增长**。$d_k$ 较大时，点积值被推到 softmax 饱和区，梯度趋近 0（梯度消失）。除以 $\sqrt{d_k}$ 把方差拉回 $1$，使 softmax 输出保持在合理区间。

> **本章核心句**：Transformer 的注意力分数 = 缩放点积相似度。

---

## 4. 皮尔逊相关系数 (Pearson Correlation)

$$\rho_{x,y} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2}\sqrt{\sum(y_i - \bar{y})^2}}$$

* **数学本质**：**中心化（Mean-centering）后的余弦相似度**。
* **平移不变性**：通过减去均值，自动校准不同 scale。

---

## 5. Jaccard 相似度

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

* **典型应用**：目标检测的 IoU、文本去重 MinHash、tokenizer 评估。

---

## 6. KL 散度与交叉熵 —— 训练流程的灵魂

前面 1–5 节都是**点之间**的相似度。但训练 LLM 时，更关心**两个概率分布**有多接近：模型预测分布 $Q$ vs 真实分布 $P$。

### 6.1 KL 散度

$$D_{KL}(P \| Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)} = \mathbb{E}_{i \sim P}\left[\log \frac{P(i)}{Q(i)}\right]$$

**性质**：
- **不对称**：$D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$
- **非负**：$D_{KL}(P\|Q) \geq 0$，$=0 \iff P = Q$（吉布斯不等式）
- **直观含义**：用分布 $Q$ 编码来自 $P$ 的样本，平均浪费多少 bit

### 6.2 交叉熵

$$H(P, Q) = -\sum_{i} P(i) \log Q(i) = \underbrace{H(P)}_{\text{常数}} + D_{KL}(P \| Q)$$

由于 $H(P)$（真实分布的熵）与模型参数无关：
$$\boxed{\text{最小化交叉熵} \iff \text{最小化 KL 散度}}$$

### 6.3 LLM 训练流程中的角色

| 阶段 | 损失/约束 | 说明 |
| :--- | :--- | :--- |
| **预训练** | $\mathcal{L} = -\sum_t \log P_\theta(x_t \mid x_{<t})$ | Next-token 交叉熵 |
| **SFT** | 同上，但只在 response 上 | Teacher forcing + 交叉熵（Ch5） |
| **知识蒸馏** | $D_{KL}(P_{teacher} \| P_{student})$ | Soft label 对齐 |
| **RLHF / PPO** | reward $- \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$ | KL penalty 防策略漂移（Ch6） |
| **DPO** | log-ratio 间接体现 KL 闭式解 | 见 Ch7 §A 推导 |

> **核心结论**：LLM 训练每一个阶段，本质都在做"用 KL/交叉熵让一个分布逼近另一个分布"。

---

# §B 模型结构（PyTorch 实现）

## B.1 Scaled Dot-Product Attention（4 行实现）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [B, H, L_q, d_k]   K: [B, H, L_k, d_k]   V: [B, H, L_k, d_v]
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)         # [B, H, L_q, L_k]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf')) # 因果/padding mask
    attn = F.softmax(scores, dim=-1)                          # 概率分布
    return attn @ V                                           # [B, H, L_q, d_v]
```

PyTorch 2.0+ 推荐直接用 `F.scaled_dot_product_attention(Q, K, V, is_causal=True)`，内部自动选择 FlashAttention 等高效实现。

## B.2 LM 损失：Cross-Entropy + ignore_index

LLM 训练的核心损失就是 **next-token prediction 交叉熵**：

```python
def lm_loss(logits, labels, ignore_index=-100):
    """
    logits: [B, L, V]    labels: [B, L]
    标签中为 -100 的位置不计入 loss（用于 mask 掉 prompt 部分）
    """
    # 错位：用 t 时刻的 logits 预测 t+1 时刻的 token
    shift_logits = logits[..., :-1, :].contiguous()       # [B, L-1, V]
    shift_labels = labels[..., 1:].contiguous()           # [B, L-1]

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
```

`ignore_index=-100` 是 PyTorch 默认行为：标签 = -100 的位置自动跳过 loss 计算与梯度回传。SFT 中通过把 prompt 部分 labels 设为 -100 实现"只对 response 算 loss"（详见 Ch5）。

## B.3 KL 散度的三种 Estimator

PPO 实现里 KL 不是直接用闭式公式，而是用 **sample-based estimator**（因为 $\pi_\theta$ 和 $\pi_{ref}$ 都是高维分布，无法穷举求和）：

```python
def kl_estimators(logp_theta, logp_ref):
    """
    logp_theta, logp_ref: shape [B, L]，每个 token 在两个 policy 下的对数概率
    返回三种 KL 近似（来自 John Schulman 博客）
    """
    log_ratio = logp_theta - logp_ref      # = log(π_θ / π_ref)

    k1 = log_ratio                                       # 无偏，但方差大
    k2 = 0.5 * log_ratio.pow(2)                          # 总是非负，但有偏
    k3 = (torch.exp(log_ratio) - 1) - log_ratio          # 无偏 + 总是非负 ⭐ TRL 默认

    return k1, k2, k3
```

* **k1**：直接 log-ratio，无偏估计，但**可能为负**（数值不稳定）
* **k2**：平方形式，必然非负但**有偏**
* **k3**：$e^x - 1 - x \geq 0$，**既无偏又非负**，是 TRL/HuggingFace 库的默认选择

> 这个细节面试常考：**为什么 PPO 实现里 KL 是 `(ratio - 1 - log_ratio)` 而不是 `log_ratio`？**

## B.4 L2 归一化与余弦相似度

```python
def cosine_sim(a, b, eps=1e-8):
    """a, b: [B, D]"""
    a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a_norm * b_norm).sum(dim=-1)              # [B]

# PyTorch 内置：
F.cosine_similarity(a, b, dim=-1)                     # 等价
```

**工程小技巧**：训练 embedding 模型时直接 `F.normalize(x, dim=-1)`，下游用内积索引即可（见 Ch3）。

---

# §C 训练与推理

## C.1 训练视角：交叉熵在 LM 中如何按 token 累加

预训练 / SFT 的 loss 实际计算流程：

```
input_ids:  [BOS]  你   好   ， 世  界 [EOS]    ← 输入
labels:     [-100, 你 , 好 , ， 世 , 界 ,EOS]   ← SFT 时 prompt 部分置 -100
                  ↑                            ← 这一位被 ignore
                                               
forward → logits: [B, L, V]
shift  → 用 logits[:, :-1] 预测 labels[:, 1:]
loss   = F.cross_entropy(logits.flatten(0,1), labels.flatten(), ignore_index=-100)
       = - (1/N_valid) * Σ log P(label_t | prefix_<t)
```

**关键点**：
- N_valid 是有效 token 数（排除 -100），而非总长度
- 每个 token 的 loss 是独立累加的（per-token CE），最后取平均
- 所以"短句"和"长句"在 loss 中的权重不同——这也是 SimPO（Ch7）等方法引入"长度归一化"的动机

## C.2 推理视角：向量检索（FAISS / Milvus）

Embedding 模型训练时用余弦相似度，推理时怎么用？

```python
import faiss
import numpy as np

# 1. 构建索引：内存中存所有文档 embedding
embeddings = np.array(doc_embeddings, dtype=np.float32)  # [N, D]
faiss.normalize_L2(embeddings)                            # ⭐ 关键：L2 归一化
index = faiss.IndexFlatIP(embeddings.shape[1])            # 内积索引（IP = Inner Product）
index.add(embeddings)

# 2. 检索：query 也归一化后用内积查
query = np.array([query_embedding], dtype=np.float32)
faiss.normalize_L2(query)
D, I = index.search(query, k=10)                          # 返回 top-10 (距离, 索引)
```

**为什么 L2 归一化后用内积代替余弦？**
归一化后 $\mathbf{x}^T\mathbf{y} = \cos\theta$，省去两次范数除法。在亿级向量库中，**内积索引（IP）比余弦索引（COS）快 30%+**。这就是 OpenAI/BGE/E5 等所有主流 embedding 模型默认输出归一化向量的原因（详见 Ch3）。

## C.3 KL 在 RLHF 推理中其实"不存在"

容易混淆的点：**KL 惩罚只在训练时存在**。
- 训练时：$\text{reward} = r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{ref})$
- 推理时：直接 sampling $y \sim \pi_\theta(\cdot \mid x)$，没有 reference model 参与

KL 的作用是在训练阶段把 $\pi_\theta$ "锚定"在 $\pi_{ref}$ 附近，训练完毕后参考模型就可以丢弃。

---

# §D 总结表

| 度量方法 | 核心属性 | 对量级敏感？ | 在 LLM/Transformer 中的角色 |
| :--- | :--- | :--- | :--- |
| **欧氏距离** | 空间位移 | **是** | 几乎不直接使用（高维退化） |
| **余弦相似度** | 向量方向 | 否 | RAG 检索、Embedding 库默认度量 |
| **点积** | 方向 + 幅值 | **是** | **Scaled Dot-Product Attention 的核心** |
| **皮尔逊系数** | 线性趋势 | 否 | 评测时衡量打分相关性 |
| **Jaccard** | 集合重叠度 | 否 | 数据去重 (MinHash)、Tokenizer 评估 |
| **KL 散度** | 分布差异（不对称） | — | 蒸馏损失、RLHF KL 惩罚 |
| **交叉熵** | 分布拟合 | — | **预训练、SFT 的损失函数** |

---

## 承上启下

本章建立的数学工具中：
- **点积** 直接通往 **Ch2 InfoNCE**（对比学习的核心打分函数）
- **L2 归一化 + 余弦** 通往 **Ch3 CLIP/SimCSE**（embedding 训练）
- **MSE = 2(1-cos)** 通往 **Ch4 BYOL**（无负样本损失设计）
- **交叉熵** 通往 **Ch5 SFT**（LM loss）
- **KL 散度** 通往 **Ch6 PPO** 和 **Ch7 DPO**（对齐算法的核心约束）

下一章进入 Ch2，看看这些工具如何组装出第一个真正的"自监督表示学习"算法。
