---
title: Chapter1 相似度度量（Similarity Measures）
categories: 学习笔记-大模型
date: 2026-03-27 12:48:00
mathjax: true
tags:
    - AI
    - AI面试知识
---

# 相似度度量（Similarity Measures）

## 1. 欧氏距离 (Euclidean Distance) —— 绝对位置的度量

欧氏距离是最基础的 $L_2$ 范数。它衡量的是 $n$ 维空间中两个点之间的直线距离。

### 数学表达
$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

### 深度理解
* **物理意义**：两点间的位移矢量长度。
* **局限性**：对特征的**量级（Scale）**极其敏感。如果特征未经过归一化，数值大的维度将主导距离计算。
* **维度灾难（Curse of Dimensionality）**：高维空间中，由于"测度集中（concentration of measure）"现象，任意两点的距离都趋于一个相近值——最远点距离与最近点距离的比值 $\to 1$。换言之，"最近邻"在高维下失去意义。这也是 LLM Embedding（通常 768~4096 维）几乎不直接用欧氏距离做检索的原因。

---

## 2. 余弦相似度 (Cosine Similarity) —— 方向的共鸣

余弦相似度通过计算向量夹角的余弦值，衡量两个向量在方向上的指向是否一致。

### 数学表达
$$S_{cos}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \frac{\sum_{i=1}^{n} x_i y_i}{\sqrt{\sum_{i=1}^{n} x_i^2} \sqrt{\sum_{i=1}^{n} y_i^2}}$$

* **取值范围**：$[-1, 1]$。$1$ 表示同向，$0$ 表示正交，$-1$ 表示反向。在文本检索中常将其线性映射到 $[0, 1]$ 作为相似度分数。

### 深度理解
* **物理意义**：衡量的是"形状"而非"大小"。在文本处理中，它能有效忽略文档长度（词频总量）的差异。
* **与欧氏距离的转换**：
    若对向量进行 $L_2$ 归一化（即 $\|\mathbf{x}\| = 1, \|\mathbf{y}\| = 1$），则欧氏距离的平方为：
    $$\|\mathbf{x} - \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\mathbf{x}^T\mathbf{y} = 2(1 - \cos(\theta))$$
    **结论**：归一化后，最小化欧氏距离等价于最大化余弦相似度。

### 在 LLM 中的角色
* **RAG / 语义检索**：FAISS、Milvus、Pinecone 等向量数据库的默认度量之一。
* **为什么 OpenAI `text-embedding-3`、BGE、E5 等 Embedding 模型输出做 L2 归一化？**
    归一化后 $\mathbf{x}^T\mathbf{y} = \cos\theta$，可以直接用**内积索引（IP index）**代替余弦索引——内积计算比余弦少了两次范数除法，在亿级向量库中能显著提速。

---

## 3. 点积 / 内积 (Dot Product) —— 注意力机制的心脏

点积是**未归一化**的余弦相似度，同时编码了"方向一致性"和"幅值大小"。

### 数学表达
$$\mathbf{x} \cdot \mathbf{y} = \sum_{i=1}^{n} x_i y_i = \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta$$

### 与余弦相似度的关系
$$\text{Dot Product} = \text{Cosine Similarity} \times \|\mathbf{x}\| \cdot \|\mathbf{y}\|$$

也就是说，点积 = 方向相似度 × 模长乘积。当向量已 L2 归一化时，二者完全等价。

### 在 Transformer 中的核心地位：Scaled Dot-Product Attention

注意力机制的打分函数本质就是点积：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

* **为什么用点积而非余弦？** 点积保留了向量幅值信息，让模型能学到"某些 query 比其他更'强'"，表达力更强；同时计算上只需一次矩阵乘法，硬件极其友好。
* **为什么要除以 $\sqrt{d_k}$？**
    设 $q, k \in \mathbb{R}^{d_k}$ 各分量独立同分布、零均值、单位方差，则：
    $$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$
    点积方差随维度 $d_k$ **线性增长**。当 $d_k$ 较大时，点积值会被推到 softmax 的饱和区，梯度趋近于 0（梯度消失）。除以 $\sqrt{d_k}$ 把方差拉回 $1$，使 softmax 输出保持在合理区间。

> 这是第一章最该记住的一句话：**Transformer 的注意力分数 = 缩放点积相似度**。

---

## 4. 皮尔逊相关系数 (Pearson Correlation) —— 消除偏见的利器

皮尔逊系数用于衡量两个变量之间的线性相关性，是推荐系统（如协同过滤）的首选。

### 数学表达
$$\rho_{x,y} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

### 深度理解
* **数学本质**：**中心化（Mean-centering）后的余弦相似度**。
* **平移不变性**：通过减去均值 $\bar{x}$，它能自动校准不同用户的打分尺度（例如：打分严苛的用户 vs 打分宽松的用户）。

---

## 5. Jaccard 相似度 —— 集合重叠的艺术

用于衡量离散集合之间的相似性。

### 数学表达
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

### 深度理解
* **典型应用**：在目标检测（Object Detection）中，衡量预测框 (BBox) 与真实框相似度的 **IoU** 指标，其数学本质就是 Jaccard 相似度。
* **适用场景**：One-hot 编码的稀疏特征、用户购买物品清单的相似度。

---

## 6. 分布间的"距离"：KL 散度与交叉熵 —— 训练流程的灵魂

前面 1–5 节都是**点之间**的相似度。但训练 LLM 时，我们更关心**两个概率分布**有多接近：模型预测分布 $Q$ vs 真实/参考分布 $P$。这类度量叫做**散度（Divergence）**，不对称，因此严格说不是"距离"，但功能上扮演相同角色。

### 6.1 KL 散度 (Kullback–Leibler Divergence)

$$D_{KL}(P \| Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$$

* **不对称性**：$D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$。
* **非负性**：$D_{KL}(P\|Q) \geq 0$，当且仅当 $P = Q$ 时取 0。
* **直观含义**：用分布 $Q$ 去编码来自 $P$ 的样本，平均会浪费多少 bit。

### 6.2 交叉熵 (Cross-Entropy)

$$H(P, Q) = -\sum_{i} P(i) \log Q(i) = H(P) + D_{KL}(P \| Q)$$

由于 $H(P)$（真实分布的熵）与模型参数无关，**最小化交叉熵 ⇔ 最小化 KL 散度**。

### 在 LLM 训练流程中的角色

| 阶段 | 损失/约束 | 作用 |
| :--- | :--- | :--- |
| **预训练 (Pretraining)** | $\mathcal{L} = -\sum_t \log P_\theta(x_t \mid x_{<t})$ | Next-token 交叉熵，本质就是让模型分布逼近数据分布 |
| **SFT** | 同上，但在指令数据上 | Teacher forcing + 交叉熵 |
| **知识蒸馏 (Distillation)** | $D_{KL}(P_{teacher} \| P_{student})$ | 学生模型对齐教师模型的 soft logits |
| **RLHF / DPO** | $\mathcal{L}_{RL} - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$ | KL 惩罚项防止策略模型偏离参考模型太远（reward hacking） |

> 把这张表记牢：**LLM 训练的每一个阶段，本质都在做"用 KL/交叉熵让一个分布逼近另一个分布"。**

---

## 7. 总结与选型指南

| 度量方法 | 核心属性 | 对量级敏感？ | 典型场景 | 在 LLM/Transformer 中的角色 |
| :--- | :--- | :--- | :--- | :--- |
| **欧氏距离** | 空间位移 | **是** | 聚类、低维几何数据 | 几乎不直接使用（高维退化） |
| **余弦相似度** | 向量方向 | 否 | NLP、Embedding 检索 | RAG 检索、Embedding 库默认度量 |
| **点积** | 方向 + 幅值 | **是** | 注意力打分 | **Scaled Dot-Product Attention 的核心** |
| **皮尔逊系数** | 线性趋势 | 否 | 协同过滤、消除用户偏好偏置 | 评测时衡量打分相关性 |
| **Jaccard** | 集合重叠度 | 否 | 文本去重、目标检测 IoU | 数据去重 (MinHash)、Tokenizer 评估 |
| **KL 散度** | 分布差异（不对称） | — | 变分推断、蒸馏 | 蒸馏损失、RLHF KL 惩罚 |
| **交叉熵** | 分布拟合 | — | 分类 / 语言建模 | **预训练、SFT 的损失函数** |

---
