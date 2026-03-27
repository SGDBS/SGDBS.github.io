---
title: LLM面试基础知识 1
date: 2026-03-27 12:48:00
mathjax: true
tags:
    - LLM
---

# Chapter 1 深度理解 AI 中的相似度度量（Similarity Measures）

## 1. 欧氏距离 (Euclidean Distance) —— 绝对位置的度量

欧氏距离是最基础的 $L_2$ 范数。它衡量的是 $n$ 维空间中两个点之间的直线距离。

### 数学表达
$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

### 深度理解
* **物理意义**：两点间的位移矢量长度。
* **局限性**：对特征的**量级（Scale）**极其敏感。如果特征未经过归一化，数值大的维度将主导距离计算。
* **面试陷阱**：在高维空间下，由于数据分布稀疏，欧氏距离的区分度会显著下降（维度灾难）。

---

## 2. 余弦相似度 (Cosine Similarity) —— 方向的共鸣

余弦相似度通过计算向量夹角的余弦值，衡量两个向量在方向上的指向是否一致。

### 数学表达
$$S_{cos}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \frac{\sum_{i=1}^{n} x_i y_i}{\sqrt{\sum_{i=1}^{n} x_i^2} \sqrt{\sum_{i=1}^{n} y_i^2}}$$



### 深度理解
* **物理意义**：衡量的是“形状”而非“大小”。在文本处理中，它能有效忽略文档长度（词频总量）的差异。
* **与欧氏距离的转换**：
    若对向量进行 $L_2$ 归一化（即 $\|\mathbf{x}\| = 1, \|\mathbf{y}\| = 1$），则欧氏距离的平方为：
    $$\|\mathbf{x} - \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\mathbf{x}^T\mathbf{y} = 2(1 - \cos(\theta))$$
    **结论**：归一化后，最小化欧氏距离等价于最大化余弦相似度。

---

## 3. 皮尔逊相关系数 (Pearson Correlation) —— 消除偏见的利器

皮尔逊系数用于衡量两个变量之间的线性相关性，是推荐系统（如协同过滤）的首选。

### 数学表达
$$\rho_{x,y} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

### 深度理解
* **数学本质**：**中心化（Mean-centering）后的余弦相似度**。
* **平移不变性**：通过减去均值 $\bar{x}$，它能自动校准不同用户的打分尺度（例如：打分严苛的用户 vs 打分宽松的用户）。

---

## 4. Jaccard 相似度 —— 集合重叠的艺术

用于衡量离散集合之间的相似性。

### 数学表达
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

### 深度理解
* **典型应用**：在目标检测（Object Detection）中，衡量预测框 (BBox) 与真实框相似度的 **IoU** 指标，其数学本质就是 Jaccard 相似度。
* **适用场景**：One-hot 编码的稀疏特征、用户购买物品清单的相似度。

---

## 5. 总结与选型指南

| 度量方法 | 核心属性 | 对量级敏感？ | 典型场景 |
| :--- | :--- | :--- | :--- |
| **欧氏距离** | 空间位移 | **是** | 聚类、低维几何数据 |
| **余弦相似度** | 向量方向 | 否 | NLP、Embedding 检索 |
| **皮尔逊系数** | 线性趋势 | 否 | 协同过滤、消除用户偏好偏置 |
| **Jaccard** | 集合重叠度 | 否 | 文本去重、目标检测 IoU |

---




# 深度进阶：对比学习（Contrastive Learning）的核心机理

对比学习（Contrastive Learning）是自监督学习（Self-supervised Learning）的明珠。其核心思想是通过构造**正负样本对**，在没有人工标注的情况下学习到具有判别力的特征表示。

---

## 1. 数学本质：InfoNCE 损失函数

现代对比学习（如 SimCLR, MoCo）大多基于 InfoNCE 损失。它本质上是一个 **$K+1$ 路分类问题**，目标是从一堆样本中准确识别出那个“正样本”。

### 公式定义
$$\mathcal{L}_{q, k_+} = -\log \frac{\exp(\text{sim}(q, k_+) / \tau)}{\sum_{i=0}^{K} \exp(\text{sim}(q, k_i) / \tau)}$$

其中：
* $\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}$ 通常使用**余弦相似度**。
* $\tau$ 为**温度参数 (Temperature)**。

### 深度追问：为什么余弦相似度要配合温度参数 $\tau$？
1.  **平滑概率分布**：余弦值的范围是 $[-1, 1]$，直接放入 Softmax 会导致概率分布过于集中，梯度消失。
2.  **难样本挖掘**：较小的 $\tau$ 会放大正负样本间的差异，迫使模型关注那些距离 $q$ 很近的“困难负样本”，从而学习到更精细的特征。

---

## 2. 三种演进范式

### 2.1 Triplet Loss (三元组损失)
早期对比学习的代表。目标是拉近 $(a, p)$，推开 $(a, n)$。
$$\mathcal{L} = \max(d(a, p) - d(a, n) + \alpha, 0)$$
* **局限**：对超参数 $\alpha$ (margin) 敏感，且挖掘“有意义”的三元组计算成本极高。

### 2.2 SimCLR (Large Batch Size)
通过强力数据增强（Crop, Color Jitter 等）产生正样本对。
* **核心策略**：通过极大的 Batch Size（如 4096）来提供成千上万个负样本。
* **工程难点**：对显存要求极高。

### 2.3 MoCo (Momentum Contrast)
将对比学习看作一个**字典查询（Dictionary Lookup）**任务。
* **创新点 1：Queue**。用一个队列维护负样本，不再受 Batch Size 限制。
* **创新点 2：Momentum Encoder**。
    $$\theta_k \leftarrow m\theta_k + (1-m)\theta_q$$
    通过动量更新 Encoder，确保队列中不同时间步存入的特征具有**一致性（Consistency）**。

---

## 3. 面试核心考点：模型坍塌（Collapse）

**问题**：如果模型为了偷懒，给所有图片都输出相同的常数向量，此时 Loss 会很低，但模型废了。这就是“模型坍塌”。

**解决方案**：
* **InfoNCE**：通过负样本（Denominator）提供的“排斥力”强行推开样本。
* **BYOL / SimSiam**：这类算法不需要负样本。它们通过 **Stop-gradient (停止梯度回传)** 和 **Predictor MLP** 打破对称性，从数学上避免了常数解。

---

## 4. 总结对比

| 方法 | 负样本需求 | 核心技巧 | 解决坍塌的方式 |
| :--- | :--- | :--- | :--- |
| **SimCLR** | 大量 (Batch-based) | Data Augmentation | 显式负样本排斥 |
| **MoCo** | 大量 (Queue-based) | Momentum Encoder | 显式负样本排斥 |
| **BYOL** | **不需要** | Dual Networks + EMA | 停止梯度 (Stop-gradient) |

---