---
title: LLM面试基础知识 1
date: 2026-03-27 12:48:00
mathjax: true
tags:
    - LLM
---

# Chapter 1 深度理解 AI 中的相似度度量（Similarity Measures）

## 1. 欧氏距离 (Euclidean Distance)
最直观的 $L_2$ 范数，衡量高维空间中两点之间的绝对位移。

**公式：**
$$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} = \|\mathbf{x} - \mathbf{y}\|_2$$

* **特点**：对数值的大小极其敏感。
* **面试考点**：为什么在计算欧氏距离前通常需要做 **Standardization（标准化）**？
    * 因为如果某个特征的量级特别大（如：年收入 vs 年龄），该特征会主导距离计算，导致模型失效。

---

## 2. 余弦相似度 (Cosine Similarity)
通过计算两个向量夹角的余弦值来衡量方向的一致性。

**公式：**
$$S_{cos}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}$$



* **数学本质**：衡量的是向量的方向而非量级。
* **面试考点：与欧氏距离的关系？**
    如果对向量进行 $L_2$ 归一化（即 $\|\mathbf{x}\| = 1, \|\mathbf{y}\| = 1$），则：
    $$\|\mathbf{x} - \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\mathbf{x}^T\mathbf{y} = 2 - 2\cos(\theta)$$
    **结论**：归一化后，欧氏距离越小，余弦相似度越大。

---

## 3. 皮尔逊相关系数 (Pearson Correlation Coefficient)
衡量两个变量之间的线性相关程度。

**公式：**
$$\rho_{x,y} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

* **数学本质**：它是**中心化（Mean-centered）后的余弦相似度**。
* **应用场景**：推荐系统中的协同过滤。它能自动修正用户评分习惯的偏差（如某些用户习惯性打高分）。

---

## 4. Jaccard 相似度 (Jaccard Similarity)
用于衡量集合的重叠度。

**公式：**
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

* **应用场景**：
    1. 文本去重（计算两篇文章词汇集的重叠）。
    2. 目标检测（Object Detection）中的 **IoU (Intersection over Union)** 实际上就是 Jaccard 相似度。

---

## 总结：如何选型？

| 场景 | 推荐度量 | 原因 |
| :--- | :--- | :--- |
| **聚类/KNN** | 欧氏距离 | 关注样本在空间的绝对差异 |
| **文本向量/Embedding** | 余弦相似度 | 消除文本长度（模长）的影响 |
| **评分系统（去偏）** | 皮尔逊系数 | 消除个体评分标准不一的噪声 |
| **离散标签/IoU** | Jaccard | 关注集合元素的重合情况 |