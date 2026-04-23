---
title: Flash Attention 从入门到入土
categories: 学习笔记- AI Infra
date: 2026-04-23 22:48:00
mathjax: true
tags:
    - AI
    - AI Infra
---


## 1.FlashAttention 整体结构

FlashAttention 是由斯坦福大学 Tri Dao 等人在 2022 年提出的一种**精确**（非近似）的注意力计算算法。它通过重新设计注意力的计算方式，大幅降低了显存占用并加速了训练与推理,现已成为现代大模型（如 GPT、LLaMA 等）的标配。

### 一、问题背景：标准 Attention 的瓶颈

在 Transformer 中，自注意力的核心计算是:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

其中 $Q, K, V \in \mathbb{R}^{N \times d}$，$N$ 是序列长度，$d$ 是每个头的维度。

**标准实现的做法**（PyTorch 默认的 naive 实现）：

1. 计算 $S = QK^T$，得到一个 $N \times N$ 的矩阵,写回 HBM（显存）
2. 计算 $P = \text{softmax}(S)$，再次写回 HBM
3. 计算 $O = PV$，写回 HBM

**两个核心问题：**

- **显存占用 $O(N^2)$**：当 $N=8192$ 时,单个注意力矩阵就要 256MB（fp32）,多头多层叠加显存爆炸
- **速度瓶颈不是算力,而是显存带宽**：GPU 的算力增长远快于显存带宽,注意力计算是典型的 **memory-bound**（访存受限）问题,大量时间花在 HBM 读写上

### 二、GPU 内存层级的关键认知

理解 FlashAttention 必须先理解 GPU 的内存结构:

| 层级 | 容量 | 带宽 | 速度 |
|------|------|------|------|
| HBM（全局显存） | 40–80 GB | ~1.5–3 TB/s | 慢 |
| SRAM（片上共享内存） | ~20 MB（A100 全部 SM 合计） | ~19 TB/s | 快约 10 倍 |

标准 attention 反复在 HBM 中读写大矩阵,SRAM 几乎没被利用。FlashAttention 的核心思路就是:**尽可能把计算留在 SRAM 里,避免往 HBM 写中间结果**。

### 三、核心思想：Tiling + Online Softmax + Recomputation

#### 1. Tiling（分块计算）

把 $Q, K, V$ 沿序列维度切成小块（block）,每次只把一小块加载到 SRAM 中计算,这样 $N \times N$ 的大矩阵永远不会被完整地实例化。

假设将 $Q$ 切成 $T_r$ 块，$K, V$ 切成 $T_c$ 块,算法变成一个双层循环: 外层遍历 $Q$ 的块,内层遍历 $K, V$ 的块。

#### 2. Online Softmax（增量式 softmax）

最大的技术难点是: **softmax 需要看到整行所有元素才能归一化**（要算分母 $\sum e^{x_i}$）,而分块计算时一次只能看到一部分。FlashAttention 采用了 Milakov & Gimelshein 提出的 online softmax 技巧。

为了数值稳定,softmax 通常减去最大值:

$$
\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j
$$

当新的一块数据进来,当前块最大值是 $m^{new}$,则可以这样更新:

$$
m^{total} = \max(m^{old}, m^{new})
$$

$$
\ell^{total} = e^{m^{old} - m^{total}} \cdot \ell^{old} + e^{m^{new} - m^{total}} \cdot \ell^{new}
$$

输出 $O$ 也可以用类似的 rescaling 进行增量更新:

$$
O^{total} = \text{diag}(e^{m^{old} - m^{total}}) \cdot O^{old} + e^{m^{new} - m^{total}} \cdot P^{new} V^{new}
$$

这样每处理完一个 $K, V$ 块,就用缩放因子修正之前累积的结果,**数学上与标准 softmax 完全等价,精度没有任何损失**。

#### 3. Recomputation（反向传播时重算）

在反向传播中,原本需要保存 $N \times N$ 的注意力矩阵 $P$ 来算梯度。FlashAttention 只保存 softmax 的统计量（每行的最大值 $m$ 和归一化因子 $\ell$,各 $O(N)$）,反向时**重新计算** $S$ 和 $P$。

看起来增加了计算量,但由于 attention 是 memory-bound 的,**省下来的 HBM 访问时间远多于重算的时间**,整体反而更快。


### 四、复杂度分析

设 $M$ 是 SRAM 大小:

| 指标 | 标准 Attention | FlashAttention |
|------|:-:|:-:|
| 计算量（FLOPs） | $O(N^2 d)$ | $O(N^2 d)$ |
| **HBM 访问量** | $O(Nd + N^2)$ | $O(N^2 d^2 / M)$ |
| **显存占用** | $O(N^2)$ | $O(N)$ |

计算量没变,但 HBM 访问量在 $d \ll M$ 的常见情况下大幅降低（$d$ 一般是 64 或 128,$M$ 在 A100 上约 100KB 级别）,这是加速的根本来源。


## 3.Online Softmax（增量式 softmax）


## To Be Continued