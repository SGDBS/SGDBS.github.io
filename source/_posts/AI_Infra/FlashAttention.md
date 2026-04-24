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


Online Softmax 是 FlashAttention 能成立的数学基石。它由 NVIDIA 的 Milakov & Gimelshein 在 2018 年的论文 *"Online normalizer calculation for softmax"* 中提出,核心思想是:**在只看过部分数据的情况下,增量地计算数值稳定的 softmax,并在新数据到来时做修正**。

下面我会从标准 softmax 的问题开始,一步步推导到 online 版本,并解释它在 FlashAttention 中的角色。

### 一、标准 Softmax 与数值稳定性问题

#### 1.1 朴素定义

给定向量 $x = (x_1, x_2, \dots, x_N)$,softmax 定义为:

$$
y_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$

#### 1.2 数值溢出问题

如果某个 $x_i$ 较大(比如 1000),$e^{x_i}$ 会溢出变成 `inf`。即使在 fp16 下,$x_i > 11$ 就会溢出($e^{11} \approx 59874$,fp16 的上限约 65504)。

#### 1.3 Safe Softmax(3-pass 版本)

解决方法是减去最大值:

$$
y_i = \frac{e^{x_i - m}}{\sum_{j=1}^{N} e^{x_j - m}}, \quad m = \max_j x_j
$$

这个变换**数学上完全等价**(分子分母同时乘以 $e^{-m}$),但把最大指数限制在 0,保证 $e^{x_i - m} \in (0, 1]$,不会溢出。

算法变成 **3 遍扫描**:

```
Pass 1: 求最大值 m
  m = -∞
  for i = 1..N:
      m = max(m, x[i])

Pass 2: 求归一化因子 ℓ
  ℓ = 0
  for i = 1..N:
      ℓ += exp(x[i] - m)

Pass 3: 计算最终输出
  for i = 1..N:
      y[i] = exp(x[i] - m) / ℓ
```

**问题**:需要 3 次读取整个 $x$ 向量,对 GPU 这种 memory-bound 场景非常不友好。如果 $x$ 太大放不进 SRAM,就得反复从 HBM 读 3 次。

### 二、Online Softmax 的核心想法

**目标**:把 Pass 1 和 Pass 2 合并成一次扫描,也就是**边扫描边同时维护 $m$ 和 $\ell$**。

难点在于:$\ell$ 的定义依赖 $m$,而 $m$ 要扫完整个数组才能确定。扫到一半时的"局部最大值"可能比真正的最大值小,之前算出来的 $\ell$ 就是错的。

**关键洞察**:如果有办法在 $m$ 发生变化时**修正** $\ell$,就可以一边扫一边算。

#### 2.1 推导修正公式

假设已经处理了前 $i$ 个元素,维护着:

- $m_i$ = 前 $i$ 个元素中的最大值
- $\ell_i = \sum_{j=1}^{i} e^{x_j - m_i}$ = 基于当前最大值的归一化因子

现在来了第 $i+1$ 个元素 $x_{i+1}$,新的最大值是:

$$
m_{i+1} = \max(m_i, x_{i+1})
$$

要得到新的 $\ell_{i+1} = \sum_{j=1}^{i+1} e^{x_j - m_{i+1}}$,观察:

$$
\ell_{i+1} = \underbrace{\sum_{j=1}^{i} e^{x_j - m_{i+1}}}_{\text{旧元素,基准从 } m_i \text{ 改为 } m_{i+1}} + e^{x_{i+1} - m_{i+1}}
$$

对旧部分做一个简单变换:

$$
\sum_{j=1}^{i} e^{x_j - m_{i+1}} = \sum_{j=1}^{i} e^{x_j - m_i} \cdot e^{m_i - m_{i+1}} = \ell_i \cdot e^{m_i - m_{i+1}}
$$

于是得到**更新公式**:

$$
\boxed{\ell_{i+1} = \ell_i \cdot e^{m_i - m_{i+1}} + e^{x_{i+1} - m_{i+1}}}
$$

**几何直觉**: 每次最大值增大(即 $m_{i+1} > m_i$),之前累积的 $\ell_i$ 是基于"旧基准" $m_i$ 的,现在基准变大了,所有旧项都得乘以 $e^{m_i - m_{i+1}} < 1$ 进行**缩小**(rescale)。如果最大值没变($m_{i+1} = m_i$),缩放因子是 $e^0 = 1$,什么都不用调整。

#### 2.2 2-pass 算法

```
Pass 1: 同时计算 m 和 ℓ
  m = -∞, ℓ = 0
  for i = 1..N:
      m_new = max(m, x[i])
      ℓ = ℓ · exp(m - m_new) + exp(x[i] - m_new)
      m = m_new

Pass 2: 计算最终输出
  for i = 1..N:
      y[i] = exp(x[i] - m) / ℓ
```

从 3 遍扫描降到了 2 遍,而且**数值稳定性与 safe softmax 完全相同**(始终保持减去当前最大值)。

### 三、分块(Block-wise)版本:FlashAttention 的真正需求

单元素的更新公式只是开胃菜。FlashAttention 需要**按块处理**——一次性算完一个块的局部 $m$ 和 $\ell$,再和已有的累积量合并。

#### 3.1 两块合并公式

设已经处理完块 A,维护着 $(m_A, \ell_A)$; 现在处理块 B,独立算出 $(m_B, \ell_B)$,其中:

$$
m_A = \max_{j \in A} x_j, \quad \ell_A = \sum_{j \in A} e^{x_j - m_A}
$$

$$
m_B = \max_{j \in B} x_j, \quad \ell_B = \sum_{j \in B} e^{x_j - m_B}
$$

合并后的全局量 $(m, \ell)$ 应为:

$$
m = \max(m_A, m_B)
$$

$$
\boxed{\ell = \ell_A \cdot e^{m_A - m} + \ell_B \cdot e^{m_B - m}}
$$

**这就是 FlashAttention 里那个看起来眼熟的公式**。推导完全同上:两块都要把基准从自己的局部最大值改到全局最大值,各自乘以对应的 $e^{m_{\text{local}} - m_{\text{global}}}$ 缩放因子。

注意这个合并操作是**满足结合律的**——合并 A、B 再合并 C,和先合并 B、C 再和 A 合并,结果一致。这使得它能很自然地并行化(比如 tree reduction)。

### 四、扩展:同时在线更新输出 $O$

在普通 softmax 里,Pass 2 要重新扫一遍算 $y_i$。但在 attention 里,我们最终要的不是 softmax 本身,而是 $O = PV$,其中 $P = \text{softmax}(S)$。FlashAttention 需要**把 $O$ 也一起在线累积**,这样整个算法变成 **1-pass**。

#### 4.1 定义未归一化的输出

定义"未归一化"的输出(只做了减最大值但没除以 $\ell$):

$$
\tilde{O}_A = \sum_{j \in A} e^{x_j - m_A} \cdot V_j
$$

处理块 B 时同理:

$$
\tilde{O}_B = \sum_{j \in B} e^{x_j - m_B} \cdot V_j
$$

合并后全局未归一化输出应为 $\tilde{O} = \sum_{j \in A \cup B} e^{x_j - m} V_j$,同样需要基准变换:

$$
\boxed{\tilde{O} = \tilde{O}_A \cdot e^{m_A - m} + \tilde{O}_B \cdot e^{m_B - m}}
$$

最后一步再除以 $\ell$ 得到真正的 $O = \tilde{O} / \ell$。

#### 4.2 伪代码(FlashAttention 核心)

```
m = -∞, ℓ = 0, O = 0   # O 此处表示未归一化的 Õ
for 每个 K, V 块 j:
    S_j = Q · K_j^T                     # 当前块的分数
    m_j = rowmax(S_j)                   # 当前块最大值
    P_j = exp(S_j - m_j)                # 当前块的未归一化权重
    ℓ_j = rowsum(P_j)
    
    m_new = max(m, m_j)
    α = exp(m - m_new)                  # 旧累积量的缩放因子
    β = exp(m_j - m_new)                # 当前块的缩放因子
    
    ℓ = α · ℓ + β · ℓ_j
    O = α · O + β · P_j · V_j           # 直接累积未归一化输出
    m = m_new

O = O / ℓ                               # 最后才归一化
```

这样,**整个 attention 只需一次扫描 $K, V$**,完全不需要实例化 $N \times N$ 的注意力矩阵。FlashAttention-2 的一个小优化就是把每一步的 `/ ℓ` 推迟到最后,避免反复做除法。

### 五、为什么数学上精确?

这点很多人初学时会怀疑:"边扫边更新,最后结果真的和一次性算一样吗?"

**答案是完全一样**,因为每次合并时:

- $m$ 始终是当前已见元素的真实最大值
- $\ell$ 通过缩放因子修正后,始终等于 $\sum_{j \in \text{已见}} e^{x_j - m_{\text{当前}}}$
- $\tilde{O}$ 同理

扫描结束时,$m$ 等于全局最大值,$\ell$ 等于全局归一化因子,$\tilde{O} / \ell$ 等于标准 attention 的输出。这和"提前知道 $m$ 再一口气算"在代数上完全等价——**没有任何近似**。

### 六、一个数值例子

以 $x = [1, 3, 2, 5]$ 为例,手工走一遍:

**标准做法**: $m = 5$,$\ell = e^{-4} + e^{-2} + e^{-3} + e^0 \approx 0.0183 + 0.1353 + 0.0498 + 1 = 1.2034$

**Online 做法**:

| 步骤 | 新元素 | $m_{\text{new}}$ | 缩放因子 $e^{m_{\text{old}} - m_{\text{new}}}$ | $\ell$ 更新 |
|------|:--:|:--:|:--:|:--|
| 初始 | — | $-\infty$ | — | $\ell = 0$ |
| 步 1 | 1 | 1 | $e^{-\infty - 1} = 0$ | $\ell = 0 \cdot 0 + e^0 = 1$ |
| 步 2 | 3 | 3 | $e^{1 - 3} = 0.1353$ | $\ell = 1 \cdot 0.1353 + e^0 = 1.1353$ |
| 步 3 | 2 | 3 | $e^{3 - 3} = 1$ | $\ell = 1.1353 \cdot 1 + e^{-1} = 1.1353 + 0.3679 = 1.5032$ |
| 步 4 | 5 | 5 | $e^{3 - 5} = 0.1353$ | $\ell = 1.5032 \cdot 0.1353 + e^0 = 0.2034 + 1 = 1.2034$ ✓ |

最后得到 $\ell = 1.2034$,与标准做法**完全一致**。

### 七、工程意义

Online softmax 看似只是一个小技巧,实际上解锁了一整类算法:

1. **FlashAttention** — 让 attention 从 $O(N^2)$ 显存降到 $O(N)$
2. **Ring Attention** — 把序列分片到多 GPU,每个 GPU 持一部分 $K, V$,通过环形通信传递部分结果,用 online softmax 合并。支持百万级 token 序列
3. **流式推理 / Prefix Caching** — 先算一部分 prefix 的 softmax 状态,后续 token 来时增量更新
4. **分布式 softmax** — 跨设备的 softmax 归一化

它的思想可以抽象为:**任何 "reduce + 归一化"的操作,只要能找到带修正因子的合并律,就能分块并行计算**。这一点也和 log-sum-exp 的并行化密切相关(实际上 $\log \ell + m$ 就是 log-sum-exp,online softmax 等价于 log-sum-exp 的在线计算)。
