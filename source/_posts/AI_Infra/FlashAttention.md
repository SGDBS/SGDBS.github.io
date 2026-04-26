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


## 2.Online Softmax


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



## 3.Recomputation(重计算)详解

Recomputation 是 FlashAttention 反向传播的核心技巧。它的本质是一个**用计算换显存**的权衡——前向时故意不保存中间结果,反向时重新算一遍。这个思想本身不是 FlashAttention 发明的(在深度学习中叫 gradient checkpointing,2016 年就有了),但 FlashAttention 把它用到了极致,并且在这里**不但不慢,反而更快**。

### 一、为什么反向传播需要中间结果?

先回顾反向传播的基本原理:链式法则需要"前向时的中间激活值"。

举个最简单的例子,$y = \sigma(Wx)$,反向求 $\frac{\partial L}{\partial W}$:

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \sigma'(Wx) \cdot x^T
$$

这里需要 $\sigma'(Wx)$,而 $\sigma'$ 依赖前向的中间量 $Wx$。所以默认情况下,**所有中间激活值都得保存到反向用完为止**。

### 二、标准 Attention 反向需要什么?

前向是:

$$
S = QK^T, \quad P = \text{softmax}(S), \quad O = PV
$$

反向时给定 $\frac{\partial L}{\partial O} = dO$,要求 $dQ, dK, dV$。推导一下(详细推导步骤请看附录A“反向传播公式推导”):

$$
dV = P^T \cdot dO
$$

$$
dP = dO \cdot V^T
$$

$$
dS = P \odot (dP - \text{rowsum}(dP \odot P)) \quad \text{(softmax 反向)}
$$

$$
dQ = dS \cdot K, \quad dK = dS^T \cdot Q
$$


**需要的中间量**:

- $P$(用于算 $dV$ 和 $dS$)—— $N \times N$ 矩阵
- $S$ 本身不一定要,因为 $P$ 已经隐含了 $S$ 的信息

所以标准实现要保存整个 $N \times N$ 的 $P$ 矩阵,反向期间一直占着显存。这又把我们带回了 $O(N^2)$。

### 三、FlashAttention 的选择:只保存最小必要信息

FlashAttention 前向结束后,**只保留两样东西用于反向**:

| 保留的量 | 形状 | 含义 |
|---------|------|------|
| $O$ | $N \times d$ | 前向的输出(本来就要保留) |
| $L$ | $N \times 1$ | 每行的 "logsumexp" 统计量 |

其中 $L$ 是每行的:

$$
L_i = m_i + \log \ell_i
$$

$m_i$ 和 $\ell_i$ 就是上一节 online softmax 里的最大值和归一化因子。**两个标量被合并成了一个**(后面会解释为什么这样更方便)。

**总共额外需要的显存**:$O(N) + O(Nd) = O(N)$,比原来省了 $N \times N$ 矩阵那么多。

### 四、反向时怎么重算?

反向传播依然是**分块**进行的,和前向一样按 tile 扫描。核心观察是:

> 给定 $Q_i, K_j, L_i$,可以**完全重建**某个小块的 $P_{ij}$。

#### 4.1 重建 $P$ 的公式

对于块 $(i, j)$:

$$
S_{ij} = Q_i K_j^T \quad \text{(在 SRAM 里重算)}
$$

$$
P_{ij} = \exp(S_{ij} - L_i) \quad \text{(直接得到归一化后的 softmax)}
$$

**这里的关键点**:$L_i = m_i + \log \ell_i$,所以

$$
\exp(S_{ij} - L_i) = \exp(S_{ij} - m_i - \log \ell_i) = \frac{\exp(S_{ij} - m_i)}{\ell_i}
$$

正好是标准 softmax 的结果。把 $m$ 和 $\ell$ 合并成 $L$ 的好处就在这:**一次减一次 exp 就拿到归一化的 $P$,不用再做除法**。

#### 4.2 反向算法(简化版)

```
输入: Q, K, V, O, dO, L, 块大小 B_r, B_c
初始化 dQ = 0, dK = 0, dV = 0 (都在 HBM)

for i = 1..T_r:                      # 遍历 Q 块(也就是遍历行)
    加载 Q_i, O_i, dO_i, L_i 到 SRAM
    
    for j = 1..T_c:                   # 遍历 K, V 块
        加载 K_j, V_j 到 SRAM
        
        # === 重算前向 ===
        S_ij = Q_i K_j^T              # 重新计算分数
        P_ij = exp(S_ij - L_i)        # 重新得到 softmax 权重
        
        # === 算梯度 ===
        dV_j += P_ij^T · dO_i         # 累加 dV
        dP_ij = dO_i · V_j^T          # P 的梯度
        
        # softmax 反向: 需要 rowsum(dP ⊙ P)
        D_i = rowsum(dO_i ⊙ O_i)      # 这是个技巧,等价于 rowsum(dP_ij ⊙ P_ij)
        dS_ij = P_ij ⊙ (dP_ij - D_i)
        
        dQ_i += dS_ij · K_j           # 累加 dQ
        dK_j += dS_ij^T · Q_i         # 累加 dK
    
    写回 dQ_i 到 HBM

写回 dK, dV 到 HBM
```

注意里面那个 $D_i = \text{rowsum}(dO_i \odot O_i)$ 的小技巧——softmax 反向本来需要 $\text{rowsum}(dP \odot P)$,但在 attention 里有个恒等式:

$$
\text{rowsum}(dP \odot P) = \text{rowsum}(dO \odot O)
$$

这意味着这个量可以直接用 $O$ 和 $dO$ 算出来,**不需要完整的 $P$ 和 $dP$**。这个 $D_i$ 本来可以预先一次性算好(形状只有 $N \times 1$),让反向循环更干净。

### 五、计算量的账:真的能接受吗?

重算显然多了 FLOPs,算算具体多多少。

**前向计算量**:

- $S = QK^T$: $2N^2 d$ FLOPs
- softmax: $O(N^2)$
- $O = PV$: $2N^2 d$ FLOPs

总共约 $4N^2 d$。

**反向计算量(标准实现,不重算)**:

- $dV = P^T dO$: $2N^2 d$
- $dP = dO V^T$: $2N^2 d$
- softmax 反向: $O(N^2)$
- $dQ = dS \cdot K$: $2N^2 d$
- $dK = dS^T Q$: $2N^2 d$

总共约 $8N^2 d$。

**FlashAttention 反向(带重算)**:

- 多算一遍 $S = QK^T$: $+2N^2 d$
- 其他同上

总共约 $10N^2 d$。

**增加了大约 25%** 的 FLOPs(从 $8N^2d$ 到 $10N^2d$)。

### 六、为什么增加 25% FLOPs 反而更快?

这是 FlashAttention 最反直觉的地方:**明明多算了,为什么还更快?**

答案是之前反复强调的一点:**attention 是 memory-bound,不是 compute-bound**。

#### 6.1 现代 GPU 的算力 / 带宽失衡

A100 的数字:

- FP16 算力:312 TFLOPS
- HBM 带宽:1.5 TB/s
- **算力/带宽比约 200 FLOPs/byte**

这意味着:每从 HBM 读一个字节,GPU 最好能做 200 次浮点运算才"划算"。如果做不到(算术强度低于这个比值),GPU 就在等数据——算力闲置。

Attention 的瓶颈就在于 $N^2$ 的中间矩阵读写,算术强度很低。

#### 6.2 标准反向 vs FlashAttention 反向的实际耗时

| 项目 | 标准反向 | FlashAttention 反向 |
|------|---------|---------------------|
| FLOPs | $8N^2d$ | $10N^2d$(多 25%) |
| HBM 读写 | $O(N^2)$(读写 $P, dP, dS$ 等大矩阵) | $O(N^2 d^2 / M)$(只读写 $Q, K, V, O, dO, dQ, dK, dV$ 和标量 $L, D$) |
| 实际瓶颈 | HBM 带宽 | 算力 |

因为 HBM 读写从 $O(N^2)$ 降到接近 $O(Nd)$(忽略 block 层面的重复访问),省掉的 HBM 访问时间 **远多于** 多算 25% FLOPs 的时间。

举个粗略的估算:若 $N = 4096, d = 64$,标准反向要读写的 $P, dP$ 等大约是 $4 \times N^2 \times 2 = 128$ MB,在 1.5 TB/s 的带宽下约耗时 85 μs。而多算的 $2N^2 d$ FLOPs 在 312 TFLOPS 下只需 **0.2 μs**。两个数量级的差距。

### 七、和通用 Gradient Checkpointing 的关系

通用的 gradient checkpointing(Chen et al. 2016)是:

- 前向只在某些"checkpoint"保存激活
- 反向时从最近的 checkpoint 重跑一段前向

FlashAttention 的 recomputation 可以看作是**针对 attention 的专用 checkpointing**:

- checkpoint 的是什么?$L$(softmax 统计量)和 $O$
- 重算的是什么?$S$ 和 $P$ 这两个大矩阵

但 FlashAttention 的做法比通用 checkpointing 更精细:

1. **融合 kernel**:重算和梯度计算在同一个 CUDA kernel 里完成,中间值只活在 SRAM,不走 HBM
2. **按需重算,粒度是块**:不是重跑整段前向,而是反向循环遍历到哪个 block 就重算哪个 block
3. **倒过来快**:通用 checkpointing 只是节省显存、会更慢;FlashAttention 因为跳过了 HBM 读写,**同时更省显存并且更快**

这是一个理论上罕见的"两全其美"——它得以成立,完全依赖于现代 GPU 算力 / 带宽的失衡。

### 八、总结

Recomputation 在 FlashAttention 中解决的核心问题是:**反向传播怎么不保存 $N \times N$ 的 $P$ 矩阵**。

关键要点:

1. **只存 $L = m + \log \ell$ 和 $O$**(都是 $O(N)$ 或 $O(Nd)$),不存 $S, P$
2. 反向时用 $Q_i, K_j, L_i$ **在 SRAM 里重算** $S_{ij}$ 和 $P_{ij}$,用完即丢
3. 利用 $\text{rowsum}(dP \odot P) = \text{rowsum}(dO \odot O)$ 这个恒等式避免需要完整的 $dP$
4. 多了约 25% 的 FLOPs,但因为 attention 是 memory-bound,省下的 HBM 读写时间远超重算的开销

更高层的启示是:在现代 GPU 上,**"少存一点、多算一点" 经常是更好的选择**。算力在持续增长,带宽却增长缓慢,这个剪刀差会让 recomputation 这种技术越来越普遍。

## 4. 附录A

{% details $Y = AB$ ->  $dA = dY \cdot B^T, \quad dB = A^T \cdot dY $ %}

**结论 1**:若 $Y = AB$,则

$$
dA = dY \cdot B^T, \quad dB = A^T \cdot dY
$$

这是矩阵求导里最基础也最重要的结论。我之前给的推导比较简略,这里从头一步步讲清楚,包括直觉、严格推导、和几种不同的验证方法。

### 一、先把问题说清楚

**设定**:
- $A \in \mathbb{R}^{m \times k}$
- $B \in \mathbb{R}^{k \times n}$  
- $Y = AB \in \mathbb{R}^{m \times n}$
- $L$ 是某个标量损失函数,依赖 $Y$(通过 $Y$ 间接依赖 $A$ 和 $B$)

**已知**:$dY := \frac{\partial L}{\partial Y} \in \mathbb{R}^{m \times n}$(上游传下来的梯度,形状和 $Y$ 一样)

**目标**:求 $dA = \frac{\partial L}{\partial A} \in \mathbb{R}^{m \times k}$ 和 $dB = \frac{\partial L}{\partial B} \in \mathbb{R}^{k \times n}$

**关键约定**:梯度 $\frac{\partial L}{\partial X}$ 的形状**总是和 $X$ 本身一样**。也就是说,$dA$ 的第 $(i,k)$ 元素表示 $\frac{\partial L}{\partial A_{ik}}$。这个约定叫 "denominator layout",是深度学习的通用惯例。

### 二、标量形式的矩阵乘法

先把矩阵乘法写成逐元素的形式:

$$
Y_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}
$$

即 $Y$ 的第 $(i,j)$ 个元素是 $A$ 的第 $i$ 行和 $B$ 的第 $j$ 列的点积。

### 三、推导 $dA$

#### 3.1 应用多变量链式法则

$L$ 依赖 $A$ 是通过所有 $Y_{ij}$ 实现的,所以:

$$
\frac{\partial L}{\partial A_{ik}} = \sum_{i'=1}^{m} \sum_{j=1}^{n} \frac{\partial L}{\partial Y_{i'j}} \cdot \frac{\partial Y_{i'j}}{\partial A_{ik}}
$$

这里 $i', j$ 遍历 $Y$ 的所有元素,因为改变 $A_{ik}$ **可能**影响 $Y$ 的每一个元素,所以要全部求和。

#### 3.2 计算 $\frac{\partial Y_{i'j}}{\partial A_{ik}}$

从 $Y_{i'j} = \sum_p A_{i'p} B_{pj}$ 出发。

**$A_{ik}$ 是否出现在 $Y_{i'j}$ 的表达式里?** 分情况讨论:

- 如果 $i' \neq i$:表达式里是 $A_{i'p}$ 的各种形式,第一个下标是 $i'$,而我们关心的 $A_{ik}$ 第一个下标是 $i$,不同。**根本不出现**,所以偏导为 0。
- 如果 $i' = i$:$Y_{ij} = \sum_p A_{ip} B_{pj} = A_{i1}B_{1j} + A_{i2}B_{2j} + \dots + A_{ik}B_{kj} + \dots$。其中只有 $p = k$ 那一项含 $A_{ik}$,系数是 $B_{kj}$。

所以:

$$
\frac{\partial Y_{i'j}}{\partial A_{ik}} = \begin{cases} B_{kj} & \text{若 } i' = i \\ 0 & \text{否则} \end{cases} = \mathbb{1}[i' = i] \cdot B_{kj}
$$

#### 3.3 代回链式法则

把这个结果代入:

$$
\frac{\partial L}{\partial A_{ik}} = \sum_{i'=1}^{m} \sum_{j=1}^{n} dY_{i'j} \cdot \mathbb{1}[i' = i] \cdot B_{kj}
$$

由于指示函数 $\mathbb{1}[i' = i]$ 只在 $i' = i$ 时为 1,外层对 $i'$ 的求和塌缩,只剩下 $i' = i$ 这一项:

$$
\frac{\partial L}{\partial A_{ik}} = \sum_{j=1}^{n} dY_{ij} \cdot B_{kj}
$$

#### 3.4 识别这是一个矩阵乘法

看这个式子的结构:$\sum_j dY_{ij} B_{kj}$。

标准矩阵乘法 $(XZ)_{ik} = \sum_j X_{ij} Z_{jk}$,需要内部下标匹配。我们现在的 $B_{kj}$ 内部下标是 $j$,但 $k$ 在**前面**,所以要把 $B$ 转置一下:$B^T_{jk} = B_{kj}$。

于是:

$$
\frac{\partial L}{\partial A_{ik}} = \sum_{j=1}^{n} dY_{ij} \cdot B^T_{jk} = (dY \cdot B^T)_{ik}
$$

写成矩阵形式:

$$
\boxed{dA = dY \cdot B^T}
$$

**形状验证**:$dY \in \mathbb{R}^{m \times n}$,$B^T \in \mathbb{R}^{n \times k}$,乘积形状是 $m \times k$,与 $A$ 一致 ✓

### 四、推导 $dB$

完全类似的过程。

#### 4.1 链式法则

$$
\frac{\partial L}{\partial B_{kj}} = \sum_{i=1}^{m} \sum_{j'=1}^{n} \frac{\partial L}{\partial Y_{ij'}} \cdot \frac{\partial Y_{ij'}}{\partial B_{kj}}
$$

#### 4.2 计算 $\frac{\partial Y_{ij'}}{\partial B_{kj}}$

从 $Y_{ij'} = \sum_p A_{ip} B_{pj'}$ 出发。

- 如果 $j' \neq j$:$B_{pj'}$ 第二个下标是 $j'$,不含 $B_{kj}$,偏导为 0。
- 如果 $j' = j$:$Y_{ij} = \sum_p A_{ip} B_{pj}$,其中 $p = k$ 的项是 $A_{ik} B_{kj}$,其他项无关。

所以:

$$
\frac{\partial Y_{ij'}}{\partial B_{kj}} = \mathbb{1}[j' = j] \cdot A_{ik}
$$

#### 4.3 代回

$$
\frac{\partial L}{\partial B_{kj}} = \sum_{i=1}^{m} dY_{ij} \cdot A_{ik}
$$

#### 4.4 识别为矩阵乘法

现在要组成一个结果形状为 $k \times n$ 的矩阵,下标是 $(k,j)$。式子里是 $\sum_i A_{ik} dY_{ij}$。

$A_{ik} = A^T_{ki}$,所以:

$$
\frac{\partial L}{\partial B_{kj}} = \sum_{i=1}^{m} A^T_{ki} \cdot dY_{ij} = (A^T \cdot dY)_{kj}
$$

写成矩阵形式:

$$
\boxed{dB = A^T \cdot dY}
$$

**形状验证**:$A^T \in \mathbb{R}^{k \times m}$,$dY \in \mathbb{R}^{m \times n}$,乘积形状 $k \times n$,与 $B$ 一致 ✓

### 五、直觉理解:为什么一定是转置?

光有严格推导还不够,这里讲两种**直觉**,帮助你以后不看推导也能秒写出来。

#### 5.1 形状匹配法(最快)

已知 $Y = AB$,形状 $(m,n) = (m,k) \cdot (k,n)$。

想求 $dA$,形状必须是 $(m,k)$。能用到的两块积木是 $dY$(形状 $m \times n$)和 $B$(形状 $k \times n$)。

要得到 $m \times k$,唯一合理的方式是 $(m \times n) \cdot (n \times k) = (m \times k)$,所以 $B$ 必须转置。

$$
dA = dY \cdot B^T
$$

同理求 $dB$,形状必须 $(k,n)$,用 $A$ 和 $dY$ 拼:$(k \times m) \cdot (m \times n) = (k \times n)$,所以 $A$ 要转置。

$$
dB = A^T \cdot dY
$$

**这个技巧几乎万能**。很多人做推导就是靠形状对齐。

#### 5.2 "什么乘什么得到 Y" 视角

看 $Y = AB$:

- $A$ 在**左边**,所以它的梯度也要"从左边构造出来",剩下的 $B$ 就得挪到右边,但为了形状对,$B$ 要转置 → $dA = dY \cdot B^T$
- $B$ 在**右边**,所以它的梯度也要"从右边构造出来",$A$ 挪到左边并转置 → $dB = A^T \cdot dY$

**助记**:梯度的公式就是把 $Y = AB$ 里的 $Y$ 换成 $dY$,另一个因子转置并保持在原来的位置。

### 六、用更紧凑的方法验证:全微分

上面是硬推,这里给一种优雅的验证方法,也是研究者常用的技巧。

**核心工具**:对标量 $L$,有

$$
dL = \text{tr}\left(\left(\frac{\partial L}{\partial X}\right)^T dX\right)
$$

意思是:$L$ 的微小变化 = 梯度与 $X$ 微小变化的"内积"(迹形式的 Frobenius 内积)。如果能把 $dL$ 写成 $\text{tr}(M^T dX)$,那么 $\frac{\partial L}{\partial X} = M$。

#### 6.1 对 $A$ 求导

$Y = AB$,固定 $B$,$Y$ 的微分为 $dY = dA \cdot B$(这里 $dA$ 是 $A$ 的微小变化,不是梯度,记号冲突但上下文清楚)。

$$
dL = \text{tr}(dY^T \cdot dY_{\text{梯度}}) = \text{tr}((dA \cdot B)^T \cdot dY_{\text{梯度}})
$$

为避免混淆,换下记号:令 $G_Y = \frac{\partial L}{\partial Y}$。

$$
dL = \text{tr}(G_Y^T \cdot dY) = \text{tr}(G_Y^T \cdot dA \cdot B)
$$

利用迹的循环性 $\text{tr}(XYZ) = \text{tr}(ZXY)$:

$$
dL = \text{tr}(B \cdot G_Y^T \cdot dA) = \text{tr}((G_Y \cdot B^T)^T \cdot dA)
$$

对比 $dL = \text{tr}\left(\left(\frac{\partial L}{\partial A}\right)^T dA\right)$,读出:

$$
\frac{\partial L}{\partial A} = G_Y \cdot B^T = dY \cdot B^T \quad \checkmark
$$

#### 6.2 对 $B$ 求导

固定 $A$,$dY = A \cdot dB$:

$$
dL = \text{tr}(G_Y^T \cdot A \cdot dB)
$$

对比 $dL = \text{tr}\left(\left(\frac{\partial L}{\partial B}\right)^T dB\right)$:

$$
\frac{\partial L}{\partial B} = (G_Y^T \cdot A)^T = A^T \cdot G_Y = A^T \cdot dY \quad \checkmark
$$

这种方法**不需要下标**,几行就搞定,是研究矩阵微分的标准套路。推荐掌握。

### 七、一个小例子手动验证

取最小的情况:$A \in \mathbb{R}^{1 \times 2}$,$B \in \mathbb{R}^{2 \times 1}$,则 $Y = AB \in \mathbb{R}^{1 \times 1}$ 是标量。

记 $A = [a_1, a_2]$,$B = [b_1, b_2]^T$,则:

$$
Y = a_1 b_1 + a_2 b_2
$$

**直接求偏导**:

- $\frac{\partial Y}{\partial a_1} = b_1$,$\frac{\partial Y}{\partial a_2} = b_2$,所以 $dA_{\text{直接}} = [b_1, b_2] = B^T$
- $\frac{\partial Y}{\partial b_1} = a_1$,$\frac{\partial Y}{\partial b_2} = a_2$,所以 $dB_{\text{直接}} = [a_1, a_2]^T = A^T$

**用公式验证**(取 $dY = 1$,因为 $Y$ 就是 $L$):

- $dA = dY \cdot B^T = 1 \cdot B^T = B^T$ ✓
- $dB = A^T \cdot dY = A^T \cdot 1 = A^T$ ✓

手算结果和公式结果完全吻合。

### 八、扩展:常见变体

掌握了基础结论,其他情况都能直接推导:

| 前向 | 反向 |
|------|------|
| $Y = AB$ | $dA = dY \cdot B^T$,$dB = A^T \cdot dY$ |
| $Y = A^T B$ | $dA = B \cdot dY^T$,$dB = A \cdot dY$ |
| $Y = AB^T$ | $dA = dY \cdot B$,$dB = dY^T \cdot A$ |
| $Y = ABC$ | $dA = dY \cdot (BC)^T$,$dB = A^T \cdot dY \cdot C^T$,$dC = (AB)^T \cdot dY$ |

规律就是前面说的:**每个因子的梯度 = 把公式里对应位置换成 $dY$,其他因子保持位置不变,但要转置**。

### 九、在 Attention 里的具体应用

回到上一节,这三处都用了结论 1:

1. **$O = PV \Rightarrow dV = P^T \cdot dO, dP = dO \cdot V^T$**
   - $P$ 在左,所以 $dV$ 公式里 $P$ 挪到左边并转置
   - $V$ 在右,所以 $dP$ 公式里 $V$ 挪到右边并转置

2. **$S = QK^T$**,看成 $S = Q \cdot (K^T)$,所以:
   - $dQ = dS \cdot (K^T)^T = dS \cdot K$
   - $d(K^T) = Q^T \cdot dS$,再转置:$dK = dS^T \cdot Q$

这正是为什么那些公式"看起来对称但不完全对称"——$Q, K$ 在前向里不是对称进入的($K$ 带了转置),反向里自然也不对称。

### 十、总结

$$
\boxed{Y = AB \implies dA = dY \cdot B^T, \quad dB = A^T \cdot dY}
$$

**三种理解角度**:

1. **标量链式法则**:逐元素写出 $Y_{ij} = \sum_p A_{ip} B_{pj}$,用多元链式法则加起来,识别出矩阵乘法结构
2. **全微分 + 迹**:$dL = \text{tr}((\partial L/\partial X)^T dX)$,用迹的循环性凑出梯度
3. **形状对齐 / 位置对应**:$dA$ 形状必须与 $A$ 一致,倒推转置位置;或者记"梯度公式就是前向公式中某个因子换成 $dY$"

三种方法殊途同归。日常使用推荐方法 3(最快),遇到陌生情况用方法 2(最严谨),方法 1 是理解背后机制的基础。

---

需要我接着讲结论 2(softmax 雅可比)的严格推导吗?那个用到 Kronecker delta 和链式法则,但逻辑和这里完全类似。


{% enddetails %}