---
title: 1. Flash Attention
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


## 2. 推理场景：Prefill / Decode 与 KV Cache

第 1 节我们用训练前向作为切入点，分析了 attention 的 memory-bound 本质。但 FlashAttention 在生产环境最大的价值其实在**推理服务**——LLM 部署后每秒成千上万次的请求都在跑这个 kernel。

推理和训练的计算模式不完全相同。GPT 类自回归模型的推理分**两个阶段**，瓶颈完全不同，对 attention kernel 的要求也不一样。理解这两个阶段之后，才能完整地看到 FlashAttention 在工程上的意义，也能理解为什么后续会演化出 FlashAttention-2、FlashDecoding 等变体。

### 一、GPT 的两阶段推理

给定一个 prompt（比如 "今天天气真好"），模型要生成后续 token。这个过程分为：

- **Prefill（预填充）**：一次性处理整个 prompt。输入是 N 个 token，模型并行算出它们的 Q, K, V，然后做一次完整的（带因果 mask 的）attention。这一步只发生一次。
- **Decode（解码）**：从生成第一个新 token 开始，每次只输入 1 个 token（上一步刚生成的），计算它的 Q, K, V，与**前面所有 token 的 K, V** 做 attention，得到 logits 后采样下一个 token。这一步循环执行，直到 EOS 或达到最大长度。

数据流如下（省略 batch 维度，单层视角）：

```
Step 0: Prefill (prompt 长度 N=6)

  Q[1..6], K[1..6], V[1..6]  ∈ R^{6×d}
              │
              ▼
  ┌──────────────────────────────────────────┐
  │  S = Q·Kᵀ ∈ R^{6×6}  (causal mask)       │
  │  P = softmax(S)                          │
  │  O = P·V    ∈ R^{6×d}                    │  ← 完整 N×N attention
  └──────────────┬───────────────────────────┘
                 │
                 ▼ 同时把 K, V 存进 cache
  ┌──────────────────────────────────────────┐
  │  KV Cache (HBM):  K[1..6],  V[1..6]      │
  └──────────────────────────────────────────┘


Step 1: Decode 生成第 7 个 token

  Q[7], K[7], V[7]  ∈ R^{1×d}   ← 只算 1 行!
       │
  ┌────┴────────────────────────────────────┐
  │  读 KV Cache: K[1..6], V[1..6]          │
  └────┬────────────────────────────────────┘
       ▼
  ┌─────────────────────────────────────────┐
  │  S = Q[7] · K[1..7]ᵀ ∈ R^{1×7}          │
  │  P = softmax(S)                         │
  │  O = P · V[1..7]    ∈ R^{1×d}           │
  └─────────────┬───────────────────────────┘
                │
                ▼ 追加 K[7], V[7]
  ┌──────────────────────────────────────────┐
  │  KV Cache (HBM):  K[1..7],  V[1..7]      │
  └──────────────────────────────────────────┘


Step 2 ~ T-1: 重复 Decode，每一步 cache 增长 1 行
```

注意几个关键点：

1. **Prefill 的计算模式和训练前向几乎一样**——都是 N×d 的 Q, K, V，都要算完整的 N×N attention 矩阵。所以**第 1 节里讲的所有 memory-bound 分析、FlashAttention 的所有优化，直接就适用于 prefill**。
2. **Decode 的 Q 只有 1 行**，根本不存在所谓的 "N×N 注意力矩阵"——这一步要算的是 1×(N+t) 的 attention，瓶颈完全是另一回事。
3. **每一步 decode 都依赖前面所有步的 K, V**——这就是 KV Cache 存在的根本原因。

### 二、为什么需要 KV Cache

不存 cache 行不行？假设不存，每生成第 t 个 token 时，要重新计算前 t-1 个 token 的 K, V。整个序列生成完，每个 token 的 K, V 平均会被重算 T/2 次——**生成 T 个 token 总共做了 O(T²) 量级的 K/V 重复投影**。

存了 cache 之后：

- 每一步 decode 只算**当前这 1 个 token** 的 Q, K, V，是 O(d²) 的开销
- 之前所有 token 的 K, V 直接从 cache 里读出来，不再重算
- 然后做 1×(N+t) 的 attention

总计算量从 O(T²) 量级降到 O(T) 量级——**计算量的问题被解决了**。

但代价是什么？**每一步 decode 都要把整个 KV cache 从 HBM 完整读出来一次**。一个典型的 7B 模型，prefill 之后 KV cache 大小为：

$$
\text{KV cache size} = 2 \times L \times N \times d_{\text{model}} \times \text{bytes}
$$

（2 是 K 和 V 两个张量,$L$ 是层数,$d_{\text{model}}$ 是隐藏维度。）以 LLaMA-7B、$N=2048$、fp16 为例,KV cache ≈ $2 \times 32 \times 2048 \times 4096 \times 2$ ≈ **1 GB**。每生成一个 token 都要把这 1 GB 从 HBM 完整搬一次到 SRAM。

**所以 KV cache 把"算力问题"转换成了"带宽问题"**——这恰好是 FlashAttention 最擅长解决的那一类问题。

### 三、Decode 的算术强度有多低

把训练前向（或 prefill）和 decode 放在一起对比：

| 阶段 | Q 形状 | KV 形状 | 主要算力 | 主要 HBM 流量 | 算术强度 |
|------|--------|---------|---------|--------------|---------|
| 训练 / Prefill | N×d | N×d | $O(N^2 d)$ | $O(N^2)$ 中间矩阵 + $O(Nd)$ 输入 | $\sim N$ (随序列长增长) |
| Decode 单步 | 1×d | (N+t)×d | $O((N+t) \cdot d)$ | $O((N+t) \cdot d)$ KV cache | **$\sim 1$ (常数!)** |

Decode 单步要做的算力大约是 $O(N \cdot d)$（Q 和 N 行 K 各做一次内积，再用 P 与 N 行 V 加权求和），读取的 KV cache 字节量也是 $O(N \cdot d)$ 量级——**算术强度只有 ~1 FLOP/byte**。

对照第 1 节给的 A100 数字：算力/带宽比 200 FLOPs/byte。这意味着 decode 时 GPU 的 312 TFLOPS 算力**几乎完全闲置**，时间全部花在等 KV cache 从 HBM 加载。如果一个 decode kernel 跑 100 μs，可能 99 μs 在搬数据，1 μs 在算。

这就是 **memory-bound 的极端版本**。第 1 节里我们用训练为例已经论证了 attention 是 memory-bound，但 decode 把这个矛盾推到了顶峰：

- 训练 / prefill：算术强度 $\sim N$（数百到数千），低于临界值,已经 memory-bound
- Decode：算术强度 $\sim 1$（常数级），低于临界值**两个数量级以上**，重度 memory-bound

### 四、FlashAttention 在推理里的角色

理解了这两个阶段，再看 FlashAttention 在推理中扮演什么角色：

**(1) Prefill：FlashAttention v1 直接适用。** Prefill 就是一次大的 attention 前向，FlashAttention 的 tiling + online softmax 完整适用。在长 prompt 场景（RAG、长文档总结、代码补全），prefill 阶段就是延迟瓶颈，FlashAttention 直接把它的延迟和显存都降下来。

**(2) Decode：v1 不够好，需要 FlashDecoding。** v1 的算法结构是"外层循环遍历 Q 块，内层循环遍历 K, V 块"。这套并行划分基于一个假设：**Q 有很多行可以分给不同的 SM 并行处理**。但在 decode 时 Q 只有 1 行，外层循环只有 1 步——**整个 GPU 的几十个 SM 里只有 1 个在干活，其余全部闲置**。

FlashDecoding 的做法是反过来：**把长长的 K, V 维度切分到多个 SM 上并行处理**，每个 SM 负责一段 K, V，独立算出局部的 attention 输出和 softmax 统计量 $(m, \ell)$，最后用 online softmax 的合并公式把所有局部结果归一成最终输出。

注意这里 online softmax 的"分块合并律"再次起到了关键作用——它使得 attention 这个看起来必须看全行才能归一的操作，可以被任意方式切分并行后再合并。在训练时它解决了"按 K, V 块**串行扫描**"的问题，在 decode 时它解决了"按 K, V 块**并行 reduce**"的问题，**同一个数学结构在两种场景里被反向利用**。

### 五、小结

把推理这条线补上之后，FlashAttention 的故事就完整了：

| 场景 | 计算模式 | 瓶颈 | FlashAttention 的角色 |
|------|---------|------|---------------------|
| 训练前向 | N×N attention | $N^2$ 中间矩阵的 HBM 读写 | v1 的 tiling + online softmax |
| 训练反向 | N×N attention + 梯度 | $N^2$ 矩阵保存 + HBM 读写 | v1 的 recomputation |
| 推理 Prefill | 同训练前向 | 同上 | v1 直接适用 |
| 推理 Decode | 1×N attention，每步读 cache | KV cache 的 HBM 读取 | FlashDecoding（K/V 维度并行） |

从工程哲学上看，可以总结出一个对偶：

- **KV Cache**：用 HBM 存储换计算（不重算 K, V）
- **FlashAttention Recomputation**：用计算换 HBM 存储（不存 $N \times N$ 矩阵）

两者都是在"算力 vs 带宽"这把剪刀差下做的工程取舍，方向相反但目的一致——**让宝贵的 HBM 带宽花在刀刃上**。

理解了这一点，再回去看后面的 Online Softmax 和 Recomputation，会发现它们不只是训练里的优化技巧，而是支撑整个现代 LLM 推理服务能跑起来的数学基础。


## 3.Online Softmax


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



## 4.Recomputation(重计算)详解

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

## 5. 附录A

{% details 结论一：若 $Y = AB$，则  $dA = dY \cdot B^T, \quad dB = A^T \cdot dY $ %}

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


{% enddetails  %}


{% details  结论二： 设 $p = \text{softmax}(s)$，则 $dS = P \odot \big(dP - \text{rowsum}(dP \odot P)\big)$ %}


**结论 2**:设 $p = \text{softmax}(s)$,其中 $s, p \in \mathbb{R}^N$,则:

$$
ds = p \odot dp - p \cdot (p^T dp) = p \odot (dp - \rho), \quad \rho = p^T dp = \sum_k p_k \, dp_k
$$

扩展到按行 softmax 的矩阵形式:

$$
dS = P \odot \big(dP - \text{rowsum}(dP \odot P)\big)
$$

下面从零开始推导。

### 一、softmax 的定义回顾

对向量 $s = (s_1, s_2, \dots, s_N) \in \mathbb{R}^N$,softmax 输出:

$$
p_i = \frac{e^{s_i}}{\sum_{k=1}^{N} e^{s_k}}, \quad i = 1, \dots, N
$$

记分母 $Z = \sum_k e^{s_k}$,则 $p_i = e^{s_i} / Z$。

**两个关键性质**:

1. $\sum_i p_i = 1$
2. $p_i > 0$ 对所有 $i$

### 二、求雅可比矩阵 $\frac{\partial p_i}{\partial s_j}$

这是整个推导的核心。我们要算:**$p_i$ 对每个 $s_j$ 的偏导**。

由于 $p_i = e^{s_i} / Z$,而 $Z$ **依赖所有的 $s_j$**(不只是 $s_i$),这是关键——改变任何一个 $s_j$ 都会通过 $Z$ 影响每个 $p_i$。

#### 2.1 分情况推导

用商的求导法则:

$$
\frac{\partial p_i}{\partial s_j} = \frac{\partial}{\partial s_j}\left(\frac{e^{s_i}}{Z}\right) = \frac{\frac{\partial e^{s_i}}{\partial s_j} \cdot Z - e^{s_i} \cdot \frac{\partial Z}{\partial s_j}}{Z^2}
$$

先算两个关键偏导:

$$
\frac{\partial e^{s_i}}{\partial s_j} = \begin{cases} e^{s_i} & \text{若 } i = j \\ 0 & \text{否则} \end{cases} = e^{s_i} \delta_{ij}
$$

$\frac{\partial Z}{\partial s_j} = \frac{\partial}{\partial s_j}\sum_k e^{s_k} = e^{s_j}$(只有 $k = j$ 那一项有贡献)

其中 $\delta_{ij}$ 是 Kronecker delta:$i = j$ 时为 1,否则为 0。

#### 2.2 代入整理

$$
\frac{\partial p_i}{\partial s_j} = \frac{e^{s_i} \delta_{ij} \cdot Z - e^{s_i} \cdot e^{s_j}}{Z^2}
$$

分子分母同除以 $Z^2$:

$$
\frac{\partial p_i}{\partial s_j} = \frac{e^{s_i}}{Z} \delta_{ij} - \frac{e^{s_i}}{Z} \cdot \frac{e^{s_j}}{Z}
$$

注意 $e^{s_i} / Z = p_i$,$e^{s_j} / Z = p_j$,所以:

$$
\boxed{\frac{\partial p_i}{\partial s_j} = p_i \delta_{ij} - p_i p_j}
$$

#### 2.3 分两种情况的形式

写得更直观一点:

$$
\frac{\partial p_i}{\partial s_j} = \begin{cases} p_i (1 - p_i) & \text{若 } i = j \\ -p_i p_j & \text{若 } i \neq j \end{cases}
$$

对角上是 $p_i(1-p_i)$,非对角是 $-p_i p_j$。注意这是一个**对称**雅可比(因为 $p_i p_j = p_j p_i$)。

#### 2.4 验证一下

简单验证:对角上 $p_i(1 - p_i) > 0$(softmax 的输出是增函数,增大 $s_i$ 会增大 $p_i$),非对角 $-p_i p_j < 0$(增大 $s_j$ 会减小别的 $p_i$),符合直觉。

还可以验证每列求和为 0:

$$
\sum_i \frac{\partial p_i}{\partial s_j} = \sum_i (p_i \delta_{ij} - p_i p_j) = p_j - p_j \sum_i p_i = p_j - p_j \cdot 1 = 0 \quad \checkmark
$$

这反映了 $\sum_i p_i = 1$ 这个约束——所有 $p_i$ 对任意 $s_j$ 的变化量加起来必须为 0。

### 三、雅可比的矩阵写法

把 $\frac{\partial p_i}{\partial s_j} = p_i \delta_{ij} - p_i p_j$ 整理成矩阵:

$$
J = \text{diag}(p) - p p^T
$$

其中:
- $\text{diag}(p)$ 是对角矩阵,对角线上是 $p_1, p_2, \dots, p_N$(对应 $\delta_{ij} p_i$ 项)
- $p p^T$ 是外积矩阵,第 $(i,j)$ 元素是 $p_i p_j$

**形状**:$J \in \mathbb{R}^{N \times N}$,$J_{ij} = \frac{\partial p_i}{\partial s_j}$。

### 四、反向传播:求 $ds$

#### 4.1 链式法则

给定上游梯度 $dp \in \mathbb{R}^N$($dp_i = \frac{\partial L}{\partial p_i}$),反向得到:

$$
ds_j = \frac{\partial L}{\partial s_j} = \sum_{i=1}^{N} \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial s_j} = \sum_{i=1}^{N} dp_i \cdot J_{ij}
$$

写成矩阵形式:

$$
ds = J^T dp = J dp \quad \text{(因为 J 对称)}
$$

#### 4.2 代入 $J$ 的表达式

$$
ds = (\text{diag}(p) - pp^T) dp = \text{diag}(p) \cdot dp - p p^T dp
$$

逐项分析:

- $\text{diag}(p) \cdot dp$:对角矩阵左乘向量,等价于**逐元素相乘**,即 $p \odot dp$
- $p p^T dp$:先算 $p^T dp$,这是个**标量**(两个向量的点积),记为 $\rho = \sum_k p_k dp_k$。然后 $p \cdot \rho$ 是每个 $p_i$ 乘以这个标量。

所以:

$$
\boxed{ds = p \odot dp - \rho \cdot p, \quad \rho = p^T dp = \sum_k p_k \, dp_k}
$$

#### 4.3 因式分解形式

把 $p$ 提出来(每项都含 $p_i$):

$$
ds_j = p_j \cdot dp_j - \rho \cdot p_j = p_j (dp_j - \rho)
$$

写成向量:

$$
\boxed{ds = p \odot (dp - \rho)}
$$

这就是结论 2 的向量形式。$\rho$ 是个标量,被**广播**到 $dp$ 的每个位置做减法。

### 五、直觉理解:$ds$ 公式在说什么

这个公式有很清晰的几何/概率直觉。

#### 5.1 $\rho = \sum_k p_k dp_k$ 是什么?

这是 **$dp$ 在 $p$ 分布下的加权平均**(因为 $\sum_k p_k = 1$,$p$ 本身是概率分布)。

#### 5.2 $(dp - \rho)$ 是什么?

是 $dp$ **减去其加权平均**,即"中心化"后的 $dp$。

#### 5.3 为什么要中心化?

因为 softmax 有个根本约束:$\sum_i p_i = 1$,所以**所有 $p_i$ 同时增加 1 单位** 是不可能的——它们必须"此消彼长"。

从梯度角度看:如果 $dp$ 是一个常量向量(所有 $dp_i$ 相等),说明损失对 $p$ 的每个分量"要求"同样的变化,但 softmax 无法同时满足(受约束)。此时应该 $ds = 0$。

验证:若 $dp_i = c$(常量),则 $\rho = \sum_k p_k \cdot c = c$,所以 $dp - \rho = 0$,进而 $ds = 0$。**完美符合直觉**。

#### 5.4 为什么乘 $p$?

中心化后,还要乘以 $p$(逐元素)。这反映了:$p_i$ 小的分量对 $s_i$ 的变化**不敏感**(因为 $p_i \approx 0$,改变 $s_i$ 影响很小)。

### 六、扩展到矩阵(按行 softmax)

在 attention 里,$P = \text{softmax}(S)$ 是按 $S$ 的每一行独立做 softmax。设 $S, P \in \mathbb{R}^{N \times N}$,则**不同行之间互不影响**——第 $i$ 行的 $P$ 只依赖第 $i$ 行的 $S$。

所以把前面的结论逐行应用即可。对第 $i$ 行:

$$
\rho_i = \sum_k P_{ik} \, dP_{ik} = (P \odot dP) \text{ 的第 } i \text{ 行之和}
$$

$$
dS_{ij} = P_{ij} (dP_{ij} - \rho_i)
$$

写成矩阵形式:

$$
\boxed{dS = P \odot (dP - \text{rowsum}(dP \odot P))}
$$

其中 $\text{rowsum}(\cdot)$ 返回一个 $N \times 1$ 的列向量,每一行是对应行的和。减法时这个列向量被**广播到整行**(每行都减同一个标量)。

### 七、一个具体例子手算验证

取 $N = 2$,$s = (1, 2)$。

**前向**:

$$
p_1 = \frac{e}{e + e^2} = \frac{1}{1 + e}, \quad p_2 = \frac{e^2}{e + e^2} = \frac{e}{1 + e}
$$

带入数值($e \approx 2.718$):$p_1 \approx 0.269$,$p_2 \approx 0.731$。

**雅可比**:

$$
J = \text{diag}(p) - pp^T = \begin{pmatrix} p_1 - p_1^2 & -p_1 p_2 \\ -p_1 p_2 & p_2 - p_2^2 \end{pmatrix} \approx \begin{pmatrix} 0.197 & -0.197 \\ -0.197 & 0.197 \end{pmatrix}
$$

(注意到 $p_1 p_2 = p_1(1-p_1) = p_2(1-p_2)$,因为 $N=2$ 时 $p_2 = 1 - p_1$。)

**假设上游梯度** $dp = (1, 0)$(即 $L = p_1$)。

**用公式算**:

- $\rho = p^T dp = p_1 \cdot 1 + p_2 \cdot 0 = p_1 \approx 0.269$
- $dp - \rho = (1 - 0.269, 0 - 0.269) = (0.731, -0.269)$
- $ds = p \odot (dp - \rho) = (0.269 \times 0.731, 0.731 \times (-0.269)) \approx (0.197, -0.197)$

**直接用雅可比**:

- $ds = J^T dp = J dp = (0.197, -0.197)$ ✓

**再直接对 $p_1$ 求偏导验证**:

- $\frac{\partial p_1}{\partial s_1} = p_1(1-p_1) \approx 0.197$ ✓
- $\frac{\partial p_1}{\partial s_2} = -p_1 p_2 \approx -0.197$ ✓

三种方法结果一致,公式正确。

### 八、为什么这个结果特别好?

在 attention 的反向里,这个公式有几个重要的工程价值:

#### 8.1 $ds$ 可以逐行、逐元素算

公式 $dS_{ij} = P_{ij}(dP_{ij} - \rho_i)$ 是一个**逐元素**操作(加上每行一个共享的 $\rho_i$)。

这意味着如果把 $P, dP$ 分块,每个 $(i, j)$ 块可以独立算 $dS_{ij}$——只需要知道对应行的 $\rho_i$ 这一个标量。这正是 FlashAttention 能分块反向的基础。

#### 8.2 $\rho_i$ 可以预计算或替换

$\rho_i = \sum_k P_{ik} dP_{ik}$ 看似需要完整的 $P$ 和 $dP$,但上一节的恒等式告诉我们:

$$
\rho_i = \text{rowsum}(P \odot dP)_i = \text{rowsum}(dO \odot O)_i =: D_i
$$

所以 $D_i$ 可以**在反向循环开始前一次性算好**(只用 $dO$ 和 $O$),每个 block 需要时直接读取。

#### 8.3 避免显式构造雅可比

若老老实实构造 $J = \text{diag}(p) - pp^T$,需要 $O(N^2)$ 存储一个稠密矩阵,然后做 $Jdp$ 这个矩阵-向量乘法 $O(N^2)$ 次运算。

用因式分解形式 $ds = p \odot (dp - \rho)$:

- 算 $\rho$:$O(N)$ 次乘加
- 减去 $\rho$:$O(N)$ 次减法
- 逐元素乘 $p$:$O(N)$ 次乘法

**总共 $O(N)$**,不需要显式构造雅可比矩阵。这在 $N$ 大(长序列)时是关键的优化。

### 九、另一种推导路径:用交叉熵直觉

如果你学过分类问题里的 softmax + 交叉熵,可能见过一个著名公式:

$$
L = -\sum_i y_i \log p_i \Rightarrow \frac{\partial L}{\partial s_j} = p_j - y_j
$$

这个公式看起来和我们推的完全不同,但其实是一致的——它是**特殊情况**。

当损失是交叉熵 $L = -\sum_i y_i \log p_i$ 时:

$$
dp_i = \frac{\partial L}{\partial p_i} = -\frac{y_i}{p_i}
$$

代入通用公式 $ds = p \odot (dp - \rho)$:

$$
\rho = \sum_k p_k \cdot \left(-\frac{y_k}{p_k}\right) = -\sum_k y_k = -1 \quad \text{(假设 } y \text{ 是 one-hot 或概率分布)}
$$

$$
ds_j = p_j \left(-\frac{y_j}{p_j} - (-1)\right) = p_j - y_j \quad \checkmark
$$

一致!这也再次验证了通用公式的正确性。交叉熵之所以那么简洁,是因为它的 $dp$ 形式特别,让 $\rho = -1$ 消得很干净。

### 十、总结

**核心结果**:

$$
\boxed{ds = p \odot (dp - \rho), \quad \rho = p^T dp}
$$

**三层理解**:

1. **数学层面**:softmax 的雅可比是 $\text{diag}(p) - pp^T$,利用这个矩阵的特殊结构(一个对角加一个秩一),$Jdp$ 有 $O(N)$ 的分解形式
2. **几何层面**:$\rho$ 是 $dp$ 在 $p$ 分布下的加权平均,$dp - \rho$ 是中心化,反映了 softmax 的"概率和为 1"约束——共同平移不改变 softmax
3. **工程层面**:分解形式省时省空间,可以**逐行、逐块**计算,让 FlashAttention 的分块反向成为可能;$\rho$ 可以预计算或用 $dO \odot O$ 替换,彻底绕过 $N \times N$ 的 $P, dP$

**在 FlashAttention 中的角色**:这个结论加上结论 1(矩阵乘的梯度)和 $\rho = \text{rowsum}(dO \odot O)$ 的恒等式,构成了 attention 反向传播分块计算的完整数学基础。


{% enddetails%}


{% details 反向传播的公式推导 %}


这个推导是理解 FlashAttention 反向的基础。我会从最基本的链式法则开始,一步步把所有梯度推出来。

### 一、前向回顾与记号约定

前向传播:

$$
S = QK^T \in \mathbb{R}^{N \times N}
$$
$$
P = \text{softmax}(S) \in \mathbb{R}^{N \times N} \quad \text{(按行做 softmax)}
$$
$$
O = PV \in \mathbb{R}^{N \times d}
$$

其中 $Q, K, V \in \mathbb{R}^{N \times d}$(为简洁省略 $\sqrt{d}$ 缩放,加回去只是每处乘个常数)。

**记号约定**(这是关键,很多推导让人糊涂就是记号乱):

$$
dX := \frac{\partial L}{\partial X}
$$

即 $dX$ 表示损失 $L$ 对 $X$ 的梯度,形状与 $X$ **完全相同**。

已知:$dO \in \mathbb{R}^{N \times d}$(从上游传下来)。

目标:求 $dQ, dK, dV$,形状分别与 $Q, K, V$ 相同。

### 二、两个基础结论

推导矩阵梯度时,以下两个结论会被反复使用。先把它们证明清楚,后面直接套用。

#### 结论 1:若 $Y = AB$,则

$$
dA = dY \cdot B^T, \quad dB = A^T \cdot dY
$$

**推导**:标量形式 $Y_{ij} = \sum_k A_{ik} B_{kj}$。

$$
dA_{ik} = \sum_{i',j} \frac{\partial L}{\partial Y_{i'j}} \cdot \frac{\partial Y_{i'j}}{\partial A_{ik}}
$$

$\frac{\partial Y_{i'j}}{\partial A_{ik}} = B_{kj} \cdot \mathbb{1}[i' = i]$,所以

$$
dA_{ik} = \sum_j dY_{ij} B_{kj} = (dY \cdot B^T)_{ik}
$$

同理得 $dB = A^T \cdot dY$。**一个助记方法**:梯度的形状必须和原矩阵一致,所以 $B$ 出现在右边就要转置,$A$ 出现在左边也要转置。

#### 结论 2:行 softmax 的雅可比

设 $p = \text{softmax}(s)$,其中 $s, p \in \mathbb{R}^N$(单独一行),则给定 $dp$,

$$
ds = p \odot \left(dp - \sum_k p_k \, dp_k\right) = p \odot dp - p \cdot (p^T dp)
$$

**推导**:softmax 的雅可比

$$
\frac{\partial p_i}{\partial s_j} = p_i \delta_{ij} - p_i p_j
$$

所以

$$
ds_j = \sum_i dp_i \cdot \frac{\partial p_i}{\partial s_j} = \sum_i dp_i (p_i \delta_{ij} - p_i p_j) = p_j dp_j - p_j \sum_i p_i dp_i
$$

写成向量形式:$ds = p \odot dp - p \cdot (\sum_i p_i dp_i)$。

令 $\rho = \sum_i p_i dp_i$(这是个标量,即 $p$ 和 $dp$ 的点积),则:

$$
ds = p \odot (dp - \rho)
$$

扩展到整个矩阵 $P$(每行独立做 softmax),逐行应用上式,$\rho$ 变成每行一个的列向量,广播到对应行:

$$
dS = P \odot (dP - \text{rowsum}(dP \odot P))
$$

这里 $\text{rowsum}(\cdot)$ 返回一个 $N \times 1$ 的列向量,广播时每一行减去自己那行的标量。

### 三、逐步推导三个梯度

#### 步骤 1:从 $O = PV$ 出发,求 $dV$ 和 $dP$

套用结论 1(令 $Y = O, A = P, B = V$):

$$
\boxed{dV = P^T \cdot dO}
$$

$$
\boxed{dP = dO \cdot V^T}
$$

**形状检查**:
- $dV$:$(N \times N)^T \cdot (N \times d) = (N \times d)$ ✓
- $dP$:$(N \times d) \cdot (d \times N) = (N \times N)$ ✓

#### 步骤 2:从 $P = \text{softmax}(S)$,求 $dS$

直接套用结论 2:

$$
\boxed{dS = P \odot \left(dP - \text{rowsum}(dP \odot P)\right)}
$$

**形状检查**:$N \times N$ ✓

#### 步骤 3:从 $S = QK^T$,求 $dQ$ 和 $dK$

把 $S = QK^T$ 写成 $S = Q \cdot (K^T)$,套用结论 1(令 $Y = S, A = Q, B = K^T$):

$$
dQ = dS \cdot (K^T)^T = dS \cdot K
$$

对 $B = K^T$:

$$
d(K^T) = Q^T \cdot dS
$$

所以

$$
dK = (d(K^T))^T = (Q^T \cdot dS)^T = dS^T \cdot Q
$$

$$
\boxed{dQ = dS \cdot K, \qquad dK = dS^T \cdot Q}
$$

**形状检查**:
- $dQ$:$(N \times N) \cdot (N \times d) = (N \times d)$ ✓
- $dK$:$(N \times N) \cdot (N \times d) = (N \times d)$ ✓

### 四、整合全流程

把五个公式串起来:

$$
\begin{aligned}
dV &= P^T \cdot dO \\
dP &= dO \cdot V^T \\
D &= \text{rowsum}(dP \odot P) \quad \text{(每行一个标量)} \\
dS &= P \odot (dP - D) \\
dQ &= dS \cdot K \\
dK &= dS^T \cdot Q
\end{aligned}
$$

标准实现里,每一步都要在 HBM 中保存 $N \times N$ 的 $P, dP, dS$,显存 $O(N^2)$,且反复读写导致速度慢。

### 五、一个关键恒等式:$D$ 可以只用 $dO$ 和 $O$ 算

FlashAttention 的一个优化点在于:那个 $D = \text{rowsum}(dP \odot P)$ 看起来需要完整的 $dP$ 和 $P$,但其实可以**完全绕过**它们。

**断言**:

$$
\text{rowsum}(dP \odot P) = \text{rowsum}(dO \odot O)
$$

**推导**:注意 $O = PV$,$dP = dO \cdot V^T$。逐元素算:

$$
(dP \odot P)_{ij} = P_{ij} \cdot (dO \cdot V^T)_{ij} = P_{ij} \sum_k dO_{ik} V_{jk}
$$

对 $j$ 求和:

$$
\sum_j (dP \odot P)_{ij} = \sum_j P_{ij} \sum_k dO_{ik} V_{jk} = \sum_k dO_{ik} \sum_j P_{ij} V_{jk}
$$

里面的 $\sum_j P_{ij} V_{jk} = (PV)_{ik} = O_{ik}$,于是:

$$
\sum_j (dP \odot P)_{ij} = \sum_k dO_{ik} O_{ik} = (dO \odot O)_i \text{ 行和}
$$

即:

$$
\boxed{D_i = \text{rowsum}(dO \odot O)_i}
$$

**这个结论很重要**,因为它意味着:
- $D \in \mathbb{R}^{N \times 1}$ 可以在反向循环**开始之前**用 $dO$ 和 $O$ 一次性算好
- 每个 block 在需要 $D_i$ 时直接读取,不需要 $P$ 或 $dP$ 的全貌
- 让分块反向在数学上变得干净:每个 block 所需的外部输入只有 $Q_i, K_j, V_j, dO_i, O_i, L_i, D_i$,全都是 $O(N)$ 或 $O(Nd)$ 的量

### 六、FlashAttention 分块反向的完整流程

有了上面的公式,可以把整个反向重写为分块形式:

```
预处理: D = rowsum(dO ⊙ O)      # 形状 N×1, 一次扫描搞定
初始化: dQ = dK = dV = 0

for i = 1..T_r:
    加载 Q_i, O_i, dO_i, L_i, D_i 到 SRAM
    
    for j = 1..T_c:
        加载 K_j, V_j 到 SRAM
        
        # --- 重算前向得到 P_ij ---
        S_ij = Q_i K_j^T
        P_ij = exp(S_ij - L_i)          # 利用 L_i 直接得到归一化的 P
        
        # --- 按上面五个公式算梯度 ---
        dV_j += P_ij^T · dO_i           
        dP_ij = dO_i · V_j^T             
        dS_ij = P_ij ⊙ (dP_ij - D_i)    # 用预计算的 D_i, 不用 rowsum
        dQ_i += dS_ij · K_j              
        dK_j += dS_ij^T · Q_i            
    
    写回 dQ_i
写回 dK, dV
```

注意 $P_{ij}, dP_{ij}, dS_{ij}$ 都是 $B_r \times B_c$ 的小块,只活在 SRAM 里,计算完就丢。整个反向期间 **HBM 中从未出现完整的 $N \times N$ 矩阵**。

### 七、一句话总结

| 推导环节 | 用到的工具 |
|---------|-----------|
| $dV, dP$ | 矩阵乘反向(结论 1) |
| $dS$ | softmax 雅可比(结论 2) |
| $dQ, dK$ | 矩阵乘反向(结论 1) |
| $D$ 的简化 | 交换求和顺序 + $O = PV$ 恒等式 |

**最核心的两个技术点**:

1. **结论 2 的 softmax 反向** 把 $dS$ 和 $P \odot dP$ 联系起来
2. **$D = \text{rowsum}(dO \odot O)$ 的恒等式** 让整个反向可以完全分块,不需要 $P, dP$ 的全貌

这两点加上前向存下来的 $L$,让 FlashAttention 反向做到了数学上与标准实现完全等价,但显存 $O(N)$、速度更快。

{% enddetails %}