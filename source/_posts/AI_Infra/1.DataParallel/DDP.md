---
title: 2. Distributed Data Parallel
categories: 学习笔记- AI Infra
date: 2026-04-28 22:48:00
mathjax: true
tags:
    - AI
    - AI Infra
---


## 一、从 DP 到 DDP:数据并行的演进

DDP 这一整套设计——多进程、Ring AllReduce、bucket 化梯度、异步通信——单看每一块都不算复杂,但它们组合在一起的方式背后有很强的工程动机。**最快理解 DDP 的路径,是先看它"反对"的那个东西**:PyTorch 早年那套 `torch.nn.DataParallel`(DP)。这一节先把 DP 怎么工作、为什么不好讲清楚,后面所有"DDP 为什么这么做"的问题都会有现成答案。

### 1.1 PyTorch 最早的方案:`nn.DataParallel`

在 DDP 出现之前,PyTorch 的多卡训练方案就是 `torch.nn.DataParallel`。它的设计思路是**单进程多线程**——一个 Python 主进程,N 张 GPU 各跑一个工作线程,所有线程共享同一份 Python 解释器,共用同一份主模型。

DP 的工作流程围绕一张被指定为 **master** 的 GPU(默认 cuda:0)展开,一次迭代可以拆成 6 步:

```
                       Master GPU (cuda:0)
                  ┌───────────────────────────┐
                  │   主模型 + optimizer       │
                  └────────────┬──────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       │ ① Replicate           │ ① Replicate           │ ① Replicate
       ▼                       ▼                       ▼
   ┌────────┐              ┌────────┐              ┌────────┐
   │ GPU 0  │              │ GPU 1  │              │ GPU 2  │
   │ model' │              │ model' │              │ model' │
   └────┬───┘              └────┬───┘              └────┬───┘
        │ ② Scatter             │ ② Scatter             │ ② Scatter
        ▼                       ▼                       ▼
      x[0:B/3]                x[B/3:2B/3]            x[2B/3:B]
        │ ③ Forward             │ ③ Forward             │ ③ Forward
        ▼                       ▼                       ▼
      out₀                    out₁                    out₂
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼ ④ Gather outputs
                  ┌───────────────────────────┐
                  │ Master: loss 计算 + backward│
                  └────────────┬──────────────┘
                               ▼ ⑤ 反向把 grad 散回各 GPU
                       各 GPU 算本地 grad
                               │
                               ▼ ⑤ Gather grads to master
                  ┌───────────────────────────┐
                  │  Master: 求和 + step()    │
                  └───────────────────────────┘
                               │
                               └──► 下一轮回到 ①
```

逐步:① **Replicate** 把主 GPU 模型复制到所有其他 GPU;② **Scatter** 把 batch 沿 batch 维切成 N 份分发出去;③ 各 GPU 在自己的 sub-batch 上**并行 forward**;④ 输出 **Gather** 回 master(因为 loss 通常需要完整 batch 才能算);⑤ 在 master 上算 loss 然后 `backward()`,反向时各 GPU 算本地参数的梯度,梯度 Gather 回 master 求和;⑥ master 调 `optimizer.step()` 更新主模型——下一轮再 Replicate 出去。

代码上 DP 极其简洁,几乎不动训练循环:

```python
import torch
import torch.nn as nn

model = nn.Linear(128, 10).cuda()                  # 默认放在 cuda:0
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# 正常训练循环——无需任何修改
for x, y in loader:
    x, y = x.cuda(), y.cuda()                      # 数据先到 cuda:0
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

"包一行就能多卡"——这种零侵入 API 让 DP 在 PyTorch 早期非常流行。但训练规模一上去,四个问题就全冒出来。

### 1.2 DP 的四个致命问题

**① Python GIL 让多线程几乎退化成串行**

DP 是单进程,N 张 GPU 跑在 N 个 Python 线程里。但 CPython 的 Global Interpreter Lock 同一时刻只允许一个线程执行 Python 字节码。每次 launch kernel、包装 tensor、走 autograd 调度都要回到 Python 层——这些步骤被 GIL 串行化后,GPU 大部分时间在等 CPU 喂指令,加速比远低于线性。

**② Master GPU 是显存和算力的双重瓶颈**

所有数据从 cuda:0 scatter 出去、所有输出 gather 回 cuda:0、loss 计算在 cuda:0、所有梯度归约在 cuda:0、optimizer.step() 在 cuda:0、更新后参数再从 cuda:0 复制广播。结果是:

- **显存不均衡**:cuda:0 同时持有 N 份输出 + N 份梯度副本 + 完整 optimizer state,实测 4 卡 DP 训练 cuda:0 显存常比其他卡多用 30~50%,经常 OOM
- **算力不均衡**:loss 计算、梯度求和、参数更新全压在 cuda:0,其他卡常在等它

后面 §3.9 会专门讲为什么 Parameter Server 架构在同构 GPU 训练里被淘汰——其实 **DP 就是单机版的 PS**,master GPU 扮演 PS 的角色。PS 的所有缺点(单点带宽、扩展性差)在 DP 上原样复现。

**③ 单机限制,完全无法跨节点**

DP 依赖单进程内的多个 CUDA device id,主机之间没有共享的 Python interpreter,所以 DP 没法跨节点。16 卡(2 机)以上的训练 DP 直接出局。

**④ 通信和计算完全串行,无法重叠**

DP 的 scatter / gather 都是 host 端的同步阻塞操作。反向算完后必须等 gather 完才能进 optimizer.step(),不存在"前几层反向还在算、后几层梯度已经在传"的可能,带宽利用率天然受限。

### 1.3 DDP 怎么逐条解决

DDP 把上面四个问题一一对应解掉:

| DP 的问题 | DDP 的回应 |
|---|---|
| GIL 让多线程串行 | **每张 GPU 一个独立进程**,各自有自己的 Python 解释器,GIL 不再是问题 |
| Master GPU 双重瓶颈 | 取消 master 角色,所有 rank 对等;梯度同步用 **Ring AllReduce** 摊平到所有链路 |
| 单机限制 | 通信走 NCCL/Gloo 后端,**天然支持跨节点 InfiniBand / Ethernet** |
| 通信无法重叠 | **bucket 化梯度 + 异步 AllReduce**,反向算前几层时后几层的梯度已经在传 |

每一条的具体实现都对应到本文后续的一节:

- "对等多进程 + Ring AllReduce" → §三 集合通信原语与 Ring AllReduce
- "bucket 化 + 异步" → §四 DDP 工作流程
- "跨节点" → 留到《多机训练与 NCCL 工程》单独展开

理解 DDP 设计的最快路径就是**记住它每一处工程选择都是在反对 DP 的某个问题**。

### 1.4 一句话总结:DP 已经不推荐用了

PyTorch 官方文档现在明确写着:

> Even on a single machine, we recommend using `DistributedDataParallel` over `DataParallel` for multi-GPU training.

`nn.DataParallel` 现在是历史遗留 API,做小规模实验、快速调试时还能凑合用,但凡是认真训练的场景都用 DDP。本文从下一节开始,所有讨论默认就是 DDP。

## 二、数学基础:为什么数据并行是正确的

### 2.1 SGD 的可分解性

考虑 batch size 为 B 的损失函数:

$$L(\theta) = \frac{1}{B}\sum_{i=1}^{B} \ell(x_i, \theta)$$

梯度:

$$\nabla L(\theta) = \frac{1}{B}\sum_{i=1}^{B} \nabla \ell(x_i, \theta)$$

把 B 个样本平均切给 N 张 GPU,每张卡处理 B/N 个样本。第 k 张卡上的局部梯度是:

$$g_k = \frac{1}{B/N}\sum_{i \in S_k} \nabla \ell(x_i, \theta) = \frac{N}{B}\sum_{i \in S_k} \nabla \ell(x_i, \theta)$$

那么 N 张卡的局部梯度的平均是:

$$\frac{1}{N}\sum_{k=1}^{N} g_k = \frac{1}{N}\sum_{k=1}^{N} \frac{N}{B}\sum_{i \in S_k} \nabla \ell(x_i, \theta) = \frac{1}{B}\sum_{i=1}^{B} \nabla \ell(x_i, \theta) = \nabla L(\theta)$$

**结论**:N 张卡分别在 B/N 个样本上算梯度,然后做平均,数学上完全等价于单卡上 batch size = B 的梯度。这就是 DDP 正确性的根基。

### 2.2 关键前提

这个等价的成立依赖三个条件:**所有卡的初始参数相同**、**每步梯度同步保持参数一致**、**优化器是确定性的**(给定相同的参数、梯度、状态,产出相同的更新)。DDP 的所有工程设计都在保障这三点。

### 2.3 一个常被忽视的细节:BatchNorm 不可分解

BN 的均值方差是在 batch 内统计的,所以 N 张卡各算各的 BN 等价于 batch size = B/N(不是 B)。这就是为什么需要 SyncBatchNorm。LayerNorm、RMSNorm 不存在这个问题,因为它们在样本内部归一化,与 batch 无关。Transformer/GPT 用 LayerNorm,所以 DDP 训练 LLM 不用担心这一点。

## 三、集合通信原语与 Ring AllReduce

DDP 反向阶段做的事看似简单——"把各卡的梯度求和后再分发回去"——但这一步是由更基础的**集合通信原语 (collective communication primitives)** 组合而成的。NCCL、MPI、Gloo 等通信库提供的就是这套零件。理解了这些原语,再回看 Ring AllReduce 就只是一种"如何高效实现 AllReduce"的具体调度。

下面所有图都假设 4 张卡(rank 0~3),每张卡持有一个或多个数据块。

### 3.1 Broadcast(广播)

由一个 root rank 把数据**复制**给所有其他 rank,所有 rank 拿到完全相同的副本。

```
        Before                       After (root = 0)
   ┌───────────────┐              ┌───────────────┐
   │ Rank 0: [ A ] │              │ Rank 0: [ A ] │
   │ Rank 1: [   ] │   ────────►  │ Rank 1: [ A ] │
   │ Rank 2: [   ] │              │ Rank 2: [ A ] │
   │ Rank 3: [   ] │              │ Rank 3: [ A ] │
   └───────────────┘              └───────────────┘
```

**用途**:DDP 初始化时把 rank 0 的模型参数同步给所有其他 rank,保证起点一致。

### 3.2 Scatter(散开)

root 把一个大张量切成 N 份,**每份发给一个 rank**(自己留一份)。

```
        Before                       After (root = 0)
   ┌──────────────────────┐       ┌───────────────┐
   │ Rank 0: [ A B C D ]  │       │ Rank 0: [ A ] │
   │ Rank 1: [         ]  │ ────► │ Rank 1: [ B ] │
   │ Rank 2: [         ]  │       │ Rank 2: [ C ] │
   │ Rank 3: [         ]  │       │ Rank 3: [ D ] │
   └──────────────────────┘       └───────────────┘
```

**和 Broadcast 的区别**:Broadcast 是"全员复制",Scatter 是"按位切分"。深度学习里直接用 Scatter 不多,但它是 ReduceScatter 的雏形。

### 3.3 Gather(聚集)

Scatter 的逆操作:每个 rank 把自己的数据**送到 root**,root 拼接成一个大张量。

```
        Before                       After (root = 0)
   ┌───────────────┐              ┌──────────────────────┐
   │ Rank 0: [ A ] │              │ Rank 0: [ A B C D ]  │
   │ Rank 1: [ B ] │   ────────►  │ Rank 1: [ B ]        │
   │ Rank 2: [ C ] │              │ Rank 2: [ C ]        │
   │ Rank 3: [ D ] │              │ Rank 3: [ D ]        │
   └───────────────┘              └──────────────────────┘
```

**用途**:常见于评估阶段把各卡的预测结果汇总到 rank 0 算 metric。

### 3.4 Reduce(归约)

每个 rank 持有同形状的张量,通过某个二元运算(`SUM` / `MAX` / `MIN` / `AVG` ...)合并到 root 上。**只有 root 拿到结果**。

```
        Before                          After (op = SUM, root = 0)
   ┌────────────────┐               ┌─────────────────────────────┐
   │ Rank 0: [ a₀ ] │               │ Rank 0: [ a₀+a₁+a₂+a₃ ]     │
   │ Rank 1: [ a₁ ] │   ─────────►  │ Rank 1: [ a₁ ]              │
   │ Rank 2: [ a₂ ] │               │ Rank 2: [ a₂ ]              │
   │ Rank 3: [ a₃ ] │               │ Rank 3: [ a₃ ]              │
   └────────────────┘               └─────────────────────────────┘
```

可以理解为 Gather 的"加法版"——Gather 是拼接,Reduce 是按位运算。

### 3.5 AllReduce(全局归约)—— DDP 的主角

Reduce + Broadcast 的组合:**所有 rank 的数据归约,且所有 rank 都拿到结果**。没有 root。

```
        Before                          After (op = SUM, σ = a₀+a₁+a₂+a₃)
   ┌────────────────┐               ┌────────────────┐
   │ Rank 0: [ a₀ ] │               │ Rank 0: [ σ ]  │
   │ Rank 1: [ a₁ ] │   ─────────►  │ Rank 1: [ σ ]  │
   │ Rank 2: [ a₂ ] │               │ Rank 2: [ σ ]  │
   │ Rank 3: [ a₃ ] │               │ Rank 3: [ σ ]  │
   └────────────────┘               └────────────────┘
```

**这就是 DDP 反向传播里干的事**:每张卡有一份梯度,AllReduce(SUM) 后所有卡都拿到全局梯度之和,再除以 N 得到平均梯度。AllReduce 是 DDP 工作流里的唯一通信操作。

### 3.6 Reduce-Scatter

每个 rank 持有完整向量 V(切成 N 块),归约之后**每个 rank 只保留结果的 1/N**。

```
         Before                            After (op = SUM)
   ┌──────────────────────┐         ┌────────────────────────┐
   │ Rank 0: [a₀ b₀ c₀ d₀]│         │ Rank 0: [ a₀+a₁+a₂+a₃ ]│
   │ Rank 1: [a₁ b₁ c₁ d₁]│ ──────► │ Rank 1: [ b₀+b₁+b₂+b₃ ]│
   │ Rank 2: [a₂ b₂ c₂ d₂]│         │ Rank 2: [ c₀+c₁+c₂+c₃ ]│
   │ Rank 3: [a₃ b₃ c₃ d₃]│         │ Rank 3: [ d₀+d₁+d₂+d₃ ]│
   └──────────────────────┘         └────────────────────────┘
```

可以理解成 Reduce 的"分布式版本":没有 root,每个 rank 都各拿走结果的一块,谁也不持有完整结果。**ZeRO/FSDP 大量用到**——梯度归约后立刻分片存储,顺手省了显存。

### 3.7 AllGather

每个 rank 持有 1/N 数据,通信后**所有 rank 都拿到完整数据**。

```
        Before                           After
   ┌───────────────┐              ┌──────────────────────┐
   │ Rank 0: [ A ] │              │ Rank 0: [ A B C D ]  │
   │ Rank 1: [ B ] │  ─────────►  │ Rank 1: [ A B C D ]  │
   │ Rank 2: [ C ] │              │ Rank 2: [ A B C D ]  │
   │ Rank 3: [ D ] │              │ Rank 3: [ A B C D ]  │
   └───────────────┘              └──────────────────────┘
```

可以理解成 Gather 的"无 root 版本":人人都参与拼接。**ZeRO-3/FSDP 在前向时把分片参数临时聚合就是用它**。

### 3.8 一个关键恒等式

把 ReduceScatter 和 AllGather 串起来,有一个 ZeRO 论文反复用到的等式:

$$\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}$$

直观看:先用 ReduceScatter 让每个 rank 拿到结果的一块,再用 AllGather 把这些块聚合给所有人——结果和直接 AllReduce 等价,**但通信量不变**。这是 Ring AllReduce 的实现思路,也是 ZeRO-1/2 把"DDP 的 AllReduce"换成"分片梯度 + 分片 optimizer"却不增加通信的根本原因。

### 3.9 朴素方案:为什么不用 Parameter Server

最直觉的 AllReduce 实现是 Parameter Server (PS):所有 worker 把梯度发给一个 master,master 求和后广播回来。设梯度大小为 V,worker 数为 N:
- master 接收带宽需求:N·V
- master 广播带宽需求:N·V
- master 是单点瓶颈,**通信量随 N 线性增长**,无法扩展

PS 在异构集群、稀疏更新场景下还有应用,但在同构 GPU 训练里早被 Ring AllReduce 取代。**§一讲到的 DP 就是把 PS 模式塞进单机**——master GPU 扮演 PS 的 master,所以 DP 在同构多卡场景下天然也是个被淘汰方案。

### 3.10 Ring AllReduce 的核心思路

把 N 张卡排成一个环 0 → 1 → 2 → ... → N−1 → 0,每张卡只和**相邻两个邻居**通信(收上游 / 发下游)。梯度切成 N 块,Ring AllReduce 严格按 §3.8 的等式做:

- **Phase 1 (Reduce-Scatter)**:N−1 轮,把"梯度求和"沿环逐步累加并分散
- **Phase 2 (AllGather)**:N−1 轮,把已求和的块沿环再传一圈,所有人都拿到所有块

每轮每张卡只发 V/N 数据、只收 V/N 数据——这正是 Ring 在大消息上能跑满带宽的原因。

### 3.11 4 卡完整走一遍

记 GPU $i$ 在 chunk $j$ 上的梯度为 $A_{ij}$,目标是让每张卡都得到

$$\sigma_j = A_{0j} + A_{1j} + A_{2j} + A_{3j},\quad j=0,1,2,3$$

**初始状态**:每张卡持有自己完整的 4 个块。

```
GPU 0:  [ A₀₀  A₀₁  A₀₂  A₀₃ ]
GPU 1:  [ A₁₀  A₁₁  A₁₂  A₁₃ ]
GPU 2:  [ A₂₀  A₂₁  A₂₂  A₂₃ ]
GPU 3:  [ A₃₀  A₃₁  A₃₂  A₃₃ ]
```

#### Phase 1: Reduce-Scatter(3 轮)

第 t 轮:GPU $i$ 把当前位置 `(i-t) mod 4` 上的内容发给 GPU $i{+}1$,GPU $i$ 把上游传来的累加到自己位置 `(i-t-1) mod 4` 上。

**Round 1 后**(每张卡有一个块完成"两卡求和"):

```
GPU 0:  [ A₀₀  A₀₁  A₀₂   A₀₃+A₃₃  ]    ← 收到 A₃₃
GPU 1:  [ A₀₀+A₁₀  A₁₁  A₁₂  A₁₃    ]    ← 收到 A₀₀
GPU 2:  [ A₂₀  A₁₁+A₂₁  A₂₂  A₂₃    ]    ← 收到 A₁₁
GPU 3:  [ A₃₀  A₃₁  A₂₂+A₃₂  A₃₃    ]    ← 收到 A₂₂
```

**Round 2 后**(每张卡有一个块完成"三卡求和"):

```
GPU 0:  [ A₀₀  A₀₁  A₀₂+A₂₂+A₃₂   A₀₃+A₃₃    ]
GPU 1:  [ A₀₀+A₁₀   A₁₁  A₁₂  A₀₃+A₁₃+A₃₃    ]
GPU 2:  [ A₀₀+A₁₀+A₂₀   A₁₁+A₂₁  A₂₂  A₂₃    ]
GPU 3:  [ A₃₀  A₁₁+A₂₁+A₃₁   A₂₂+A₃₂  A₃₃    ]
```

**Round 3 后**(每张卡恰好持有 1 个**全局已求和**的块,即 σ):

```
GPU 0:  [ A₀₀          σ₁              A₀₂+A₂₂+A₃₂   A₀₃+A₃₃    ]
GPU 1:  [ A₀₀+A₁₀      A₁₁             σ₂            A₀₃+A₁₃+A₃₃]
GPU 2:  [ A₀₀+A₁₀+A₂₀  A₁₁+A₂₁         A₂₂           σ₃         ]
GPU 3:  [ σ₀           A₁₁+A₂₁+A₃₁     A₂₂+A₃₂       A₃₃        ]
```

到这里 Reduce-Scatter 阶段完成:GPU 3 拿到 σ₀,GPU 0 拿到 σ₁,GPU 1 拿到 σ₂,GPU 2 拿到 σ₃,**和 §3.6 的 ReduceScatter 语义完全一致**(只不过这里数据原本就在每张卡上,没有 root)。

#### Phase 2: AllGather(3 轮)

把刚刚那 4 个 σ 块沿环再传一圈,每轮每张卡多拥有一个 σ。这里只关心 σ 块,其他位置的中间值 Phase 2 不再用,直接覆盖丢弃。

**Round 1 后**:
```
GPU 0:  [  σ₀     σ₁      ·       ·   ]
GPU 1:  [  ·      σ₁      σ₂      ·   ]
GPU 2:  [  ·      ·       σ₂      σ₃  ]
GPU 3:  [  σ₀     ·       ·       σ₃  ]
```

**Round 2 后**:
```
GPU 0:  [  σ₀     σ₁      ·       σ₃  ]
GPU 1:  [  σ₀     σ₁      σ₂      ·   ]
GPU 2:  [  ·      σ₁      σ₂      σ₃  ]
GPU 3:  [  σ₀     ·       σ₂      σ₃  ]
```

**Round 3 后**(全部到齐):
```
GPU 0:  [  σ₀     σ₁      σ₂      σ₃  ]
GPU 1:  [  σ₀     σ₁      σ₂      σ₃  ]
GPU 2:  [  σ₀     σ₁      σ₂      σ₃  ]
GPU 3:  [  σ₀     σ₁      σ₂      σ₃  ]
```

完美——每张卡都拿到完整的全局求和梯度,即 AllReduce 的结果。**总共 2(N−1) = 6 轮通信,每轮每张卡只收发 V/N = V/4 数据**。

### 3.12 通信量推导

每张卡每轮发送 V/N 数据。两个阶段共 2(N−1) 轮。每张卡的总发送量:

$$T_{\text{send}} = 2(N-1) \cdot \frac{V}{N} = 2V \cdot \frac{N-1}{N}$$

当 N 很大时,**每张卡发送量趋近 2V,与 N 无关**——这就是 Ring AllReduce 能扩展到几千卡的关键。延迟是 O(N)(轮数),但每轮数据量小,带宽利用率高。

对比 PS 方案的 O(N·V) 单点带宽,Ring 把流量摊平到了所有链路上,这也是为什么现代多机训练几乎清一色用 Ring 类拓扑。

### 3.13 Tree AllReduce 与 NCCL 的实际选择

Ring 的延迟是 O(N),每一轮只能等上游把数据推过来才能继续。在**小消息**场景(几 MB 以下),N 张卡跑 2(N−1) 轮的延迟会盖过通信量节省的好处。

NCCL 还提供 **Tree AllReduce**:把节点组织成二叉树,延迟变 O(log N),适合小张量。NCCL 会根据消息大小自动切拓扑——大消息走 Ring(带宽友好),小消息走 Tree(延迟友好)。但 Ring 是理解所有变体的基础,Tree/双二叉树/2D-Ring 本质都是"如何把 ReduceScatter + AllGather 调度得更快"的不同答案。

## 四、DDP 的工作流程

### 4.1 完整生命周期

对照 §一 讲的 DP 流程读这一节会更清晰——DDP 的每一步都在反对 DP 的某个设计。

第一步是**初始化进程组**:每张 GPU 一个进程(不是 DP 的线程),通过 NCCL 通信后端组成进程组,每个进程有 rank 和 world_size。GPU 训练几乎一律选 NCCL——它是 NVIDIA 为 GPU 优化的集合通信库,直接走 NVLink/PCIe/InfiniBand,绕过 CPU 内存拷贝,内置 Ring/Tree 等高效拓扑。Gloo 是通用 CPU 后端,在 GPU 训练里性能差很多,只在 CPU-only 或调试场景才用。

第二步是**模型构造时的状态同步**:DDP 把 rank 0 的模型参数和 buffers 通过 Broadcast 发给所有其他 rank(就是 §3.1 那个原语),保证所有进程从相同状态开始。这一步等价于 DP 每轮都要做的 Replicate,但 DDP **只在初始化时做一次**——后续靠梯度同步隐式保持参数一致,不需要每步重新广播。

第三步是**反向传播中的梯度同步**:每个 step 各进程独立做前向(无通信),反向时通过 autograd hook 触发 bucket 级的**异步** AllReduce,通信和计算重叠。这一步的细节会在 §4.3 单独展开。

第四步是**参数更新**:各进程独立调用 optimizer.step(),由于参数、梯度、优化器状态完全一致,更新结果也完全一致——这里没有 DP 那种"只在 master 上 step"的中心化操作。

把这四步串起来看,DDP 的"参数始终一致"其实靠**三道保证**叠加:① 初始化时 Broadcast rank 0 参数,起点一致;② 每个 step 反向后 AllReduce 让所有 rank 的梯度一致;③ 优化器是确定性的——相同参数、相同梯度、相同状态产出相同的更新。三者共同作用,跨 rank 的参数除浮点累加顺序导致的极小数值差外,始终完全相同。

### 4.2 关键工程优化:Gradient Bucketing

如果每个参数算完梯度就发一次 AllReduce,小消息太多,通信效率低(NCCL 的小消息延迟开销大)。DDP 把参数按一定大小(默认 25MB)打包成 bucket,一个 bucket 内所有参数的梯度都算好后,一次性发起 AllReduce。

更精妙的是 bucket 划分按"反向传播顺序"逆序排列。反向是从最后一层往前算,所以**最后一层的参数最先算完梯度**,DDP 把最后一层放到第一个 bucket,前面的反向还在进行时,这个 bucket 已经可以**异步**发起 AllReduce,实现**计算与通信重叠**。这里"异步"是个独立的大话题——CUDA stream、`async_op=True`、`handle.wait()` 等机制如何让通信不卡反向——我把它单独写在了 [《GPU 训练里的异步计算》](AsyncCompute.md) 里,本节不再展开。

### 4.3 一个工程坑:动态计算图 / 未使用参数

DDP 的异步机制依赖一个隐含假设:**每个被 register_hook 的参数都会在反向中收到梯度**。如果某些参数在某次 forward 里被分支跳过(例如 if 分支、条件 routing、MoE),它们的 grad hook 永远不会触发,对应的 bucket 永远凑不齐,AllReduce 不发起,所有 rank 卡死等通信——程序就这样挂住或者报错。

两种解法:

```python
# 方案 1:让 DDP 在反向开始前先扫一遍计算图,标记哪些参数没用
model = DDP(model, device_ids=[rank], find_unused_parameters=True)
```

`find_unused_parameters=True` 让 DDP 在每个 forward 之后遍历计算图,把没参与的参数对应的 bucket 提前标记为"无需通信",反向时就不会卡住。代价是有不小的开销(遍历计算图本身要时间),官方建议**只在确实有动态分支时才开**。

```python
# 方案 2(推荐):重构模型,让所有参数每次都参与 forward
```

更干净的做法是把动态分支改成数学等价的稠密计算(比如 mask 而不是 if),或者把不同分支的参数放进不同模块、用不同 DDP wrap。MoE 等真正的稀疏架构会用专门的并行策略(Expert Parallel),不靠 `find_unused_parameters` 兜底。

## 五、关键代码:从最小可用版本理解

### 5.1 最小可运行的 DDP 训练脚本

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

def setup():
    # torchrun 会自动设置 RANK, LOCAL_RANK, WORLD_SIZE 环境变量
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

class ToyDataset(Dataset):
    def __init__(self, n=10000):
        self.x = torch.randn(n, 128)
        self.y = torch.randint(0, 10, (n,))
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

def main():
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
    # 关键:用 DDP 包装模型
    model = DDP(model, device_ids=[local_rank])

    dataset = ToyDataset()
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        sampler.set_epoch(epoch)  # 关键:让 shuffle 每个 epoch 不同
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()           # 反向时 DDP 自动同步梯度
            optimizer.step()

        if dist.get_rank() == 0:
            print(f"epoch {epoch} loss {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    main()
```

启动命令:

```bash
torchrun --nproc_per_node=4 train.py
```

### 5.2 DDP 内部做了什么:伪代码版

下面是 DDP 关键逻辑的简化伪代码,帮你理解其内部实现:

```python
class DistributedDataParallel:
    def __init__(self, module, device_ids):
        self.module = module

        # 1. 广播参数,保证所有 rank 起点一致
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

        # 2. 把参数划分到 buckets(按反向顺序的逆序)
        params_in_reverse = list(self.module.parameters())[::-1]
        self.buckets = self._build_buckets(params_in_reverse, bucket_size_mb=25)

        # 3. 给每个参数注册 grad hook
        for bucket in self.buckets:
            for param in bucket.params:
                param.register_hook(lambda grad, p=param, b=bucket:
                                     self._on_grad_ready(p, b, grad))

    def _on_grad_ready(self, param, bucket, grad):
        bucket.mark_ready(param)
        # bucket 内所有参数梯度都准备好了,异步发起 AllReduce
        if bucket.all_ready():
            bucket.handle = dist.all_reduce(
                bucket.flat_grad,
                op=dist.ReduceOp.SUM,
                async_op=True
            )

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _wait_and_average(self):
        # backward 结束后,等所有 bucket 的 AllReduce 完成,然后除以 world_size
        for bucket in self.buckets:
            bucket.handle.wait()
            bucket.flat_grad.div_(dist.get_world_size())
```

几个关键点:

**hook 机制**:`param.register_hook` 在该参数的梯度被计算出来时触发回调。DDP 利用这一点知道"哪个参数的梯度已经算好了"。

**bucket.flat_grad**:bucket 内所有参数的梯度被拼成一个连续的 flat tensor 再做 AllReduce,避免多次小通信。AllReduce 完成后再 view 回各参数的 .grad。

**async_op=True**:AllReduce 异步发起,不阻塞反向传播。前面层的反向计算可以与后面层的通信并行。

**除以 world_size**:NCCL 的 AllReduce 是求和,DDP 在最后除以 N 得到平均梯度,保证语义和单卡 batch size = B 一致。

### 5.3 几个工程上必须知道的 API

```python
# 屏障同步,所有 rank 卡到这里再继续
dist.barrier()

# 跨 rank 求和,常用于汇总 loss/accuracy 用于打印
loss_tensor = torch.tensor(loss.item(), device=device)
dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
avg_loss = loss_tensor.item() / dist.get_world_size()

# 只在 rank 0 保存 checkpoint
if dist.get_rank() == 0:
    torch.save(model.module.state_dict(), "ckpt.pt")  # 注意 model.module
dist.barrier()  # 其他 rank 等保存完再继续

# SyncBatchNorm:把所有 BN 替换成跨卡同步版本
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[local_rank])

# 各 rank 用不同 seed,避免数据增强 / dropout 在所有 rank 上完全同步
torch.manual_seed(base_seed + dist.get_rank())
```

注意 `model.module` 这个细节:DDP 包装后,原模型在 `.module` 属性里,保存 state_dict 通常用 `model.module.state_dict()` 而不是 `model.state_dict()`,这样保存的是不带 DDP 前缀的干净权重。

**关于随机种子**:DataLoader 的样本顺序由 `DistributedSampler` 处理(下一节单独讲),不需要你管。但 dropout、数据增强、初始化等用到的全局随机状态如果各 rank 完全一样,所有 rank 就在用同样的随机数做同样的扰动,失去了"多 rank 看不同样本"的统计意义。所以 `torch.manual_seed(base_seed + rank)` 是标准做法。

### 5.4 DistributedSampler:数据怎么分到不同 rank 上

§5.1 的最小训练脚本里用了 `DistributedSampler`,但只一行带过。这个组件是 DDP 数据流的关键,值得单独拆开讲。

#### 它解决什么问题

DDP 每个 rank 是独立进程,各自跑自己的 DataLoader——如果不做任何协调,**所有 rank 会读到完全相同的数据**(同一个 dataset、同一个默认 sampler),那 N 张卡只是把同一份梯度算 N 遍,加速为零。

`DistributedSampler` 的工作就是:**保证 N 个 rank 拿到不重叠、不遗漏、覆盖整个 dataset 的样本切分**,等价于"先把 dataset shuffle 一次,再分成 N 段,rank k 拿第 k 段"。

#### 工作机制

简化版伪代码:

```python
class DistributedSampler:
    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, seed=0, drop_last=False):
        self.dataset = dataset
        self.num_replicas = num_replicas or dist.get_world_size()
        self.rank = rank or dist.get_rank()
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last
        # 每个 rank 拿到的样本数(向上取整,可能 padding)
        if drop_last:
            self.num_samples = len(dataset) // self.num_replicas
        else:
            self.num_samples = math.ceil(len(dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # 关键:所有 rank 用相同 seed+epoch,得到相同的 shuffle 顺序
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # 长度不整除时:drop_last=False 从头补齐;drop_last=True 直接截断
        if not self.drop_last:
            indices += indices[:(self.total_size - len(indices))]
        else:
            indices = indices[:self.total_size]

        # 切片:rank k 拿 indices[k::num_replicas]——和其他 rank 不重叠
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch
```

#### 几个关键点

**① 所有 rank 共享同一个 shuffle 种子**。`g.manual_seed(self.seed + self.epoch)` 在所有 rank 上结果一样,所以**各 rank 看到的 shuffle 顺序完全一致**,然后 `indices[rank::num_replicas]` 按步长切片,各 rank 各拿一份不重叠的索引。这里有个常见误解:DistributedSampler 不需要跨 rank 通信来协商谁拿什么,完全靠"种子一致 + 确定性算法"达到一致。

**② 必须每个 epoch 调 `sampler.set_epoch(epoch)`**。如果不调,`self.epoch` 永远是 0,每个 epoch 的 shuffle 都一模一样——等于 epoch 1、2、3 都在重复 epoch 0 的样本顺序,优化轨迹会失真。这是初学者最容易忘的一行:

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)    # 必不可少
    for x, y in loader:
        ...
```

**③ 长度不整除时的两种行为**。默认 `drop_last=False`:sampler 会从 indices 开头补齐,凑足 `total_size = ceil(len(dataset) / N) * N` 个样本(尾部样本会被重复一次)。如果想严格丢掉尾部不能整除的部分,设 `drop_last=True`。LLM 训练通常用 `drop_last=True` 避免最后一个 batch 大小不一致,以及避免重复样本污染统计量。

**④ 和 DataLoader 的关系**。把 sampler 传给 DataLoader 时,**不能再设 DataLoader 的 `shuffle=True`**,这两者互斥——PyTorch 会直接报错。正确写法:

```python
loader = DataLoader(
    dataset, batch_size=64,
    sampler=sampler,        # 不要再写 shuffle=True
    num_workers=2,
    pin_memory=True,
)
```

shuffle 的工作交给 DistributedSampler,DataLoader 只负责按 sampler 给的索引取数据。

#### 和随机种子的协作

§5.3 提到 `torch.manual_seed(base_seed + rank)` 是为了让各 rank 的 dropout / 数据增强用不同的随机数。这跟 DistributedSampler 的种子是**两件独立的事**:

- **DistributedSampler 的 seed** 决定"哪张卡拿哪些样本",必须**所有 rank 一致**
- **`torch.manual_seed` 的 seed** 决定"模型内部的随机操作"(dropout、数据增强、初始化),应该**每张卡不同**

两者放一起用没问题,因为 DistributedSampler 在 `__iter__` 里用的是自己内部的 `torch.Generator()` 实例,不受全局 seed 影响。

#### 评估时的小坑

eval 阶段也经常用 DistributedSampler(`shuffle=False`),让各 rank 拿固定切片,跑完后 AllGather 预测结果。但有个坑:**默认 `drop_last=False` 的 padding 会让评估样本数多于 dataset 真实大小**(尾部样本被重复一次)。严格评估时要么 `drop_last=True`(丢掉零头),要么自己写逻辑去掉 padding 部分。Hugging Face 的 `Accelerate` / `Trainer` 内部都做了这个处理,自己手写时记得自己处理。

## 六、有效 batch size 与学习率

DDP 下的 effective batch size:

$$B_{\text{effective}} = B_{\text{per\_gpu}} \times N_{\text{gpus}} \times N_{\text{accum}}$$

其中 N_accum 是梯度累积步数。学习率按 **Linear Scaling Rule** 缩放:相对于基准 batch size,batch 扩大 k 倍,学习率也扩大 k 倍。这个 rule 在 ResNet 训练上验证有效,LLM 训练有时用平方根缩放或者实际调试。

## 七、梯度累积与 no_sync

如果想做梯度累积,naive 写法每个 micro-batch 都触发 AllReduce,通信浪费。正确做法:

```python
for i, (x, y) in enumerate(loader):
    is_last = (i + 1) % accum_steps == 0
    # 非最后一个 micro-batch 时,禁用梯度同步
    ctx = model.no_sync() if not is_last else nullcontext()
    with ctx:
        loss = criterion(model(x), y) / accum_steps
        loss.backward()
    if is_last:
        optimizer.step()
        optimizer.zero_grad()
```

`model.no_sync()` 是 DDP 提供的上下文管理器,在它作用域内 backward 不触发 AllReduce,只在最后一步累积完所有梯度后做一次同步,通信量减少 accum_steps 倍。

## 八、进阶:静态图模式与通信 hook

前面 §四 讲 DDP 工作流程时一直假设默认模式——每次 forward 都重新 trace 一遍计算图,反向时 hook 触发 bucket 通信。这套默认行为在大多数场景够用,但有两个进阶机制经常出现在生产代码里,它们正好对应"DDP 怎么变得更快、怎么变得更可定制":**`static_graph=True`** 让 DDP 利用计算图不变的假设做激进优化;**`register_comm_hook`** 让你完全接管梯度通信策略,做压缩、量化、跨节点协议替换等定制化通信。

### 8.1 静态图模式:`static_graph=True`

#### 起因

§4.3 提到 `find_unused_parameters=True` 是为了应付动态分支(if、MoE 这类),代价是**每次 forward 后都要遍历计算图**找哪些参数被用过。这个遍历本身在大模型上会是几个百分点的开销——反向计算非常快的层,遍历开销看起来就明显。

但绝大多数模型其实**计算图是不变的**——同一个 batch、同一个 forward 路径、同样的算子调用顺序、所有参数都参与。如果你能告诉 DDP "我保证计算图永远不变",DDP 就可以做几件激进的优化。

#### 工作机制

打开 `static_graph=True`:

```python
model = DDP(model, device_ids=[local_rank], static_graph=True)
```

DDP 在**第一步训练**结束时,把所有参数的反向触发顺序、bucket 分配、通信调度全部记录下来,后续每一步直接复用这套调度,不再做任何 runtime check。具体能省的东西包括:

- **不再每步遍历计算图**找未使用参数(因为假设全部都用)
- **bucket 顺序固定**,可以做更激进的预先调度
- **autograd 的某些 hook 可以预编译**而不是每步注册

实测在 Llama-7B / 8 卡 DDP 上,static_graph 比默认快 5-15%(模型越简单、反向越快、收益越明显)。

#### 限制

但它要求**计算图严格不变**:

- 不能有任何 dynamic control flow(if/while 取决于 tensor value)
- 所有参数每步都必须收到梯度
- 不能动态添加/移除子模块

实际上现代 LLM 训练绝大多数都满足——Transformer block 完全静态。但 MoE / Mixture-of-Experts 不行,因为不同 token 走不同 expert,部分 expert 参数某些步可能完全没参与。**MoE 训练用 `find_unused_parameters=True`,稠密 LLM 用 `static_graph=True`**——记住这条二选一规则即可。

#### `static_graph` 与 `find_unused_parameters` 互斥

两个开关不能同时打开:静态图假设所有参数都参与,而 find_unused 假设可能不参与,逻辑冲突。PyTorch 会在你两个都设 True 时报错。

实战决策树:

```
模型有动态分支吗?(MoE、条件 routing、某些 detection 模型)
       │
   ┌───┴───┐
  是      否
   │       │
   ▼       ▼
find_unused   static_graph=True
=True         (推荐,快 5-15%)
```

#### 与 `torch.compile` 的关系

`torch.compile` 在 PyTorch 2.x 后默认就要求计算图静态(动态分支会触发 graph break、降级到 eager)。所以 **`torch.compile` + DDP + `static_graph=True`** 是 LLM 训练的现代标配——这三者的假设完全一致,组合起来既快又稳。

注意 `torch.compile` 包装的位置:**先 `compile` 再 `DDP`**:

```python
model = build_model().cuda()
model = torch.compile(model, mode="default")     # 先 compile
model = DDP(model, device_ids=[local_rank], static_graph=True)  # 再 DDP
```

反过来 DDP 会包一层 `module` 干扰 compile 的图捕获,可能完全失效。

### 8.2 通信 hook:接管梯度同步

#### 起因

DDP 默认通信是 NCCL AllReduce,数据类型跟 .grad 一样(通常 FP32)。这套方案对大多数场景够用,但有两类需求 default 满足不了:

**第一类:跨节点带宽紧张**。如果你跨节点用的是 100Gbps Ethernet 而不是 200/400Gbps IB,梯度通信会是瓶颈。这时候**用 BF16 做通信**(带宽减半,精度几乎无损)是非常划算的优化。

**第二类:超大规模训练 + 梯度有冗余**。1024 卡训练里,即使每张卡发送 2V 总数据,聚合带宽压力也很大。如果能在传输前**压缩梯度**(用低秩近似、TopK 稀疏化等),通信量可以再降 2-10 倍,代价是少量精度损失。

`register_comm_hook` 就是 DDP 提供的"开放接口",让你把默认的 AllReduce 换成任意自定义的通信逻辑。

#### 接口

```python
def my_hook(state, bucket: dist.GradBucket) -> torch.futures.Future:
    """
    输入:bucket 内的梯度(flatten 后的一个大 tensor)
    输出:Future,resolve 之后 bucket 里的 .grad 应该是同步好的最终梯度
    """
    grad = bucket.buffer()
    # ... 自定义通信 ...
    fut = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True).get_future()
    return fut.then(lambda f: f.value()[0] / dist.get_world_size())

model = DDP(model, device_ids=[local_rank])
model.register_comm_hook(state=None, hook=my_hook)
```

DDP 内部不再自己做 AllReduce,而是 bucket 凑齐后调用你的 hook,等返回的 future resolve 之后认为通信完成。

#### 内置 hook 1: `fp16_compress_hook`

PyTorch 自带的最常用 hook——把 FP32 .grad 压成 BF16/FP16 通信:

```python
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

model.register_comm_hook(state=None, hook=default_hooks.fp16_compress_hook)
```

工作流程:

```
.grad (FP32) ──► cast to FP16 ──► AllReduce (FP16) ──► cast back FP32 ──► .grad
                  ↓                                          ↓
              省一半带宽                                  精度损失极小
                                                        (LLM 几乎无影响)
```

**通信量减半,精度几乎无损**,在跨节点训练中是默认开启项之一。BF16 版本叫 `bf16_compress_hook`,语义类似但用 BF16(范围更广,更稳)。

注意这只影响**通信中的 dtype**,本地 .grad 和 master 参数仍然是 FP32,与 §MixedPrecision 那篇讲的概念完全独立。

#### 内置 hook 2: `PowerSGD`

更激进的压缩——基于矩阵的低秩分解:

```python
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook

state = powerSGD_hook.PowerSGDState(
    process_group=None,
    matrix_approximation_rank=1,    # 低秩 rank,越小压得越狠
    start_powerSGD_iter=1000,        # 训练初期不用,防影响 warmup
)
model.register_comm_hook(state, powerSGD_hook.powerSGD_hook)
```

PowerSGD 把每个 bucket 的梯度看作矩阵 $G \in \mathbb{R}^{m \times n}$,然后用 power iteration 找一个低秩近似 $G \approx P Q^T$,只通信小得多的 $P, Q$。压缩比 = $mn / (mr + nr) = mn / r(m+n)$,**rank=1 时压缩 50-100 倍**。

代价是精度——低秩近似丢掉了梯度的高频成分。实际经验:

- 视觉 / NLP 小模型:**通常能用,精度损失 < 1%**
- LLM 预训练:**经验上影响较大**,梯度方向被低秩约束后 loss 曲线会变差,目前生产里几乎不用
- 慢网络的 fine-tuning:**收益最大**,通信瓶颈严重时能直接拉训练速度 5-10×

PowerSGD 的实战价值在 LLM 预训练里有限,但在**带宽差但又必须分布式**的场景(教育机构、跨地理区域)是救命稻草。

#### 自己写一个 hook:示例

理解了接口,你完全可以写自己的通信策略。比如**只通信 Top-K 大梯度**:

```python
def topk_hook(state, bucket):
    grad = bucket.buffer()
    k = max(1, int(grad.numel() * 0.01))    # 只发 1% 最大的
    
    # 找 top-k 的位置和值
    abs_grad = grad.abs()
    _, indices = torch.topk(abs_grad, k)
    values = grad[indices]
    
    # 用稀疏 AllReduce(实际要自己实现完整逻辑,这里简化)
    # 通信完毕后把稀疏梯度还原回完整梯度,其余位置置 0
    
    # ...省略完整实现...
    return future
```

这只是示例,实际 TopK SGD 还要处理误差补偿、稀疏 AllReduce、numerical stability 等问题。但接口本身确实是这么开放——DDP 把"梯度同步"这个动作完全暴露给你,只要你的 hook 返回的 .grad 在所有 rank 上语义一致,DDP 不做额外干预。

### 8.3 实战中怎么用这两个开关

把上面的内容串起来,典型 LLM 训练的 DDP 配置:

```python
# 1. 模型构建
model = build_llm_model().cuda()

# 2. (可选)torch.compile 加速
model = torch.compile(model, mode="default")

# 3. DDP 包装 + 静态图
model = DDP(
    model,
    device_ids=[local_rank],
    static_graph=True,                # 稠密模型,假设计算图不变
    bucket_cap_mb=25,                  # bucket 大小,默认 25MB 通常合适
    gradient_as_bucket_view=True,      # 让 .grad 直接是 bucket 的 view,省一次 copy
)

# 4. 跨节点带宽紧张时打开 BF16 通信
if dist.get_world_size() > 8:    # 多节点
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
    model.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)
```

这套配置在 8-256 卡的 LLM 训练上是经过实战验证的、最贴近极限的 DDP 设置。再往上(几千卡)就是 FSDP 或 TP+PP 的领域了。

## 九、小结:DDP 的能力边界

把前面讲的串起来,DDP 这套机制实际**解决了**三件事:

- **算力扩展**:N 张卡的算力被同时利用,样本吞吐近似线性增长
- **通信效率**:Ring AllReduce 让每张卡的发送量趋近 2V、与 N 几乎无关,加上 bucket + 异步重叠,通信被反向计算掩盖
- **正确性**:Broadcast 起点 + AllReduce 同步 + 确定性优化器,数学上严格等价于单卡 batch = B·N 的训练

回头看 §一 的 DP 四问题,DDP 也都给出了具体答案:多进程绕开 GIL、对等架构消灭 master 瓶颈、NCCL 后端支持跨节点、bucket + async 让通信和计算重叠。

但它**没解决**显存扩展——每张卡仍然要存完整的参数、梯度、optimizer state(7B 模型就是 112 GB,单卡 80G 装不下)。这是 ZeRO/FSDP、TP/PP 出场的地方:DDP 是数据并行的"基线",上面分别从"切状态"和"切模型"两个维度继续扩展,共同支撑起现代大模型训练的 3D 并行。

理解 DDP 之后再看 ZeRO,会发现它本质就是把 §3.5 那个 AllReduce 拆成 §3.8 的恒等式 `ReduceScatter + AllGather`,中间塞进分片的 optimizer step——通信量不变,显存却线性下降,这是后续 ZeRO 篇要展开的故事。
