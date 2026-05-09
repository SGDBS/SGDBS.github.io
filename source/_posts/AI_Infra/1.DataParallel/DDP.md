---
title: 1. Distributed Data Parallel
categories: 学习笔记- AI Infra
date: 2026-04-28 22:48:00
mathjax: true
tags:
    - AI
    - AI Infra
---


## 一、数学基础:为什么数据并行是正确的

### 1.1 SGD 的可分解性

考虑 batch size 为 B 的损失函数:

$$L(\theta) = \frac{1}{B}\sum_{i=1}^{B} \ell(x_i, \theta)$$

梯度:

$$\nabla L(\theta) = \frac{1}{B}\sum_{i=1}^{B} \nabla \ell(x_i, \theta)$$

把 B 个样本平均切给 N 张 GPU,每张卡处理 B/N 个样本。第 k 张卡上的局部梯度是:

$$g_k = \frac{1}{B/N}\sum_{i \in S_k} \nabla \ell(x_i, \theta) = \frac{N}{B}\sum_{i \in S_k} \nabla \ell(x_i, \theta)$$

那么 N 张卡的局部梯度的平均是:

$$\frac{1}{N}\sum_{k=1}^{N} g_k = \frac{1}{N}\sum_{k=1}^{N} \frac{N}{B}\sum_{i \in S_k} \nabla \ell(x_i, \theta) = \frac{1}{B}\sum_{i=1}^{B} \nabla \ell(x_i, \theta) = \nabla L(\theta)$$

**结论**:N 张卡分别在 B/N 个样本上算梯度,然后做平均,数学上完全等价于单卡上 batch size = B 的梯度。这就是 DDP 正确性的根基。

### 1.2 关键前提

这个等价的成立依赖三个条件:**所有卡的初始参数相同**、**每步梯度同步保持参数一致**、**优化器是确定性的**(给定相同的参数、梯度、状态,产出相同的更新)。DDP 的所有工程设计都在保障这三点。

### 1.3 一个常被忽视的细节:BatchNorm 不可分解

BN 的均值方差是在 batch 内统计的,所以 N 张卡各算各的 BN 等价于 batch size = B/N(不是 B)。这就是为什么需要 SyncBatchNorm。LayerNorm、RMSNorm 不存在这个问题,因为它们在样本内部归一化,与 batch 无关。Transformer/GPT 用 LayerNorm,所以 DDP 训练 LLM 不用担心这一点。

## 二、集合通信原语与 Ring AllReduce

DDP 反向阶段做的事看似简单——"把各卡的梯度求和后再分发回去"——但这一步是由更基础的**集合通信原语 (collective communication primitives)** 组合而成的。NCCL、MPI、Gloo 等通信库提供的就是这套零件。理解了这些原语,再回看 Ring AllReduce 就只是一种"如何高效实现 AllReduce"的具体调度。

下面所有图都假设 4 张卡(rank 0~3),每张卡持有一个或多个数据块。

### 2.1 Broadcast(广播)

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

### 2.2 Scatter(散开)

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

### 2.3 Gather(聚集)

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

### 2.4 Reduce(归约)

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

### 2.5 AllReduce(全局归约)—— DDP 的主角

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

### 2.6 Reduce-Scatter

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

### 2.7 AllGather

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

### 2.8 一个关键恒等式

把 ReduceScatter 和 AllGather 串起来,有一个 ZeRO 论文反复用到的等式:

$$\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}$$

直观看:先用 ReduceScatter 让每个 rank 拿到结果的一块,再用 AllGather 把这些块聚合给所有人——结果和直接 AllReduce 等价,**但通信量不变**。这是 Ring AllReduce 的实现思路,也是 ZeRO-1/2 把"DDP 的 AllReduce"换成"分片梯度 + 分片 optimizer"却不增加通信的根本原因。

### 2.9 朴素方案:为什么不用 Parameter Server

最直觉的 AllReduce 实现是 Parameter Server (PS):所有 worker 把梯度发给一个 master,master 求和后广播回来。设梯度大小为 V,worker 数为 N:
- master 接收带宽需求:N·V
- master 广播带宽需求:N·V
- master 是单点瓶颈,**通信量随 N 线性增长**,无法扩展

PS 在异构集群、稀疏更新场景下还有应用,但在同构 GPU 训练里早被 Ring AllReduce 取代。

### 2.10 Ring AllReduce 的核心思路

把 N 张卡排成一个环 0 → 1 → 2 → ... → N−1 → 0,每张卡只和**相邻两个邻居**通信(收上游 / 发下游)。梯度切成 N 块,Ring AllReduce 严格按 §2.8 的等式做:

- **Phase 1 (Reduce-Scatter)**:N−1 轮,把"梯度求和"沿环逐步累加并分散
- **Phase 2 (AllGather)**:N−1 轮,把已求和的块沿环再传一圈,所有人都拿到所有块

每轮每张卡只发 V/N 数据、只收 V/N 数据——这正是 Ring 在大消息上能跑满带宽的原因。

### 2.11 4 卡完整走一遍

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

到这里 Reduce-Scatter 阶段完成:GPU 3 拿到 σ₀,GPU 0 拿到 σ₁,GPU 1 拿到 σ₂,GPU 2 拿到 σ₃,**和 §2.6 的 ReduceScatter 语义完全一致**(只不过这里数据原本就在每张卡上,没有 root)。

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

### 2.12 通信量推导

每张卡每轮发送 V/N 数据。两个阶段共 2(N−1) 轮。每张卡的总发送量:

$$T_{\text{send}} = 2(N-1) \cdot \frac{V}{N} = 2V \cdot \frac{N-1}{N}$$

当 N 很大时,**每张卡发送量趋近 2V,与 N 无关**——这就是 Ring AllReduce 能扩展到几千卡的关键。延迟是 O(N)(轮数),但每轮数据量小,带宽利用率高。

对比 PS 方案的 O(N·V) 单点带宽,Ring 把流量摊平到了所有链路上,这也是为什么现代多机训练几乎清一色用 Ring 类拓扑。

### 2.13 Tree AllReduce 与 NCCL 的实际选择

Ring 的延迟是 O(N),每一轮只能等上游把数据推过来才能继续。在**小消息**场景(几 MB 以下),N 张卡跑 2(N−1) 轮的延迟会盖过通信量节省的好处。

NCCL 还提供 **Tree AllReduce**:把节点组织成二叉树,延迟变 O(log N),适合小张量。NCCL 会根据消息大小自动切拓扑——大消息走 Ring(带宽友好),小消息走 Tree(延迟友好)。但 Ring 是理解所有变体的基础,Tree/双二叉树/2D-Ring 本质都是"如何把 ReduceScatter + AllGather 调度得更快"的不同答案。

## 三、DDP 的工作流程

### 3.1 完整生命周期

第一步是**初始化进程组**:每张 GPU 一个进程,通过 NCCL 通信后端组成进程组,每个进程有 rank 和 world_size。GPU 训练几乎一律选 NCCL——它是 NVIDIA 为 GPU 优化的集合通信库,直接走 NVLink/PCIe/InfiniBand,绕过 CPU 内存拷贝,内置 Ring/Tree 等高效拓扑。Gloo 是通用 CPU 后端,在 GPU 训练里性能差很多,只在 CPU-only 或调试场景才用。

第二步是**模型构造时的状态同步**:DDP 把 rank 0 的模型参数和 buffers 通过 Broadcast 发给所有其他 rank(就是 §2.1 那个原语),保证所有进程从相同状态开始。

第三步是**反向传播中的梯度同步**:每个 step 各进程独立做前向(无通信),反向时通过 autograd hook 触发 bucket 级的**异步** AllReduce,通信和计算重叠。这一步的细节会在 §3.3 单独展开。

第四步是**参数更新**:各进程独立调用 optimizer.step(),由于参数、梯度、优化器状态完全一致,更新结果也完全一致。

把这四步串起来看,DDP 的"参数始终一致"其实靠**三道保证**叠加:① 初始化时 Broadcast rank 0 参数,起点一致;② 每个 step 反向后 AllReduce 让所有 rank 的梯度一致;③ 优化器是确定性的——相同参数、相同梯度、相同状态产出相同的更新。三者共同作用,跨 rank 的参数除浮点累加顺序导致的极小数值差外,始终完全相同。

### 3.2 关键工程优化:Gradient Bucketing

如果每个参数算完梯度就发一次 AllReduce,小消息太多,通信效率低(NCCL 的小消息延迟开销大)。DDP 把参数按一定大小(默认 25MB)打包成 bucket,一个 bucket 内所有参数的梯度都算好后,一次性发起 AllReduce。

更精妙的是 bucket 划分按"反向传播顺序"逆序排列。反向是从最后一层往前算,所以**最后一层的参数最先算完梯度**,DDP 把最后一层放到第一个 bucket,前面的反向还在进行时,这个 bucket 已经可以**异步**发起 AllReduce,实现**计算与通信重叠**。这里"异步"是个独立的大话题——CUDA stream、`async_op=True`、`handle.wait()` 等机制如何让通信不卡反向——我把它单独写在了 [《GPU 训练里的异步计算》](AsyncCompute.md) 里,本节不再展开。

### 3.3 一个工程坑:动态计算图 / 未使用参数

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

## 四、关键代码:从最小可用版本理解

### 4.1 最小可运行的 DDP 训练脚本

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

### 4.2 DDP 内部做了什么:伪代码版

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

### 4.3 几个工程上必须知道的 API

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

**关于随机种子**:DataLoader 的样本顺序由 `DistributedSampler` 处理(内部用相同 seed + rank 做 shuffle,保证不同 rank 拿到不重叠的数据),不需要你管。但 dropout、数据增强、初始化等用到的全局随机状态如果各 rank 完全一样,所有 rank 就在用同样的随机数做同样的扰动,失去了"多 rank 看不同样本"的统计意义。所以 `torch.manual_seed(base_seed + rank)` 是标准做法。

## 五、有效 batch size 与学习率

DDP 下的 effective batch size:

$$B_{\text{effective}} = B_{\text{per\_gpu}} \times N_{\text{gpus}} \times N_{\text{accum}}$$

其中 N_accum 是梯度累积步数。学习率按 **Linear Scaling Rule** 缩放:相对于基准 batch size,batch 扩大 k 倍,学习率也扩大 k 倍。这个 rule 在 ResNet 训练上验证有效,LLM 训练有时用平方根缩放或者实际调试。

## 六、梯度累积与 no_sync

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

## 七、小结:DDP 的能力边界

把前面讲的串起来,DDP 这套机制实际**解决了**三件事:

- **算力扩展**:N 张卡的算力被同时利用,样本吞吐近似线性增长
- **通信效率**:Ring AllReduce 让每张卡的发送量趋近 2V、与 N 几乎无关,加上 bucket + 异步重叠,通信被反向计算掩盖
- **正确性**:Broadcast 起点 + AllReduce 同步 + 确定性优化器,数学上严格等价于单卡 batch = B·N 的训练

但它**没解决**显存扩展——每张卡仍然要存完整的参数、梯度、optimizer state(7B 模型就是 112 GB,单卡 80G 装不下)。这是 ZeRO/FSDP、TP/PP 出场的地方:DDP 是数据并行的"基线",上面分别从"切状态"和"切模型"两个维度继续扩展,共同支撑起现代大模型训练的 3D 并行。

理解 DDP 之后再看 ZeRO,会发现它本质就是把 §2.5 那个 AllReduce 拆成 §2.8 的恒等式 `ReduceScatter + AllGather`,中间塞进分片的 optimizer step——通信量不变,显存却线性下降,这是后续 ZeRO 篇要展开的故事。
