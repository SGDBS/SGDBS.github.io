---
title: 1. 训练显存预算与激活优化
categories: 学习笔记- AI Infra
date: 2026-05-08 20:00:00
mathjax: true
tags:
    - AI
    - AI Infra
---


训练大模型时,显存几乎永远是瓶颈。这一篇是整个 AI Infra 系列后续章节(DDP / ZeRO / Pipeline Parallel / Tensor Parallel)共同的**问题陈述**——先把"训练时显存到底被什么吃掉"算清楚,再讲两个最常用的工程降显存手段:**Micro-batch / 梯度累积** 和 **Activation Checkpointing**,最后顺手把"梯度裁剪"这个训练护栏一起讲掉(它和显存无关,但每个训练循环都要写)。

显存压力可以被拆成两个正交方向:**激活值的"宽度"**(同时塞进来的样本数)和**激活值的"深度"**(单个样本经过的层数)。Micro-batch 解决前者,Re-materialization (Activation Checkpointing) 解决后者,两者既可以独立使用,也常常叠加。

## 一、训练显存的构成

### 1.1 精度和字节数

| 数据类型 | 每个数占用 |
|---|---|
| FP32(单精度)| 4 字节 |
| FP16 / BF16(半精度)| 2 字节 |
| FP8 | 1 字节 |
| INT8 | 1 字节 |

记住:**显存 = 张量元素个数 × 每个元素的字节数**。所有计算都是这个公式,复杂的只是"元素个数怎么数"。

训练时显存主要被四块吃掉:模型参数、梯度、优化器状态、激活值。前三块跟 batch size 和序列长度无关,**真正爆炸的是激活值**——它正比于 `batch × seq_len × hidden × layer_num`。

### 1.2 模型参数 (Parameters)

**是什么**:模型里所有可学习的权重,比如 `nn.Linear` 的 `weight` 和 `bias`、Embedding 表、LayerNorm 的 γ 和 β 等。前向要用它算输出,反向要用它算梯度,整个训练过程都得在显存里。

**怎么数元素个数**:以一个标准 Transformer 层为例(hidden size = h,FFN 中间维度通常 = 4h):

Attention 部分:
- Q、K、V 投影矩阵:3 × (h × h) = 3h²
- Output 投影:h × h = h²
- 小计:4h²

FFN 部分:
- 第一个线性层:h × 4h = 4h²
- 第二个线性层:4h × h = 4h²
- 小计:8h²

LayerNorm:2 × 2h = 4h(很小,常忽略)

**单层合计 ≈ 12h²**,整个模型(L 层)参数量 ≈ **12 × L × h²**(再加上 embedding 和 lm_head,约 2 × V × h,V 是词表大小)。

**LLaMA-7B 实例**:L = 32,h = 4096,12 × 32 × 4096² ≈ 6.4B,加 embedding 等 ≈ 7B。
- FP32 训练:7B × 4B = **28 GB**
- 混合精度(参数存 BF16):7B × 2B = **14 GB**

### 1.3 梯度 (Gradients)

反向传播算出来的 $\partial L / \partial \theta$,**每个参数对应一个梯度**,所以张量形状和参数完全一样。反向算完后优化器要用梯度去更新参数,在更新之前必须留着。

**大小**:和参数一样大。混合精度下梯度通常也是 BF16,7B 模型 ≈ **14 GB**。

### 1.4 优化器状态 (Optimizer States)

优化器自己维护的辅助变量。Adam 系列要存两个:
- **一阶动量 m**(梯度的指数滑动平均)
- **二阶动量 v**(梯度平方的指数滑动平均)

每个都和参数一样大。

**关键陷阱:精度**。即使用混合精度训练,**优化器状态几乎总是用 FP32**(不然数值会爆炸/下溢);而且通常还会**额外保留一份 FP32 的参数副本**(叫 master weights),更新时在 FP32 上做。

混合精度 Adam 的完整账单(7B 模型):

| 项目 | 精度 | 大小 |
|---|---|---|
| 参数(计算用) | BF16 | 14 GB |
| 梯度 | BF16 | 14 GB |
| 参数副本(master) | FP32 | 28 GB |
| Adam m | FP32 | 28 GB |
| Adam v | FP32 | 28 GB |
| **小计** | | **112 GB** |

这就是"**模型参数量 × 16~20 字节**"经验公式的来源——7B 模型光这部分就要 100+ GB,**单张 80GB 的 A100/H100 都装不下**,必须 ZeRO/FSDP 切分。

### 1.5 激活值 (Activations)——真正的"浮动"部分

**是什么**:前向传播每一层的**输出张量**和**中间张量**,例如 Linear 层的输入、Attention 的 softmax 矩阵、GeLU 的输入等。

**为什么需要存**:反向算梯度时要用到。例如对于 $y = W \cdot x$,反向算 $\partial L / \partial W = (\partial L / \partial y) \cdot x^T$,**必须知道 x**;对于 $y = \text{ReLU}(x)$,反向需要知道 x 的符号。每一层都有这种"反向需要的中间值",前向时不存就没法反向。

**怎么数:和 batch、序列长度强相关**。这是激活和前三项最大的区别——**激活正比于 batch_size × seq_len**。

以一个 Transformer 层、batch = b、seq_len = s、hidden = h 为例(BF16):

Attention 内部:
- 输入 x:b·s·h
- Q、K、V:3 × b·s·h
- Attention score 矩阵 QKᵀ:b · num_heads · s · s ← **这个是 O(s²),长序列下会爆**
- Softmax 输出:b · num_heads · s · s
- Attention 输出:b·s·h

FFN 内部:
- 输入:b·s·h
- 中间激活(4h 维度):b·s·4h
- GeLU 输出:b·s·4h

LayerNorm 等:还有几个 b·s·h。

粗略估算,**单层激活 ≈ (10~20) × b·s·h + 2 × b·num_heads·s²**,每个数 2 字节。

**LLaMA-7B 实例**,b = 1,s = 8192,h = 4096,num_heads = 32:
- b·s·h = 1 × 8192 × 4096 = 33.5 M 个数 → 67 MB(BF16)
- 单层 Transformer "线性部分"激活 ≈ 15 × 67 MB ≈ **1 GB**
- Attention 矩阵 b·num_heads·s² = 1 × 32 × 8192² = 2.1 B 个数 → **4.3 GB**(!!)
- 单层合计 ≈ 5+ GB
- **32 层 ≈ 160+ GB**

这就是为什么长序列训练**激活值才是头号显存杀手**,也是为什么 FlashAttention(不存那个 s² 矩阵)和 activation checkpointing 这么重要。

### 1.6 总账:7B 模型

| 类别 | 占用 | 是否随 batch/seq 变化 |
|---|---|---|
| 参数(BF16)| 14 GB | 否 |
| 梯度(BF16)| 14 GB | 否 |
| Master 参数 + Adam m + Adam v(FP32)| 84 GB | 否 |
| 激活值 | ~160 GB | **是,正比于 b·s** |
| **总计** | **~270 GB** | |

单张 H100 80GB 显然装不下。所以实际训练必须组合用:
- **ZeRO/FSDP**:把参数 + 梯度 + 优化器状态切分到多卡(解决前三项)
- **Activation Checkpointing**:丢掉中间激活反向时重算,把 160 GB 砍到 ~30 GB
- **FlashAttention**:消掉那个 s² 的 attention 矩阵
- **Tensor/Pipeline 并行**:进一步切分

**速记公式**:
- 纯推理(前向):参数 + 一份激活 ≈ 参数量 × 2B + 少量激活
- 训练(混合精度 + Adam):静态部分(不随 batch 变)≈ 参数量 × **16~20 字节**;动态部分(激活)≈ b · s · h · L · (10~20) × 2B + b · h · s² · 2B
- **判断瓶颈**:模型大但 batch 和 seq 小 → 静态部分占主导,用 ZeRO/FSDP;模型一般但 seq 很长 → 激活占主导,用 checkpointing + FlashAttention

## 二、Micro-batch:压缩激活的"宽度"

### 2.1 先理清三个 batch 概念

这是初学者最容易混的地方,先把术语钉死:

| 术语 | 含义 |
|---|---|
| **Global batch size** | 一次参数更新真正"看到"的样本总数。决定优化轨迹(学习率、收敛性)。 |
| **Mini-batch / per-GPU batch** | 单卡单次前向处理的样本数。受显存限制。 |
| **Micro-batch** | 在梯度累积或流水线并行中,把 mini-batch 进一步切分后的更小单位。 |

关系:**Global batch = micro-batch × 累积步数 × 数据并行卡数**

### 2.2 核心问题:想要的 batch 太大塞不下

假设你在训练一个模型,理论上 global batch = 256 才能稳定收敛(学习率是按这个调的),但单卡显存只能塞 batch = 8。三种思路:

1. **直接缩小 batch**:batch = 8 训练。问题:梯度噪声大,可能不收敛或要重调学习率。
2. **加机器**:32 张卡数据并行,每卡 batch = 8。问题:没那么多卡。
3. **梯度累积(micro-batch)**:单卡跑 32 次 batch = 8,把梯度累加起来再更新。**等价于** batch = 256,只用 1 张卡。

micro-batch 就是第三种思路的具体实现。

### 2.3 标准训练循环 vs 梯度累积

标准训练循环(无累积):

```python
for batch in dataloader:
    optimizer.zero_grad()        # 清空梯度
    loss = model(batch)          # 前向
    loss.backward()              # 反向,梯度写入 .grad
    optimizer.step()             # 用 .grad 更新参数
```

梯度累积版本(4 个 micro-batch 累加成 1 个 global batch):

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, micro_batch in enumerate(dataloader):
    loss = model(micro_batch) / accumulation_steps   # 注意除法!
    loss.backward()              # 梯度累加到 .grad(PyTorch 默认行为)

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()         # 累计够了再更新
        optimizer.zero_grad()    # 然后清空
```

### 2.4 为什么要除以 accumulation_steps

PyTorch 的 `loss.backward()` **默认是把新梯度加到 `.grad` 上**(不是覆盖)。所以连续 4 次 backward 后,`.grad` 是 4 个 micro-batch 梯度之和。

但损失函数(比如 CrossEntropy)通常在 batch 维度上是**取平均**的,不是求和。如果直接累加 4 次的梯度,得到的是"4 倍的平均梯度",等价于学习率被偷偷放大了 4 倍。所以要么把 loss 除以累积步数(常用),要么把累加的梯度除以累积步数。

**数学等价性**:设 micro-batch 损失 $L_i = \frac{1}{m}\sum_{j} \ell_j$(m 是 micro-batch size)

累加 k 步后梯度 $= \sum_{i=1}^{k} \nabla L_i = \frac{1}{m} \sum_{i,j} \nabla \ell_j$

而真正大 batch 的梯度 $= \frac{1}{km} \sum_{i,j} \nabla \ell_j$

**前者是后者的 k 倍**,所以要除以 k 才严格等价。

### 2.5 一个完整的可跑例子

```python
import torch
import torch.nn as nn

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

micro_batch_size = 8
global_batch_size = 256
accumulation_steps = global_batch_size // micro_batch_size  # = 32

model.train()
optimizer.zero_grad()

for step, batch in enumerate(dataloader):
    inputs, targets = batch
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()

    # 每 32 个 micro-batch 才真正更新一次
    if (step + 1) % accumulation_steps == 0:
        # 可选:梯度裁剪要在 step 之前、累积完之后做
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()
```

注意点:
- **梯度裁剪**必须在累积完之后、`step()` 之前做,不能每个 micro-batch 都裁
- **学习率调度器** `scheduler.step()` 也应该跟着 `optimizer.step()` 走,而不是每个 micro-batch 都调
- **DataLoader 的 batch_size** 设成 micro_batch_size,而不是 global

### 2.6 显存到底省了多少

回到第一节的显存四块:

| 类别 | 是否被 micro-batch 影响 |
|---|---|
| 参数 | ❌ 不变 |
| 梯度 | ❌ 不变(还是参数同大小,累加在同一份 buffer 里)|
| 优化器状态 | ❌ 不变 |
| **激活值** | ✅ **正比于 micro-batch size,按比例下降** |

所以 micro-batch **只省激活值这一块**。如果你的瓶颈是参数 + 优化器(小模型大 batch),micro-batch 帮不上忙——那是 ZeRO/FSDP 的领域。

例子:7B 模型 + s = 8192,激活值 ~160 GB(b = 1)。如果你想跑 b = 8,激活会变 1280 GB——必须靠 micro-batch 把单步 batch 控制在 1。

### 2.7 代价:吞吐 vs 显存

| 维度 | 影响 |
|---|---|
| **数学等价性** | 几乎完全等价于大 batch 训练 |
| **计算量** | 一样(总样本数没变)|
| **wall-clock 时间** | 略增。原因:每次 micro-batch 都有 kernel launch 开销、optimizer step 的间隔变长(次要)|
| **BatchNorm** | ⚠️ **不等价**!BN 在每个 micro-batch 内单独算均值方差,统计量不准。所以现代大模型基本不用 BN,用 LayerNorm/RMSNorm(这些不依赖 batch 维度,完全等价)|
| **DataLoader** | 不变,只是消费速度变成原来的 1/k |

### 2.8 在分布式训练里的微妙之处

#### DDP 的同步开销

DDP 默认每次 `backward()` 后会**自动 all-reduce 所有梯度**(跨卡同步)。如果 micro-batch 累积 32 次,就会 all-reduce 32 次,通信浪费严重。解决:在前 31 次累积时关闭同步,最后一次再同步。

```python
for i, micro_batch in enumerate(dataloader):
    is_last_micro = (i + 1) % accumulation_steps == 0

    # 用 no_sync 上下文管理器跳过 all-reduce
    context = model.no_sync() if not is_last_micro else nullcontext()
    with context:
        loss = model(micro_batch) / accumulation_steps
        loss.backward()

    if is_last_micro:
        optimizer.step()
        optimizer.zero_grad()
```

`model.no_sync()` 会让 backward 只把梯度累加到本地 `.grad`,不触发跨卡同步。最后一次正常 backward 时会一次性 all-reduce 累加好的总梯度。这能省掉 (k-1)/k 的通信开销。

#### 流水线并行中的 micro-batch

micro-batch 在流水线并行(GPipe、1F1B 等)里有**另一个完全不同的用途**——不是为了省显存,而是为了**填充流水线、减少 stage 之间的空闲气泡 (bubble)**。单 micro-batch 的 naive pipeline 会让大部分 stage 在大部分时间空等;切成多个 micro-batch 后,它们像水流一样在不同 stage 上同时跑,把 bubble 压到 $(P-1)/(M+P-1)$——其中 $P$ 是 stage 数,$M$ 是 micro-batch 数。

调度算法的演进、bubble 公式的完整推导、以及 1F1B / Interleaved / Zero Bubble 等具体方案,详见 **Ch2 Pipeline Parallel**。这里只需要记住:**梯度累积 + micro-batch 是流水线并行能跑起来的前提**,Megatron-LM、DeepSpeed Pipeline 都是这套逻辑。

## 三、Re-materialization:压缩激活的"深度"

### 3.1 核心思想:用计算换显存

普通反向传播需要每一层的前向激活(因为链式法则要用到 $\partial L / \partial x$ 和前向时存的中间值)。

**Re-materialization 的策略**:前向时**故意丢掉**中间激活,只在少数"检查点"位置保留;反向需要某层激活时,**从最近的检查点重新前向算一次**得到。

这是一笔交易:
- **省**:丢掉的那些层的激活显存
- **付**:反向时多做一次前向计算(约多 1/3 的总计算量)

### 3.2 具体操作流程

#### 不用 checkpoint 的标准流程

```
前向:x0 → [Layer1] → a1 → [Layer2] → a2 → [Layer3] → a3 → [Layer4] → a4 → loss
       存:    a1,         a2,         a3,         a4 (全部保留)

反向:从 loss 开始,依次用 a4, a3, a2, a1 计算梯度
```

显存占用 = 4 层激活全在。

#### 用 checkpoint 的流程

假设把 4 层分成 2 个 segment,每个 segment 只在**入口**存激活:

```
前向:x0 → [Layer1 → Layer2] → a2 → [Layer3 → Layer4] → a4 → loss
       存 x0           丢 a1   存 a2          丢 a3   存 a4

反向(处理 segment 2,需要 a3):
  - 从 a2 重新前向算 Layer3 → 得到 a3(重算!)
  - 用 a3 反向 Layer4 和 Layer3
反向(处理 segment 1,需要 a1):
  - 从 x0 重新前向算 Layer1 → 得到 a1(重算!)
  - 用 a1 反向 Layer2 和 Layer1
```

显存峰值 = 检查点的激活 + 当前正在重算的那个 segment 的激活,远小于全存。

### 3.3 怎么选检查点位置(segment 大小)

这是经典的**时间-空间权衡**。设总层数 L,每隔 k 层放一个检查点:
- 检查点数量:L/k
- 每个 segment 重算时的激活峰值:k 层
- 总激活显存 ∝ L/k + k

求导得最优 **k = √L**,激活显存从 O(L) 降到 **O(√L)**,重算开销约一次额外前向。这是 Chen et al. 2016 那篇 *Training Deep Nets with Sublinear Memory Cost* 的核心结论。

实践中,Transformer 通常**每个 Transformer block 整体作为一个 checkpoint 单元**(因为 block 内部的 attention 中间矩阵最占显存,整体重算最划算),不一定严格按 √L。

### 3.4 PyTorch 里怎么用

#### 方式一:`torch.utils.checkpoint.checkpoint`

包裹一个函数或子模块即可:

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x

# 在外层 forward 里用 checkpoint 包装
def forward(self, x):
    for block in self.blocks:
        x = checkpoint(block, x, use_reentrant=False)
    return x
```

PyTorch 会自动:前向时不存这个 block 内部的激活,反向时重新调用 `block(x)` 拿到所需中间值。

#### 方式二:`checkpoint_sequential`(针对 nn.Sequential)

```python
from torch.utils.checkpoint import checkpoint_sequential

# 把 24 层切成 4 个 segment,每 6 层一段
x = checkpoint_sequential(self.blocks, segments=4, input=x, use_reentrant=False)
```

#### 注意事项

- **`use_reentrant=False`** 是新 API,更稳定,建议总是设这个。
- **RNG 状态**:checkpoint 默认会保存随机数生成器状态(dropout 等),保证两次前向结果一致,否则梯度会错。
- **不能在 checkpoint 内部修改全局状态**(比如往 list 里 append 东西,第二次前向会重复)。
- **和 `torch.no_grad()` 不兼容的地方要小心**:被 checkpoint 包的函数内部不能整体处于 no_grad,否则没法反向。

### 3.5 进阶变体

**Selective checkpointing(选择性重算)**:不是所有激活都同样昂贵——attention 的 softmax 矩阵是 O(s²) 但计算便宜(重算划算),FFN 的 GeLU 输出是 O(s·h) 计算贵(不如直接存)。新的做法(如 Megatron-LM 的 selective recompute)只对"显存占比高、重算便宜"的部分重算,最优化时间-空间比。

**FlashAttention**:本质上也是 attention 的 re-materialization——softmax 中间矩阵不存,反向时按 tile 重算,但因为是 fused kernel,重算几乎不增加 wall-clock 时间。这是目前长序列训练的事实标准。

**CPU offload + checkpoint**:把检查点激活临时挪到 CPU 内存,进一步省 GPU 显存,代价是 PCIe 传输时间。DeepSpeed 和 FSDP 都支持。

### 3.6 什么时候不该用

- 模型本来就小,激活不是瓶颈:纯亏算力。
- 推理(inference):没有反向传播,不需要存激活,无意义。
- 已经用了 FlashAttention 且瓶颈在 FFN:先看 selective 方案。

## 四、两者的对比与组合

### 4.1 核心区别

**Micro-batch(梯度累积)**:把一个大 batch 切成若干个小 batch 顺序前向 + 反向,梯度累加后再更新参数。降低的是**激活值的"宽度"**——同一时刻只需要存一个 micro-batch 的激活。

**Re-materialization / Activation Checkpointing**:前向时只保存少数几个"检查点"层的激活,丢弃中间激活;反向传播需要时**重新前向计算**一次得到丢失的激活。降低的是激活值的"深度"——同一个 micro-batch 内只保存部分层的激活。

### 4.2 详细对比

| 维度 | Micro-batch | Re-materialization |
|---|---|---|
| **省显存原理** | 减小 batch 维度,激活张量变小 | 丢弃中间层激活,反向时重算 |
| **省的是哪部分** | 激活显存 ∝ batch_size | 激活显存 ∝ 层数(只存检查点) |
| **额外计算开销** | 几乎没有(只是把大 batch 拆成多次) | 多一次前向传播,约 +33% 计算量 |
| **对收敛的影响** | 数学上等价于大 batch(BN 除外)| 完全等价,不改变数值 |
| **对吞吐的影响** | 通信/启动开销略增,但常用于流水线并行重叠 | 计算变多,单步变慢 |
| **能否突破单样本显存** | 不能。batch = 1 仍 OOM 时无效 | 能。即使 batch = 1 也能让超深模型跑起来 |
| **典型组合** | 流水线并行的基本单元 | 长序列、大模型、Transformer 深层堆叠 |

### 4.3 什么时候用哪个

- **batch 太大塞不下** → 用 micro-batch 最划算,几乎免费。
- **模型太深 / 序列太长**,单个样本的激活就已经爆了 → micro-batch 没用,必须 re-materialization。
- **流水线并行(如 GPipe、1F1B)** → 两者通常一起上:micro-batch 让流水线有东西流动并减少 bubble,re-materialization 进一步压缩每个 stage 的激活峰值。

### 4.4 一个直观的数量级感受

假设 Transformer 层数 L = 48,micro-batch = 1,序列长度 s = 8192:
- 不省:激活显存 ∝ L × s(48 份)
- 只用 micro-batch(已经是 1 了):没法再降
- 加 re-materialization(每 √L 设一个检查点):激活显存 ∝ √L × s(≈ 7 份),代价是反向时多算一次前向

记忆口诀:**micro-batch 解决"一次塞太多样本",re-materialization 解决"一个样本本身就太大"**,两者正交,按瓶颈选用或叠加。

### 4.5 显存吃紧时的调参顺序

1. 先开混合精度(BF16/FP16)—— 几乎免费
2. 开 activation checkpointing —— 用 33% 算力换 √L 的激活显存
3. 开 FlashAttention —— 直接消掉 attention 那个 O(s²) 大头
4. micro-batch 梯度累积 —— 控制激活宽度
5. 不够再上 ZeRO-2/3 或 FSDP —— 切分参数/优化器
6. 还不够上张量并行、流水线并行

## 五、梯度裁剪 (Gradient Clipping):训练的安全护栏

前面三节都在讨论"显存怎么省",这一节谈"训练怎么不崩"。梯度裁剪和显存优化是正交的工程手段,但几乎所有大模型训练脚本都会带它,所以放在这里一起讲清楚。

### 5.1 解决什么问题

反向传播算出的梯度,有时会因为某一步特别陡峭(比如 RNN 的长程依赖、Transformer 早期的 attention 不稳定、loss spike 等)突然变得非常大。这时候参数更新

$$\theta \leftarrow \theta - \eta \cdot g$$

就会一次跨出去太远,直接把模型推进 loss 曲面的一个糟糕区域,表现为 **loss 突然飙到 NaN 或者训练曲线崩盘**。裁剪就是给 g 的范数设个天花板,超过就缩回来——是大模型训练里防止 loss spike 的标准护栏。

### 5.2 两种常见做法

#### 按范数裁剪 (clip by global norm)——最常用

$$g \leftarrow g \cdot \min\left(1, \frac{\text{max\_norm}}{\|g\|_2}\right)$$

把所有参数的梯度拼起来当一个大向量,算 L2 范数;超过阈值就整体等比例缩小。**方向不变,只是步长变小**——这是关键,保证优化方向仍然是真实梯度方向,只是被压扁了。

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

返回值是裁剪前的梯度范数,通常会顺手 log 出来监控训练健康度。

#### 按值裁剪 (clip by value)

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

把每个梯度元素 clamp 到 `[-c, c]`。简单粗暴但**会改变方向**(因为不同维度被独立 clip),现代 LLM 训练几乎不用,这里只作了解。

### 5.3 在训练循环里的位置

必须在 `backward()` 之后、`optimizer.step()` 之前,这样裁剪的是已经累计好的、即将被用来更新的那份梯度:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

**梯度累积场景下要在累积完之后裁**(就像第 2.5 节里写的那样),不是每个 micro-batch 都裁——否则裁的是局部梯度,语义不对,且会让真实的 global 梯度方向失真。

### 5.4 阈值怎么选

| 场景 | 典型阈值 |
|---|---|
| LLM 预训练 (GPT、LLaMA、PaLM) | **1.0**(事实标准) |
| LLM 微调 | 0.5 ~ 1.0 |
| RNN/LSTM | 5.0 ~ 10.0(梯度本身就大) |

可以一开始训练时把 `grad_norm` 打印出来看,正常稳定后通常在 0.1 ~ 几之间;如果经常超 100、1000 就说明训练有问题(学习率太大、初始化炸、数据有脏样本),光靠裁剪压不住,要从根上排查。

### 5.5 混合精度下的坑

用 `GradScaler` 做 FP16 训练时,梯度被 scale 放大过(防止 underflow),**必须先 unscale 再裁剪**,否则裁的是放大后的梯度,阈值就失效了:

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)        # 先还原回真实梯度尺度
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

BF16 没有 loss scaling,直接裁就行,这也是 LLM 训练偏爱 BF16 的小理由之一。

### 5.6 在分布式训练里的微妙之处

**DDP**:`clip_grad_norm_` 算的是单卡本地的范数,但 DDP 在 backward 中已经做了 AllReduce 让各卡梯度一致,所以各卡裁出来的结果也一致,语义正确。

**FSDP/ZeRO-3**:每张卡只持有 1/N 的梯度分片,`clip_grad_norm_` 直接调用会得到错误的范数。FSDP 提供了专门的 `model.clip_grad_norm_(max_norm)` 方法,内部会做跨卡归一化,**必须用这个**而不是 `torch.nn.utils.clip_grad_norm_`。

**流水线并行**:每个 stage 只持有一部分参数,需要在所有 stage 间 AllReduce 范数后再统一缩放,Megatron-LM、DeepSpeed Pipeline 内部已经处理好了。

### 5.7 一句话总结

梯度裁剪就是**梯度的限速器**——不改变更新方向,只在某次梯度异常大时把步长拉回安全区间,几乎所有 LLM 训练脚本都会带上 `clip_grad_norm_(..., 1.0)` 这一行,组合上 BF16、warmup、合理初始化,基本就能把"loss 突然 NaN"挡在门外。

## 六、面试高频追问

**Q1: 训练时显存到底被什么吃掉了?**

四块:参数、梯度、优化器状态、激活。前三块跟 batch 和序列长度无关,大小 ≈ 参数量 × 16~20 字节(混合精度 + Adam);第四块正比于 b · s · h · L,长序列下是绝对的大头。前者用 ZeRO/FSDP 切分,后者用 checkpointing + FlashAttention 解决。

**Q2: 为什么混合精度训练里优化器状态还要 FP32?**

数值稳定性。Adam 的二阶动量 v 是梯度平方的滑动平均,BF16/FP16 容易下溢到 0,导致更新方向错误;参数 master 也要 FP32,否则小学习率下 `param += lr * grad` 在半精度里直接被舍掉。这就是为什么"参数 × 16 字节"——2 (BF16 param) + 2 (BF16 grad) + 4 (FP32 master) + 4 (m) + 4 (v)。

**Q3: Micro-batch 和 ZeRO/FSDP 的省显存有什么本质区别?**

Micro-batch 只省**激活值**(动态部分),不动参数/梯度/优化器状态。ZeRO/FSDP 只切**参数 + 梯度 + 优化器状态**(静态部分),不动激活。两者完全正交,常一起用。

**Q4: 为什么梯度累积时 loss 要除以 accumulation_steps?**

PyTorch 的 backward 默认是把新梯度**累加**到 `.grad`(不是覆盖)。CrossEntropy 等 loss 通常在 batch 维度取平均,如果直接 backward 累积 k 次,得到的是"k 倍的平均梯度",等价于学习率被偷偷放大 k 倍。除以 k 才能严格等价于"在 k·m 个样本上算一次大梯度"。

**Q5: Re-materialization 为什么是 +33% 算力?**

完整训练一步 = 1 次前向 + 1 次反向。反向里链式法则的计算量大约是前向的 2 倍(要算 $\partial L/\partial x$ 和 $\partial L/\partial W$ 两个东西)。所以 forward : backward ≈ 1 : 2,总共 3 份算力。Re-materialization 多做 1 次前向,变成 4 份算力,**+33%**。

**Q6: DDP + 梯度累积怎么避免每次 micro-batch 都 all-reduce?**

用 `model.no_sync()` 上下文管理器,在累积阶段(前 k-1 次)关闭 DDP 的梯度同步,只让最后一次 backward 触发一次 all-reduce 同步累加好的总梯度。能省 (k-1)/k 的通信开销。

**Q7: BatchNorm 在梯度累积下为什么不等价?**

BN 的均值方差是在 batch 维度内统计的,每个 micro-batch 单独算 BN 等价于 batch size = m 而不是 k·m,统计量不准会影响收敛。LayerNorm/RMSNorm 在样本内部归一化,与 batch 无关,完全等价——这也是现代 LLM 用 LayerNorm 的另一个工程理由。

**Q8: Transformer 训练里检查点要切多细?**

实践中**每个 Transformer block 整体作为一个 checkpoint 单元**最常见,因为 block 内部的 attention 矩阵和 FFN 中间激活最占显存,整体重算划算。理论上 √L 是最优,但工程上简单粗暴按 block 切就够用,且更易和 FlashAttention、selective recompute 等配合。

**Q9: FlashAttention 和 activation checkpointing 是同一个东西吗?**

思想一脉相承——都是"前向不存中间结果,反向需要时重算"——但层次不同。Checkpoint 是层级别的(整层重算),FlashAttention 是 kernel 内部的 tile 级别(softmax 矩阵根本不写回 HBM,反向按 tile 重算)。FlashAttention 在 kernel fusion 加持下,重算几乎不增加 wall-clock 时间,所以是无脑开;checkpoint 是按需开,因为有明确的算力代价。

**Q10: 7B 模型 + 32k 序列要怎么训练?**

按瓶颈拆:参数+梯度+OS = 112 GB,激活在 32k 下能上千 GB。组合:**bf16 + FlashAttention(消 s²)+ activation checkpointing(消 L)+ FSDP/ZeRO-3(切静态部分)**,如果还紧再加 micro-batch=1 + 梯度累积、CPU offload。这就是现代长序列大模型训练的标准菜单。

**Q11: 梯度裁剪用 clip by norm 还是 clip by value?**

LLM 训练几乎只用 **clip by global norm**(`clip_grad_norm_`)。原因:它把所有参数的梯度看成一个大向量做整体缩放,**保留方向只压步长**;而 clip by value 是逐元素 clamp,会让不同维度被独立 clip 掉,改变梯度方向,优化轨迹会偏离真实梯度。GPT、LLaMA、PaLM 都是 `max_norm=1.0` 这套。

**Q12: FSDP/ZeRO-3 下为什么不能直接用 `torch.nn.utils.clip_grad_norm_`?**

因为每张卡只持有 1/N 的梯度分片,本地算出来的范数是 $\|g_{\text{local}}\|$,而不是真正的全局 $\|g\|$。直接用会导致各卡按错的尺度缩放,梯度被裁得过狠或过松。FSDP 提供了 `model.clip_grad_norm_(max_norm)`,内部先 AllReduce 各分片的平方和、开方得到全局范数,再做缩放——必须用这个 API。Megatron-LM、DeepSpeed Pipeline 同理,框架内部都做了跨 rank 的范数归一化。
