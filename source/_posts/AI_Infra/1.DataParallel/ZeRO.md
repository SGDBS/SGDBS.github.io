---
title: 5. ZeRO 与 FSDP
categories: 学习笔记- AI Infra
date: 2026-04-28 22:48:00
mathjax: true
tags:
    - AI
    - AI Infra
---


ZeRO 是 DeepSpeed 团队在 2019 年提出的显存优化方案,核心思想是**消除数据并行中的显存冗余**。DDP 让每张卡都存完整的模型参数、梯度、优化器状态,这些是冗余的——ZeRO 把它们分片到不同卡上,大幅降低单卡显存占用,让训练超大模型成为可能。PyTorch 原生的 FSDP (Fully Sharded Data Parallel) 是 ZeRO-3 的等价实现。

## 一、动机:DDP 的显存瓶颈

### 1.1 训练时显存的构成

训练一个模型,单卡显存被以下几部分占用:

**模型参数 (Parameters)**:前向反向都需要,通常用 fp16/bf16 存储。

**梯度 (Gradients)**:反向传播产生,通常和参数同 dtype。

**优化器状态 (Optimizer States)**:Adam/AdamW 需要存 fp32 的参数副本(master weights)、一阶动量 m、二阶动量 v,共 3 份 fp32 状态。

**激活值 (Activations)**:前向产生,反向计算梯度时用,这部分通过 activation checkpointing 优化。

**临时缓冲区**:通信、计算中间结果等。

### 1.2 关键计算:7B 模型训练的显存账

以 7B 参数 + 混合精度 (bf16 训练 + fp32 master) + AdamW 为例,单卡显存需求:

- bf16 参数: $7B \times 2 = 14$ GB
- bf16 梯度: $7B \times 2 = 14$ GB  
- fp32 master 参数: $7B \times 4 = 28$ GB
- fp32 一阶动量 m: $7B \times 4 = 28$ GB
- fp32 二阶动量 v: $7B \times 4 = 28$ GB

合计 **112 GB**,A100 80G 单卡装不下。这还没算激活值和临时缓冲区。

如果按 "参数 + 梯度 + optimizer states" 拆解,通用公式是 **每参数 16 字节**(2 + 2 + 4 + 4 + 4)。这个 16 字节是 ZeRO 论文的标准假设。

### 1.3 DDP 的冗余

N 张卡每张都存完整的 16×P 字节,总占用 N×16×P。但实际上,**这些状态在每张卡上都一模一样**(因为参数同步、梯度 AllReduce 后相同、optimizer 更新确定),完全冗余。

ZeRO 的洞察:**既然冗余,就分片存储,需要时再通信获取**。

## 二、ZeRO 三个 Stage:逐级分片

ZeRO 分三个 stage,逐步分片更多内容,显存节省越来越多,通信开销也越来越大。

> 下面三节统一用 **N = 4 卡**、**P = 模型参数量**(以"个数"计)、混合精度训练(fp16 参数 + fp16 梯度 + fp32 OS)的设定来举例。bf16 数值精度不同但占 2 字节/参数,与 fp16 一样。

#### 显存基线对照(每卡持有的内容,DDP)

```
DDP (无分片):
  GPU0: [── P (2P 字节) ──][── G (2P 字节) ──][──── OS (12P 字节) ────]
  GPU1: [── P ───────────][── G ───────────][──── OS ─────────────]
  GPU2: [── P ───────────][── G ───────────][──── OS ─────────────]
  GPU3: [── P ───────────][── G ───────────][──── OS ─────────────]
        每卡 16P 字节,4 卡共 64P,其中 48P 完全冗余
```

OS 内部由三份组成:fp32 master 参数(4P)、Adam 一阶动量 m(4P)、二阶动量 v(4P),共 12P 字节。

---

### 2.1 ZeRO-1: 分片 Optimizer States

**核心思想**:既然 optimizer states 在每张卡上完全相同,那就分成 N 段,每张卡只存自己那一段、只负责更新对应那一段参数。

#### 显存布局

```
ZeRO-1 (只分片 OS):
  GPU0: [── P (2P) ──][── G (2P) ──][OS₀ (3P)]   ← OS 的第 0/4 段
  GPU1: [── P (2P) ──][── G (2P) ──][OS₁ (3P)]
  GPU2: [── P (2P) ──][── G (2P) ──][OS₂ (3P)]
  GPU3: [── P (2P) ──][── G (2P) ──][OS₃ (3P)]
                                     ↑
                          每卡只存 12P/4 = 3P 字节 OS
```

每卡占用 = 2P + 2P + 12P/N。N=4 时单卡 7P;N=64 时单卡约 4P。

#### 一次训练迭代的完整流程

```
┌──────────────────────────────────────────────────────────┐
│ Step 1: Forward (与 DDP 一致)                              │
│   每卡用完整 P 在不同 mini-batch 上算前向,得到 loss_k        │
└──────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│ Step 2: Backward (与 DDP 一致)                             │
│   每卡算出完整梯度 G_k(2P 字节,与 P 同形状)                │
└──────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│ Step 3: 梯度 Reduce-Scatter ★ 替代 DDP 的 AllReduce         │
│                                                            │
│   把 G 沿参数维度切成 4 段。各卡之间互相通信,                 │
│   每卡只收到"自己负责那 1/4 参数的累加梯度":                 │
│                                                            │
│     GPU0 收到: Ḡ[0..¼]   = Σ_k G_k[0..¼]                  │
│     GPU1 收到: Ḡ[¼..½]   = Σ_k G_k[¼..½]                  │
│     GPU2 收到: Ḡ[½..¾]   = Σ_k G_k[½..¾]                  │
│     GPU3 收到: Ḡ[¾..1]   = Σ_k G_k[¾..1]                  │
│                                                            │
│   通信完后各卡丢弃其他 3/4 梯度,只保留自己这 1/4。            │
└──────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│ Step 4: 本地 optimizer step                               │
│   GPU_k:  P_new[k段] = AdamW(P[k段], Ḡ[k段], OS_k)        │
│   每卡只算 1/N 的 Adam 更新,计算量也分摊                    │
└──────────────────────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│ Step 5: 参数 AllGather ★ 替代 AllReduce 的后半段             │
│                                                            │
│   每卡把更新好的 1/4 参数广播,所有卡拿到完整新 P             │
│                                                            │
│   GPU0:  P_new[0..¼] ──┐                                  │
│   GPU1:  P_new[¼..½] ──┼─→ 互相 AllGather                 │
│   GPU2:  P_new[½..¾] ──┤   后每卡都有完整 P_new             │
│   GPU3:  P_new[¾..1] ──┘                                  │
└──────────────────────────────────────────────────────────┘
```

#### 关键洞察

DDP 的 `AllReduce(G)` 在数学上 = `Reduce-Scatter(G)` + `AllGather(G)`。ZeRO-1 把这两步**拆开**,在中间塞进了一个"用本地分片做 optimizer step"的操作——通信总量没变,但 OS 不必再每卡都存一份。

#### 显存与通信

- 单卡显存:2P + 2P + 12P/N → N→∞ 时趋于 4P,约为 DDP 的 **1/4**
- 通信量:Reduce-Scatter (V) + AllGather (V) = 2V,**与 DDP 完全相同**

---

### 2.2 ZeRO-2: 分片 Optimizer States + Gradients

**新增动作**:不只是 OS 分片,**梯度也分片**。这对应一个观察——既然每卡只更新 1/N 参数,那它根本不需要其他 (N-1)/N 的梯度,算完就该扔掉。

#### 显存布局

```
ZeRO-2 (分片 OS + G):
  GPU0: [── P (2P) ──][G₀ (P/2)][OS₀ (3P)]
  GPU1: [── P (2P) ──][G₁ (P/2)][OS₁ (3P)]
  GPU2: [── P (2P) ──][G₂ (P/2)][OS₂ (3P)]
  GPU3: [── P (2P) ──][G₃ (P/2)][OS₃ (3P)]
                       ↑
              每卡梯度只占 2P/4 = 0.5P 字节
```

每卡占用 = 2P + 2P/N + 12P/N = 2P + 14P/N。N→∞ 时趋于 2P,约为 DDP 的 **1/8**。

#### 与 ZeRO-1 的关键差异:逐层 Reduce-Scatter

ZeRO-1 是反向**全部跑完**后,对完整梯度做一次 Reduce-Scatter——这意味着完整梯度曾经在卡上同时存在过(峰值显存仍然占 2P)。ZeRO-2 把它**逐层立刻处理**:

```
反向时序 (模型有 3 层 L1, L2, L3,反向顺序 L3 → L2 → L1):

ZeRO-1:
  ┌─ ∇L3 算出 ─┐┌─ ∇L2 算出 ─┐┌─ ∇L1 算出 ─┐┌─ Reduce-Scatter 全部 G ─┐
  │            ││            ││            ││                        │
  └────────────┘└────────────┘└────────────┘└────────────────────────┘
  ↑ 这段时间峰值显存里有完整 G(2P)

ZeRO-2:
  ┌─ ∇L3 算出 ─┐┌─ RS L3 ─┐┌─ ∇L2 算出 ─┐┌─ RS L2 ─┐┌─ ∇L1 算出 ─┐┌─ RS L1 ─┐
  │            ││ + 丢弃  ││            ││ + 丢弃  ││            ││ + 丢弃  │
  └────────────┘└─────────┘└────────────┘└─────────┘└────────────┘└─────────┘
  ↑ 任何时刻 G 只占 1/N(已分片) + 1 层临时(尚未 RS)
```

每算完一层梯度,立刻 Reduce-Scatter,只保留自己负责那部分,其余立刻释放。这样**梯度峰值显存就变成了 2P/N**。

#### 完整流程

1. **Forward**:与 ZeRO-1 完全一致
2. **Backward + 逐层 Reduce-Scatter**:每反向出一层梯度,立刻 RS,只保留自己那 1/N
3. **本地 optimizer step**:与 ZeRO-1 一致
4. **参数 AllGather**:与 ZeRO-1 一致

#### 显存与通信

- 单卡显存:2P + 14P/N,N=4 时 5.5P,N=64 时约 2.2P
- 通信量:**仍然 = DDP**(Reduce-Scatter 的总通信量与一次性做不变,只是切碎了)

---

### 2.3 ZeRO-3: 分片 Optimizer States + Gradients + Parameters

**最激进**:连参数本身也分片。每张卡平时只持有 1/N 的参数,需要哪一层时再临时聚合。

#### 显存布局

```
ZeRO-3 (全部分片):
  GPU0: [P₀ (P/2)][G₀ (P/2)][OS₀ (3P)]
  GPU1: [P₁ (P/2)][G₁ (P/2)][OS₁ (3P)]
  GPU2: [P₂ (P/2)][G₂ (P/2)][OS₂ (3P)]
  GPU3: [P₃ (P/2)][G₃ (P/2)][OS₃ (3P)]
        ↑         ↑          ↑
        全部按参数维度分 N 段,各卡只持 1/N
```

每卡占用 = 16P/N,**显存随 N 线性下降**。

#### 一次训练迭代:逐层 AllGather → 计算 → 丢弃

设模型有 3 层 L1, L2, L3,每层参数本来分散在 4 卡上(以 L1 为例:GPU0 存 P₁⁰、GPU1 存 P₁¹、...)。

```
═══════════════════ 前向阶段 ═══════════════════

Step F1: 算 L1
  ┌────────────────────────────────────────────────────────┐
  │ ① AllGather L1 参数:                                    │
  │     GPU0 [P₁⁰] ┐                                        │
  │     GPU1 [P₁¹] ├─→ 各卡临时合成完整 P₁ = [P₁⁰|P₁¹|P₁²|P₁³] │
  │     GPU2 [P₁²] │                                        │
  │     GPU3 [P₁³] ┘                                        │
  │ ② 各卡用完整 P₁ 在自己 batch 上算 L1 前向,存激活           │
  │ ③ 丢弃 P₁ 中非本卡持有部分,显存回到分片状态                │
  └────────────────────────────────────────────────────────┘

Step F2: 算 L2  (同上,AllGather → 计算 → 丢弃)
Step F3: 算 L3  (同上)
                            ▼  loss

═══════════════════ 反向阶段 ═══════════════════

Step B3: L3 反向
  ┌────────────────────────────────────────────────────────┐
  │ ① AllGather L3 参数(再聚一次,前向时已丢)                  │
  │ ② 各卡算 ∇L3(完整大小)                                    │
  │ ③ Reduce-Scatter ∇L3:每卡只留自己那 1/4 梯度 G₃ᵏ          │
  │ ④ 丢弃 L3 完整参数                                        │
  └────────────────────────────────────────────────────────┘

Step B2: L2 反向 (同上)
Step B1: L1 反向 (同上)

══════════════════ 优化器更新阶段 ══════════════════

各卡:P_kᵏ ← AdamW(P_kᵏ, G_kᵏ, OS_kᵏ)   ← 全部本地操作
                          ↑
              不需要 AllGather!参数本来就只存 1/N
```

#### 通信开销与重叠优化

每层前向 1 次 AllGather、每层反向 1 次 AllGather + 1 次 Reduce-Scatter。设总参数量 V:

- DDP 通信量:`AllReduce(G) = 2V`
- ZeRO-3 通信量:`AllGather(P) [前向] + AllGather(P) [反向] + Reduce-Scatter(G) [反向] = 3V`

通信量约为 DDP 的 **1.5×**,但因为是**逐层**进行的,可以用**预取(prefetch)**把通信藏到计算后面:

```
预取后的前向时序 (forward_prefetch=True):

时间 ──────────────────────────────────────────────────→
 计算流: │  L1 前向计算  │  L2 前向计算  │  L3 前向计算  │
         ↑              ↑              ↑
 通信流: │AG L1│AG L2(重叠)│AG L3(重叠)│
                ↑通信和计算并发,wall-time ≈ 纯计算时间
```

实际开销往往远低于 1.5×,这就是为什么 ZeRO-3/FSDP 在多机大模型训练里仍然实用。

#### 显存与通信小结

- 单卡显存:**16P/N**,理论上 N 足够大可以训练任意大的模型
- 通信量:**~1.5× DDP**,但可与计算重叠

### 2.4 三个 stage 对比表

| Stage | 分片内容 | 单卡显存(每参数) | 通信量(相对 DDP) | 适用场景 |
|---|---|---|---|---|
| DDP | 无 | 16 字节 | 1× | 小模型 |
| ZeRO-1 | OS | ~4 字节 | 1× | 中等模型 |
| ZeRO-2 | OS + G | ~2 字节 | 1× | 较大模型 |
| ZeRO-3 | OS + G + P | 16/N 字节 | ~1.5× | 超大模型 |

## 三、关键操作的数学与通信原理

### 3.1 AllReduce = Reduce-Scatter + AllGather

这是 ZeRO 设计的核心数学等价:

$$\text{AllReduce}(x) = \text{AllGather}(\text{Reduce-Scatter}(x))$$

Reduce-Scatter:N 张卡每张有完整的向量 V,经过 Reduce-Scatter 后,每张卡持有 V 的不同 1/N 块,这块是所有卡上对应位置的累加和。

AllGather:N 张卡每张有 V 的不同 1/N 块,经过 AllGather 后,每张卡都拿到完整的 V。

通信量分别都是 V·(N-1)/N ≈ V,合起来 2V,这正好等于 Ring AllReduce 的通信量。

**ZeRO-1/2 的关键**:既然反正要做 AllReduce,不如在 Reduce-Scatter 之后停下来,各卡先用自己那 1/N 梯度更新自己那 1/N 参数,再 AllGather 参数。通信量没增加,但 optimizer states 可以分片存储。

### 3.2 ZeRO-3 的额外通信

ZeRO-3 多出来的通信是前向和反向各需要把分片的参数临时 AllGather 起来。设有 L 层,每层参数大小 V_l,前向通信量 Σ V_l = V(总参数量),反向再来一次 V,加上原本的 Reduce-Scatter V,总共 3V。相比 DDP 的 2V,通信增加 50%。

但这个增加可以通过**预取 (prefetch)** 优化:在计算第 i 层时,异步发起第 i+1 层的 AllGather,通信和计算重叠,实际开销可以接近 0。

## 四、ZeRO 的扩展:CPU/NVMe Offload

### 4.1 ZeRO-Offload (基于 ZeRO-2)

把 fp32 optimizer states 和参数更新计算 offload 到 CPU 内存:GPU 只做前向反向,CPU 做 optimizer.step()。

每个 step 的流程:GPU 反向产生 fp16 梯度 → 传给 CPU → CPU 用 fp32 状态做 Adam 更新 → 把更新后的 fp16 参数传回 GPU。

**好处**:GPU 显存进一步下降,因为 fp32 状态完全不占 GPU。代价是 CPU-GPU 之间的 PCIe 传输、CPU 算 Adam 速度慢。适合通信带宽不是瓶颈的场景(单机多卡或参数量极大)。

### 4.2 ZeRO-Infinity (基于 ZeRO-3)

进一步把状态 offload 到 NVMe SSD,理论上可以训练 trillion 级别参数的模型,但 NVMe 带宽远低于内存,实际吞吐很低。

## 五、PyTorch FSDP:ZeRO-3 的官方实现

FSDP (Fully Sharded Data Parallel) 是 PyTorch 原生的 ZeRO-3 实现,API 比 DeepSpeed 更 PyTorch 化。核心概念:

**FlatParameter**:把一组参数(通常是一个 Transformer block)flatten 成一个一维 tensor,然后切分到 N 张卡。这个粒度叫 **FSDP unit**,通常用 `auto_wrap_policy` 自动按 Transformer block 划分。

**前向**:进入一个 unit 前 AllGather 该 unit 的参数,前向完丢弃(reshard)。

**反向**:进入一个 unit 前 AllGather,梯度算完后 Reduce-Scatter,只保留自己那 1/N。

### 5.1 FSDP 关键代码

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

# 假设这是 GPT 的 Transformer block
class TransformerBlock(nn.Module):
    ...

model = build_gpt_model().to(device)

# 自动按 TransformerBlock 划分 FSDP unit
auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)

# 混合精度配置
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,      # 参数用 bf16
    reduce_dtype=torch.bfloat16,      # 梯度通信用 bf16
    buffer_dtype=torch.bfloat16,
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mp_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    device_id=torch.cuda.current_device(),
    # cpu_offload=CPUOffload(offload_params=True),  # 可选 offload
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for x, y in loader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

`ShardingStrategy` 的几种选项:
- `FULL_SHARD` = ZeRO-3
- `SHARD_GRAD_OP` = ZeRO-2
- `NO_SHARD` = DDP
- `HYBRID_SHARD` = 节点内 ZeRO-3,节点间 DDP(权衡通信和显存)

### 5.2 HYBRID_SHARD:大规模训练的实用选择

跨节点通信比节点内 NVLink 慢很多。FULL_SHARD 在多机训练时,跨节点 AllGather 会成为瓶颈。

HYBRID_SHARD 的策略是:在一个节点内(8 卡)做 ZeRO-3 分片,跨节点之间做 DDP 复制。这样每个节点都有完整模型副本,跨节点只做梯度 AllReduce(像 DDP),节点内做 ZeRO-3 通信(走 NVLink,快)。

**适用条件**:模型能在一个节点(8×80GB = 640GB)内放下。LLaMA-65B、7B 都符合,这是当前主流大模型预训练的实用配置。

## 六、ZeRO 与其他并行的组合

ZeRO/FSDP 是**数据并行的优化版**,它和其他并行可以组合使用,这是大模型训练的标配。

**ZeRO + Tensor Parallel**:节点内用 TP 切分单层(8 卡 TP),节点间用 ZeRO-3 或 HYBRID_SHARD。Megatron-DeepSpeed 框架就是这么做的。

**ZeRO + Pipeline Parallel**:PP 把不同层放到不同卡,每个 pipeline stage 内部再用 ZeRO 优化。

**3D 并行 = DP (ZeRO) × TP × PP**:GPT-3、LLaMA 这类模型预训练的标准配置。

## 七、面试高频追问

**Q1: ZeRO-1/2 为什么通信量和 DDP 相同?**

因为 DDP 的梯度 AllReduce 在数学上等价于 Reduce-Scatter + AllGather。ZeRO-1/2 把它拆成两步,中间插入分片的 optimizer step,通信总量没变,但 optimizer states 可以分片存。

**Q2: ZeRO-3 的额外通信能否被隐藏?**

可以。前向时计算第 i 层的同时,异步预取第 i+1 层的参数 AllGather。反向同理。预取做得好的话,通信几乎完全和计算重叠,实际 wall time 增加很少。这正是 FSDP 实现里 `forward_prefetch` 和 `backward_prefetch` 参数的作用。

**Q3: ZeRO-3 vs Tensor Parallel 怎么选?**

ZeRO-3 是数据并行的扩展,各卡有不同数据;TP 是模型并行,各卡有同样数据但模型切分。

**ZeRO-3 优点**:实现简单(基本无侵入)、扩展性好(随卡数线性减显存)、通用性强。**缺点**:每个 forward/backward 都要通信参数,通信量大,跨节点会瓶颈。

**TP 优点**:通信发生在层内,可以走 NVLink,延迟低;每张卡 batch 不缩水。**缺点**:实现复杂(需要改写每个算子)、不能跨节点(NVLink 才行)、扩展上限受限于节点内卡数(通常 8)。

实际:**节点内 TP,节点间 ZeRO/PP**,各取所长。

**Q4: ZeRO-3 训练时 checkpoint 怎么保存?**

每张卡只持有部分参数,直接 save 会得到分片 ckpt。两种做法:

(1) **Sharded checkpoint**:每张卡保存自己的分片,加载时按相同切分恢复。优点是快,缺点是必须用相同 world_size 加载。

(2) **Full checkpoint**:用 `FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT)` 配合 `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)`,所有 rank 把分片 AllGather 到 rank 0 的 CPU 内存,合并成完整 state_dict 保存。优点是兼容单卡推理,缺点是 rank 0 需要足够 CPU 内存。

**Q5: ZeRO 解决了显存,但优化器更新会不会变慢?**

不会,反而变快。因为每张卡只更新 1/N 参数,Adam 的计算量被分摊。瓶颈通常在通信而非 optimizer 计算。

**Q6: 为什么 LLM 训练通常用 bf16 而非 fp16?**

bf16 的指数位和 fp32 一样多(8 位),动态范围广,不容易溢出和下溢;fp16 的指数位只有 5 位,训练大模型容易 NaN,需要 loss scaling 配合。bf16 不需要 loss scaling,训练更稳定。代价是尾数精度稍低,但 LLM 训练对此不敏感。

**Q7: ZeRO 和梯度累积怎么配合?**

完全兼容。ZeRO 在每次 backward 时做梯度的 Reduce-Scatter,多次累积下来梯度仍然分片。FSDP 提供 `no_sync` 上下文(类似 DDP),累积阶段不通信,最后一个 micro-batch 再通信。但 ZeRO-3 的 `no_sync` 会保留完整梯度,占显存,使用前要权衡。

**Q8: 模型有 30B 参数,8×A100 80G,你会怎么训练?**

30B × 16 字节 = 480GB,单卡 80G 装不下,DDP 不行。

8×80G = 640GB,FSDP/ZeRO-3 全分片后每卡需要 60GB 参数+梯度+OS,加上激活,可以训练。**首选 FSDP FULL_SHARD + bf16 + activation checkpointing**,如果显存还紧张可以加 CPU offload。

如果是多机,选 HYBRID_SHARD 减少跨节点通信。
