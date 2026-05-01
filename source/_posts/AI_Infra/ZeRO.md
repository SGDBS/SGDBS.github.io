---
title: ZeRO (Zero Redundancy Optimizer)从入门到入土
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

### 2.1 ZeRO-1: 分片 Optimizer States

每张卡只持有 1/N 的 optimizer states (master weights、m、v)。每张卡负责更新自己那 1/N 的参数。

**工作流程**:

前向反向和 DDP 完全一致,各卡有完整参数、计算完整梯度、做梯度 AllReduce。区别在 optimizer.step():

每张卡只更新自己负责的那 1/N 参数(用本地的 1/N optimizer states 和对应的 1/N 梯度)。更新后,通过 **AllGather** 让所有卡都拿到完整的更新后参数,继续下一步训练。

**显存**:每参数从 16 字节降到 4 + 12/N ≈ 4 字节(N 大时),显存大约降到 1/4。

**通信**:相比 DDP 多了一次参数 AllGather,但 AllReduce 本身可以拆解为 Reduce-Scatter + AllGather,所以巧妙地把"梯度 AllReduce"换成"梯度 Reduce-Scatter + 参数 AllGather",**通信量与 DDP 相同**。这是 ZeRO-1 的精妙之处。

### 2.2 ZeRO-2: 分片 Optimizer States + Gradients

在 ZeRO-1 基础上,梯度也分片:每张卡只持有 1/N 的梯度。

**工作流程**:

反向传播时,梯度计算出来后做 **Reduce-Scatter**(而不是 AllReduce):每张卡只接收自己负责那 1/N 参数对应的梯度的累加结果。其他参数的梯度算完就丢弃。

每张卡用本地的 1/N optimizer states 和 1/N 梯度更新自己那 1/N 参数,然后 AllGather 同步参数。

**显存**:每参数从 16 字节降到 2 + 14/N ≈ 2 字节,显存大约降到 1/8。

**通信量**:仍然和 DDP 相同(AllReduce = Reduce-Scatter + AllGather)。

### 2.3 ZeRO-3: 分片 Optimizer States + Gradients + Parameters

最激进的方案:连参数本身都分片,每张卡只持有 1/N 的参数。

**工作流程**:

前向时,每遇到一层,通过 AllGather 把这层的参数从所有卡上聚合起来(临时获得完整参数),计算完前向就把不属于自己的参数丢弃。

反向时,同样需要 AllGather 这层参数才能计算梯度。梯度算完后做 Reduce-Scatter,每张卡只保留 1/N 的梯度。

参数更新和 ZeRO-2 类似,各卡更新自己那 1/N 参数,**不再需要参数 AllGather**(因为参数本身就分片存储)。

**显存**:每参数从 16 字节降到 16/N 字节,**显存随 N 线性下降**。理论上 N 足够大,可以训练任意大的模型。

**通信量**:相比 DDP 增加约 50%。前向多了一次 AllGather(临时聚合参数),反向多了一次 AllGather + Reduce-Scatter,但因为是逐层进行的,可以和计算重叠。

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
