---
title: 1. GPipe:朴素流水线与 micro-batch
categories: 学习笔记- AI Infra
date: 2026-05-10 14:00:00
mathjax: true
tags:
    - AI
    - AI Infra
---


这一篇是 Ch2 Pipeline Parallel 的开篇。GPipe 是 Google 在 2018 年提出的第一代实用流水线方案,它的核心贡献只有一个——**用 micro-batch 把朴素流水线的 bubble 压下去**——但围绕这一招衍生出来的 bubble 公式、激活显存账、调度顺序,几乎是后续所有 PP 算法(1F1B / Interleaved / Zero Bubble / DualPipe)的共同语言。把 GPipe 吃透,后面的进化都只是在它的基础上做局部优化。

按之前的风格,这篇仍然按 **起因 → 数学/系统原理 → 工程细节 → 直觉与踩坑** 推进。最后会顺带说清楚 GPipe 为什么注定要被 1F1B 接班——这是 Ch2 第二篇的引子。

## 一、起因:数据并行 + ZeRO 解决不了的场景

### 1.1 ZeRO 的天花板

Ch1 的 ZeRO/FSDP 已经把"参数 + 梯度 + 优化器状态"切到了多卡上,理论上 N 张卡可以训 N 倍大的模型。但实际上 ZeRO-3 有两个工程上的硬天花板:

**第一,激活显存切不掉**。ZeRO 切的是静态状态(参数/梯度/optimizer),激活值仍然完整地待在每张卡上。前面 `MemoryBudget` 算过,7B + 32k 序列下激活就要上百 GB——ZeRO 对它一筹莫展,只能靠 activation checkpointing。但 AC 也只能把激活降到 $O(\sqrt{L})$,**一旦单层激活本身就装不下,任何 AC 都救不了**。

**第二,跨节点 AllGather 太慢**。ZeRO-3 forward 时需要把当前层的参数从所有 rank AllGather 回来,backward 时再 AllGather 一次。这套通信发生在每一层、每一步,**节点内吃 NVLink 没问题(900 GB/s),跨节点吃 IB(50 GB/s)就直接拖慢 5-10 倍**。Llama-3 405B 这种规模,光 ZeRO-3 在 1024 卡上的 AllGather 就把训练时间吃掉一半。

简单说:**模型再大一点、序列再长一点,ZeRO 就够不着了**。

### 1.2 流水线并行的核心思想

GPipe 的解法和 ZeRO 完全不同——它不切"状态",而是**按层切模型**:

```
原模型 (32 层 Transformer):
  Layer 0 → Layer 1 → ... → Layer 31

切成 4 个 stage(P=4 张 GPU):
  GPU 0:  Layer 0  ~ Layer 7    (stage 0)
  GPU 1:  Layer 8  ~ Layer 15   (stage 1)
  GPU 2:  Layer 16 ~ Layer 23   (stage 2)
  GPU 3:  Layer 24 ~ Layer 31   (stage 3)
```

每张卡只持有原模型的 1/P 参数,激活也只存自己负责的那段层数。**显存压力直接从 O(L) 降到 O(L/P)**,这就是 PP 能跑超大模型的根本原因。

代价是:**stage 之间必须串行**。Stage 1 要等 Stage 0 把激活送过来才能开始算,Stage 2 要等 Stage 1……一个 batch 流完 P 个 stage 需要 P 倍的时间。

### 1.3 与 DP / TP 的对比

| 并行方式 | 切什么 | 通信原语 | 通信频率 | 适合的硬件层级 |
|---|---|---|---|---|
| **Data Parallel** | 切 batch,模型完整复制 | AllReduce | 每 step 一次 | 跨节点都行 |
| **ZeRO / FSDP** | 切 batch + 切静态状态 | AllGather / ReduceScatter | 每层一次 | 节点内最佳 |
| **Tensor Parallel** | 切单层内的矩阵 | AllReduce | 每层 2-4 次 | **必须节点内**(NVLink) |
| **Pipeline Parallel** | 切层 | P2P send/recv | 每个 stage 边界一次 | **跨节点首选**(IB 也够) |

PP 最大的工程优势是**通信量最小**——只有 stage 边界的激活/梯度需要 P2P 传输,而且只发给相邻 rank,不需要 AllReduce 这种全局集合通信。这就是为什么工业训练的拓扑通常是 **节点内 TP + 跨节点 PP**——把贵的通信放在 NVLink,便宜的通信留给 IB。

## 二、Naive Pipeline:能跑但是慢

### 2.1 单 batch 的调度图

最朴素的实现:一个 batch 顺序通过所有 stage,forward 流到底,backward 回来,然后下一个 batch。假设每个 stage 的 forward 时间为 $T_f$,backward 为 $T_b$:

```
t:        0   1   2   3   4   5   6   7
Stage 0:  F   ·   ·   ·   ·   ·   ·   B
Stage 1:  ·   F   ·   ·   ·   ·   B   ·
Stage 2:  ·   ·   F   ·   ·   B   ·   ·
Stage 3:  ·   ·   ·   F   B   ·   ·   ·
```

(为了图清晰,这里假设 $T_f = T_b = 1$,P=4 个 stage)

肉眼可见:**任何时刻都只有一个 stage 在干活,其他 3 个在等**。

### 2.2 利用率分析

总时间 = forward 流过 P 个 stage + backward 回来 P 个 stage = $2P \cdot T$(假设 $T_f = T_b = T$)。

每个 stage 真正干活的时间只有 forward + backward = $2T$。

利用率:

$$
\text{utilization} = \frac{2T}{2P \cdot T} = \frac{1}{P}
$$

**P 张卡只达到了单卡 1/P 的利用率**——4 卡的 PP 比单卡训练还慢!这显然不能用。

问题的根源很清楚:**一次只送一个 batch,流水线根本"流"不起来**。这就是 GPipe 要解决的问题。

## 三、GPipe 的核心招式:Micro-batch + Pipeline

### 3.1 思想:把 batch 切成 micro-batch 顺序发射

GPipe 的洞察:**既然 stage 之间是串行的,那就让多个样本"同时在不同 stage 上跑"**。

具体做法:把一个 mini-batch(假设 batch_size = 32)切成 M 个 micro-batch(假设 M = 8,每个 micro-batch_size = 4),按顺序往流水线里塞。Stage 0 不必等 micro-batch 0 走完整个流水线再发 micro-batch 1——它做完 micro-batch 0 就立刻开始 micro-batch 1。

这样多个 micro-batch 像水流一样**同时**流过不同的 stage,前面的 stage 不再空等。

这里的 micro-batch 和 Ch1 `MemoryBudget` §2 里梯度累积的 micro-batch **本质上是同一种东西**——都是把大 batch 切成 M 份,分别做 forward + backward,梯度累加到同一份 buffer,最后才 `optimizer.step()`。区别只在目的:梯度累积是为了让大 batch 的激活别同时塞进单卡;GPipe 是为了让多个 micro-batch 在不同 stage 上同时跑、填满流水线。所以一句话概括:**GPipe = 流水线 + 梯度累积**——Ch1 §2.4 里"loss 必须除以 accumulation_steps"那个细节,在 GPipe 里也得原样照搬,否则梯度尺度会偏。

### 3.2 GPipe 完整调度图

P = 4,M = 4,假设 $T_f = T_b = 1$,**先全部 forward,再全部 backward**(这是 GPipe 的关键特征):

```
t:        0   1   2   3   4   5   6   7   8   9   10  11  12  13
Stage 0:  F0  F1  F2  F3  ·   ·   ·   ·   ·   ·   B3  B2  B1  B0
Stage 1:  ·   F0  F1  F2  F3  ·   ·   ·   ·   B3  B2  B1  B0  ·
Stage 2:  ·   ·   F0  F1  F2  F3  ·   ·   B3  B2  B1  B0  ·   ·
Stage 3:  ·   ·   ·   F0  F1  F2  F3  B3  B2  B1  B0  ·   ·   ·
          |←─ warmup ─→|              |←─── steady ───→|←cooldown→|
```

观察这张图(请认真对照,后面所有公式都基于它):

- **Warmup 阶段**(t = 0~3):流水线在"灌水",前 P-1 个时刻总有 stage 在等
- **Steady 阶段**(t = 3~7 forward / t = 7~10 backward):中间这段是流水线满载状态,效率最高
- **Cooldown 阶段**(t = 11~13):反向走完最后几个 stage,后 P-1 个时刻又开始有 stage 空闲

forward 全部完成在 t = P + M - 1 = 7 时刻,然后立刻进入 backward,backward 全部完成在 t = 2(M + P - 1) = 14 时刻。

注意 **GPipe 是严格"先 F 后 B"**——所有 micro-batch 的 forward 跑完才开始任何 backward。这是它和 1F1B 的核心区别(1F1B 会立刻交替),也是 GPipe 激活显存爆炸的根源(下一节会算)。

### 3.3 Bubble 公式推导

设 $T_f = T_b = T$(简化,实际两者不一定相等)。

**总时间**:从 stage 0 开始第一个 forward,到 stage 0 完成最后一个 backward,需要

$$
T_{\text{total}} = (M + P - 1)(T_f + T_b) = 2(M + P - 1)T
$$

直觉:从 micro-batch 0 进入 stage 0,到 micro-batch 0 在 stage P-1 完成 forward,经过 P 个时间单位;然后 M-1 个 micro-batch 依次涌入,每个再加 1 个时间单位,共 $M + P - 1$ 个时间单位完成所有 forward。Backward 完全对称,再加 $M + P - 1$ 个时间单位。

**有效计算时间**:每个 stage 真正在算的时间是 $M$ 个 forward + $M$ 个 backward = $2MT$。

**Bubble 时间**:

$$
T_{\text{bubble}} = T_{\text{total}} - 2MT = 2(P-1)T
$$

**Bubble 比例**(无效时间 / 总时间):

$$
\text{bubble ratio} = \frac{T_{\text{bubble}}}{T_{\text{total}}} = \frac{2(P-1)T}{2(M+P-1)T} = \boxed{\frac{P-1}{M+P-1}}
$$

这个公式是整个 PP 算法演进的"基线"——后续所有调度算法都是想办法把分子降低(Zero Bubble)或者把分母提高(Interleaved)。

### 3.4 数值直觉:M 取多少够用

把上面的公式代几个数,感受一下 bubble 怎么随 M 变化(P = 8,工业典型设置):

| M | bubble ratio | 利用率 |
|---|---|---|
| 1 | 7/8 = 87.5% | 12.5%(等于 naive,完全没用)|
| 2 | 7/9 ≈ 78% | 22% |
| 4 | 7/11 ≈ 64% | 36% |
| 8 | 7/15 ≈ 47% | 53% |
| **16** | 7/23 ≈ **30%** | 70% |
| **32** | 7/39 ≈ **18%** | 82% |
| 64 | 7/71 ≈ 10% | 90% |
| 128 | 7/135 ≈ 5% | 95% |

工业上的经验法则:**M 至少要是 P 的 4 倍**(对应 ~70% 利用率才划算),典型设置是 $M = 4P \sim 8P$。Megatron-LM、DeepSpeed Pipeline 默认值就在这个区间。

但 M 不能无限大——下一节会看到,**M 大了激活显存会爆**。这个 "想要小 bubble vs 不想 OOM" 的矛盾是 GPipe 的核心痛点,也是后续 1F1B 的动机。

## 四、激活显存:GPipe 的代价

### 4.1 在飞 micro-batch 数

回到 §3.2 那张调度图,看每个 stage 在某个时刻"压着多少个 micro-batch 的激活"。

以 stage 0 为例:

- t = 0:刚算完 F0,激活计数 = 1(等 F0 反向时用)
- t = 1:算完 F1,激活计数 = 2(F0 + F1 都还没反向)
- t = 2:算完 F2,激活计数 = 3
- t = 3:算完 F3,激活计数 = **4**(峰值!所有 micro-batch 的 forward 都跑完了,但反向一个还没开始)
- t = 4 ~ 9:stage 0 闲着,激活计数保持 4
- t = 10:开始 B3,反向消耗 F3 的激活,计数降到 3
- t = 11:B2,计数 2
- t = 12:B1,计数 1
- t = 13:B0,计数 0

**Stage 0 的激活峰值是 M 个 micro-batch 的激活**。也就是说,GPipe 切了 M 个 micro-batch 之后,stage 0 上的激活总量并没有减少——单个 micro-batch 的激活变小了 M 倍,但同时压着 M 个,**乘起来不变**。

### 4.2 所有 stage 的激活峰值都 = M

对 stage k 同样的方法分析:

- Stage k 的第一个 forward 在 t = k 完成,第 M 个 forward 在 t = k + M - 1 完成
- 但 stage k 的第一个 backward 要等 stage P-1 把所有 forward 跑完、backward 再从 stage P-1 一路传回来,中间隔很久
- 这段窗口里,stage k **一直压着 M 个 micro-batch 的激活,从不释放**

所以 GPipe 在**每个 stage 上的激活峰值都 = M × 单 micro-batch 激活**,没有 stage 0 / stage P-1 的差别。这是 GPipe "全 F 后全 B" 调度的直接后果——任何 stage 都得等所有 micro-batch 反向才能开始消耗激活。

### 4.3 用 in-flight 框架精确化显存公式

把激活峰值写成更通用的形式:

$$
\text{stage 显存峰值} = (\text{在飞 micro-batch 数}) \times (\text{单 micro-batch 激活})
$$

代入 GPipe(在飞数 = M):

$$
\text{GPipe 峰值} = M \times A_{\text{per-micro}} = M \cdot m \cdot A_{\text{sample}}
$$

**这里有一个常见的混淆要小心**:乘积 $M \cdot m$ 实际上就是 global batch size $B$。如果 **B 固定**(给定 batch 大小,只是切成更多 micro-batch),那 GPipe 的激活峰值 = $B \cdot A_{\text{sample}}$,**和 M 完全无关**——M 上去 m 下来,代数上抵消。

但工程中**真正被固定的不是 B,而是 m**。原因:m 太小会让单个 forward 的 GPU kernel 跑不满(矩阵乘的有效维度不够,Tensor Core 利用率掉),70B 量级的大模型,m 通常已经压到 1~2 这种"地板值",再小也省不下去。

这种"m 固定"模式下,M 是被 bubble 公式 $\frac{P-1}{M+P-1}$ 拽着往上加的——

- 想降 bubble → 加大 M
- m 是地板值,不能动 → 加大 M 等于加大 $B = M \cdot m$
- 加大 B → 激活峰值 $B \cdot A_{\text{sample}}$ 线性涨

所以 GPipe 的根本困境其实是:**bubble 和显存通过 B 耦合在一起**。你不能既要小 bubble 又要小显存——加大 M 必然加大 B,加大 B 必然加大显存。GPipe 的 M 被显存死死限在 $4P \sim 8P$,bubble 怎么也压不下来。

下一章 1F1B 的核心贡献,就是把"在飞数"从 M 解放出来——让它只和 P 有关,从而 M 可以独立加大而不带任何代价。

| 方案 | 在飞数(stage 0) | 显存峰值 |
|---|---|---|
| 不切 micro-batch | 1 | $B \cdot A_{\text{sample}}$ |
| 梯度累积(单卡顺序) | 1 | $\frac{B}{M} \cdot A_{\text{sample}}$ ← M 真正起作用 |
| **GPipe** | **M** | $M \cdot m \cdot A_{\text{sample}} = B \cdot A_{\text{sample}}$ ← M 被 m 抵消,只看 B |
| 1F1B(下一篇) | **P** | $P \cdot m \cdot A_{\text{sample}}$ ← 和 M 无关 |

### 4.4 为什么 GPipe 必须配 activation checkpointing

正因为激活显存没省,GPipe 的论文从一开始就**强制配合 activation checkpointing**——前向时只在每个 stage 入口存激活,中间层全部丢掉,反向时按需重算。

效果:

| 方案 | 激活显存(stage 0)|
|---|---|
| GPipe,无 AC | $M \times L_{\text{per-stage}} \times \text{激活} / L_{\text{per-stage}}$(全部层激活) |
| GPipe + AC | $M \times \text{stage 入口激活}$(只存边界) |

激活从"所有层"降到"只有 stage 边界",通常能降一个数量级。代价是 GPipe + AC 的总计算量比 naive 多一次 forward(约 +33%),叠加 PP 本身的 bubble(假设 30%),GPipe 的总开销是 naive 的 $1.33 \times 1.3 \approx 1.73$ 倍——**用 1.7 倍算力换"能跑得起来"**。

### 4.5 三个旋钮的耦合关系

GPipe 调参其实是在三个量之间找平衡:

```
                    ┌─ M (micro-batch 数)
                    │
                    ├─ B/M (单个 micro-batch 大小)
                    │
                    └─ P (stage 数)
```

- M ↑ → bubble 降,但激活显存升
- B/M ↑ → 激活显存升,但 kernel 跑得更高效(避免太小 batch 的 GPU 利用率不足)
- P ↑ → 单 stage 模型更小,但 bubble 升、跨节点通信也增多

工业经验:**先定 P(由模型大小和单卡显存决定),再选 B/M(让单 micro-batch GPU 利用率 > 70%),最后用 M 把 bubble 压到能接受的范围**。如果三者怎么配都不满足,就要换 1F1B / Interleaved。

## 五、工程实现

### 5.1 怎么把模型切成 stage

最简单的均匀切层:

```python
import torch.nn as nn

def split_model_by_layer(model, num_stages):
    """假设 model 是一个 nn.Sequential 或 layer list"""
    layers = list(model.children())
    L = len(layers)
    assert L % num_stages == 0, "层数要整除 stage 数,否则得手动平衡"
    
    layers_per_stage = L // num_stages
    stages = []
    for i in range(num_stages):
        stage_layers = layers[i * layers_per_stage : (i+1) * layers_per_stage]
        stages.append(nn.Sequential(*stage_layers))
    return stages
```

这是教学版,**生产代码要做的远不止这些**:

- **embedding 和 LM head 的归属**:embedding 通常和 stage 0 绑,lm_head 和 stage P-1 绑;但两者在 LLM 中常常**共享权重**,这时候要么放弃共享,要么在 stage 0 和 stage P-1 之间做一次 AllReduce 同步梯度
- **均匀切层不一定均衡**:第一层(带 embedding)和最后一层(带 lm_head + loss)显存压力天然更大,实际生产代码会按 FLOPs 或显存做加权切分。Megatron-LM 的 `--num-layers-per-virtual-pipeline-stage` 就是给这个用的
- **LayerNorm / Dropout 等小算子**:跨 stage 时要小心权重位置,通常和它依附的主体层放一起

### 5.2 Micro-batch 数据流与 send/recv

GPipe 调度的核心代码长这样(简化版,只关心 stage k):

```python
def gpipe_step(stage_module, micro_batches, prev_rank, next_rank, rank, world_size):
    """rank: 当前 stage 编号 (0 ~ P-1)"""
    M = len(micro_batches)
    activations = []  # 存所有 micro-batch 的输出,反向用

    # ===== 全部 forward =====
    for i in range(M):
        if rank == 0:
            # 第一个 stage 直接读 micro-batch
            x = micro_batches[i]
        else:
            # 其他 stage 从前一个 stage 接收激活
            x = recv_tensor(src=prev_rank)
        
        x = stage_module(x)
        activations.append(x)
        
        if rank != world_size - 1:
            # 不是最后一个 stage 的话,把激活发给下一个
            send_tensor(x, dst=next_rank)

    # ===== 全部 backward =====
    for i in reversed(range(M)):  # 反向按相反顺序
        if rank == world_size - 1:
            # 最后一个 stage 自己算 loss、自己起反向
            loss = compute_loss(activations[i], targets[i]) / M
            grad = torch.autograd.grad(loss, activations[i])[0]
        else:
            # 其他 stage 从后一个 stage 接收梯度
            grad = recv_tensor(src=next_rank)
        
        # 反向算这一段的梯度,顺便把梯度往前传
        input_grad = backward_through_stage(activations[i], grad)
        
        if rank != 0:
            send_tensor(input_grad, dst=prev_rank)

    # ===== 一次性更新参数 =====
    optimizer.step()
    optimizer.zero_grad()
```

几个关键工程点:

- `loss / M`:和 Ch1 梯度累积里的除法一个道理,**M 个 micro-batch 的梯度累加成一份,要除以 M 才数学等价于 batch_size = M·m 的训练**
- `optimizer.step()` 在所有 M 个 micro-batch 反向完之后才调用一次——本质上 GPipe 就是"流水线版的梯度累积"
- `send_tensor / recv_tensor` 用 `torch.distributed.isend / irecv` 或 NCCL 的 `ncclSend / ncclRecv`,**只对相邻 rank 通信**

### 5.3 P2P 通信的实现细节

NCCL 从 2.7 版本开始正式支持 `ncclSend` / `ncclRecv`,这是 PP 的通信基础。PyTorch 上对应:

```python
import torch.distributed as dist

# 异步 send
req_send = dist.isend(tensor=x, dst=next_rank)

# 异步 recv(必须先分配好 buffer)
recv_buf = torch.empty_like(x)
req_recv = dist.irecv(tensor=recv_buf, src=prev_rank)

# 等待完成
req_send.wait()
req_recv.wait()
```

**几个常见踩坑**:

1. **shape 必须事先匹配**。`irecv` 要求接收方提前知道张量形状,所以 PP 框架通常会在第一个 micro-batch 之前先发一次"shape 协商"消息,或者要求所有 micro-batch shape 严格一致(不然变长序列就麻烦)
2. **dtype 也要一致**。混合精度下激活通常是 BF16,梯度可能是 FP32,要分开处理
3. **死锁**。如果两个 stage 都在 `recv` 而没人 `send`,直接卡死。GPipe 的"先全 F 再全 B"调度天然避免了这个问题,但写自定义调度要非常小心
4. **CUDA stream 管理**。计算和通信最好在不同 stream 上,这样 send/recv 能藏在下一个 micro-batch 的 forward 后面——这是 PP 实现"通信隐藏"的关键

### 5.4 PyTorch 中的官方接口

PyTorch 2.4 起把 PP 的官方实现归到 `torch.distributed.pipelining`(原 PiPPy 项目并入主线):

```python
from torch.distributed.pipelining import pipeline, ScheduleGPipe, SplitPoint

# 1. 把模型切成 stage(自动追踪)
pipe = pipeline(
    module=model,
    mb_args=(example_input,),
    split_spec={"layers.8": SplitPoint.BEGINNING,
                "layers.16": SplitPoint.BEGINNING,
                "layers.24": SplitPoint.BEGINNING},  # 4 个 stage 的切点
)

# 2. 取出当前 rank 的 stage
stage = pipe.build_stage(stage_index=rank, device=device)

# 3. 用 GPipe 调度跑
schedule = ScheduleGPipe(stage, n_microbatches=8, loss_fn=loss_fn)

# 4. 训练
for batch in dataloader:
    if rank == 0:
        schedule.step(batch.input)
    elif rank == world_size - 1:
        losses = []
        schedule.step(target=batch.target, losses=losses)
    else:
        schedule.step()
    
    optimizer.step()
    optimizer.zero_grad()
```

PyTorch 同时提供了 `ScheduleGPipe / Schedule1F1B / ScheduleInterleaved1F1B` 等多个调度器——只要切完 stage,换一个 schedule 类就能切换算法,不用动模型代码。这是 2024 年起 PyTorch PP 主推的工作流。

DeepSpeed Pipeline 和 Megatron-LM 的接口风格不同,但抽象层级一致(切 stage + 选 schedule + 用 micro-batch),原理都是这一套。

### 5.5 PP × DP 的拓扑组合

实际生产里很少单独用 PP——通常和 DP 一起组合。把 N 张卡先按 PP 切成 P 份(每份是一个 stage),然后**每个 stage 内部再做 DP**:每个 stage 占 N/P 张卡,这 N/P 张卡上做数据并行,看不同的 micro-batch 子集。

实现上要建两个 process group:

- `pp_group`:同一个 DP rank 内、跨 stage 的 P2P send/recv 通信
- `dp_group`:同一个 stage 内、跨 DP rank 的梯度 AllReduce

举例:32 张卡,P=4 PP × 8 DP。所有 stage 0 的 8 张卡共享一个 dp_group(stage 0 内部做 DP),每张 stage 0 的卡和它对应的 stage 1/2/3 卡组成一条 pp_group 链。Megatron-LM 内部就是这么建 communicator 的。

这套组合已经能跑到几千卡规模。再叠上节点内 TP,就是 3D 并行的标准配方——`(节点内 TP) × (跨节点 PP) × (跨节点 DP)`,**通信量从大到小匹配带宽从大到小**。完整的 process group 设计、rank 到三维坐标 `(pp_rank, tp_rank, dp_rank)` 的映射,放到 Ch2 后面 `Topology3D.md` 里讲。

## 六、GPipe 的边界:为什么 1F1B 接班

把前面所有内容串起来,GPipe 的根本问题是:

**问题一:激活显存随 M 线性增长**

§4.1 推导过,stage 0 上的激活峰值正比于 M 个 micro-batch。M 越大 bubble 越小,但 M 越大 OOM 风险越高——这两个目标直接打架。

**问题二:反向必须等所有 forward 完成**

§3.2 调度图里,stage 0 在 t=4 之后就闲了 6 个时间单位,直到 t=10 才开始反向。这段时间它只是在"占着激活"——既不在算前向(因为 micro-batch 已经发完),也不在算反向(因为反向还没传到这里)。这是纯浪费。

1F1B(One Forward One Backward,Ch2 第二篇要讲)的核心就是**修这两个问题**:

- 让反向尽早开始(不等所有 forward 完),反向一开始就能释放对应的激活显存
- 在 steady 阶段严格交替 1F1B,**激活峰值只和 P 相关,不和 M 相关**

代价是调度复杂度上升、send/recv 的依赖关系更复杂。但有一个反直觉的事实必须提前点明:**1F1B 和 GPipe 的总 bubble 公式完全相同**——都是 $\frac{P-1}{M+P-1}$,因为 warmup + cooldown 的总时长只由 P-1 决定,和 forward / backward 是否交替无关。

**1F1B 的胜负手是"操作意义上"的**——bubble 公式只是一张地图,GPipe 拿着这张地图但被显存死死困在原地走不远(M 取不到 100、1000),1F1B 把显存的束缚解开,M 才能真正放大,公式那一项才能真正趋近 0。**算法的真正贡献往往是把不可达变可达,而不是把极限本身往下挪**。

所以工业上 GPipe 几乎没人用了,**Megatron-LM、DeepSpeed Pipeline 的默认调度都是 1F1B**,2023 年之后又出现了 Interleaved 1F1B、Zero Bubble Pipeline、DualPipe(DeepSeek V3)等更激进的方案。但所有这些算法的概念母版都是 GPipe——bubble 公式、激活账、micro-batch 调度顺序、AC 强绑定,都是从 GPipe 沿用下来的。把 GPipe 吃透,后面的演进只是在它的基础上做局部优化;不学透它,后面那些"省 bubble"、"省激活"的招式都无从理解。

---

**Zero Bubble 的种子已经埋好**:GPipe 和 1F1B 都把 forward 和 backward 当成"原子操作"。但 backward 其实可以再拆——

$$
\frac{\partial L}{\partial x} = W^T \cdot g_{\text{out}} \quad \text{(B,有链式依赖,必须串行往上游传)}
$$
$$
\frac{\partial L}{\partial W} = g_{\text{out}} \cdot x^T \quad \text{(W,只要 } g_{\text{out}} \text{ 和保存的 } x \text{ 都在,任意时刻都能算)}
$$

W 没有链式依赖,**可以塞进 bubble 时刻去算**——这就是 2023 年 Zero Bubble Pipeline 的核心招式。GPipe 和 1F1B 都把 B 和 W 绑成一个原子的 backward,所以 bubble 公式 $\frac{P-1}{M+P-1}$ 就是它们的下限;一旦把 B 和 W 拆开,这个下限就被打穿了。
