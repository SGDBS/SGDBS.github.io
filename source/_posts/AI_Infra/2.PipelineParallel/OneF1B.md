---
title: 2. 1F1B
categories: 学习笔记- AI Infra
date: 2026-05-11 14:00:00
mathjax: true
tags:
    - AI
    - AI Infra
---


GPipe 把朴素模型并行救活了,但它有一个工程上的死结:**bubble 想小就得加大 M,M 一大显存就炸**。这一篇讲的 **1F1B(One Forward One Backward)**是 PP 的工业标准——它的 bubble 公式和 GPipe 完全一样,但因为巧妙地把"在飞 micro-batch 数"从 M 降到 P,**让 M 可以独立加大,真正把 bubble 压下去**。Megatron-LM、DeepSpeed Pipeline 的默认调度都是它。

按 Ch2 第一篇 GPipe 的风格,这一篇也走"故事 → 推导 → 工程"的脉络。但会跳过 GPipe 已经讲过的基础(bubble 公式、micro-batch 是什么、AC 怎么配),专注在 **1F1B 比 GPipe 多做了什么、为什么这一招是物理下限**。

## 一、起因:GPipe 撞墙之后

GPipe 论文 2018 年底挂出来,第一波工业用户(主要是 Google、微软、百度)很快就上手了。微软研究院的一个团队(Aaron Harlap、Deepak Narayanan 等)在训机器翻译模型时,直接撞上了 GPipe 的硬天花板——

具体场景:训一个 20 亿参数的 Transformer 翻译模型,8 张 V100,每张 16 GB 显存。他们已经把 m 压到不能再小(单 micro-batch 的 GPU 利用率已经濒临崩溃),M 撑到 16 也只能让 bubble 降到 ~30%。再大就直接 OOM——GPipe 的 stage 0 同时压 16 个 micro-batch,活生生把显存撑爆。

他们的脑回路特别朴素:**反向传播为什么非要等所有 forward 都完成?**

回头看 GPipe 调度图(Ch2 第一篇 §3.2),stage 0 在做完第 M 个 forward 之后,会干等很久才开始反向。这段时间里,stage P-1 已经把 F0 算完了——B0 完全可以**立刻**启动。它需要的只是 stage P-1 自己刚算的输出和 loss,根本不需要 F1、F2、F3 跑完。

这个观察直接催生了 **PipeDream**(NeurIPS 2019,*PipeDream: Generalized Pipeline Parallelism for DNN Training*),核心招式就是 **One Forward One Backward(1F1B)**——做完一个 forward 就让 backward 尽早开始。

(PipeDream 原始论文里其实还有个**异步版本**,用一个叫 weight stashing 的 hack 对付"参数版本不一致"——这个 hack 后来被证明数值不稳定,几乎没人用。NVIDIA 在 Megatron-LM 里把它改成同步版,叫 PipeDream-Flush,简称 1F1B。生产里说"1F1B"默认指同步版。这一点在 §5.1 详细讲。)

## 二、推导:从约束反推 1F1B 的形状

GPipe §4.3 推导过一个通用公式:

$$
\text{stage 显存峰值} = (\text{在飞 micro-batch 数}) \times (\text{单 micro-batch 激活})
$$

GPipe 的在飞数是 M——我们的目标是把它降到 $O(P)$。这一节从约束推导 1F1B 必然长成的样子。

### 2.1 目标:在飞数从 M 降到 O(P)

为什么 $O(P)$ 而不是 $O(1)$?因为流水线本身的存在就要求"管道里同时有 P 个 stage 在干活",每个 stage 至少压着 1 个 micro-batch 的激活——一共至少 P 个。**P 是物理下限**,任何 PP 算法都不可能低于 P。

如果能做到任意 stage 上的在飞数都 ≤ P,那么 M 就和显存彻底解耦——M 加到 100、1000 都不影响显存。这是 1F1B 的设计目标。

### 2.2 从 stage P-1 的自然节奏开始

考虑最下游的 stage P-1。它有什么"自然节奏"?

- 收到 F0 的输入 → 算 F0 → 算 loss → **立刻就能算 B0**(它有 $g_{\text{out}}$,有刚算出的激活,B 的所有前置条件都满足)
- B0 算完后送梯度给 stage P-2,然后该干什么?
  - 选项 A:收 F1 的输入,算 F1
  - 选项 B:闲着(B0 已经送出去,没什么可等)

显然 A。继续推:

- 算 F1 → 立刻算 B1(同样的逻辑)
- B1 → F2 → B2 → F3 → B3 …

**stage P-1 的自然节奏就是严格交替 F-B-F-B-F-B…**

这个节奏下,stage P-1 在任意时刻最多压着 1 个 micro-batch 的激活——F 一算完立刻被 B 消费掉,显存峰值 = $1 \cdot A_{\text{per-micro}}$。

### 2.3 节奏沿流水线传播到所有 stage

现在看 stage P-2 收到的输入:

- 从 stage P-3 来的 forward 激活(按上游的发射节奏)
- 从 stage P-1 来的 backward 梯度(按 F-B 交替的节奏)

两边交替到达 → **stage P-2 自然落入 F-B 交替节奏**。

但有个细节:**上游 stage 必须先攒一些 forward,流水线才能填起来**。这就是 warmup 阶段。

stage k 的 warmup 长度 = "我必须先做几个 forward,才能让反向第一次回到我这里"——把这个数算清楚就是 **P - k 个 forward**:

- stage 0:warmup P 个 F(然后等 B0 从 stage P-1 一路传回来)
- stage 1:warmup P-1 个
- ...
- stage P-1:warmup 1 个(F0 一完成立刻 B0,几乎没 warmup)

## 三、1F1B 完整调度图

### 3.1 调度图(P=4, M=6)

```
Time:  0   1   2   3   4   5   6   7   8   9   10  11  12  13  14
S0:    F0  F1  F2  F3  ·   ·   ·   B0  F4  B1  F5  B2  B3  B4  B5
S1:    ·   F0  F1  F2  F3  ·   B0  F4  B1  F5  B2  B3  B4  B5  ·
S2:    ·   ·   F0  F1  F2  B0  F3  B1  F4  B2  F5  B3  B4  B5  ·
S3:    ·   ·   ·   F0  B0  F1  B1  F2  B2  F3  B3  F4  B4  F5  B5
       |←  warmup  →|←─── 1F1B 严格交替 ───→|← cooldown →|
```

(这是简化的理想调度示意——真实生产代码里 cooldown 段会有少量额外的等待气泡(因为 backward 也要按 stage 依次传回),但概念结构就是这样。完整时间分析见 §4.3。)

读图要点:

- **Warmup 期**:每个 stage 攒几个 F(stage 0 攒 P=4 个,stage P-1 只攒 1 个),上游长、下游短
- **Steady 1F1B 期**:每个 stage 严格 F-B-F-B 交替
- **Cooldown 期**:最后清空所有未完成的 backward

### 3.2 三阶段的工作量分布

| 阶段 | stage k 做的事 | 步数 |
|---|---|---|
| Warmup | 连续 P-k 个 F | P-k |
| Steady | F-B 严格交替的 F-B 对 | M-(P-k) |
| Cooldown | 剩余的 B | P-k |

合计每个 stage 都做 M 个 F + M 个 B,总工作量和 GPipe 一致。区别全在**顺序**上。

## 四、显存与 bubble 账本

### 4.1 stage 0 的在飞数逐拍追踪

对照 §3.1 的调度图,看 stage 0 的"持有激活集合"和计数:

| t | stage 0 操作 | 持有的激活 | 计数 |
|---|---|---|---|
| 0 | F0 | {F0} | 1 |
| 1 | F1 | {F0, F1} | 2 |
| 2 | F2 | {F0, F1, F2} | 3 |
| 3 | F3 | {F0, F1, F2, F3} | **4** ← warmup 峰值 |
| 4 | · 等待 B0 | {F0, F1, F2, F3} | 4 |
| 5 | · 等待 B0 | {F0, F1, F2, F3} | 4 |
| 6 | · 等待 B0 | {F0, F1, F2, F3} | 4 |
| 7 | B0 | {F1, F2, F3} | 3 ← 释放 F0 激活 |
| 8 | F4 | {F1, F2, F3, F4} | **4** ← steady 回到峰值 |
| 9 | B1 | {F2, F3, F4} | 3 |
| 10 | F5 | {F2, F3, F4, F5} | 4 |
| 11 | B2 | {F3, F4, F5} | 3 |
| 12 | B3 | {F4, F5} | 2 |
| 13 | B4 | {F5} | 1 |
| 14 | B5 | {} | 0 |

**峰值始终 = 4 = P,不管 M 怎么变**。M=6 是 4,M=60 也是 4,M=600 还是 4。

这就是 1F1B 的核心机制:进入 steady 期之后,**每做完一个 B 就立刻释放一份激活**,然后做一个 F 把空位填回去——F 和 B 在 stage 0 上严格 1:1 抵消,在飞数像被钉死在 P 一样。

### 4.2 为什么 P 是物理下限

P 这个数字不是设计选择,**是物理依赖逼出来的**。

- stage 0 在 t=0 发出 F0
- F0 沿流水线流到 stage P-1,需要 P-1 个时间单位,到达时刻 t = P-1
- stage P-1 在 t=P-1 完成 F0 后**立刻**启动 B0(这是 1F1B 的核心招式)
- B0 从 stage P-1 反向流回 stage 0,又需要 P-1 个时间单位,到达 stage 0 时刻约 t = 2P-1

在 B0 回到 stage 0 之前的这 2P-1 个时间单位里,stage 0 没有任何方式释放激活——它只能塞下"这段时间能做的 F 数",刚好就是 **P 个**(每个时间单位一个 F)。

**P 就是这个时间窗口里 stage 0 能容纳的 F 数量上限,也是激活在飞的物理下限**。1F1B 把它榨到了下限,不能再压。

### 4.3 bubble 公式与 GPipe 完全相同

数一下 §3.1 调度图里 stage 0 的空闲格子——t=4, 5, 6 是 warmup 末尾(3 = P-1 格),等 B0 从最下游绕一圈回来。把整个 timeline 加起来:

- 总时间 $= 2(M+P-1) T$
- bubble $= 2(P-1) T$
- **bubble ratio $= \dfrac{P-1}{M+P-1}$** ← 和 GPipe 完全一样

为什么一样?因为 warmup + cooldown 的总时长由 P-1 决定(forward 要流过 P 级,backward 要流回 P 级),这是**物理依赖**,改不了。1F1B 只是把 F 和 B 的相对顺序重新排,**没有改变流过的总时长**。

### 4.4 真正的胜负手:操作意义上的胜利

bubble 公式和 GPipe 一样,但 1F1B 仍然是工业标准——胜负手是**操作意义上**的:

| | bubble 公式 | M 的实际可达上限 | 实际能达到的 bubble |
|---|---|---|---|
| GPipe | $\frac{P-1}{M+P-1}$ | ~4P(被显存限死) | ~20-30% |
| 1F1B | $\frac{P-1}{M+P-1}$ | ~100P+(只受 dataloader 速度限) | < 1% |

两个算法的 bubble 公式像是同一张地图——GPipe 拿着这张地图,但被显存死死困在原地走不远;1F1B 把显存的束缚解开,M 才能真正放大,公式那一项才能真正趋近 0。

工程系统优化里这种"公式没改,可达性变了"的胜利很常见——很多算法的真正贡献是把不可达变可达,而不是把极限本身往下挪。

### 4.5 1F1B vs GPipe 总览

| | GPipe | 1F1B |
|---|---|---|
| bubble 公式 | $\frac{P-1}{M+P-1}$ | $\frac{P-1}{M+P-1}$ |
| 在飞数(stage 0) | M | **P** |
| 显存峰值(stage 0) | $M \cdot m \cdot A_{\text{sample}}$ | $P \cdot m \cdot A_{\text{sample}}$ |
| 想加大 M(降 bubble) | 显存爆炸 | **显存不变,随便加** |
| 调度复杂度 | 简单(全 F + 全 B) | 复杂(warmup + 交替 + cooldown) |
| 必须配 AC | 是 | 仍然推荐 |
| 数学语义 | 严格 = batch size B 的 SGD | 严格 = batch size B 的 SGD(同步版) |

## 五、工程实现

### 5.1 同步还是异步:PipeDream-async vs PipeDream-Flush

最原始的 PipeDream(2019)是**异步**的——每个 micro-batch 用当时 stage 上的参数 forward,反向时再用反向那一刻的参数算梯度。问题是 **forward 和 backward 用的参数版本可能不一致**:

```
t=0: F0 用参数 W₀ 算激活
t=1: optimizer step → W₀ 变成 W₁
t=2: B0 用参数 W₁ 算梯度   ← 不一致!
```

PipeDream 的 hack 叫 **weight stashing**——每次 forward 时把当时的参数 $W_0$ 存下来,反向时用 stashed $W_0$。代价是显存翻倍存参数副本,且**梯度仍然是有偏的**(stash 期间损失了更新信息),收敛性比同步差。

**PipeDream-Flush**(NVIDIA Megatron 改造版)放弃了异步,改成严格同步——所有 M 个 micro-batch 用同一份参数 forward 和 backward,M 个梯度累加后才 step。这和 GPipe 的语义完全一致(等价于 batch size $M \cdot m$ 的 SGD),只是调度变成 1F1B。

**生产里"1F1B"默认指 PipeDream-Flush 同步版**。后面所有讨论都是这个版本。

### 5.2 代码结构:warmup + steady + cooldown

伪代码,只看 stage k 的视角:

```python
def one_f_one_b_step(stage_module, M, P, k, optimizer):
    """1F1B schedule for one rank (stage k)"""
    activations = {}        # 存还没反向的 micro-batch 激活
    n_warmup = P - k        # warmup 期的 forward 数

    # ===== Warmup: 连续 P-k 个 forward,不做 backward =====
    for i in range(n_warmup):
        x = recv_from_prev() if k > 0 else load_micro_batch(i)
        out = stage_module.forward(x)
        activations[i] = (x, out)
        if k < P - 1:
            send_to_next(out)

    # ===== Steady: F-B 严格交替 =====
    for i in range(n_warmup, M):
        # 先做 forward
        x = recv_from_prev() if k > 0 else load_micro_batch(i)
        out = stage_module.forward(x)
        activations[i] = (x, out)
        if k < P - 1:
            send_to_next(out)

        # 再做对应的 backward(从最早未反向的 micro-batch 开始)
        bw_id = i - n_warmup
        x_saved, out_saved = activations.pop(bw_id)
        if k == P - 1:
            grad_out = compute_loss_grad(out_saved) / M  # 注意除以 M
        else:
            grad_out = recv_from_next()
        grad_in = stage_module.backward(x_saved, out_saved, grad_out)
        if k > 0:
            send_to_prev(grad_in)

    # ===== Cooldown: 把剩下的 backward 做完 =====
    for bw_id in sorted(activations.keys()):
        x_saved, out_saved = activations.pop(bw_id)
        if k == P - 1:
            grad_out = compute_loss_grad(out_saved) / M
        else:
            grad_out = recv_from_next()
        grad_in = stage_module.backward(x_saved, out_saved, grad_out)
        if k > 0:
            send_to_prev(grad_in)

    # ===== 累加完所有梯度后才 step =====
    optimizer.step()
    optimizer.zero_grad()
```

注意几点:

- **三阶段在代码里清晰分开**——边界明确,每段做的事不一样
- **`activations` 字典**保存"已 forward 但未 backward"的激活,字典大小就是在飞数,**最多 P**
- **`loss / M`**:和 Ch1 梯度累积 + GPipe 一样的除法,保持数学等价

### 5.3 send/recv 依赖比 GPipe 复杂得多

GPipe 的通信是"所有 F 一波,所有 B 一波",依赖图简单。1F1B 的通信是 F 和 B 交叉的,**依赖图复杂得多**:

```
S0 → S1: F0, F1, F2, F3, F4, F5             (forward 激活)
S1 → S0: B0, B1, B2, B3, B4, B5             (backward 梯度)
两条流交错且严格有时序约束
```

工程上必须用**异步通信** (`dist.isend / dist.irecv`) + **CUDA stream 协作**,让 send/recv 和计算重叠。最常见的两个坑:

1. **死锁**:如果两端都 `irecv` 而没人 `isend`,直接卡死。GPipe 的天然 phase 分隔避免了这个问题;1F1B 必须仔细排发送顺序——通常的做法是 "先 send 再 recv",或者用 `batch_isend_irecv` 把成对的操作打包
2. **buffer 复用**:同一个张量 shape 反复 send/recv,buffer 池要管好,否则反复 `cudaMalloc` 拖慢

NCCL 2.7+ 原生支持 `ncclSend / ncclRecv`,这是 1F1B 通信的底座。

### 5.4 PyTorch / Megatron 中的接口

PyTorch 2.4+ 的 `torch.distributed.pipelining` 直接提供 1F1B 调度:

```python
from torch.distributed.pipelining import pipeline, Schedule1F1B, SplitPoint

pipe = pipeline(
    module=model,
    mb_args=(example_input,),
    split_spec={"layers.8":  SplitPoint.BEGINNING,
                "layers.16": SplitPoint.BEGINNING,
                "layers.24": SplitPoint.BEGINNING},
)
stage = pipe.build_stage(stage_index=rank, device=device)

# 唯一变化:把 ScheduleGPipe 换成 Schedule1F1B
schedule = Schedule1F1B(stage, n_microbatches=64, loss_fn=loss_fn)
```

唯一的代码变化就是 `ScheduleGPipe → Schedule1F1B`——调度算法换了,模型代码、切层、send/recv 全不动。

Megatron-LM 通过 `--pipeline-model-parallel-size P` 和 `--num-microbatches M` 开 1F1B,默认就是 1F1B(没有 GPipe 选项)。DeepSpeed Pipeline 类似。生产里你**几乎找不到不用 1F1B 的 PP 训练**。

## 六、1F1B 的边界:还能继续压吗?

1F1B 已经把 PP 的两个核心指标都推到了"看起来不能再优"的位置:

- **bubble**:$\frac{P-1}{M+P-1}$,M 大就趋近 0
- **显存**:在飞数 = P,达到物理下限

但仍有两个进一步优化的方向,对应后面两篇笔记:

### 6.1 单 stage 粒度太粗 → Interleaved 1F1B

bubble 公式里的 P-1 是 "warmup + cooldown 的总长度",由 stage 数决定。能不能把每个 stage 切得更细?Megatron 2021 年的 Interleaved 1F1B 把单 stage 切成 V 个 "virtual stage"(每个 virtual stage 持有 L/(PV) 层,在 PP 维度上交错排布),bubble 进一步压到:

$$
\text{bubble} = \frac{P-1}{V \cdot M + P - 1}
$$

V = 2 时分母翻倍,bubble 直接减半。代价是通信量增多(同样的激活要跨 stage 多次发送)。

### 6.2 B 和 W 还没拆 → Zero Bubble Pipeline

回到 GPipe 篇 §6 末尾埋的种子:**backward 其实可以再拆**——

$$
\frac{\partial L}{\partial x} = W^T \cdot g_{\text{out}} \quad \text{(B,有链式依赖,必须串行往上游传)}
$$
$$
\frac{\partial L}{\partial W} = g_{\text{out}} \cdot x^T \quad \text{(W,没有链式依赖,任意时刻都能算)}
$$

1F1B 把 B 和 W 绑成一个原子的 backward 操作,所以 bubble 公式 $\frac{P-1}{M+P-1}$ 就是它的下限。**2023 年的 Zero Bubble Pipeline** 把 B 和 W 拆开——W 没有链式依赖,可以**塞进 bubble 时刻**(warmup 末尾的空闲、cooldown 末尾的空闲)去算。bubble 从此被 W 占满,理论上可以做到接近 0。

DeepSeek V3 的 DualPipe 走得更远——双向流水线 + 计算通信重叠,把跨节点 AllReduce 也藏进 bubble。这些都是 1F1B 之上的进化分支。

---

把 GPipe 和 1F1B 两章串起来看,概念地图大致是:

- **GPipe**:把朴素 model parallel 救活,但 bubble 和显存通过 B 耦合,M 加不上去
- **1F1B**:把"在飞数"从 M 降到 P,**让 M 和显存解耦,bubble 才真正可压**
- **Interleaved 1F1B / Zero Bubble**:继续往两个方向压的进化(切细 stage 维度 / 拆 backward)

下一篇讲 Interleaved 1F1B。
