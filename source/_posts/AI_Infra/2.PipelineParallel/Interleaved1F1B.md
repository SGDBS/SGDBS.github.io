---
title: 3. Interleaved 1F1B
categories: 学习笔记- AI Infra
date: 2026-05-10 14:00:00
mathjax: true
tags:
    - AI
    - AI Infra
---


## 1. 故事线:为什么会出现 interleaved

### 1.1 前一代留下的问题

Gpipe 解决了"pipeline 怎么并行"的问题,但激活内存爆炸——in-flight 的 micro-batch 数 = $M$。

1F1B(One Forward One Backward)解决了显存:forward 一个就尽快 backward 一个,in-flight 激活数从 $M$ 降到 $\approx P$。

但 1F1B 没动 bubble。bubble ratio(以 ideal compute 为分母,Megatron 论文风格):

$$\text{bubble ratio}_{\text{1F1B}} = \frac{P-1}{M}$$

(以总 step 为分母的版本是 $\frac{P-1}{M+P-1}$,两者等价,只是基准不同。)

### 1.2 为什么不能继续靠增大 $M$ 来减 bubble

$M = \dfrac{\text{global\_bs}}{\text{DP\_size} \cdot \text{micro\_bs}}$,想增大 $M$ 只有三条路,全部堵死:

| 路径 | 为什么堵死 |
|---|---|
| 减小 DP size | 大规模训练 $N = \text{DP} \cdot \text{TP} \cdot \text{PP}$,TP 被单机 NVLink 卡在 $\leq 8$,PP 增大反而恶化 bubble,DP 只能往上。 |
| 减小 micro batch size | Tensor Core 有最小高效 GEMM shape;太小退化成 memory-bound,kernel launch overhead 占比上升。 |
| 增大 global batch size | 被 scaling law / critical batch size 锁死(超过临界值泛化变差),GPT-3 的 3.2M tokens 已是仔细调过的上限。 |

**结论:$M$ 在大规模训练中是被算法/拓扑/硬件三方共同锁死的常数,工程上不能当自由变量。** 只能去攻击公式的另一端——分子里的 $P-1$ 也动不了(因为 $P$ 由模型大小决定),那就只能往公式里**塞一个新的自由变量**。

这就是 Megatron-LM(Narayanan et al. 2021)的 interleaved 出现的根本动机。

---

## 2. 从约束反推:为什么 interleaved 必须长成这个形状

### 2.1 核心操作:把每个 device 的连续层切成 $v$ 个不连续 chunk

朴素 PP(4 stage,16 层):

```
Device 0: Layer 0-3
Device 1: Layer 4-7
Device 2: Layer 8-11
Device 3: Layer 12-15
```

Interleaved($v=2$):

```
Device 0: Layer 0-1,  Layer 8-9
Device 1: Layer 2-3,  Layer 10-11
Device 2: Layer 4-5,  Layer 12-13
Device 3: Layer 6-7,  Layer 14-15
```

整个系统等效成一个 $P \cdot v$ 长的 **virtual pipeline**,virtual stage round-robin 分配到 device 上。一个 micro-batch 走完整个网络要在 device 间穿梭 $v$ 次。

### 2.2 bubble 公式的推导

bubble 的本质 = pipeline 启动/排空时间 = $(P-1) \times$ 单个 chunk 的执行时间。

chunk 变小 $v$ 倍 → 启动/排空时间变小 $v$ 倍 → bubble 时间 $\div v$。

稳态计算量不变(总层数 × 总 micro-batch 不变)。所以:

$$\boxed{\text{bubble ratio}_{\text{interleaved}} = \frac{1}{v} \cdot \frac{P-1}{M}}$$

### 2.3 schedule 时序对照(关键画面)

朴素 1F1B 每个 F/B 占 $v$ 个时间单位,interleaved 每个 chunk 占 1 个单位:

```
朴素 1F1B (P=4, v=2 等效):
D0: [F1   ][F2   ][F3   ][F4   ][B1   ]...
D1:     [F1   ][F2   ][F3   ][F4   ][B1   ]...
D2:         [F1   ][F2   ][F3   ][B1   ]...
D3:             [F1   ][B1   ][F2   ][B2   ]...
    └─── warmup = (P-1)·v ───┘

Interleaved (P=4, v=2):
D0: F1.0 F2.0 F3.0 F4.0 F1.4 F2.4 F3.4 F4.4 B1.4 F5.0 ...
D1:      F1.1 F2.1 F3.1 F4.1 F1.5 F2.5 F3.5 F4.5 B1.5 ...
D2:           F1.2 F2.2 F3.2 F4.2 F1.6 F2.6 F3.6 F4.6 ...
D3:                F1.3 F2.3 F3.3 F4.3 F1.7 B1.7 F2.7 ...
    └── warmup = (P-1) ──┘   ↑ 注意:micro-batch 绕回 D0
```

直观:**chunk 更小,pipeline 填满更快,启动 bubble 砍 $v$ 倍**。代价是 activation 要"绕回头"传输,通信次数 $\times v$。

---

## 3. $v$ 的三个约束(为什么实际取 $v \in \{2, 3, 4\}$)

bubble 公式看似 $v$ 越大越好,但实际很少超过 8。三个约束把 $v$ 压住:

### 3.1 通信约束

每个 micro-batch 在 device 间穿梭 $v$ 次,P2P 通信次数从 $P-1$ 变成 $v(P-1)$。每次通信量(activation tensor)不变,但 chunk 计算时间 $\div v$,**通信/计算比恶化 $v$ 倍**。

P2P 能和计算 overlap,前提是 chunk 计算时间 $\geq$ 通信时间。$v$ 太大,overlap 失效,通信暴露。

- NVLink 内:$v$ 可拉到 4–8
- 跨节点(IB):$v=2$ 就到头

### 3.2 激活内存约束(关键且反直觉)

**naive 直觉**:每个 chunk 内存 $\div v$,份数 $\times v$,应该抵消 → 内存不变。

**实际公式**:

$$\text{激活内存}_{\text{interleaved}} = P \cdot \frac{v+1}{2} \times \text{(1F1B 单份激活内存)}$$

| $v$ | 内存倍数 vs 1F1B |
|---|---|
| 1 | 1.0× (退化为 1F1B) |
| 2 | 1.5× |
| 4 | 2.5× |
| 8 | 4.5× |

**为什么不抵消——virtual pipeline 位置的不对称性**:

1F1B 中,device 0 持有 $P$ 个激活,device $P-1$ 持有 1 个(头部多尾部少的阶梯)。

Interleaved 中,一个物理 device 持有 $v$ 个 chunk,每个 chunk 在 virtual pipeline 中位置不同:

- 最头部 chunk:持有 $Pv$ 个激活
- 次头部 chunk:持有 $P(v-1)$ 个激活
- ...

一个 device 把这 $v$ 个 chunk 的激活份数全加起来:

$$\sum_{i=0}^{v-1}(Pv - iP) = P \cdot \frac{v(v+1)}{2}$$

每份大小 $1/v$,总内存 $= P \cdot \frac{v+1}{2}$。

**记忆点:份数 × 大小本应抵消,但因为 $v$ 个 chunk 在 virtual pipeline 中位置不对称(头部持有的激活份数远多于尾部),加和后留下了 $(v+1)/2$ 倍净增长。**

类比:4 个收银台,interleaved 让每个台管 2 个虚拟队列。台 0 同时管"长度 8 的队"和"长度 4 的队",负担是 8+4=12,不是单看最长的那个。

### 3.3 整除约束

每个 device 持有 $L/(P \cdot v)$ 层,必须为整数。

- GPT-3 175B(96 层),$P=8$:$v \in \{1, 2, 3, 4, 6, 8, 12\}$
- LLaMA-3 70B(80 层),$P=8$:$v \in \{1, 2, 5\}$,$v=4$ 直接不行

(也是为什么 transformer 层数常取 12/24/32/48/64/80/96 这种高分解数——配合 PP 调度。)

---

## 4. 实际工程选 $v$ 的逻辑

不是"bubble 最小",而是 **"内存 budget 允许下尽量大"**:

```
v ↑  →  bubble ↓                  (好)
v ↑  →  P2P 通信/计算比恶化       (overlap 失效风险)
v ↑  →  激活内存 × (v+1)/2        (OOM 风险)
v 必须整除 L/P                      (离散选择)
```

典型搭配:

- $v$ 选 2–4
- 配 selective activation recomputation 抵消内存增长
- TP 留在单机内($\leq 8$),PP 跨机,$v$ 在跨机场景取小

**规模越大,收益越显著**:小规模 $M$ 还撑得住 bubble 不痛;一旦上千卡 $M$ 被压到个位数,interleaved 就从"优化"变成"必需"。

---

## 5. 核心公式 cheat sheet

| 量 | 1F1B | Interleaved($v$) |
|---|---|---|
| bubble ratio(/ideal) | $\dfrac{P-1}{M}$ | $\dfrac{1}{v} \cdot \dfrac{P-1}{M}$ |
| P2P 通信次数(per micro-batch) | $P-1$ | $v(P-1)$ |
| 激活内存(以 1F1B 单份为单位) | $P$ | $P \cdot \dfrac{v+1}{2}$ |
| 每个 chunk 层数 | $L/P$ | $L/(Pv)$ |

---

## 6. 一句话总结

> **$M$ 在大规模训练中被锁死,bubble 公式 $\frac{P-1}{M}$ 没法从分母动手,interleaved 引入一个纯调度层面的新自由变量 $v$,把 pipeline 在调度上虚拟拉长 $v$ 倍、每个 chunk 缩薄 $v$ 倍,启动/排空 bubble 因此 $\div v$。代价是通信次数 $\times v$ 和激活内存 $\times \frac{v+1}{2}$,实际工程取 $v \in \{2,3,4\}$。**