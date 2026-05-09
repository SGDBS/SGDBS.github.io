---
title: 4. 混合精度训练
categories: 学习笔记- AI Infra
date: 2026-05-09 11:00:00
mathjax: true
tags:
    - AI
    - AI Infra
---


这一篇是后续 DDP / ZeRO / Pipeline 章节都会反复用到的"基石设定"。所有大模型训练都默认开混合精度——但**到底什么是混合,混到什么程度,为什么不能直接全用 FP16,为什么 BF16 来了之后反而更简单了**——很多人只知道"加个 `autocast` 就行"。这一篇把每个细节都拆开讲清楚:从浮点数的位级布局,到 loss scaling 的数学,到 `autocast` 的 op 级派发机制,再到混合精度怎么和 DDP/FSDP 协作。最后看一眼 H100/Hopper 引入的 FP8 训练。

## 一、起因:为什么需要混合精度

### 1.1 FP32 训练的两笔账

深度学习十年前几乎全部用 **FP32**(每个数 4 字节)。在小模型时代这没问题——ResNet-50 参数 25M,训练时显存压力主要在数据。但到了 LLM 时代,FP32 同时撞上两面墙:

**显存墙**。前面 `MemoryBudget` 那篇里算过:7B 模型 + Adam,显存账是参数 + 梯度 + (master + m + v) ≈ $16P$ 字节。如果全用 FP32,**这个 16 要变成 28**——参数 4、梯度 4、master 4、m 4、v 4、再加上 4 字节的 FP32 激活,7B 模型直接 200 GB 起步。

**算力墙**。Tensor Core 这种专用矩阵乘单元,从 V100 起就为半精度优化:V100 的 FP32 算力 15.7 TFLOPS,FP16 算力 125 TFLOPS——**相差 8 倍**。A100 上 BF16/FP16 是 312 TFLOPS、FP32 只有 19.5 TFLOPS。H100 上更夸张,BF16 是 989 TFLOPS、FP32 67 TFLOPS。如果不用半精度,你买的 GPU 算力一大半在闲着。

合起来一句话:**全用 FP32 训练,既塞不下也跑不快**。

### 1.2 半精度的麻烦

半精度的好处显而易见:存得下、算得快。但直接把 FP32 换成 FP16 时,Google/Baidu 在 2017 年前后踩了大量坑——loss 训着训着就 NaN、梯度突然全为 0、或者训练曲线在某个 epoch 突然崩盘。问题的根源不是 bug,而是**半精度的数值范围根本不够装训练时见到的数**。

这就是混合精度——准确说叫 **Automatic Mixed Precision (AMP)**——存在的理由:**用半精度去算耗时大头(矩阵乘),用单精度去算敏感的部分(loss、reduction、参数更新)**,再用一些数值技巧(loss scaling、master weights)把半精度的数值范围问题修补回来。

## 二、浮点格式的位级真相

要理解混合精度为什么"修补"得动,必须先看清半精度到底有多窄。一个浮点数由三部分组成:

$$x = (-1)^{\text{sign}} \cdot 2^{\text{exp} - \text{bias}} \cdot (1.\text{mantissa})_2$$

**sign**(符号位)总是 1 位。**exp**(指数位)决定**动态范围**——能表示多大或多小的数。**mantissa**(尾数位)决定**精度**——两个相邻可表示数之间能有多近。

### 2.1 四种格式的字节布局

```
                     ┌─────────────────────────────────────────────┐
FP32   (4 字节)      │ S │ EEEEEEEE │ MMMMMMMMMMMMMMMMMMMMMMM │     │
                     │ 1 │   8      │           23             │     │
                     └─────────────────────────────────────────────┘
                       ↑      ↑                ↑
                     符号   指数(广)        尾数(高精度)

                     ┌─────────────────────┐
BF16   (2 字节)      │ S │ EEEEEEEE │ MMMMMMM │
                     │ 1 │   8      │   7     │
                     └─────────────────────┘
                       ↑     ↑          ↑
                     符号  和 FP32 一样的 8 位指数
                                      尾数被砍到 7 位
                                      → 范围广,精度差

                     ┌─────────────────────┐
FP16   (2 字节)      │ S │ EEEEE │ MMMMMMMMMM │
                     │ 1 │  5    │     10     │
                     └─────────────────────┘
                       ↑     ↑          ↑
                     符号  指数砍到 5 位
                                      尾数 10 位,比 BF16 准
                                      → 精度好,范围窄

                     ┌─────────────┐
FP8 E4M3 (1 字节)    │ S │ EEEE │ MMM │      ← 训练前向用,精度优先
                     │ 1 │  4   │ 3   │
                     └─────────────┘

                     ┌─────────────┐
FP8 E5M2 (1 字节)    │ S │ EEEEE │ MM │      ← 训练反向用,范围优先
                     │ 1 │  5    │ 2  │
                     └─────────────┘
```

**关键观察**:BF16 的指数位**和 FP32 完全一样多**(8 位),所以 BF16 能表示的最大/最小数和 FP32 一致;它只是把尾数砍掉了。FP16 反过来,牺牲指数换精度。

### 2.2 数值范围对比

| 格式 | 最大正常数 | 最小正常数 | 相对精度(eps) |
| --- | --- | --- | --- |
| FP32 | $\sim 3.4 \times 10^{38}$ | $\sim 1.2 \times 10^{-38}$ | $\sim 1.2 \times 10^{-7}$ |
| BF16 | $\sim 3.4 \times 10^{38}$ | $\sim 1.2 \times 10^{-38}$ | $\sim 7.8 \times 10^{-3}$ |
| FP16 | $65504$ | $\sim 6.1 \times 10^{-5}$ | $\sim 9.8 \times 10^{-4}$ |
| FP8 E4M3 | $448$ | $\sim 1.95 \times 10^{-3}$ | $\sim 0.125$ |
| FP8 E5M2 | $57344$ | $\sim 1.5 \times 10^{-5}$ | $\sim 0.25$ |

注意 FP16 那一行:**最大数只有 65504**。训练 Transformer 时,某些梯度很容易超过这个值;某些梯度又会小于 $6 \times 10^{-5}$ 直接 underflow 到 0。这就是 FP16 训练不稳的根源。

而 BF16 的范围和 FP32 一模一样——它只是精度差了大概一个数量级,但深度学习对相对精度其实不敏感(梯度 noise 本来就比这大),所以 BF16 训练**几乎不需要任何额外技巧**就能跑稳。

### 2.3 为什么 BF16 是大模型训练的事实标准

把上面的事实串起来:

- FP16 范围窄,**梯度容易 underflow / overflow**,需要配 loss scaling、需要 GradScaler 动态调整,工程链条复杂
- BF16 范围 = FP32 范围,**直接换上去就跑**,不需要 loss scaling、不需要 GradScaler
- 现代 GPU(A100 起)BF16 和 FP16 的算力相同,**没有性能差**

所以 2022 年之后训练 LLM 几乎没人用 FP16 了:GPT-3、PaLM、Llama 全系列、Mistral、DeepSeek-V3 都是 BF16 训练。FP16 主要剩在**老 GPU(V100 没有原生 BF16 支持)** 和**推理**(推理对范围要求低、对精度要求高)。

## 三、Loss Scaling:FP16 训练的"数值油门"

虽然 LLM 用 BF16,但理解 loss scaling 仍然重要——它是历史上"半精度训练"得以工作的核心技巧,Vision/CV 大量代码到现在还在用 FP16 + GradScaler。

### 3.1 问题:梯度 underflow

考虑一个 ReLU 网络的反向传播,某层激活的导数 $\frac{\partial L}{\partial x}$ 在训练后期通常落在 $10^{-5}$ ~ $10^{-7}$ 量级——对 FP32 完全无压力,但**已经在 FP16 的可表示下界附近**。再乘上 weight 的反向(通常更小一两个数量级),很多梯度会直接被舍入到 0,等价于这个参数永远不更新。

直方图大概长这样:

```
FP32 梯度的分布(对数尺度):
            count
              │
              │      ████████
              │   █████████████
              │ ███████████████████
              │ ███████████████████████
              │ █████████████████████████████████
              ├─────────────────────────────────────► |grad|
              10⁻¹⁰  10⁻⁸   10⁻⁶   10⁻⁴   10⁻²   10⁰

FP16 表示能力的下界 ≈ 6×10⁻⁵
              ▲ 这一点以下的梯度全部丢失 = underflow
```

底下被吃掉一大坨——大概率包含了**对训练真正重要的小梯度信号**。

### 3.2 解法:把 loss 整体放大 K 倍

数学上,梯度对 loss 是线性的:

$$\nabla_\theta (K \cdot L) = K \cdot \nabla_\theta L$$

如果在反向传播开始之前把 loss 乘以一个大的常数 $K$,整个梯度图都被同步放大 $K$ 倍。原本落在 $10^{-7}$ 的梯度,放大 $K = 2^{16} = 65536$ 倍后变成 $\sim 10^{-2}$,稳稳地落在 FP16 范围内。

更新参数之前,再把累加好的梯度除以 $K$ 还原回真实尺度,在 FP32 master weights 上做更新。整个流程:

```
forward (FP16/BF16) ──► loss (FP32) ──► loss × K ──► backward (FP16) 
                                                          │
                                                          ▼
                                                   gradient (FP16, 被放大 K 倍)
                                                          │
                                                   unscale: g /= K  (转 FP32)
                                                          │
                                                   ┌──────┴──────┐
                                                   ▼             ▼
                                              clip_grad_norm   是否 inf/nan?
                                                   │             │
                                                   ▼            是 → 跳过这步
                                              optimizer.step
                                              (在 FP32 master 上)
```

### 3.3 静态 vs 动态 loss scaling

最简单的方案是**静态**——固定 $K = 2^{15}$ 或 $2^{16}$。但训练后期梯度可能变大,固定的 $K$ 又会让梯度 overflow。

**动态 loss scaling**(PyTorch 的 `GradScaler` 用的方法):

- 初始 $K = 2^{16}$
- 每步反向后检查梯度是否有 inf/nan
  - 没有:这步正常更新;每 $N$ 步(默认 2000)把 $K$ 翻倍 → 适应可能变小的梯度
  - 有:**跳过这步参数更新**(梯度被污染了);把 $K$ 减半 → 适应可能变大的梯度

这个机制就像汽车的自动变速箱——根据路况(梯度大小)自动调挡(scale 大小),代价是偶尔丢一两步训练。实际训练里 skip 比例通常 < 0.1%,可以忽略。

### 3.4 PyTorch 标准写法

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in loader:
    optimizer.zero_grad()

    # 1. forward 在 autocast 区域内,自动用 FP16 跑矩阵乘等
    with autocast(dtype=torch.float16):
        pred = model(x)
        loss = criterion(pred, y)

    # 2. backward 之前先把 loss 放大
    scaler.scale(loss).backward()

    # 3. 梯度裁剪之前必须先 unscale,否则裁的是放大版的梯度
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # 4. step 内部:检查 inf/nan → 决定是否 skip;调整 scale
    scaler.step(optimizer)
    scaler.update()
```

四个 GradScaler 调用的语义:

| 调用 | 干啥 |
| --- | --- |
| `scaler.scale(loss)` | 返回 loss × K,backward 后 .grad 也是放大的 |
| `scaler.unscale_(optimizer)` | 把 optimizer 管的所有参数的 .grad 除以 K,转回真实尺度 |
| `scaler.step(optimizer)` | 检查 inf/nan;无则 step,有则跳过 |
| `scaler.update()` | 根据这一步的结果调整 K(翻倍或减半) |

### 3.5 BF16 不需要这套吗?

不需要。BF16 的范围 = FP32,梯度不会 underflow / overflow,所以 loss scaling 完全没意义。BF16 训练的代码就纯粹:

```python
with autocast(dtype=torch.bfloat16):
    pred = model(x)
    loss = criterion(pred, y)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

少了 GradScaler 这一整套,代码干净得多。这是 LLM 全面切到 BF16 的另一个工程理由——**调试链路短**。

## 四、autocast:op 级 dtype 派发

### 4.1 autocast 在干什么

`torch.amp.autocast` 是一个**上下文管理器**,它不改变张量的存储 dtype,而是**在 PyTorch 的 op dispatcher 里插入一层**——每次调用一个 op 时,先看这个 op 在 autocast 名单里属于哪一档,然后把输入 cast 成对应 dtype 再算。

PyTorch 把 op 分成三类:

- **白名单(用半精度跑)**:矩阵乘、卷积、各种线性层——计算量大、对精度不敏感
- **黑名单(强制 FP32)**:`softmax`、`log`、`exp`、`pow`、loss、归一化、reduction——数值敏感
- **未列入(随输入)**:大多数 elementwise op,跟着输入的 dtype 走

```python
with autocast(dtype=torch.bfloat16):
    a = torch.randn(1024, 1024, device="cuda")  # FP32
    b = torch.randn(1024, 1024, device="cuda")  # FP32

    c = a @ b           # 白名单 → 输入会被自动 cast 成 BF16,输出 BF16
    d = c.softmax(-1)   # 黑名单 → 输入被 cast 回 FP32,输出 FP32
    e = d * 2.0         # 未列入 → 跟 d,FP32
```

这就是"混合"两个字的来源:**同一段代码里不同 op 跑在不同精度上**,而且对用户透明——你只管写正常 PyTorch 代码,autocast 帮你决定每步用什么 dtype。

### 4.2 模型权重的 dtype 与 autocast 的关系

这里是初学者最容易混的地方。**模型参数本身仍然是 FP32**——你 `model = MyModel().cuda()` 时它默认是 FP32。autocast 只是**在 op 调用时临时把输入和权重一起 cast 成 BF16/FP16** 算一遍,算完不会改变权重原本的存储 dtype。

```
model parameters (FP32)            ← 永远以 FP32 存
       │
       ├── 进入 autocast 区域
       │      │
       │      ▼  (临时 cast 成 BF16)
       │   matmul / conv ──► 输出 BF16
       │      │
       │      ▼  (黑名单 op 把它 cast 回 FP32)
       │   softmax / log ──► 输出 FP32
       │
       └── 离开 autocast 区域,后续操作维持 FP32
```

为什么参数要保持 FP32?**这就是所谓的 master weights**。优化器更新 $\theta \leftarrow \theta - \eta g$ 中,$\eta$ 通常 $10^{-4}$,$g$ 量级 $10^{-3}$,乘起来 $10^{-7}$——在 BF16 里(精度只有 $10^{-3}$)这个更新会被直接舍入掉,等价于学习率为 0。所以**最终的参数加法必须在 FP32 上做**。

这也是为什么 ZeRO/FSDP 的 `MixedPrecision` 配置里有三个独立的 dtype——参数计算用什么 / 梯度通信用什么 / 缓冲区用什么——但**优化器内部的 master 权重永远 FP32**(除非你显式开 `optim_in_bf16` 那种激进选项)。

### 4.3 一个常踩的坑:custom kernel 不会被 autocast 转

PyTorch 内置的 op 都注册了 autocast 行为,但**你自己写的 CUDA kernel 不会**。如果训练里用了一个第三方库(比如 xformers、Apex 的某个 fused op),它在 autocast 区域里**可能不会自动转 dtype**——你给它 FP32 输入它就老老实实 FP32 算,看起来 autocast"没生效"。

排查方法:在 autocast 区域里 `print(x.dtype)`,如果矩阵乘后的张量还是 FP32 就说明这个 op 没注册 autocast。解决方案:要么手动 `x = x.to(torch.bfloat16)`,要么让库支持 autocast(`@torch.amp.custom_fwd`/`custom_bwd` 装饰器)。

## 五、混合精度 × DDP / FSDP

### 5.1 DDP + AMP

最简单的组合。DDP 不关心 dtype,它只管"把 .grad AllReduce 一下"。autocast + GradScaler 在 forward / backward 里独立运作,DDP 在 backward 末尾接管 .grad 的同步。

```python
model = DDP(model.cuda(), device_ids=[local_rank])
scaler = GradScaler()

with autocast(dtype=torch.bfloat16):
    loss = criterion(model(x), y)

scaler.scale(loss).backward()   # DDP hook 在反向中触发,AllReduce .grad
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

唯一一个细节:DDP 的梯度通信(AllReduce)用的是 .grad 当前的 dtype。如果你想**用 BF16 通信**省一半带宽,有两种做法:

```python
# 做法 1:把 backward 通信 dtype 设成 BF16(FSDP 直接支持,见下面)
# DDP 这里要走 communication hook(详见 DDP.md 的 comm hook 部分)

# 做法 2:让 backward 出来的 .grad 直接是 BF16
# 默认 FP32 master 模型在 autocast 反向后 .grad 仍是 FP32
# 想要 BF16 通信,需要用 FSDP 或者手写 hook
```

### 5.2 FSDP 的 MixedPrecision 配置

FSDP 把混合精度拆成**三个独立可调的 dtype**:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,     # 参数计算时的 dtype
    reduce_dtype=torch.bfloat16,    # 梯度通信(ReduceScatter)用的 dtype
    buffer_dtype=torch.bfloat16,    # buffers (BN running_mean 等) 的 dtype
)

model = FSDP(model, mixed_precision=mp_policy, ...)
```

三个 dtype 的含义:

| 字段 | 控制什么 | 影响 |
| --- | --- | --- |
| `param_dtype` | 前向反向时的参数和激活计算 dtype | 算力(用 BF16 才能跑 Tensor Core) |
| `reduce_dtype` | 梯度跨卡通信的 dtype | 通信带宽(BF16 比 FP32 减半) |
| `buffer_dtype` | non-trainable buffers 的存储 dtype | 显存(很小,影响有限) |

**关键工程优化**:`reduce_dtype=torch.bfloat16` 让 FSDP 在 ReduceScatter 时把 FP32 .grad 临时 cast 成 BF16 再通信,**通信量直接减半**。代价是数值精度——大模型训练里这点精度损失对收敛几乎没影响,所以是默认推荐配置。

而 `optimizer state` 不在 MixedPrecision 控制范围内——FSDP 默认把分片的 optimizer states 仍然以 FP32 存,这就是 master weights 的物理实现。

### 5.3 一张图看清谁是什么 dtype

```
                  ┌─────────────────────┐
                  │ FSDP shard (1/N)    │
                  │  ┌───────────────┐  │
                  │  │ FP32 master   │  │  ← optimizer 看到的(更新在这里发生)
                  │  │     ↕         │  │
                  │  │ FP32 m, v     │  │
                  │  └───────────────┘  │
                  └──────────┬──────────┘
                             │
                  forward 时 cast 成 param_dtype
                             ↓
                  ┌─────────────────────┐
                  │ AllGather → BF16    │  ← 临时聚成完整层参数,BF16 算
                  │ (param_dtype)       │
                  └─────────────────────┘
                             ↓
                  ┌─────────────────────┐
                  │ matmul/conv (BF16)  │  ← Tensor Core 跑得飞起
                  │ activations (BF16)  │
                  └─────────────────────┘
                             ↓
                  ┌─────────────────────┐
                  │ softmax/loss (FP32) │  ← autocast 黑名单提升回 FP32
                  └─────────────────────┘

                  反向时再走一遍 AllGather,梯度 BF16
                             ↓
                  ┌─────────────────────┐
                  │ ReduceScatter BF16  │  ← reduce_dtype 控制
                  │ → 落回每卡 1/N grad │
                  └─────────────────────┘
                             ↓
                  在 FSDP 内部 cast 回 FP32 master,做 step
```

理解了这张图,FSDP 的混合精度配置就不再神秘——**每一层 cast 都对应一个具体的工程目的**(算力 / 显存 / 通信),没有一处是为了"看起来高级"。

## 六、FP8 训练:Hopper 时代的下一步

H100/Hopper 引入了对 FP8 的硬件支持,Tensor Core FP8 算力是 BF16 的 2 倍——训练时长直接砍半的诱惑。NVIDIA 的 transformer-engine 和后来的 Megatron-Core 把它产品化,Llama-3 的训练里就有部分 FP8 推理实验。

### 6.1 两种 FP8 各管一段

FP8 不像 BF16 那样能"一刀切"用在所有地方,因为只有 8 位实在太窄了——必须用两种格式分别处理前向和反向:

- **E4M3**(4 指数 + 3 尾数):**前向用**。范围窄($\pm 448$)但精度相对好,适合 activation 和 weight
- **E5M2**(5 指数 + 2 尾数):**反向用**。范围广($\pm 57344$)但精度差,适合 gradient(梯度数值范围跨好几个数量级)

```
                forward 计算:           backward 计算:
                  使用 E4M3                 使用 E5M2
                ┌─────────────┐           ┌─────────────┐
                │ act × weight│           │ dL/dy × W^T │
                │ → BF16 累加 │           │ → BF16 累加 │
                └─────────────┘           └─────────────┘
                       ↓                          ↑
                  量化回 E4M3              量化回 E5M2
```

### 6.2 Per-tensor scaling

FP8 范围太窄,**单一 scale 装不下**整个 batch 不同 tensor 的数值范围。所以每个 tensor 都需要带一个 scale factor——存储时:

$$x_{\text{stored}} = \text{round}(x_{\text{real}} / s)$$

读取时反量化回来 $x_{\text{real}} = x_{\text{stored}} \times s$。$s$ 通常根据 tensor 当前 batch 的 amax(绝对值最大值)动态计算,使得 max value 正好用满 FP8 的最大可表示数。这套 scale 的管理就是 transformer-engine 的核心抽象。

### 6.3 工程现状

到 2026 年,FP8 训练基本成熟,但还没有像 BF16 那样"换上去就能用":

- **稳定性**:某些层(尤其 attention 输出的 softmax 之前)对 FP8 敏感,需要保留 BF16
- **精度损失**:对小模型(7B 以下)有可观察的 loss 差异;大模型(70B+)基本看不出
- **生态**:transformer-engine 主要支持 NVIDIA 自家的 Megatron-Core 和 NeMo,PyTorch 原生 FSDP + FP8 还在快速演进

实战推荐:**70B+ 训练且预算紧张**才考虑 FP8;否则 BF16 仍然是默认选择。

## 七、关于精度的几个有用直觉

把整篇串起来,留几条能记得住的判断准则:

**第一**,**为什么 master weights 必须 FP32**——不是因为参数本身需要 FP32,而是因为 `param += lr * grad` 这个更新里 `lr * grad` 太小,半精度直接舍掉。如果你在 LR warmup 阶段看到 loss 完全不动,先怀疑这个。

**第二**,**为什么 LLM 全部 BF16 而 CV 还有 FP16**——LLM 的梯度数值范围跨好几个数量级(transformer 的 layernorm + softmax 让数值非常 wild),FP16 撑不住。CV 的 ResNet 类网络相对温和,FP16 + GradScaler 在那个数据分布下足够用。**所以遇到新模型,先用 BF16 训,稳了再考虑切 FP16 省那点尾数精度的事**(实际上几乎没人这么干)。

**第三**,**为什么 FSDP 的 reduce_dtype 默认是 FP32 而不是 BF16**——FSDP 默认配置保守,不打开 BF16 通信。**主动把它改成 BF16 通常能省 30-50% 通信时间**,是现代大模型训练的"免费午餐"——记得你自己加上。

**第四**,**autocast 不是万能解决方案**——它只覆盖 PyTorch 内置 op。如果你的代码里有自定义 CUDA / Triton kernel 或者第三方库,autocast 不会自动给它们转 dtype,需要你手动 cast 输入或在 kernel 里 dtype-aware。这是"训练曲线和理论不符"的最常见 footgun 之一。

**第五**,**mixed precision 不会"调小学习率"**——很多人以为换 BF16 / FP16 之后要相应缩小学习率,这是错的。Master weights 是 FP32,优化器仍然在 FP32 上更新,与学习率配置完全无关。学习率只跟模型规模、batch size、warmup 有关。如果换混合精度后必须改 LR 才能稳,大概率是 GradScaler 没配好或者有 dtype 不匹配。
