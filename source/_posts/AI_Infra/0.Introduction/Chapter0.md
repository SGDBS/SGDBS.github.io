---
title: 0. 总览:为什么需要 AI Infra,以及笔记讲什么
categories: 学习笔记- AI Infra
date: 2026-05-09 10:00:00
mathjax: true
tags:
    - AI
    - AI Infra
---


**模型代码只是冰山的一角**。一个 7B 的 Transformer 模型核心代码不到一千行,而训练它的 infra 代码——3D 并行、混合精度、checkpoint、容错、推理服务——动辄几万行。算法工程师和系统工程师之间的分工边界,这十年里一直在向"谁先理解 infra,谁先能 ship 大模型"那一侧倾斜。这门课就是想把这条边界本身讲清楚。

## 一、起因:为什么是现在,为什么是这套话题

### 1.1 算力膨胀与单卡显存的脱节

过去十年模型参数量大致每 18 个月涨 10 倍(从 BERT-base 110M 到 GPT-3 的 175B、再到 Llama-3 的 405B、Gemini Ultra 的万亿级 MoE),而单卡显存只在四五年里翻了一倍多——V100 16GB → A100 40GB / 80GB → H100 80GB → B200 192GB。**两条曲线的剪刀差就是 AI Infra 这门工程学科存在的根本原因**:既然单卡塞不下,就必须把模型、梯度、激活、优化器状态切碎到几百乃至上万张 GPU 上,同时还要让它们协同得像一台机器。

光"切碎"不够,还要让切出来的小块在通信延迟、网络拓扑、kernel 启动开销、numerical stability 等等因素下仍然能高效训练。这一切堆叠起来就是现代分布式训练系统。

### 1.2 训练 vs 推理的不同瓶颈

训练阶段,瓶颈是**显存 + 通信**——参数+梯度+优化器要 $16P$ 字节($P$ 是参数量),激活要 $b \cdot s \cdot h \cdot L$ 量级,跨卡 AllReduce 又要把吞吐压住。

推理阶段,瓶颈是**KV cache + 内存带宽**——KV cache 随上下文长度线性增长,decode 阶段每个 token 都要把全部参数从 HBM 读一遍,GPU 算力闲置但内存带宽打满。

这两套问题的应对手段几乎完全不同(训练用 ZeRO/PP/TP,推理用 PagedAttention/continuous batching/spec decoding),所以这门笔记会把它们分章独立讲。

### 1.3 工程主线:3D 并行 + Kernel + 推理

把现代大模型训练 + 部署的所有 infra 技术抽象出来,实际上就三件事:

1. **怎么把模型切到很多卡上**——这一族的答案是 3D 并行(数据并行 / 流水线并行 / 张量并行)
2. **怎么让单卡的每个 kernel 跑得更快**——FlashAttention、fused kernel、quantization、Triton 编译
3. **怎么把训练好的模型用最低的延迟和最高的吞吐服务出去**——KV cache 管理、continuous batching、speculative decoding

这门笔记五大章的拆分,就是按这条主线展开的。

## 二、主线:训练系统的三个正交轴

理解后续四五章之前,先把这张图刻在脑子里——所有并行策略都是这张图上某个或多个维度的组合。

```
                   ┌─────────────────────────────────────────┐
                   │   3D 并行的三个轴 (彼此正交)              │
                   ├─────────────────────────────────────────┤
                   │                                          │
   Data Parallel   │   每张卡看不同 batch、有相同的(或可临时   │
   (Ch1: DDP/ZeRO) │   聚齐的)完整模型                        │
                   │   通信:AllReduce / ReduceScatter /      │
                   │         AllGather                        │
                   │                                          │
   Pipeline Para.  │   每张卡只持有部分层、看相同 micro-batch │
   (Ch2)           │   通信:相邻 stage 的 P2P send/recv      │
                   │                                          │
   Tensor Parallel │   每张卡持有同一层的不同切片、看相同 batch│
   (Ch3)           │   通信:层内的 AllReduce / AllGather     │
                   │                                          │
                   └─────────────────────────────────────────┘

                                  ↓ 单卡内部继续优化 ↓

                   ┌─────────────────────────────────────────┐
                   │   GPU Kernel 优化 (Ch4)                  │
                   │   FlashAttention / Triton / Fused kernel │
                   │   → 不切模型,只让 kernel 跑得贴近硬件极限 │
                   └─────────────────────────────────────────┘

                                  ↓ 训练完之后 ↓

                   ┌─────────────────────────────────────────┐
                   │   Inference Infra (Ch5)                  │
                   │   KV cache / vLLM / Speculative Decode   │
                   │   → 训练完模型,怎么用最少卡跑最多请求     │
                   └─────────────────────────────────────────┘
```

三个并行轴**正交**意味着可以任意组合——典型工业训练用的就是 **3D 并行 = DP × PP × TP**。比如训练 Llama-3 405B 用了 16-way TP × 16-way PP × N-way DP,加起来几千张 H100。理解每个轴是什么、对什么负责、什么时候上,是整套系统的脊椎骨。

## 三、五大章脉络

### Ch1. Data Parallel(数据并行 + 显存切分)

**对应问题**:模型本身能塞进单卡(或在 N 卡之间均匀分片后能塞进),只是数据吞吐不够、显存还紧。

**主线**:从最经典的 Distributed Data Parallel(DDP)出发,理解集合通信原语和 Ring AllReduce;然后引入 ZeRO/FSDP,把"参数同步"重新定义为"参数分片",把数据并行从"复制状态"演进到"分片状态"。这一章顺手把训练系统里几乎所有人都要写的东西都讲掉:**显存预算、混合精度、激活检查点、梯度累积、梯度裁剪、多机启动、checkpoint**。

读完这一章,你应该能在 8×A100 上跑起 7B–65B 规模的训练,理解 NCCL 在干什么、为什么 BF16 比 FP16 稳定、以及 `optimizer.step()` 之前的那一行 `clip_grad_norm_` 为什么必须存在。

### Ch2. Pipeline Parallel(流水线并行)

**对应问题**:模型本身大到任何 ZeRO 配置都塞不下单卡(典型阈值:百亿参数 + 长序列),或者跨节点带宽差到 ZeRO-3 的 AllGather 无法藏在计算后面。

**主线**:从 Naive Pipeline 的 bubble 公式出发,讲透 GPipe、PipeDream、1F1B、Interleaved 1F1B、Zero Bubble Pipeline 几代调度的演进。Pipeline 的核心是**调度问题**——同样的硬件、同样的通信原语,不同的发射顺序对应非常不同的吞吐。

读完这一章,你应该能看懂 Megatron-LM 里 `--pipeline-model-parallel-size` 到底在干什么、为什么 micro-batch 数对吞吐这么敏感、以及为什么 2023 年突然冒出来的 Zero Bubble PP 把 bubble 干到了几乎为 0。

### Ch3. Tensor Parallel(张量并行)

**对应问题**:单层(尤其 attention 和 FFN)算一次就把单卡显存吃满,需要把"一个 nn.Linear"内部切到多卡上同时算。

**主线**:从 Megatron-LM 的列切 / 行切组合出发,讲清楚 attention 和 FFN 在 TP 下如何放置 AllReduce、为什么"列切 + 行切"组合可以省掉中间 AllReduce、以及 Sequence Parallel 怎么把 LayerNorm 这类元素级操作的激活也分片。

读完这一章,你应该能理解 TP 为什么必须在节点内部(NVLink)、跨节点会瓶颈,理解 TP × DP × PP 的拓扑映射(典型:节点内 TP,跨节点 PP+DP),以及为什么 TP rank 数通常是 8(对应一台 DGX 节点的 8 GPU)。

### Ch4. GPU Kernel 优化

**对应问题**:模型已经切好了,但单 GPU 上的 kernel 还远未跑到硬件极限——大量时间花在 HBM ↔ SRAM 数据搬运上,而不是真正的浮点运算上。

**主线**:从 GPU 内存层级(HBM 1TB/s vs SRAM 19TB/s)开始,理解为什么"看起来 attention 是 $O(s^2)$ 计算,但其实是 $O(s^2)$ memory I/O 在限制速度"。然后讲透 FlashAttention 1/2/3 的 tiling + online softmax + recomputation 三件套,顺势讲推理时的 PagedAttention、Fused RMSNorm/RoPE/SwiGLU,以及如何用 Triton 自己写一个有竞争力的 kernel。

读完这一章,你应该能区分"算法瓶颈"和"内存瓶颈",理解 FA 为什么是 LLM 训练的事实标准,以及拿到一个新模型时如何用 profiler 找到最该 fuse 的几个 kernel。

### Ch5. Inference Infrastructure(推理基础设施)

**对应问题**:训练好的模型怎么以 1ms 级 TTFT、几十 token/s 吞吐稳定服务,同时把硬件成本压到最低。

**主线**:从推理两阶段(prefill 计算密集 / decode 内存密集)出发,讲透 KV cache 的物理布局、PagedAttention 怎么把碎片化干掉、continuous batching 为什么对 SLA 至关重要、speculative decoding 怎么用一个小模型把大模型推理拉快 2-3 倍、以及量化(GPTQ / AWQ / FP8)怎么在精度几乎不掉的前提下把延迟和显存都砍半。

读完这一章,你应该能搭出一个生产级 vLLM/TGI 服务,理解它在不同流量模式下的 throughput-latency 曲线,以及为什么"训练强于推理"未必意味着 ship 成功——很多模型死在了推理成本上。

## 四、读者画像与前置知识

### 4.1 这套笔记假设你已经会的东西

- **PyTorch 基础**:能写一个完整的 train loop,知道 `nn.Module` / `loss.backward()` / `optimizer.step()` / `DataLoader` 的角色
- **Transformer 结构**:知道 multi-head attention、FFN、LayerNorm、RoPE 大概怎么算,不要求会推导反向公式(Ch4 会顺便补)
- **基本 GPU 概念**:知道 GPU 有"算力"和"显存",HBM 是"主存",大概知道 NVLink、PCIe 是不同带宽的链路
- **基础 Linux**:能跑 `ssh`、`tmux`、`nvidia-smi`、能看懂一个简单的 bash 脚本

### 4.2 不要求会的(笔记会顺便补)

- 集合通信原语(Broadcast/Reduce/AllReduce/AllGather/ReduceScatter):**Ch1 第二节**会从零讲清楚
- CUDA stream/event 模型与异步执行:**Ch1 异步计算篇**专门展开
- 浮点格式(FP32/BF16/FP16/FP8)的字节布局与数值特性:**Ch1 混合精度篇**
- HBM/SRAM 内存层级:**Ch4 开篇**

### 4.3 从硕士实习到工业级

如果你是研究方向的研究生,这套笔记希望让你能**看懂 Megatron-LM、DeepSpeed、FSDP、vLLM 的源码**——它不是源码导读,但读完每一章你应该能直接打开对应仓库的核心文件,知道每段在干什么、为什么这么写。

如果你已经在岗,这套笔记的目标是**让你能够独立设计一个新模型的训练 + 推理方案**,知道在什么参数量、什么硬件、什么流量下选哪种并行/kernel/serving 组合,以及在出问题时知道用什么工具(profiler / nsys / `CUDA_LAUNCH_BLOCKING`)定位到根因。

## 五、笔记的写作风格

每一篇文章会按"**起因 → 数学/系统原理 → 工程实现细节 → 直觉与踩坑**"四段式展开:

- **起因**:这个东西为什么会被发明,它要解决前面没解决的什么问题
- **原理**:数学等价性、复杂度、通信量、显存账本——能算清楚的全部算清楚
- **工程**:PyTorch / NCCL / CUDA 的实际 API 是怎么把原理映射成代码,典型配置参数是怎么取的
- **直觉**:在哪些场景会失效,常见 footgun,以及"为什么这是合理的"——把每段不容易直觉化的设计配上一个可以记住的比喻

我刻意**没有**单独的"面试高频追问"小节——这种格式会把知识点和场景分离,读起来像背诵卡片,而不是建立一张可推理的概念地图。我会把那些"高频被问"的点直接以"为什么这样设计 / 在什么情况下会怎样 / 不这么做会出什么问题"的方式融进正文。读完一遍,该会的都会;遇到面试场景,自然就能讲出来。

## 六、章节内部阅读顺序

每一章内部不一定要严格按文件序号读。下面给一个粗略的依赖图,方便你按需跳读:

```
Ch1 内部:
  MemoryBudget(问题陈述,所有人先读)
       ├── DDP ── AsyncCompute(DDP 的重叠机制)
       │           │
       │           └── ZeRO(ZeRO-3 的 prefetch 复用 AsyncCompute 的概念)
       │
       ├── MixedPrecision(独立,任何时候都可以读)
       │
       └── MultiNode + Checkpoint(生产化,可放最后)

Ch2 → Ch3 → Ch4 三章彼此独立,但都假设已经读过 Ch1 的 DDP 和 AsyncCompute

Ch5 是相对独立的一章,只需要 Transformer 知识和 Ch4 的 KV cache 直觉
```

## 七、一句话总结

**AI Infra 不是把模型代码工程化**——它是另一个学科,关心的是显存、带宽、延迟、numerical stability、硬件拓扑这些算法工程师通常不主动思考的问题。在大模型时代,**会写模型 + 不会切并行 = 永远只能跑别人切好的开源版本**;而懂 infra 的人,可以在新硬件、新架构、新场景出现的第一天就知道怎么把它跑起来、调到极致。

这套笔记希望让你成为后者。
