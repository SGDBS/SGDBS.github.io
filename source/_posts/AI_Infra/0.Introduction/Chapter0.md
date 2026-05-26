---
title: 0. 总览:2026 年的 AI Infra 知识框架
categories: 学习笔记- AI Infra
date: 2026-05-26 10:00:00
mathjax: true
tags:
    - AI
    - AI Infra
---


**模型代码只是冰山的一角**。一个 7B 的 Dense Transformer 核心代码不到一千行,一个 600B 的 MoE 模型核心代码也就两三千行;而训练它的 infra 代码——5D 并行、混合精度、FP8、checkpoint、容错、RLHF rollout、推理服务——动辄几万到十几万行。算法工程师和系统工程师之间的分工边界,这十年里一直在向"谁先理解 infra,谁先能 ship 大模型"那一侧倾斜。这套笔记就是想把这条边界本身讲清楚。

这是 **2026 年版**的 Chapter 0,相比 2025 年的初版,主要补了三件事:
1. **第 4、第 5 条并行轴**——Expert Parallel(EP)和 Context Parallel(CP),不再只讲 DP×PP×TP
2. **训练 × 推理的交叉课题**——RLHF Infra、disaggregated serving,这两条不在原来的"训练或推理"框架里
3. **编译器栈与训练 Ops**——`torch.compile` / DTensor / FSDP2、数据 pipeline、可观测性、训练稳定性,这些在 1000+ 卡 LLM 集群里跟并行同等重要

整套笔记的 GPU Kernel 学习已经拆到姊妹系列 `CUDA 算子` 里(详见第七节),AI Infra 主线只在"系统视角下的 kernel"这一章引用它。

## 一、起因:为什么是现在,为什么是这套话题

### 1.1 算力膨胀与单卡显存的脱节

过去十年模型参数量大致每 18 个月涨 10 倍(BERT-base 110M → GPT-3 175B → Llama-3 405B → Llama-4 / DeepSeek-V3 / Mixtral / Gemini 系列的 600B-2T MoE),而单卡显存只在五六年里翻了一倍多:V100 16GB → A100 40/80GB → H100 80GB → H200 141GB → B200 192GB → B300 288GB。**两条曲线的剪刀差就是 AI Infra 这门工程学科存在的根本原因**:既然单卡塞不下,就必须把模型、梯度、激活、优化器状态切碎到几百乃至上万张 GPU 上,同时还要让它们协同得像一台机器。

光"切碎"不够,还要让切出来的小块在通信延迟、网络拓扑、kernel 启动开销、numerical stability 等等因素下仍然能高效训练。这一切堆叠起来就是现代分布式训练系统。

### 1.2 训练 vs 推理的不同瓶颈

训练阶段,瓶颈是**显存 + 通信**——参数+梯度+优化器要 $16P$ 字节($P$ 是参数量,FP8 训练能压到 $\sim 10P$),激活要 $b \cdot s \cdot h \cdot L$ 量级,跨卡 AllReduce 又要把吞吐压住。

推理阶段,瓶颈是**KV cache + 内存带宽**——KV cache 随上下文长度线性增长,decode 阶段每个 token 都要把全部参数从 HBM 读一遍,GPU 算力闲置但内存带宽打满。

这两套问题的应对手段几乎完全不同(训练用 ZeRO/PP/TP/EP/CP,推理用 PagedAttention / continuous batching / speculative decoding / disagg),所以这门笔记把它们分块独立讲。

### 1.3 2024-2026 的几个新变量

相比 2023 年那套"3D 并行就够了"的世界,这两年发生了几件事:

**MoE 成为前沿模型的事实选择**。Mixtral 8×7B、DeepSeek-V3 671B、Llama-4 都是 MoE。MoE 训练在 DP/PP/TP 之外多出来一条**Expert Parallel(EP)**轴,带来 all-to-all 通信和 token 路由的全新工程问题。**3D 并行 → 4D 并行**。

**长上下文从奢侈品变成必备**。32K、128K、甚至 1M token 的训练和推理都成了主流。激活随 $s$ 线性、attention 随 $s^2$,TP 和 ZeRO 都救不了,需要一条独立的**Context Parallel(CP)**轴。**4D 并行 → 5D 并行**。

**RLHF / Post-training 工程量膨胀**。一个 PPO/GRPO 训练同时跑 actor + ref + critic + reward 四个模型,rollout 用 vLLM,训练用 Megatron,两套系统要交换权重。这是"训练"和"推理"的耦合,跨越了原来的章节边界。

**FP8 训练正式落地**。Hopper 引入的 FP8 在 2024-2025 年大量进入生产(DeepSeek-V3 就是 FP8 训练的旗舰案例),Blackwell 进一步引入 MXFP8 / MXFP4(微缩放浮点),把训练精度推到 4-bit 边缘。这让"精度"从"开 BF16 还是 FP16"的二选一,变成了一条独立的设计轴。

**编译器栈和图层抽象成熟**。`torch.compile` / Inductor 不再是玩具,FSDP2 用 per-parameter sharding 取代了 FSDP1 的 FlatParameter,DTensor 成为 PyTorch 官方的并行原语。**底层正在从"手写并行"向"声明性并行"迁移**——你描述 sharding,framework 自己生成通信。

**千卡集群的可观测性问题**。一旦集群上千,silent data corruption(SDC)、slow node、网络抖动都从"偶发"变成"几乎每个 step 都遇到一两个"。**没有 MFU dashboard 和 SDC 检测,根本不知道训练在不在 work**。

这五件事拼起来,2026 年的 AI Infra 比 2023 年的版本至少多了 50% 的内容。

## 二、主线:现代 AI Infra 的五个维度

把训练 + 推理这套技术抽象出来,实际上是**五个维度**互相组合:

```
                  ┌────────────────────────────────────────────────────┐
                  │   现代 AI Infra 的五个维度                            │
                  ├────────────────────────────────────────────────────┤
                  │                                                    │
   切模型/并行     │   DP / PP / TP / EP / CP — 五条彼此正交的轴         │
   (Ch1-5)        │   通信:AllReduce / P2P / AllGather / All-to-All    │
                  │                                                    │
   卡内/Kernel    │   FA / fused kernel / Triton / KV PagedAttention   │
   (Ch6, 引用    │   → 不切模型,只让 kernel 跑得贴近硬件极限           │
    CUDA 系列)    │                                                    │
                  │                                                    │
   编译器栈       │   torch.compile / DTensor / FSDP2 / CUDA Graph     │
   (Ch7)          │   → 从"手写并行"向"声明性并行"迁移                  │
                  │                                                    │
   训练 Ops       │   数据 pipeline / MFU / SDC / 容错 / loss spike    │
   (Ch8)          │   → 让 1000+ 卡训练真的能稳定跑两个月而不出事        │
                  │                                                    │
   推理 & 训推交叉 │   PagedAttention / continuous batch / spec decode   │
   (Ch9-11)       │   prefill-decode disagg / RLHF / FP8 量化           │
                  │   → 训练完模型后,怎么用最少卡跑最多请求             │
                  │                                                    │
                  └────────────────────────────────────────────────────┘
```

五个并行轴(DP / PP / TP / EP / CP)**正交**意味着可以任意组合——典型工业训练用的就是 **5D 并行 = DP × PP × TP × EP × CP**。比如训练 DeepSeek-V3 用了 EP=64 × PP=16 × DP=多路 × ZeRO,加起来 2048 张 H800;Llama-3 405B 是 TP=16 × PP=16 × DP=多路 × CP(长序列阶段);Llama-4 进一步把 EP 拉到 128。**理解每个轴是什么、对什么负责、什么时候上,是整套系统的脊椎骨**。

下面把每条线展开,然后给出十二章的详细脉络。

## 三、十二章脉络

### Part I:训练并行化(Ch1-5)

#### Ch1. Data Parallel(数据并行 + 显存切分)— 已完成

**对应问题**:模型本身能塞进单卡(或在 N 卡之间均匀分片后能塞进),只是数据吞吐不够、显存还紧。

**子文章**:
- `MemoryBudget` — 训练显存账本、micro-batch、activation checkpointing、梯度裁剪
- `DDP` — 集合通信原语、Ring AllReduce、bucketing、DistributedSampler
- `AsyncCompute` — CUDA stream 模型、计算-通信重叠、DDP/ZeRO-3/PP 的异步范式
- `MixedPrecision` — FP32/BF16/FP16/FP8 字节布局、loss scaling、autocast、与 FSDP 的协作
- `ZeRO` — ZeRO-1/2/3、FSDP、ZeRO-Offload、HYBRID_SHARD
- `MultiNode` — 进程组、NCCL 工程、网卡选择、拓扑感知、elastic 训练
- `Checkpoint` — DDP/FSDP 三种 state_dict、torch.distributed.checkpoint、异步 ckpt

**读完之后**:能在 8×A100 上跑起 7B-65B 规模的 Dense 训练,看懂 NCCL 在干什么、为什么 BF16 比 FP16 稳定。

#### Ch2. Pipeline Parallel(流水线并行)

**对应问题**:模型本身大到任何 ZeRO 配置都塞不下单卡(典型阈值:百亿参数 + 长序列),或者跨节点带宽差到 ZeRO-3 的 AllGather 无法藏在计算后面。

**子文章**(✓ 已写 / ☐ 待写):
- ✓ `GPipe` — bubble 公式、in-flight micro-batch、激活峰值账本、为什么必须配 AC
- ✓ `OneF1B`(1F1B)— 从 M 降到 P 的在飞数、warmup/steady/cooldown 三阶段、PipeDream-Flush
- ✓ `Interleaved1F1B` — 把 stage 切成 chunk、$v$ 取 2/3/4 的三大约束
- ☐ `ZeroBubble` — 把 backward 拆成 B 和 W、让 W 填进 bubble、bubble 接近 0 的代价
- ☐ `DualPipe` — DeepSeek-V3 的双向流水、forward 和 backward 共用一条流水、节省 1/2 bubble
- ☐ `PPCombined` — PP 和其他轴的拓扑映射、Megatron 的 `--pipeline-model-parallel-size` 实战

**读完之后**:看懂 Megatron-LM / DeepSpeed Pipeline 的源码,能根据 $P / M / \text{节点带宽}$ 选合适的调度。

#### Ch3. Tensor Parallel + Sequence Parallel

**对应问题**:单层(尤其 attention 和 FFN)算一次就把单卡显存吃满,或单层激活在序列维度上过大。

**子文章**:
- ☐ `MegatronTP` — 列切/行切组合、Attention 的 ColumnLinear+RowLinear、FFN 的 4h 维度切分
- ☐ `AllReduceElision` — 为什么"列切 + 行切"能消掉中间 AllReduce、forward 1 次 AllReduce vs naive 的 2 次
- ☐ `SequenceParallel` — LayerNorm / Dropout 的序列维度分片、与 TP 的 g/g̃ 通信耦合、Megatron-SP 的具体实现
- ☐ `TopoMapping` — TP rank=8 的物理原因(NVLink 内部全互连)、TP×DP×PP×EP×CP 的层级映射、为什么 TP 不能跨节点

**读完之后**:能理解为什么 TP 必须在节点内部、TP rank 通常是 8,知道 Sequence Parallel 是 TP 的"激活优化补丁"而不是独立轴。

#### Ch4. Expert Parallel(MoE 训练)— 新增

**对应问题**:MoE 模型(Mixtral / DeepSeek-V3 / Llama-4)的 experts 太多,放不进单卡甚至单节点;每个 token 经过的 expert 路径不一样,通信模式从 AllReduce 变成 All-to-All。

**子文章**:
- ☐ `MoEBasics` — MoE 数学定义、top-k routing、稀疏激活、为什么 GShard / Switch Transformer 是起点
- ☐ `ExpertParallel` — 把不同 experts 切到不同 GPU 上、token routing 的两次 all-to-all(dispatch + combine)
- ☐ `LoadBalance` — auxiliary loss、capacity factor、token drop vs no-drop、DeepSeek 的无辅助 loss 平衡
- ☐ `GroupedGEMM` — 把"稀疏专家"批处理回 dense GEMM 的工程技巧、cuBLAS / cutlass / Triton 的 grouped MM 实现
- ☐ `DeepEP` — DeepSeek 开源的 all-to-all 通信库、低延迟 / 高吞吐两套 kernel、与 NVSHMEM 的关系
- ☐ `EPxOther` — EP × TP × PP × DP 的组合策略、为什么 EP 通常和 DP 而不是 TP 组合

**读完之后**:能读懂 DeepSeek-V3 / Mixtral 的训练代码、估算一个 MoE 模型在不同 EP 配置下的通信开销、知道什么时候 token drop 是必要的。

#### Ch5. Context Parallel(长上下文训练)— 新增

**对应问题**:序列长度 $s \geq 32K$ 时,激活 $b \cdot s \cdot h \cdot L$ 跑飞,attention 矩阵 $s^2$ 跑飞,所有现存的 DP/TP/PP/EP 都不沿序列维度切——必须有第 5 条轴专门切序列。

**子文章**:
- ☐ `WhyCP` — 长上下文显存账本、为什么 TP 和 ZeRO 都救不了序列维度、激活 vs attention 矩阵的两个 $s$ 依赖
- ☐ `RingAttention` — 把 Q/K/V 按序列切到 N 张卡、KV 沿环传递、N 步算完整个 attention、与 FlashAttention 的协作
- ☐ `Ulysses` — DeepSpeed Ulysses 的 all-to-all 方案、把序列维度的并行重映射成 head 维度的并行
- ☐ `MegatronCP` — Megatron 的 CP 实现、与 SP/TP 的关系、A2A 通信成本
- ☐ `SequencePacking` — variable-length 序列怎么 pack、document mask、FlexAttention 的 packing 支持

**读完之后**:能选 Ring / Ulysses / Megatron-CP 中的一种实现 128K+ 训练,理解 SP 和 CP 的区别(SP 切的是元素级激活,CP 切的是 attention 本身)。

### Part II:单卡 Kernel 与编译器(Ch6-7)

#### Ch6. GPU Kernel(系统视角)

**对应问题**:模型已经切好了,但单 GPU 上的 kernel 还远未跑到硬件极限——大量时间花在 HBM ↔ SRAM 数据搬运上,而不是真正的浮点运算上。

> **本章定位**:**不**教你写 CUDA / Triton。kernel 编程的学习路径在姊妹系列 `CUDA 算子`(`source/_posts/CUDA/`)里独立展开。本章只讲**对训练 infra 决策有影响的 kernel**——它们改变了"什么模型能训"、"什么并行策略可行"的边界。

**子文章**(✓ 已写 / ☐ 待写):
- ✓ `FlashAttention` — HBM I/O 瓶颈、tiling、online softmax、recomputation、FA1/FA2/FA3 演进、与 causal mask / varlen 的协作
- ✓ `KVCache` — Prefill / Decode 两阶段、KV cache 的物理布局、decode 的算术强度
- ☐ `FusedNorm` — RMSNorm / LayerNorm 的 fused kernel、和 RoPE / residual add 的 fusion、Liger Kernel
- ☐ `RoPE` — 在哪一层 fuse RoPE、与 Q/K projection 的协作、长上下文的 RoPE scaling 工程
- ☐ `KernelImpactOnInfra` — kernel 速度怎么改写 PP/TP 的切分阈值、举三个真实例子(FA 改变 TP 阈值、grouped GEMM 改变 EP 可行性、fused MoE kernel 改变 expert routing 策略)

**读完之后**:看到一篇新论文(新 attention 变体、新 MoE 路由),能在 30 分钟内判断"这个 kernel 在工业上能不能用"。

#### Ch7. 编译器栈与图层抽象 — 新增

**对应问题**:1000+ 卡训练的并行配置爆炸:DP × TP × PP × EP × CP × micro-batch × bucket size × overlap 策略 × ckpt 频率……手工调参没法穷举。同时 `torch.compile` / Inductor 在 2024-2025 年成熟到可以替代很多手写 fused kernel,DTensor / FSDP2 让"声明 sharding,framework 生成通信"成了主流路径。

**子文章**:
- ☐ `TorchCompile` — Inductor pipeline(Dynamo → AOTAutograd → Inductor → Triton)、和 fused kernel 的关系、guard 机制、graph break 的代价
- ☐ `CUDAGraph` — capture / replay、与 `torch.compile` 的协作、与 NCCL 的兼容性、训练 step 的"硬件级 ahead-of-time"
- ☐ `DTensor` — PyTorch 的并行原语、placement(Shard / Replicate / Partial)、redistribution 的代价模型
- ☐ `FSDP2` — per-parameter sharding 取代 FlatParameter、与 DTensor 的统一、和 ZeRO-3 的差异
- ☐ `GSPMDvsManual` — 声明性并行 vs 手写并行的边界、什么时候 framework 自动 sharding 不够、Megatron 为什么仍然手写

**读完之后**:能在一个新模型上同时尝试三种并行配置(`torch.compile + FSDP2` / `Megatron 手写` / `DeepSpeed`),知道每种的开发成本和性能上限。

### Part III:训练 Ops(Ch8)

#### Ch8. 训练 Ops — 新增(合并若干子主题)

**对应问题**:并行配置对了 ≠ 训练能跑完。1000+ 卡集群上,数据 pipeline 拖后腿、loss spike 不收敛、单卡 SDC 污染梯度、慢节点拖死整 step——这些是"已经把分布式调通了"的人才会遇到的问题。

**子文章**:
- ☐ `DataPipeline` — tokenizer 工程、sequence packing、streaming dataset(WebDataset / Mosaic / IterableDataset)、shuffle 策略、与 CP 的协作
- ☐ `Observability` — MFU / HFU 计算公式、`torch.profiler` 实战、Nsight Systems timeline 解读、通信效率指标(algo_bw vs bus_bw)
- ☐ `Stability` — loss spike 的几类 root cause、grad NaN 诊断流程、determinism / reproducibility 配置、与 mixed precision 的协作
- ☐ `SDCDetection` — silent data corruption 是怎么发生的、Meta / Google 的检测策略、定期 sanity check 的工程实现
- ☐ `FaultTolerance` — torchrun elastic、慢节点(straggler)mitigation、自动重启、训练健康度评估

**读完之后**:能为一个新启动的千卡训练设计完整的 monitoring / alerting 流程,看到 loss 曲线异常时能在 1 小时内定位到是数据问题、kernel 问题、硬件问题还是配置问题。

> **设计选择**:数据、可观测、稳定性、容错这四块本来可以各自独立成章,但它们的共同特征是"训练已经跑起来之后才会遇到",而且工程逻辑互相耦合(loss spike 诊断需要 profiler、SDC 检测需要数据 pipeline 的配合),所以放在同一章里。

### Part IV:推理(Ch9-10)

#### Ch9. 推理基础

**对应问题**:训练好的模型怎么以 1ms 级 TTFT、几十 token/s 吞吐稳定服务,同时把硬件成本压到最低。

**子文章**:
- ☐ `PrefillDecode` — 两阶段瓶颈差异(prefill compute-bound / decode memory-bound)、为什么 TTFT 和 TPOT 是两个独立指标
- ☐ `PagedAttention` — KV cache 的物理碎片化问题、虚拟分页、block table、内存共享、vLLM 的核心创新
- ☐ `ContinuousBatching` — iteration-level scheduling、为什么 static batching 在变长生成里吞吐糟糕、preemption 策略
- ☐ `ChunkedPrefill` — 把长 prefill 切成多段,和 decode 拼一个 batch、解决 long-prompt 阻塞 decode 的问题
- ☐ `InferenceQuantization` — GPTQ / AWQ / SmoothQuant、weight-only vs activation 量化、FP8 推理与 INT8 推理的精度-延迟权衡

**读完之后**:能基于 vLLM 搭一个生产推理服务,知道在不同流量模式(prefill-heavy / decode-heavy / 混合)下应该怎么调度。

#### Ch10. 推理进阶

**对应问题**:推理基础解决了"单实例怎么高效",进阶要解决"多实例怎么部署"、"长尾延迟怎么压"、"多租户怎么共享"。

**子文章**:
- ☐ `SpeculativeDecoding` — draft model / Medusa / EAGLE / Lookahead / 自推测、verification 数学、acceptance rate 的工程优化
- ☐ `Disaggregation` — Splitwise / DistServe / Mooncake 的 prefill-decode 分离、KV transfer over RDMA、分离的收益与代价
- ☐ `PrefixCache` — 系统级 KV cache 复用、LMCache 风格的跨请求共享、prefix 命中率与显存代价权衡
- ☐ `LoRAServing` — S-LoRA / Punica 的多 LoRA 服务、共享 base model、unified paging
- ☐ `StructuredOutput` — JSON mode / Guided generation / FSM-based decoding、Outlines / xgrammar 的实现思路
- ☐ `InferenceStack` — vLLM / SGLang / TensorRT-LLM / TGI 的架构对比,各自的设计取舍

**读完之后**:能根据业务场景(长上下文 RAG / 代码补全 / 工具调用)选合适的推理栈组合,知道每个优化的命中条件。

### Part V:训推交叉与横切话题(Ch11-12)

#### Ch11. RLHF / Post-Training Infra — 新增

**对应问题**:Post-training(SFT / DPO / PPO / GRPO)的 infra 不在原来"训练"和"推理"任一框架内——它**同时**用训练系统和推理系统,而且要让两者高频交换权重。

**子文章**:
- ☐ `RLHFOverview` — SFT vs DPO vs PPO vs GRPO 的 infra 差异、为什么 GRPO 的 rollout 量比 PPO 大几倍
- ☐ `RolloutTrainColocate` — actor 和 rollout 共卡 vs 分离、内存切换策略、权重热更新(NCCL broadcast vs IPC)
- ☐ `AsyncRL` — off-policy 的 staleness 问题、流水线化 actor 更新与 rollout 生成
- ☐ `RLHFStacks` — OpenRLHF / verl / Nemo-Aligner / Axolotl 的架构对比、Megatron + vLLM 的拼装方式

**读完之后**:能搭一个能跑 GRPO 的 RLHF 系统(几十卡到几百卡),知道权重交换、KV cache 失效、rollout 长尾这些"训推交叉"特有的工程问题。

#### Ch12. 量化与现代硬件 — 新增

**对应问题**:FP8 训练已经从实验进入主流(DeepSeek-V3),Blackwell 引入了 MXFP8 / MXFP4 这套微缩放浮点;同时 GB200 NVL72 把 72 张 GPU 做成一个 unified memory domain,Grace-Hopper 把 CPU-GPU 统一寻址——硬件层正在快速变化,**很多原来正确的 infra 决策在新硬件上不再成立**(比如 TP rank=8 的物理逻辑在 NVL72 上需要重新审视)。

**子文章**:
- ☐ `FP8Training` — Hopper FP8 全流程(per-tensor scaling、delayed scaling、E4M3 vs E5M2 的取舍)、DeepSeek-V3 的 fine-grained scaling
- ☐ `MicroscalingFP` — Blackwell 的 MXFP8 / MXFP4、per-block scaling、与 FP8 的兼容性
- ☐ `ModernHardware` — Blackwell B200/B300 关键变化、GB200 NVL72 的 unified memory domain、Grace-Hopper 的 CPU-GPU 统一寻址、MIG 多实例 GPU
- ☐ `HardwareDrivenRewrite` — 硬件变化如何反向改写 infra 决策、举三个例子(NVLink Switch 让 TP rank 突破 8、unified memory 简化 offload、TMA 让 FA 反向更激进)

**读完之后**:能看懂一份 Blackwell 集群规划文档,知道一个原本在 H100 上设计的训练方案到 B200 上需要改哪些参数。

## 四、读者画像与前置知识

### 4.1 这套笔记假设你已经会的东西

- **PyTorch 基础**:能写一个完整的 train loop,知道 `nn.Module` / `loss.backward()` / `optimizer.step()` / `DataLoader` 的角色
- **Transformer 结构**:知道 multi-head attention、FFN、LayerNorm、RoPE 大概怎么算,不要求会推导反向公式
- **基本 GPU 概念**:知道 GPU 有"算力"和"显存",HBM 是"主存",大概知道 NVLink、PCIe 是不同带宽的链路
- **基础 Linux**:能跑 `ssh`、`tmux`、`nvidia-smi`、能看懂一个简单的 bash 脚本
- **基础数学**:线性代数(矩阵乘 / 转置 / 分块)、概率(softmax / KL)、最优化(SGD / Adam 直觉)

### 4.2 不要求会的(笔记会顺便补)

- **集合通信原语**(Broadcast/Reduce/AllReduce/AllGather/ReduceScatter/All-to-All):Ch1 DDP 与 Ch4 MoE 各补一半
- **CUDA stream/event 模型与异步执行**:Ch1 AsyncCompute 专门展开
- **浮点格式**(FP32/BF16/FP16/FP8/MXFP):Ch1 MixedPrecision + Ch12 量化
- **HBM/SRAM 内存层级、roofline、合并访存**:在姊妹系列 `CUDA 算子` 的 Chapter 0 已经讲透,Ch6 直接引用
- **MoE 数学结构**:Ch4 第一篇从零讲
- **RLHF 算法直觉**(PPO 的 advantage、KL penalty、GRPO 怎么省掉 critic):Ch11 第一篇补

### 4.3 从硕士实习到工业级

如果你是研究方向的研究生,这套笔记希望让你能**看懂 Megatron-LM、DeepSpeed、FSDP2、vLLM、SGLang、verl 的源码**——它不是源码导读,但读完每一章你应该能直接打开对应仓库的核心文件,知道每段在干什么、为什么这么写。

如果你已经在岗,这套笔记的目标是**让你能够独立设计一个新模型的训练 + 推理方案**,知道在什么参数量、什么硬件、什么流量下选哪种并行/kernel/serving 组合,以及在出问题时知道用什么工具(profiler / nsys / `CUDA_LAUNCH_BLOCKING` / NCCL log)定位到根因。

## 五、笔记的写作风格

每一篇文章按"**起因 → 数学/系统原理 → 工程实现细节 → 直觉与踩坑**"四段式展开:

- **起因**:这个东西为什么会被发明,它要解决前面没解决的什么问题
- **原理**:数学等价性、复杂度、通信量、显存账本——能算清楚的全部算清楚
- **工程**:PyTorch / NCCL / CUDA / Megatron / vLLM 的实际 API 是怎么把原理映射成代码,典型配置参数是怎么取的
- **直觉**:在哪些场景会失效,常见 footgun,以及"为什么这是合理的"——把每段不容易直觉化的设计配上一个可以记住的比喻

笔记**不**单独列"面试高频追问"——这种格式会把知识点和场景分离,读起来像背诵卡片,而不是建立一张可推理的概念地图。"高频被问"的点会以"为什么这样设计 / 在什么情况下会怎样 / 不这么做会出什么问题"的方式融进正文。读完一遍,该会的都会;遇到面试场景,自然就能讲出来。

## 六、章节内部 / 跨章阅读顺序

不一定要严格按编号读。下面是依赖图,方便按需跳读:

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

Ch2 / Ch3 / Ch4 / Ch5 — 都假设你读过 Ch1 的 DDP 和 AsyncCompute
  ├── Ch2 (PP) — 与 Ch3 独立
  ├── Ch3 (TP+SP) — 与 Ch2 独立
  ├── Ch4 (EP) — 推荐先读 Ch3,因为 EP 经常和 TP 组合
  └── Ch5 (CP) — 推荐先读 Ch3 的 SP,因为 CP 是"沿序列切"的极限

Ch6 (Kernel 系统视角) — 与 Ch2-5 独立,只需要 Transformer 知识
  ├── FlashAttention 是 Ch5 (CP) 的前置
  └── KVCache 是 Ch9 (推理基础) 的前置

Ch7 (编译器栈) — 推荐放 Ch1-5 之后读,这样你才知道 framework 在自动化什么

Ch8 (训练 Ops) — 跟所有前面的章节都有交集,但本身相对独立,适合实习开始前读

Ch9 / Ch10 (推理) — 只需要 Transformer + Ch6 的 KVCache 直觉,完全可以跳着先读

Ch11 (RLHF) — 训练 + 推理两边都要先有基础,放在最后

Ch12 (量化与硬件) — 横切话题,任何时候都可以查,但放在最后能形成"硬件变化怎么影响所有前面章节"的回顾视角
```

## 七、和姊妹系列的关系

这套笔记**不**教你写 CUDA / Triton。GPU kernel 编程是另一个学科,有自己的学习路径,在我的另一个系列 `CUDA 算子`(`source/_posts/CUDA/`)里展开。两条线的分工:

| | AI Infra(本系列) | CUDA 算子(姊妹系列) |
| --- | --- | --- |
| 核心问题 | 模型怎么散到 1000+ 卡上 | 单张卡上的 kernel 怎么贴近硬件极限 |
| 视角 | 系统、集群、分布式 | 单卡硬件、指令、访存 |
| 代码层级 | PyTorch / Megatron / vLLM 这一层 | CUDA / Triton / PTX 这一层 |
| 典型话题 | 5D 并行、RLHF、推理服务 | 合并访存、tiling、warp shuffle |
| 对 kernel 的态度 | 当作黑盒,但要知道它的复杂度和显存账 | 亲手写,profile,push 到 cuBLAS 性能 |

**强烈建议两条线并行读**。原因:

- Ch1 ZeRO 的 AllGather + CUDA 系列的 matmul tile——本质都是"把数据拉进局部 buffer 然后多次复用",一个发生在 GPU 之间,一个发生在 SRAM 内部
- Ch1 的 RingAllReduce 的 tree reduction + CUDA 系列的 warp reduction——同一种算法在不同尺度上的体现
- Ch6 FlashAttention 系统视角 + CUDA 系列的手写 FA——理论 + 实操,缺一个都不完整
- Ch12 的量化与硬件 + CUDA 系列的 Hopper TMA / wgmma——硬件特性反向决定 infra 决策

如果你时间有限,可以先把本系列 Ch1-5 主线读完(理解"模型怎么切"),再回头去 CUDA 系列学 kernel,再回 Ch6-12 学 kernel 和编译器对 infra 的反作用。

## 八、和 2025 年初版 Chapter 0 的差异(给已经读过旧版的人)

旧版的核心论点是"3D 并行 = DP × PP × TP"。本版扩成"5D 并行 + 编译器栈 + 训练 Ops + 训推交叉"。具体差异:

1. **并行轴从 3 个扩到 5 个**:加上 EP(MoE 需要)和 CP(长上下文需要)
2. **新增 Ch7 编译器栈**:`torch.compile` / DTensor / FSDP2 / CUDA Graph 在 2024-2025 已经是主流
3. **新增 Ch8 训练 Ops**:数据 pipeline / MFU / SDC / 容错 / loss spike — 千卡训练的"看不见的工程量"
4. **新增 Ch11 RLHF Infra**:训推交叉,GRPO 时代的 infra
5. **新增 Ch12 量化与现代硬件**:FP8 训练落地、Blackwell / NVL72 改写硬件假设
6. **Kernel 章节降权**:从"完整的 GPU 编程教程"压缩成"系统视角下的几个关键 kernel",编程教程独立到姊妹系列

旧版的 Ch4 / Ch5(GPU Kernel + Inference)在新版里成了 Ch6 / Ch9-10,内容大幅扩展。**章节编号有变,但 Ch1 完全不动,已经读过的可以直接跳到 Ch2 继续**。

## 九、一句话总结

**AI Infra 不是把模型代码工程化**——它是另一个学科,关心的是显存、带宽、延迟、numerical stability、硬件拓扑、集群可靠性这些算法工程师通常不主动思考的问题。在 2026 年的大模型时代,**会写模型 + 不会切并行 = 永远只能跑别人切好的开源版本**;而懂 infra 的人,可以在新硬件、新架构、新场景出现的第一天就知道怎么把它跑起来、调到极致、稳定服务。

这套笔记希望让你成为后者。
