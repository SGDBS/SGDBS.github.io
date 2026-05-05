---
title: AI Infra学习计划
categories: 日记
date: 2026-04-23 22:48:00
mathjax: true
tags:
    - 日记
---


# AI Infra 学习清单（按技术栈分类）

按 **公用 → 推理 → 训练** 三大块组织，每块内部分 ⚡**速通**（面试高频，优先学）和 🐢**慢补**（基础功底，长期积累）。

---

## 一、公用技术栈

这部分是无论做推理还是训练都必须掌握的，**性价比最高**。

### ⚡ 速通部分（面试高频）

**1. Transformer 与 LLM 原理**
- Transformer 完整结构：MHA、FFN、LayerNorm/RMSNorm、残差
- Decoder-only 架构（GPT/LLaMA 系）
- RoPE 位置编码原理
- SwiGLU / GeGLU 激活函数
- Pre-norm vs Post-norm
- 主流模型架构差异：LLaMA、Qwen、DeepSeek、Mixtral

**2. Attention 机制全家桶**
- MHA 标准多头注意力
- MQA、GQA 演进
- MLA（DeepSeek）：低秩压缩 + 解耦 RoPE
- Sliding Window Attention
- Linear Attention / Lightning Attention（了解）
- 各变种的 KV Cache 大小对比

**3. 高性能算子（必备硬实力）**
- 必须能手写：Reduce、Softmax、Online Softmax、LayerNorm、RMSNorm、GEMM
- 算子融合思想：QKV 融合、RMSNorm + Residual、SwiGLU
- FlashAttention v1 / v2 / v3 原理与演进
- Tiling、Online Softmax、Recomputation 三大核心思想

**4. GPU 架构与硬件认知**
- SM、Warp、Thread Block、Grid 层级
- 内存层级：Register、Shared Memory、L1/L2、Global、HBM
- Tensor Core 使用
- NVIDIA 架构演进：Ampere → Hopper → Blackwell
- H100 / H200 / B200 的关键参数（带宽、算力、显存）

**5. 并行通信原语**
- AllReduce、AllGather、ReduceScatter、All-to-All、Broadcast
- Ring vs Tree 算法
- NCCL 基本使用
- NVLink、NVSwitch、PCIe、InfiniBand、RDMA

**6. 性能分析方法论**
- Roofline 模型
- 算术强度（Arithmetic Intensity）计算
- Compute-bound vs Memory-bound 判断
- Nsight Compute / Nsight Systems 基本使用

**7. 手撕题准备**
- LeetCode Hot 100（保持算法手感）
- LeetGPU：Reduce、GEMM、Softmax
- 算法手撕：MHA forward、RoPE、Sampling

### 🐢 慢补部分（基础功底）

**1. CUDA 编程深入**
- 编程模型完整理解
- Shared Memory 优化与 Bank Conflict
- Memory Coalescing
- Warp 级原语：`__shfl_sync`、`__ballot_sync`
- Occupancy 分析与优化
- Async Copy（`cp.async`）
- Tensor Core 编程（WMMA / MMA API）
- CUTLASS / CuTe 框架
- PTX 阅读基础
- CUDA Graph

**2. Triton 编程**
- Triton 编程模型与 CUDA 的差异
- 用 Triton 实现 GEMM、Softmax、FlashAttention
- Autotuning 机制
- 与 PyTorch 集成

**3. C++ 进阶**
- 现代 C++（11/14/17/20）特性
- 模板与模板元编程
- 移动语义、智能指针
- 多线程：`std::thread`、`std::atomic`、内存序
- pybind11
- CMake 构建系统

**4. 计算机系统基础（CSAPP 重点章节）**
- Ch3：机器级表示
- Ch5：程序优化（流水线、SIMD）
- Ch6：存储层级（Cache 原理）
- Ch9：虚拟内存（理解 PagedAttention 必备）
- Ch12：并发编程

**5. 计算机体系结构**
- 流水线、乱序执行、分支预测
- Cache 一致性、内存模型
- NUMA 架构
- SIMD 指令集

**6. PyTorch 内部机制**
- Tensor / Storage / Stride
- Autograd 机制
- Dispatcher、ATen、c10
- 自定义 CUDA 算子（cpp_extension）
- `torch.library` 注册算子
- torch.compile / Dynamo / AOTAutograd / Inductor

**7. 编译器基础（选学）**
- LLVM 基础
- MLIR Dialect 概念
- TVM 调度原语
- XLA HLO IR

---

## 二、推理技术栈

岗位最多、最热门，**强烈建议作为主攻方向**。

### ⚡ 速通部分（面试高频）

**1. 推理两阶段与关键指标**
- Prefill 阶段特性（compute-bound）
- Decode 阶段特性（memory-bound）
- TTFT、TPOT/ITL、Throughput、Goodput
- QPS、并发数、SLA 关系
- 这是几乎所有推理优化的出发点，必须烂熟

**2. KV Cache 管理**
- KV Cache 本质与必要性
- 内存占用计算公式
- **PagedAttention**（vLLM 核心）：
  - 类比 OS 虚拟内存分页
  - Block Table、逻辑/物理块映射
  - 内存碎片化解决思路
- **RadixAttention**（SGLang）：前缀树 + LRU
- **Prefix Cache**：系统提示词、Few-shot 共享场景
- KV Cache 量化（INT8 / FP8）

**3. 推理调度核心**
- Static Batching 的问题
- **Continuous Batching / In-flight Batching**（Orca）
- **Chunked Prefill**：长 prefill 切片混合 decode
- **PD 分离架构**：DistServe、Mooncake
- KV Cache 在 PD 之间的传输策略
- Priority Scheduling

**4. 解码策略与投机解码**
- Greedy / Top-k / Top-p / Temperature
- **Speculative Decoding** 数学正确性
- **EAGLE / EAGLE-2 / EAGLE-3**：原理、性能
- **Medusa**：多头并行预测
- **MTP**（DeepSeek-V3）：多 token 预测
- Lookahead Decoding（了解）

**5. 量化（推理重头戏）**
- 对称 vs 非对称量化
- Per-tensor / Per-channel / Per-group
- W8A8、W4A16、W4A8、FP8 各场景
- **GPTQ**：基于 OBQ 的逐层量化
- **AWQ**：Activation-aware
- **SmoothQuant**：迁移激活 outlier
- **FP8**：E4M3 / E5M2 格式
- KV Cache 量化（KIVI 思路）

**6. MoE 推理**
- 路由机制：Token Choice / Expert Choice
- Top-k 选择与 Auxiliary Loss
- Aux-Loss-Free（DeepSeek）
- **All-to-All 通信**瓶颈
- **EPLB**（Expert Parallel Load Balancer）
- Shared Expert 设计

**7. 推理并行策略**
- TP（Tensor Parallel）：QKV 怎么切、FFN 怎么切
- DP（Data Parallel）：replica 模式
- EP（Expert Parallel）：MoE 必备
- PP（推理少用，长序列偶尔）
- Attention DP + FFN TP 混合
- 各策略通信开销分析

**8. vLLM 源码精读**
- `LLM.generate()` 调用链
- `Scheduler`：调度逻辑
- `BlockManager`：KV Cache 块管理
- `ModelRunner`：模型前向调度
- `AttentionBackend`：FlashAttention / FlashInfer 集成
- 整体架构图能默写

**9. 必读推理论文**
- PagedAttention（vLLM）
- Orca（Continuous Batching）
- DistServe / Mooncake（PD 分离）
- SARATHI（Chunked Prefill）
- EAGLE 系列、Medusa
- GPTQ、AWQ、SmoothQuant
- DeepSeek-V2（MLA）、DeepSeek-V3（MTP）

### 🐢 慢补部分（基础功底）

**1. 框架对比与深入**
- SGLang 源码（结构比 vLLM 清晰，新手友好）
- TensorRT-LLM：NVIDIA 官方，性能最强
- LMDeploy（商汤）
- llama.cpp（CPU / 边缘端，C++ 写得很好）
- MLC-LLM（跨平台）

**2. 推理算子优化深入**
- FlashAttention 完整实现细节
- FlashInfer 库
- PagedAttention CUDA kernel 实现
- Continuous Batching 的算子要求
- 各种 fused kernel 的实现

**3. 量化算法深入**
- 量化误差分析
- Hessian 矩阵在量化中的应用（GPTQ 数学基础）
- Outlier 处理的多种思路
- QAT vs PTQ
- SpinQuant、QuaRot 等旋转矩阵方法

**4. 服务化工程**
- Tokenizer 性能优化
- Batch 调度算法细节
- 多模型混合部署
- LoRA / Multi-LoRA 推理
- 长上下文（128K+）优化
- 流式输出（SSE / WebSocket）

**5. 推理硬件深入**
- H100 vs H200 vs B200 详细对比
- MIG（Multi-Instance GPU）
- NVLink 拓扑与通信优化
- 国产硬件：昇腾、寒武纪、壁仞、摩尔线程

**6. 推理压测与运维**
- 压测工具：vLLM benchmark、GenAI-Perf
- 监控指标：GPU 利用率、KV Cache 命中率
- 容量规划

---

## 三、训练技术栈

岗位相对少但门槛高，如果只投推理可弱化此部分。

### ⚡ 速通部分（面试高频）

**1. 分布式并行三件套**
- **DDP**（Data Parallel）：基本数据并行
- **ZeRO 1/2/3**（DeepSpeed）：
  - ZeRO-1：优化器状态分片
  - ZeRO-2：+ 梯度分片
  - ZeRO-3：+ 参数分片
  - ZeRO-Offload / ZeRO-Infinity
- **FSDP**（PyTorch 原生）：与 ZeRO-3 思路类似
- **TP**（Tensor Parallel，Megatron）：
  - Column Parallel / Row Parallel
  - QKV、FFN 的切分方式
  - 通信模式
- **PP**（Pipeline Parallel）：
  - GPipe
  - 1F1B
  - Interleaved 1F1B
  - Zero Bubble Pipeline
- **3D 并行**：DP + TP + PP 组合
- **SP**（Sequence Parallel）：长序列必备
- **CP**（Context Parallel）：超长上下文

**2. 混合精度训练**
- FP16 / BF16 差异（动态范围 vs 精度）
- Loss Scaling（FP16 必备）
- FP8 训练（Hopper 架构）
- Master Weights（FP32 主权重）

**3. 训练显存优化**
- **激活重计算**（Gradient Checkpointing / Activation Recomputation）
- 选择性重计算
- CPU Offload
- 优化器状态分片

**4. 通信与计算 Overlap**
- TP 中的通信隐藏
- PP 中的 bubble 优化
- ZeRO 的 prefetch 机制

**5. 必读训练论文**
- Megatron-LM 系列（v1/v2/v3）
- ZeRO 系列论文
- GPipe、PipeDream、Zero Bubble
- LLaMA 训练报告
- DeepSeek-V3 训练报告（FP8 训练）

### 🐢 慢补部分（基础功底）

**1. 训练框架源码**
- Megatron-LM 源码
- DeepSpeed 源码
- PyTorch FSDP 源码
- ColossalAI、veScale、TorchTitan

**2. 优化器与训练动力学**
- Adam / AdamW 实现细节
- 学习率调度：warmup、cosine
- 梯度裁剪
- Loss spike 处理
- 训练稳定性技巧

**3. 数据并行细节**
- DataLoader 优化
- 数据混合策略
- Streaming Dataset
- Tokenization 加速

**4. 训练系统工程**
- Checkpoint 保存与恢复
- Failure Recovery
- Elastic Training
- 大规模集群调度

**5. 大规模训练实战经验**
- 通信拓扑优化
- 网络带宽规划
- Storage I/O 优化
- 训练效率分析（MFU / HFU）

**6. RLHF / 后训练（新热点）**
- PPO / DPO / GRPO 算法
- RLHF 训练系统设计
- 多模型协同训练（Actor / Critic / Reward / Reference）
- 框架：OpenRLHF、verl、TRL

---

## 四、整体策略建议

### 学习优先级总结

```
最高优先级 ⭐⭐⭐（公用 ⚡ + 推理 ⚡）
└── 这部分决定能否过面试

中等优先级 ⭐⭐（公用 🐢 + 推理 🐢）
└── 这部分决定面试深度和入职后能否快速上手

较低优先级 ⭐（训练 ⚡）
└── 如果只投推理岗可以放在最后；如果想做全栈则上提

可选 ☆（训练 🐢）
└── 入职后再补也来得及
```

### 路径推荐

**纯推理路径**（推荐你走）
```
公用 ⚡ → 推理 ⚡ → 公用 🐢 → 推理 🐢 → 训练 ⚡（点到为止）
```

**全栈路径**（更卷但天花板更高）
```
公用 ⚡ → 推理 ⚡ + 训练 ⚡ 并行 → 公用 🐢 → 推理 🐢 → 训练 🐢
```

### 关键提醒

1. **公用 ⚡ 是地基**：Attention、KV Cache、Roofline、CUDA 基础——这些不扎实，推理和训练都学不深
2. **推理 ⚡ 性价比最高**：vLLM 源码 + PagedAttention + Continuous Batching + 量化，这套组合拳几乎覆盖 80% 推理面试
3. **训练 ⚡ 选择性学**：投推理岗时面试官会问"是否了解训练"，能讲清楚 ZeRO 和 Megatron TP 就够了，不需要深挖
4. **慢补别拖太久**：慢补不是不学，是节奏放慢；CUDA 和 PyTorch 内部机制至少要持续推进
5. **每学完一块做手撕练习**：光看不写等于白学，这是 ICPC 选手的天然优势，要发挥出来