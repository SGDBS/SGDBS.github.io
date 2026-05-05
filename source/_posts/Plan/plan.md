---
title: AI Infra学习计划
categories: 日记
date: 2026-04-23 22:48:00
mathjax: true
tags:
    - 日记
---


按"先顶层、后底层"的顺序组织，每项都是**具体要学/要做的内容**，不带时间分配。

---

## 第一部分：快车道（面试高频，优先学）

### 1. LLM 推理基础认知

**模型结构**
- Transformer 完整结构：Embedding、MHA、FFN、LayerNorm/RMSNorm、残差连接
- Decoder-only 架构（GPT/LLaMA 系）和 Encoder-Decoder 的区别
- RoPE 位置编码原理与实现
- SwiGLU、GeGLU 等激活函数
- Pre-norm vs Post-norm

**推理两阶段**
- Prefill 阶段：计算特点、并行性、瓶颈
- Decode 阶段：自回归特性、为什么慢
- Prefill 和 Decode 的算术强度差异
- 为什么 Prefill 是 compute-bound，Decode 是 memory-bound

**关键指标**
- TTFT（Time To First Token）
- TPOT / ITL（Inter-Token Latency）
- Throughput（tokens/s）
- Goodput（满足 SLO 的有效吞吐）
- QPS、并发数、SLA 的关系
- Roofline 模型与算术强度计算

### 2. Attention 机制及变种

- **MHA**：标准多头注意力的计算流程、参数量、KV Cache 大小
- **MQA**：Multi-Query Attention，所有 head 共享 KV
- **GQA**：Grouped-Query Attention，分组共享，LLaMA-2/3 使用
- **MLA**：Multi-head Latent Attention，DeepSeek 使用
  - 低秩压缩思想
  - 解耦 RoPE 的设计
  - 为什么比 GQA 省 KV Cache
- **Sliding Window Attention**：Mistral 使用
- **Linear Attention / Lightning Attention**：MiniMax 路线
- **Sage Attention**：W8A8 量化 attention

### 3. KV Cache 相关

- KV Cache 的本质和必要性
- KV Cache 内存占用计算公式
- **PagedAttention**（vLLM）：
  - 借鉴操作系统虚拟内存分页
  - Block Table、Logical/Physical Block 映射
  - 解决内存碎片化
- **RadixAttention**（SGLang）：
  - 基于前缀树的 KV 复用
  - LRU 淘汰策略
- **Prefix Cache**：系统提示词、Few-shot 共享
- KV Cache 量化（INT8、FP8）

### 4. 推理服务调度

- **Static Batching**：传统方式的问题
- **Continuous Batching / In-flight Batching**：
  - Orca 论文核心思想
  - Iteration-level scheduling
- **Chunked Prefill**：
  - 长 prefill 切分
  - 与 decode 混合调度
  - 对 TTFT 和 TPOT 的影响
- **Priority Scheduling**：优先级调度
- **PD 分离架构**：
  - DistServe 设计思路
  - Mooncake 架构
  - KV Cache 传输策略

### 5. 解码策略

- Greedy / Top-k / Top-p / Temperature
- Beam Search 及其在 LLM 中的应用
- **Speculative Decoding**：
  - 数学正确性证明
  - Draft Model + Target Model
- **EAGLE / EAGLE-2 / EAGLE-3**：原理与实现
- **Medusa**：多头并行预测
- **MTP**（Multi-Token Prediction）：DeepSeek-V3 内置投机
- **Lookahead Decoding**

### 6. 量化与压缩

**量化基础**
- 对称 vs 非对称量化
- Per-tensor / Per-channel / Per-group 量化
- W8A8、W4A16、W4A8、FP8 各种组合的适用场景

**主流量化方法**
- **GPTQ**：基于 OBQ 的逐层量化
- **AWQ**：Activation-aware Weight Quantization
- **SmoothQuant**：迁移激活值难度到权重
- **SpinQuant**：旋转矩阵抑制 outlier
- **FP8**：E4M3 / E5M2 格式

**KV Cache 量化**
- INT8 / INT4 KV Cache
- 动态 vs 静态量化

**其他压缩技术**
- 知识蒸馏（了解即可）
- 结构化稀疏 / 非结构化稀疏（了解即可）
- 投机解码可视为一种"软压缩"

### 7. 并行策略（推理视角）

- **TP**（Tensor Parallelism）：
  - Megatron 的 column / row parallel
  - QKV、FFN 怎么切
  - 通信原语 AllReduce / AllGather
- **DP**（Data Parallelism）：推理中通常指 replica
- **EP**（Expert Parallelism）：MoE 专用
- **PP**（Pipeline Parallelism）：推理中较少用
- **SP**（Sequence Parallelism）：长序列场景
- **Attention DP + FFN TP** 混合策略
- 各种并行的通信开销分析

### 8. MoE 相关

- MoE 基本结构：Router、Expert、Top-k 选择
- Token Choice vs Expert Choice
- **All-to-All 通信**：MoE 推理的核心瓶颈
- 负载均衡：Auxiliary Loss、Aux-Loss-Free（DeepSeek）
- **EPLB**（Expert Parallel Load Balancer）：DeepSeek 开源
- Shared Expert 设计
- DeepSeek-V3、Qwen-MoE、Mixtral 的 MoE 对比

### 9. 推理框架

**vLLM 源码精读路径**
- `LLM.generate()` 调用链
- `Scheduler`：请求调度逻辑
- `BlockManager`：KV Cache 块管理
- `ModelRunner`：模型前向调度
- `AttentionBackend`：FlashAttention / FlashInfer 集成
- `Worker`：分布式 worker
- 整体架构图能默写

**SGLang 核心**
- RadixAttention 实现
- Frontend 语言（结构化生成）
- 与 vLLM 的设计差异

**其他框架了解**
- TensorRT-LLM：NVIDIA 官方，性能最强
- LMDeploy：商汤开源
- llama.cpp：CPU / 边缘端
- MLC-LLM：跨平台

### 10. 高性能算子（必备硬实力）

**必须能手写的 kernel**
- Vector Add（入门）
- Reduce（7 个版本，Mark Harris PPT）
- Softmax / Online Softmax / Safe Softmax
- LayerNorm / RMSNorm
- GEMM：naive → tiled → shared memory → tensor core
- Element-wise 算子融合
- Attention（简化版 FlashAttention）

**算子融合**
- QKV projection 融合
- RMSNorm + Residual 融合
- SwiGLU 融合
- Attention + Bias + Mask 融合

**FlashAttention 系列**
- v1：Tiling + Online Softmax + Recomputation
- v2：减少非矩阵乘运算、warp 级并行优化
- v3：Hopper 架构（WGMMA、TMA、异步）
- 能讲清楚 forward 和 backward 的差异

### 11. 硬件与通信

**GPU 架构**
- SM、Warp、Thread Block、Grid 层级
- 内存层级：Register、Shared Memory、L1/L2、Global、HBM
- Tensor Core 的使用
- NVIDIA 架构演进：Volta → Turing → Ampere → Hopper → Blackwell
- H100 / H200 / B200 关键参数（带宽、算力、显存）

**通信**
- AllReduce、AllGather、ReduceScatter、Broadcast、All-to-All
- Ring vs Tree 算法
- NCCL 基本使用
- NVLink、NVSwitch、PCIe、InfiniBand、RDMA
- 通信与计算 overlap

**国产硬件（选学）**
- 昇腾、寒武纪、壁仞、摩尔线程基本了解

### 12. 手撕题准备

- **LeetCode Hot 100**（保持手感）
- **LeetGPU**：Reduce、GEMM、Softmax、Attention
- **算法手撕**：
  - MHA forward（含 KV Cache 版本）
  - GQA / MLA 实现
  - Decoder Layer 完整实现
  - MoE Top-k Routing
  - RoPE 实现
  - Sampling 实现（top-k / top-p）
  - Beam Search

---

## 第二部分：慢车道（底层基础，长期积累）

### 1. CUDA 编程

**入门**
- CUDA 编程模型：Thread / Block / Grid
- Kernel 启动、同步原语
- 内存类型与使用
- Stream 与异步执行

**进阶**
- Shared Memory 优化、Bank Conflict
- Memory Coalescing
- Warp 级原语：`__shfl_sync`、`__ballot_sync`
- Occupancy 分析与优化
- Async Copy（`cp.async`）
- Tensor Core 编程（WMMA / MMA）

**高级**
- CUTLASS / CuTe 框架
- PTX 阅读
- SASS 阅读（深度调优）
- CUDA Graph

**工具**
- Nsight Compute：kernel 级性能分析
- Nsight Systems：系统级 timeline 分析
- `nvcc` 编译选项与优化

### 2. Triton

- Triton 编程模型与 CUDA 的差异
- `tl.program_id`、`tl.load`、`tl.store`
- Tiling 与 Block Pointer
- Autotuning
- 用 Triton 写 GEMM、Softmax、FlashAttention
- Triton 与 PyTorch 集成

### 3. 计算机系统基础

**CSAPP 重点章节**
- Ch3：机器级表示
- Ch5：程序优化（Profile、流水线、SIMD）
- Ch6：存储层级（Cache 原理）
- Ch9：虚拟内存（理解 PagedAttention 必备）
- Ch12：并发编程

**体系结构**
- 流水线、乱序执行、分支预测
- Cache 一致性、内存模型
- NUMA 架构
- 《计算机体系结构：量化研究方法》选读

### 4. C++ 进阶

- 现代 C++（11/14/17/20）特性
- 模板与模板元编程
- 移动语义与右值引用
- 智能指针
- 多线程：`std::thread`、`std::atomic`、内存序
- pybind11：Python 绑定
- CMake 构建系统

### 5. PyTorch 内部机制

- Tensor / Storage / Stride
- Autograd：动态图、反向传播
- Dispatcher 机制
- ATen / c10
- 自定义 CUDA 算子（torch.utils.cpp_extension）
- `torch.library` 注册算子
- **torch.compile**：Dynamo / AOTAutograd / Inductor
- CUDA Graph 与 PyTorch 集成

### 6. 编译器与中间表示（选学）

- LLVM 基础
- MLIR 概念与 Dialect
- TVM：调度原语、AutoTVM
- XLA：HLO IR
- TorchInductor 工作流程

### 7. 训练系统（如果想了解全栈）

**分布式训练**
- DDP（Distributed Data Parallel）
- ZeRO 1/2/3（DeepSpeed）
- FSDP（Fully Sharded Data Parallel）
- Megatron-LM 的 3D 并行
- 流水线并行：GPipe、1F1B、Interleaved 1F1B、Zero Bubble

**训练优化**
- 混合精度训练（FP16 / BF16 / FP8）
- 梯度累积
- 激活重计算（Gradient Checkpointing）
- Optimizer 状态分片

**框架**
- Megatron-LM 源码
- DeepSpeed 源码
- PyTorch FSDP 源码
- ColossalAI、veScale

---

## 第三部分：必读论文

### 经典必读
- Attention is All You Need（Transformer）
- GPT-3 / LLaMA / LLaMA-2 / LLaMA-3 技术报告
- FlashAttention v1 / v2 / v3
- PagedAttention（vLLM 论文）
- Orca（Continuous Batching）
- Megatron-LM 系列（即使做推理也要懂）

### 推理优化
- DistServe（PD 分离）
- Mooncake（Kimi 的架构）
- SARATHI（Chunked Prefill）
- SGLang / RadixAttention
- Splitwise（微软 PD 分离）

### 投机解码
- Speculative Decoding（Google / DeepMind 原始论文）
- Medusa
- EAGLE / EAGLE-2 / EAGLE-3
- Lookahead Decoding

### 量化
- GPTQ
- AWQ
- SmoothQuant
- LLM.int8()
- KIVI（KV Cache 量化）

### Attention 变种
- MQA、GQA 论文
- DeepSeek-V2（MLA）
- DeepSeek-V3（MTP）

### MoE
- Switch Transformer
- DeepSeek-MoE
- DeepSeek-V3
- Mixtral 技术报告

---

## 第四部分：项目（至少完成 1 个）

### 难度递增
1. **手写 GEMM**：从 naive 到 cuBLAS 70%+ 性能，配 Nsight 分析报告
2. **复现 FlashAttention**：用 Triton 或 CUDA，对比官方性能
3. **迷你 vLLM**：实现 PagedAttention + Continuous Batching
4. **量化工具**：实现 GPTQ 或 AWQ，量化一个 7B 模型
5. **vLLM / SGLang PR**：解决 issue 或新增 feature
6. **复现一篇 SOTA 论文**：DistServe / Mooncake / EAGLE 等

---

## 第五部分：信息源（持续跟进）

### 中文资源
- 知乎专栏：方佳瑞、游凯超、SiriusNEO、马骏 等推理领域博主
- 公众号：AI闲谈、GiantPandaLLM、关于NLP那些你不知道的事
- 苏剑林博客（科学空间）：算法原理讲解最透
- 微信公众号：DeepSeek、智源、潞晨等技术团队官方号

### 英文资源
- HuggingFace 博客
- vLLM / SGLang 官方博客
- NVIDIA Developer 博客
- PyTorch 官方博客
- Lilian Weng 博客（OpenAI）

### 论文跟踪
- arXiv：cs.DC、cs.LG、cs.AR
- 顶会：MLSys、OSDI、SOSP、ATC、ASPLOS、ISCA、NSDI
- Papers With Code：LLM Inference 类目

### 代码仓库（Star 关注更新）
- vllm-project/vllm
- sgl-project/sglang
- NVIDIA/TensorRT-LLM
- Dao-AILab/flash-attention
- NVIDIA/cutlass
- triton-lang/triton
- deepseek-ai/* 系列

---

## 学习内容自查清单

学完一个模块后问自己：

- [ ] 能用一段话向非专业人士解释这是什么吗？
- [ ] 能向面试官 5 分钟讲清楚原理和优化点吗？
- [ ] 能在白板上手写核心代码或画核心架构图吗？
- [ ] 知道这个技术的来源论文和当前 SOTA 吗？
- [ ] 知道它的局限性和改进方向吗？