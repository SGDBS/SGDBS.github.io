---
title: 0.5 GPU Kernel 学习路线图:CUDA 与 Triton 从入门到生产
categories: 学习笔记- AI Infra
date: 2026-05-09 14:00:00
mathjax: true
tags:
    - AI
    - AI Infra
    - CUDA
    - Triton
---


这是一份**与 AI Infra 主线并行**的 GPU 算子学习路线。前面 Chapter0 讲的是"模型怎么散到几百张卡上",这一份讲的是"每张卡上的 kernel 怎么贴近硬件极限"。两条线互相催化——只学 Infra 容易变成"调包侠",只学 Kernel 容易变成"单机仙人",**配合学才能在 2026 年的大模型工程里站得住**。

文档本身是一张地图:把整个学习过程切成 9 章,每章给出**必须掌握的知识点**和**对应的动手任务**。读者(包括我自己)按这张图走,一边学一边写笔记,每完成一章就在对应的章节文件夹里产出一篇 markdown。

## 一、为什么需要这条线

### 1.1 Infra 上限被算子抬

前面 AI Infra Ch1 写完之后,你会发现一个反直觉的事实:**几乎所有 Infra 优化的天花板,最终都被某个具体 kernel 的速度决定**。

- DDP 的反向重叠能藏多少通信?——藏不下的部分恰好是反向 kernel 太快、AllReduce 来不及发完的那一段
- ZeRO-3 的 prefetch 能不能完全掩盖 AllGather?——取决于这一层 forward kernel 跑多久
- FlashAttention 让长序列训练变得可能,**直接重新定义了 Pipeline / TP 的切分阈值**

不懂 kernel,你就只能在"调 batch_size、调 bucket_cap、改 ZeRO stage"这些 framework 旋钮上打转;懂了之后,你能写出真正改变性能曲线的优化。

### 1.2 算子工程师的 Infra 盲区

反过来也成立。只会写单 kernel 的人,经常做出在单卡 benchmark 里很漂亮、但放进真实训练里没用的优化——因为他不知道这个 kernel 在整个训练 step 中只占 3%,或者它的输出张量会被立刻送进通信通道、根本不需要那么"完美"。

**Kernel 设计必须服务于 Infra 的全局调度**,不然就是孤岛。

### 1.3 这条路线想让你长成什么样

走完这 9 章,你应该:

- 能用 Triton 独立写出一个 attention kernel,达到 PyTorch SDPA 60-80% 性能
- 能读懂 FlashAttention / Liger Kernel / ThunderKittens 等主流仓库的 Triton 代码
- 能用 ncu / nsys profile 出任何 kernel 的瓶颈,**判断是 compute-bound 还是 memory-bound**
- 能把自己写的 kernel 注册成 PyTorch custom op,无缝接进 DDP/FSDP 训练流程
- 看到一篇新论文(Mamba、Linear Attention、新的 MoE)的 kernel,能在一周内自己复现一个 80% 性能版本

不要求亲手写满血 CUDA(那是另一个领域,需要再花半年)——但**要能读懂 CUDA**,因为大量 production 代码、cuBLAS / cuDNN / NCCL 的实现都还是 CUDA。

## 二、战略选择:为什么主修 Triton

2026 年的现实:**新写的 GPU kernel,80% 是 Triton**。FA2/FA3、Mamba、MoE expert kernel、各种 fused norm/RoPE——主流仓库的"性能关键代码"几乎清一色 Triton 文件。原因:

- **生产力差 5-10 倍**:同一个 kernel,Triton 100 行,CUDA 800 行
- **性能损失 < 15%**:对绝大多数 kernel,Triton 编译出来的代码已经接近手写 CUDA 极限
- **可读性远好**:三年后回头看自己的 Triton 代码还能改;三年后看自己的 CUDA 代码就跟看天书差不多

但 **CUDA 心智模型必学**——不是亲手写,而是知道 SM、warp、shared memory、HBM 这些底层名词在指什么。Triton 只是把 CUDA 的 thread-level 编程换成 block-level,但**底下还是同一套硬件**,完全不懂 CUDA 的人写出来的 Triton 也是瞎调。

所以路线图安排:**Ch1-2 先把硬件 + CUDA 基础打住,Ch3 起切到 Triton 主战场**,Ch9 再回头看 CUDA 的高级特性(到那时你已经有需求了,学得快)。

## 三、前置知识

和 AI Infra 几乎完全一样,所以你边学 Infra 边学 Kernel 没有额外门槛:

- **PyTorch 熟练**:能写完整 train loop
- **Transformer 数学**:attention / layernorm / softmax 的**前向 + 反向**公式都能推。FA 反向写不出来 90% 是因为 softmax/matmul 反向没推熟
- **C 基础语法**:if / for / 指针,不需要会 OOP / 模板
- **基本 Linux**:能用 nvidia-smi、能跑 bash

不要求会的(路线里会顺便补):
- GPU 硬件细节:Ch1 系统讲
- CUDA 语法:Ch2 速通
- 性能分析工具(ncu / nsys):Ch8 专门讲

## 四、九章结构总览

```
Ch1.  GPU 硬件心智模型              ← 没硬件直觉,后面全是死记硬背
Ch2.  CUDA 编程基础                ← 能读 CUDA 代码,会写最简单 kernel
Ch3.  Triton 入门                  ← 主力工具登场
Ch4.  矩阵乘:GPU 编程的基石         ← 占整个领域 50% 心智份额
Ch5.  Reduction 与归一化           ← softmax / layernorm 的工程化
Ch6.  Fused Kernel 设计            ← 现代 LLM 的吞吐源泉
Ch7.  FlashAttention 实战          ← 集大成,写出来就出师
Ch8.  Profile, Roofline 与生产集成  ← 让 kernel 进入真实训练流程
Ch9.  CUDA 深入(可选)            ← Triton 撞墙时的工具
```

预估时长:**全职学 8-12 周,业余学 4-6 个月**。Ch4 和 Ch7 是两个明显的难度跳跃,各预留 2 周。

## 五、各章详细学习目标

### Ch1. GPU 硬件心智模型

**核心问题**:GPU 和 CPU 在硬件设计上为什么完全不同?为什么写得对的 GPU 代码能比 CPU 快 100 倍,写错了又能慢 10 倍?

**知识点**:

- GPU vs CPU 的设计哲学:**高吞吐 vs 低延迟**,核心数 vs 单核能力
- **SM (Streaming Multiprocessor)** 是什么,一张 H100 / A100 / 4090 各有几个
- 线程层级:**Thread → Warp(32 个 thread)→ Thread Block → Grid**
- 内存层级与带宽数字感:
  - Register / SRAM(shared memory)/ L1 / L2 / HBM
  - HBM ~1-3 TB/s,SRAM ~19 TB/s,**两个数差 10 倍是所有 kernel 优化的起点**
- **Tensor Core vs CUDA Core**:为什么 BF16 比 FP32 快 8 倍
- Warp scheduler 与 occupancy 概念
- **Compute-bound vs memory-bound**:用 arithmetic intensity 判断
- Roofline model 直觉版

**任务**:

- [ ] 读 PMPP(*Programming Massively Parallel Processors*)4th edition Ch 1-3
- [ ] 看 GPU Mode 社区前 3 节录播(Mark Saroufim 等组织,YouTube 上有完整录像)
- [ ] 跑 `nvidia-smi -q`,**画一张自己 GPU 的硬件图**(SM 数、SRAM/Register 容量、HBM 带宽)
- [ ] 用 nvidia-smi 数字,算出自己 GPU 的 BF16 算力 / HBM 带宽比(即 arithmetic intensity 阈值,memory-bound 与 compute-bound 的分界线)
- [ ] 写一篇笔记 `Ch1.HardwareMentalModel.md`,把这套层级讲清楚

### Ch2. CUDA 编程基础

**核心问题**:在 CUDA 里写一个最简单的 kernel,需要哪些零件?为什么读 CUDA 代码不用想着"从头实现"——只需要做到能看懂。

**知识点**:

- nvcc 编译流程:`.cu` → PTX → SASS
- **Kernel function 语法**:`__global__`、`__device__`、`__host__` 的区别
- 索引体系:`threadIdx / blockIdx / blockDim / gridDim`
- Kernel launch 语法:`kernel<<<grid, block>>>(args)`
- 内存管理:`cudaMalloc / cudaMemcpy / cudaFree`
- 同步:`__syncthreads()`(block 内)、`cudaDeviceSynchronize()`(host)
- Shared memory:`__shared__` 修饰符
- 常见错误处理:`cudaGetLastError()`,asynchronous error 的延迟暴露
- **简单了解**:Tensor Core 的 PTX 指令(`mma.sync`)、warp shuffle(`__shfl_*`)

**任务**:

- [ ] 装好 CUDA toolkit,nvcc 跑通 `hello_world.cu`
- [ ] **CUDA 写 vector_add**:理解 grid/block 怎么分
- [ ] **CUDA 写 transpose**:naive 版本 + shared memory 版本,profile 对比(应该能看到 shared memory 版本快 5-10 倍)
- [ ] 读 PMPP Ch 4-5
- [ ] 找一段 cuBLAS sgemm 的反汇编(`cuobjdump --dump-sass`),**不要求看懂全部,但能认得出 fmla / lds / sts 这些 SASS 指令**
- [ ] 写笔记 `Ch2.CUDABasics.md`

### Ch3. Triton 入门

**核心问题**:Triton 比 CUDA 高效在哪?它是怎么把"thread 级编程"换成"block 级编程"的?

**知识点**:

- Triton 与 CUDA 的关系:Triton 编译器把 Triton IR 翻译成 PTX
- 安装与环境:`pip install triton`,确认 GPU 兼容
- 核心装饰器:`@triton.jit`、`@triton.autotune`
- **Block-level programming**:你只写"一个 block 在做什么",thread 级调度由编译器处理
- API 入门:`tl.program_id`、`tl.arange`、`tl.load`、`tl.store`、`tl.dot`
- Pointer arithmetic:Triton 用裸指针,要自己算 offset
- **Mask 处理边界**:这是 Triton 最常见的 bug 来源
- 三个调优旋钮:`BLOCK_SIZE`、`num_warps`、`num_stages`
- Triton 与 CUDA 写法的对照表

**任务**:

- [ ] 跑通 Triton 官方 tutorial 01-02(vector_add、softmax)
- [ ] 把 Ch2 的 vector_add 改写成 Triton,profile 对比 CUDA 版,**期望性能基本一致**
- [ ] 跑通 Sasha Rush *GPU Puzzles* 前 5 题(github 上的 srush/GPU-Puzzles)
- [ ] **完成 Triton-Puzzles 前 10 题**(srush 的姊妹项目)
- [ ] 自己写一个简单的 dropout kernel(Triton)
- [ ] 写笔记 `Ch3.TritonIntro.md`,**做一张 CUDA ↔ Triton 对照表**

### Ch4. 矩阵乘:GPU 编程的基石

**核心问题**:为什么 matmul 是 GPU 编程的"通用衡量尺"?同样一个矩阵乘,怎么从 cuBLAS 5% 性能爬到 80%?

**知识点**:

- Block matrix multiplication 数学回顾
- **Tiling 思想**:为什么要分块,M、N、K 各自怎么切
- Shared memory blocking:把 tile 加载进 SRAM 后多次复用
- **Bank conflict**:shared memory 的访问模式陷阱
- Tensor Core 触发条件(Triton 自动判定,但要知道何时不触发)
- Persistent kernel pattern
- K-loop 展开 / Software pipelining
- **Split-K**:小矩阵的瘦长 K 怎么处理
- Roofline 应用:matmul 在哪些 size 下是 memory-bound

**任务**:

- [ ] **Naive matmul (Triton)**:每次访存乘加,profile 看是 compute-bound 还是 memory-bound(基本是 memory-bound)
- [ ] **Tiled matmul (Triton)**:用 `BLOCK_M`、`BLOCK_N`、`BLOCK_K` 分块,**目标:打到 cuBLAS 50% 性能**
- [ ] 调优 BLOCK 大小,观察 ncu profile 变化(occupancy、SRAM 利用率)
- [ ] 加 `num_stages` 看流水线效果,**目标:cuBLAS 70-80% 性能**
- [ ] (可选)用 CUDA 写一个 simple version,与 Triton 性能对比
- [ ] (可选)实现 split-K 优化,处理小矩阵
- [ ] 写笔记 `Ch4.Matmul.md`,**附性能对比表 + ncu profile 截图**

**这一章是最大的难度跳跃,预留 2 周。完成后你已经站到 GPU 编程的入门线之上。**

### Ch5. Reduction 与归一化

**核心问题**:softmax / layernorm 这些"看起来很简单"的算子,在 GPU 上怎么写才不慢?Reduction 这个模式的并行套路是什么?

**知识点**:

- Reduction 的并行模式:**tree reduction**(对数复杂度)
- Warp-level reduction(warp shuffle,`__shfl_xor` / `__shfl_down`)
- Block-level reduction(shared memory + sync)
- Multi-block reduction 的两种方式:**atomic add** vs **二次 reduce**
- **Numerical stability**:softmax 的 `-max` trick,为什么必须
- Online algorithms:**Welford algorithm** 一遍算出 mean + var
- One-pass vs two-pass softmax
- LayerNorm vs RMSNorm:为什么 LLM 偏爱后者

**任务**:

- [ ] **Sum reduction (Triton)**:atomic add 版本和 two-pass 版本各写一遍,profile 对比
- [ ] **Softmax (Triton)**:带 numerical stability,处理变长 mask
- [ ] **LayerNorm forward (Triton)**:用 Welford 一遍算 mean + var
- [ ] **LayerNorm backward (Triton)**:**梯度公式自己推一遍**(对 γ、β、x 各求一次)
- [ ] **RMSNorm 前向 + 反向**(LayerNorm 的简化,LLM 必备)
- [ ] Profile 看 SRAM 利用率,与 cuBLAS / PyTorch 比较吞吐
- [ ] 写笔记 `Ch5.ReductionAndNorm.md`

### Ch6. Fused Kernel 设计

**核心问题**:什么样的 op 该 fuse,什么时候不该?为什么 fused kernel 是现代 LLM 训练的吞吐源泉?

**知识点**:

- **为什么 fuse**:省 kernel launch + 省 HBM round-trip
- 算密度阈值:何时 fuse 收益最大(memory-bound 时)
- 经典 fusion 模式:
  - Elementwise chain(GeLU + dropout + bias_add)
  - GEMM + epilogue(matmul 后接 activation)
  - 内层 fusion(attention 内部)
- **RoPE**:在哪个层级 fuse,与 Q/K projection 的关系
- **SwiGLU/GeGLU**:gate 计算 + 主计算的 fusion
- Auto-fusion(`torch.compile`)vs hand-fusion 的边界
- 常见的 fused kernel 库:Liger Kernel、ThunderKittens、Apex

**任务**:

- [ ] 写 **fused dropout + GeLU + bias_add (Triton)**:对比"三个独立 kernel"和"一个 fused kernel"的吞吐,**期望提速 1.5-3 倍**
- [ ] 写 **RoPE (Triton)**:前向 + 反向
- [ ] 写 **fused RMSNorm + 一段 elementwise**(模拟 LLM 中常见的 norm + add residual 模式)
- [ ] 读 **Liger Kernel** 或 **ThunderKittens** 仓库的 README 和一两个核心 kernel,学他们的 fusion 风格
- [ ] (可选)用 `torch.compile` 跑同一段 op,对比手写 Triton 和编译器自动 fusion
- [ ] 写笔记 `Ch6.FusedKernel.md`

### Ch7. FlashAttention 实战

**核心问题**:为什么 standard attention 慢的根源是 HBM I/O?Online softmax 怎么把 attention 矩阵彻底从内存里移除?Flash 反向为什么是"recomputation 友好"?

**知识点**:

- Standard attention 的内存访问模式:为什么 attention 是 memory-bound 而不是 compute-bound
- **Tiling for attention**:Q 行、K/V 列怎么分块
- **Online softmax 的代数推导**(这是最关键的数学,先吃透再写代码)
- **Recomputation in backward**:为什么不存中间 softmax 矩阵,反向时按 tile 重算反而更快
- 工程处理:causal mask、KV cache、varlen(变长序列)、ALiBi、softcap
- FA1 / FA2 / FA3 的演进:
  - FA1:首版,确立 tiling + online softmax + recomputation
  - FA2:换轴(Q 当外层),提升 Tensor Core 利用率
  - FA3:Hopper 上的 TMA 和 wgmma 充分利用

**任务**:

- [ ] **Pre-FA**:实现 standard attention(Triton),profile 观察 attention matrix 占多少 HBM 写
- [ ] **FA1 forward (Triton)**:tiling + online softmax,对照 Tri Dao 的论文
- [ ] **FA1 backward (Triton)**:**真正的硬骨头**,自己推 dQ / dK / dV 公式,然后写出来
- [ ] 与 PyTorch SDPA 性能对比,**目标 SDPA 60-80% 性能**
- [ ] 加 causal mask 支持
- [ ] (高阶)读 FA2 paper,理解为什么换轴
- [ ] (高阶)读 `flash-attn` 仓库的 Triton 实现,看 production 代码处理多少 corner case
- [ ] 写笔记 `Ch7.FlashAttention.md`,**这一篇可以写得很长,把数学推导也放进来**

**这一章是路线图的 capstone。完成后你已经是行业里"能写算子的人"。预留 2-3 周。**

### Ch8. Profile, Roofline 与生产集成

**核心问题**:你写的 kernel 怎么进入真实 LLM 训练?怎么科学诊断瓶颈而不是盲调?

**知识点**:

- **三件套**:
  - `nsys` (Nsight Systems):看 timeline、kernel launch、stream 占用
  - `ncu` (Nsight Compute):看单 kernel 的 occupancy、SRAM、寄存器
  - `torch.profiler`:看 PyTorch 层 wall-time
- **Roofline model**:画 arithmetic intensity vs FLOPS 图,判断瓶颈
- 常见瓶颈模式:
  - Occupancy 低 → register pressure 太大
  - SRAM 利用率低 → tile 太小
  - HBM 带宽打满 → memory-bound,fuse 上游
- **PyTorch 集成 API**:
  - `torch.library.custom_op` 注册自定义 op
  - `torch.autograd.Function` 注册反向
  - 与 `torch.compile` 的协作:`@torch.library.custom_op("ns::name", mutates_args=())`
- **与分布式训练的连接**:
  - 把 Triton kernel 用进 DDP 的 comm hook
  - 与 FSDP 的 mixed precision 协作
  - 与 CUDA Graph 的协作

**任务**:

- [ ] 用 **ncu profile** Ch4 写的 matmul,识别 occupancy / SRAM 利用率瓶颈,迭代一版优化
- [ ] 用 **nsys** profile 一个真实训练 step 的 timeline,数 kernel launch 个数
- [ ] 把 Ch7 的 FA kernel **注册成 PyTorch custom op**,通过 `torch.library` 替换掉默认 SDPA
- [ ] 跑通完整 LLM 训练循环(forward + backward + optimizer),loss 曲线和 PyTorch SDPA 一致
- [ ] **对接 AI Infra Ch1**:写一个 Triton-based 的 BF16 AllReduce comm hook,在 DDP 下跑通(参考 Ch1 DDP §7.2)
- [ ] 写笔记 `Ch8.ProfileAndIntegration.md`,**附完整 nsys / ncu 截图与解读**

### Ch9. CUDA 深入(可选)

**核心问题**:Triton 撞墙时怎么办?那些"非要 CUDA 不可"的场景长什么样?

只有在前 8 章都做完、**确实碰到 Triton 性能差距的场景**时再来这一章。绝大多数人第一年用不到。

**知识点**:

- Warp-level primitives:`__shfl_*`、`__ballot_*`、`__any/all_*`
- Cooperative groups:`thread_block`、`thread_block_tile`
- **Async copy**:`cp.async`(Ampere+)、`cp.async.bulk`(Hopper)
- **TMA (Tensor Memory Accelerator)**:Hopper 的硬件级数据搬运
- **Distributed shared memory**:Hopper 的 cluster 概念
- Tensor Core MMA at PTX level:`mma.sync.aligned.*`
- **Persistent kernel + producer/consumer pattern**(FA3 用的)
- 与 cuBLAS / cuDNN 的 interop

**任务**:

- [ ] 用 **warp shuffle** 改写 Ch5 的 reduction,profile 看快了多少
- [ ] 用 **cp.async** 改进 Ch4 的 matmul,看能否让 `num_stages` 开得更大
- [ ] (高阶)读一段真实的 cuBLAS GEMM 反编译代码,**不要求懂全部,但能识别主要 phases**
- [ ] (高阶)阅读 FA3 的 CUDA 实现,理解 TMA 和 wgmma 的用法
- [ ] 写笔记 `Ch9.CUDADeepDive.md`,**只写你真正用到的**,不堆砌

## 六、与 AI Infra 的交叉学习时间表

**强烈建议两条线交替推进**——同一周学一段 Infra 配一段 Kernel,概念互相印证,记得快忘得慢。

```
                 AI Infra                      GPU Kernel
─────────────────────────────────────────────────────────────────────
Week 1   Ch1 MemoryBudget                Ch1 GPU 硬件心智模型
Week 2   Ch1 DDP (主体)                  Ch1 完成 + GPU Puzzles 1-10
Week 3   Ch1 DDP (深入) + AsyncCompute   Ch2 CUDA 基础 + vector_add
Week 4   Ch1 MixedPrecision              Ch3 Triton 入门
Week 5   Ch1 ZeRO                        Ch4 matmul (开干)
Week 6   Ch1 MultiNode                   Ch4 matmul (打到 cuBLAS 70%)
Week 7   Ch1 Checkpoint                  Ch5 reduction + softmax
Week 8   Ch1 复习 / 过 PR                 Ch5 layernorm 前后向
Week 9   Ch2 PipelineParallel (开始)     Ch6 fused kernel
Week 10  Ch2 PipelineParallel            Ch7 FA1 forward (开始)
Week 11  Ch2 PipelineParallel            Ch7 FA1 forward (调优)
Week 12  Ch3 TensorParallel (开始)       Ch7 FA1 backward (硬骨头)
Week 13  Ch3 TensorParallel              Ch7 FA1 backward + 与 SDPA 对比
Week 14  Ch3 TensorParallel              Ch8 profile + custom op
Week 15  Ch4 GPUKernel (Infra 视角)      Ch8 完成 + 集成进 LLM
之后      Ch5 Inference                   Ch9 CUDA 深入(按需)
```

**横向连接的几个高光时刻**:

- Week 5:Infra 的 ZeRO AllGather + Kernel 的 matmul tile,**两者本质都是"把数据拉进局部 buffer 然后多次复用"**——一个发生在 GPU 之间,一个发生在 SRAM 内部
- Week 7:Infra 的 RingAllReduce 的 tree reduction + Kernel 的 warp reduction,**同一种算法在不同尺度上的体现**
- Week 12-13:写完 FA backward 之后再回头看 Infra Ch1 的 activation checkpointing,你会发现"重算换显存"这个思想已经贯通从 kernel 内部到训练循环外部
- Week 15:把自己写的 FA 注册成 custom op 接进训练,**那一刻 Infra 和 Kernel 在你脑子里彻底打通**

## 七、资源清单(按 Tier 分级)

### Tier 1 必备

- **PMPP**(*Programming Massively Parallel Processors* 4th edition):Wen-mei Hwu / David Kirk 经典,Ch 1-7 反复读
- **Triton 官方 tutorials**:github 上 `triton-lang/triton` 仓库的 `python/tutorials/`,从 01 走到 09
- **Sasha Rush 的 GPU Puzzles**:github 上 `srush/GPU-Puzzles`,互动式
- **Sasha Rush 的 Triton-Puzzles**:同上但专门 Triton
- **GPU Mode**(原 CUDA Mode)社区:Discord + YouTube 录播,2024 年起最活跃的现代 GPU 学习社区

### Tier 2 高度推荐

- **Horace He** *Making Deep Learning Go Brrrr From First Principles*:讲性能直觉,**Ch1-4 完成后再读**,会有大量"原来如此"的瞬间
- **Lei Mao 博客**(leimao.github.io):CUDA 与 Triton 短文集,需要某个具体概念时搜
- **Tri Dao** 的 FlashAttention 系列论文(v1 / v2 / v3):每读一遍多懂一层
- **Liger Kernel** / **ThunderKittens** 源码:看 production fused kernel 怎么写的
- **NVIDIA CUDA C Programming Guide**:**当字典用,不要从头读**

### Tier 3 进阶 / 按需

- **CUTLASS** 仓库:NVIDIA 的高性能 GEMM template 库,看顶级 CUDA 是什么样
- **PTX ISA 文档**:写 Triton 调优时偶尔需要看生成的 PTX
- **SASS / NVDisasm**:看反汇编,Ch9 才用到
- **Pearl Cao**、**Aleksa Gordić**、**Mark Saroufim** 等人的个人博客 / Twitter:跟近期 trends

## 八、几条避坑准则

把前面 Infra 学习的经验也搬过来:

**第一,不要从读 cuBLAS / cuDNN 源码开始**。它们是工业产品,一万个分支处理 corner case,初学者读完只会 demoralized。从干净的教学代码开始(Triton tutorials、PMPP 例子、GPU Puzzles)。

**第二,不要追求一上来就 SOTA**。第一版 matmul 跑到 cuBLAS 30% 性能就很好了,**理解为什么剩下 70% 在哪里损失**,比硬调参重要。Tri Dao 写出 FA1 之前自己也写过几十版垃圾版本。

**第三,Profile 工具早用早受益**。`ncu` 第一次看像天书,但坚持用一周就好。**它会直接告诉你"这个 kernel 的瓶颈是 memory bandwidth 不是 compute"**——这个信息光看代码看不出来。从 Ch4 开始每个任务都要 profile。

**第四,数学先推清楚再写 kernel**。FlashAttention 反向写不出来 90% 是因为 softmax 反向公式没推熟。**反向公式抄一遍、自己手算一个 3×3 例子,再去写 kernel**。

**第五,加入社区**。GPU Mode Discord 是 2024 年以来这个领域最活跃的地方,FA、Mamba 等论文作者都在那写 paper reading。**跟着读三个月你的视野会大变**。

## 九、附录:ICPC / 强 C++ 背景的修订路径

前面 §二 默认推荐"Triton 主修,CUDA 学心智模型"——这个建议是为**PyTorch 出身、几乎不写 C++** 的读者准备的。如果你来自 ICPC、OI、或者其他重度 C++ 竞赛/工程背景,这个默认路径反而是浪费——你的优势是 Triton 完全用不上的。

这一节单独写一个**修订版路径**,只针对这类读者。

### 9.1 为什么默认计划不适合你

CUDA 比 Triton 多出来的"门槛",对一般 PyTorch 用户是巨大障碍,对你几乎不存在:

- **C++ 语法**:对你为 0。别人需要一周适应的指针、模板、内存管理,你睡着都会写
- **手动管理寄存器 / shared memory**:这正是你 ICPC 里手写线段树、莫队、cache-aware 排序时一直在做的事
- **constant factor 优化思维**:CUDA 编程 80% 是 constant factor,你的"卡常"经验直接迁移
- **读别人的代码**:CUTLASS 模板、cuBLAS 反汇编、Tri Dao 的 FA 实现——别人觉得"看不懂",你会觉得"和读 ICPC 神仙代码差不多"

反过来,Triton 对你的吸引力反而小了一些——它的 block-level 抽象**屏蔽了 thread 级控制**。对 PyTorch 用户这是恩赐,对你是信息损失:**你想看清楚每个 thread 在哪里、在干什么**。

但 Triton 仍然要学,理由有二:

- **生产代码**:2026 年的新论文 80% 用 Triton,不会读会落后
- **快速实验**:写一个性能 80% 的版本验证想法时,Triton 比 CUDA 快 5 倍

所以**修订版策略:CUDA 主修,Triton 当快速原型工具,两者并行**。

### 9.2 ICPC ↔ GPU 概念映射

ICPC 训练让你已经掌握了一堆 GPU 编程里的核心思想,只是名字不同:

| ICPC 你已经会的 | GPU 上对应什么 | 在哪一章用上 |
| --- | --- | --- |
| 线段树 / 分治 reduction | Warp shuffle → Block → Grid 三层 reduction | Ch5 |
| 莫队块大小调参 | `BLOCK_M/N/K` tile size 调优 | Ch4-5 |
| Cache-friendly 数据布局 | SRAM-friendly tiling | Ch4 全章 |
| 位运算 + bitset | Warp ballot / vote / mask | Ch5、Ch7 |
| 双指针滑动窗口 | Persistent kernel + producer/consumer | Ch7、Ch9 |
| 预处理 + ST 表 | 反向用——FA 反向是"放弃预处理换 SRAM" | Ch7 |
| 树状数组 prefix sum | GPU scan(Hillis-Steele / Blelloch) | Ch5 |
| 离线算法处理依赖 | CUDA Stream + Event 建依赖图 | Ch8 |
| 启发式 + 暴力剪枝 | Autotune 搜 BLOCK / num_warps 配置 | Ch4 |
| 树状/堆基数据结构选择 | 选 atomic 还是 two-pass reduction | Ch5 |

**带着这张映射读 PMPP**,你会发现作者讲三页的并行模式对你就是"这不就是某某算法的 GPU 版"。学习速度通常比默认读者快 2-3 倍。

### 9.3 修订版章节配置

每章的**任务清单替换为 CUDA + Triton 双实现**,profile 对比。具体调整:

#### Ch1 GPU 硬件心智模型

不变,但你应该 3-5 天搞定。**额外任务**:用 `cuobjdump --dump-sass` 看一段 cuBLAS sgemm 的 SASS,**ICPC 选手应该能直接读懂大部分指令**(fmla/lds/sts/bra),这是 Tier 3 资源里我提前推给你的部分。

#### Ch2 + Ch3 合并为一周

不再有"先 CUDA 再 Triton"的分阶段,直接合并:

- 第 1 天:**CUDA syntax cheatsheet**(找一份 1 页的速查,不需要看 PMPP Ch4-5 完整章节)
- 第 2 天:Triton 官方 tutorial 01-02
- 第 3-5 天:**vector_add 写两版,transpose 写两版**(CUDA naive / CUDA shared mem / Triton),profile 对比

**关键习惯**:从这一周开始,**每个 kernel 都写两版**,看 Triton 编译器在哪些地方比手写 CUDA 差(通常差在 register allocation 和 shared memory layout 的细节)。

#### Ch4 矩阵乘(2 周,你的主战场)

这是你和默认读者差距最大的一章。任务升级:

- [ ] CUDA naive matmul(1 小时,主要是熟语法)
- [ ] **CUDA shared memory tiled matmul**(自己推 tile 怎么放,目标 cuBLAS 30%)
- [ ] **CUDA + Tensor Core MMA**(用 `wmma` API 或者直接 PTX `mma.sync`,目标 cuBLAS 60%)
- [ ] **CUDA + cp.async (Ampere+) + double buffering**(目标 cuBLAS 75%)
- [ ] Triton tiled matmul(看 `tl.dot` 自动调用 Tensor Core,对比性能)
- [ ] **读 CUTLASS 的 device-level GEMM 实现**——这一步默认读者放在 Tier 3,你在这周就该看
- [ ] 写笔记 `Ch4.Matmul.md`,**带性能阶梯表 + 各版本的 ncu profile**

完成这一章你应该能直接做 GEMM 性能工程的工作。

#### Ch5 Reduction(1.5 周)

你的强项。任务从 warp shuffle 直接起步:

- [ ] **CUDA warp-shuffle reduction**(`__shfl_xor`),立刻就用 warp-level 原语
- [ ] **CUDA block-level reduction**(warp shuffle + shared memory 两层)
- [ ] CUDA grid-level reduction(atomic vs two-pass 对比)
- [ ] Triton 对照版本
- [ ] Softmax / LayerNorm / RMSNorm 各双版本
- [ ] **额外:写一个 prefix sum (scan)**,**这对应你 ICPC 里的树状数组**——Hillis-Steele 算法本质就是分治,你应该 30 分钟搞定

注意:默认计划把 warp shuffle 放到 Ch9 高级章节,**你完全不需要等到那时**。

#### Ch6 Fused Kernel(1 周)

不变,但加一个任务:

- [ ] **读 Liger Kernel 或 ThunderKittens 的核心 kernel**(主要看 fusion 设计,不是 syntax)
- [ ] CUDA + Triton 各写一个 fused dropout+gelu+bias

#### Ch7 FlashAttention(2-3 周)

最大的硬骨头,任务也最丰富:

- [ ] Standard attention CUDA 版(1 天,看清楚 attention matrix 怎么爆显存)
- [ ] **FA1 forward CUDA 版**(用 shared memory tiling)
- [ ] FA1 forward Triton 版(对比哪种写法编译器更友好)
- [ ] **FA1 backward CUDA 版**——硬骨头中的硬骨头,自己推 dQ/dK/dV
- [ ] FA1 backward Triton 版
- [ ] **读 FA2 paper + Tri Dao 的 CUDA 实现**(默认计划这一步是"高阶可选",你必做)
- [ ] (Hopper GPU 才需要)读 FA3 的 TMA + wgmma 实现

完成后你站到了"能复现顶级 paper kernel"的位置。

#### Ch8 + Ch9 合并

默认计划把 CUDA 高级特性(warp shuffle、cp.async、TMA、Tensor Core MMA)单独放 Ch9,**对你完全没必要分章**——这些你在 Ch4-7 已经全部用上了。

修订版的 Ch8 直接合并 Ch9 内容:

- Profile 工具(ncu / nsys)
- Roofline 分析
- Cooperative groups、warp-level primitives 系统化整理
- Cp.async / TMA 的工程模式
- PyTorch custom op 集成
- DDP comm hook 集成

这章应该 2 周左右,完成后就**结业**——你应该能独立做大模型 kernel 的优化工作。

### 9.4 资源 Tier 调整

ICPC 背景下,默认 Tier 表整体往前提一档:

**Tier 1 必备(你)**:

- PMPP Ch 1-7(默认读者也是)
- **CUTLASS 仓库的 device-level GEMM**(默认读者 Tier 3)
- **Tri Dao FA1/FA2/FA3 论文 + 代码**(默认读者 Tier 2)
- **Lei Mao 的 CUDA 深度文章**(默认读者 Tier 2)
- Triton 官方 tutorials
- GPU Mode YouTube

**Tier 2 你应该看(默认读者用不上)**:

- **NVIDIA CUDA C Programming Guide** 当字典(默认读者只用 Tier 2 末位)
- **PTX ISA 文档**(默认读者 Tier 3)
- **Maxas / Volta-tuning blog 系列**——给会读 SASS 的人看的 SGEMM 极限优化
- CUTLASS template metaprogramming 部分(C++ 模板对你不是障碍)

**跳过的(对你没价值)**:

- 大量"GPU 入门"博客——它们花 50% 篇幅讲 C++ 基础

### 9.5 调整后的时间表

```
Week 1:    Ch1 硬件(3-5 天) + 自学 CUDA syntax(2 天)
Week 2:    Ch2+3 合并(vector_add / transpose 双版本)+ GPU Puzzles
Week 3-4:  Ch4 matmul:CUDA 四档 + Triton + CUTLASS 阅读
Week 5:    Ch5 reduction + softmax(双版本)
Week 6:    LayerNorm/RMSNorm 双版本 + Ch6 fused kernel
Week 7-9:  Ch7 FlashAttention(forward + backward 双版本 + FA2/3 阅读)
Week 10:   Ch8 profile + 集成(包含原 Ch9 内容)
Week 11+:  Open-ended:复现新论文 kernel,接进 AI Infra Ch1 的 DDP comm hook
```

整体周期从默认的 12-15 周压缩到 **10-11 周**,主要是 Ch1-3 大幅压缩 + Ch9 不需要单独章。

### 9.6 几条针对你的额外建议

**第一,kernel 评估指标不止"和 cuBLAS 比性能"**。ICPC 里你已经有"分析自己代码瓶颈"的肌肉,迁移过来:每个 kernel 都用 ncu 看 occupancy / SRAM utilization / register pressure,**就当成"卡常时看 cache miss"**。

**第二,SASS 比你想的好读**。第一次看像天书,但 ICPC 选手通常 1-2 天就能识别主要 phases(矩阵乘里的 fmla、shared memory 加载的 lds.shared、寄存器搬运的 mov)。**FA、CUTLASS 的优化文章经常贴 SASS 截图**,能读懂 SASS 等于解锁第二条信息源。

**第三,competitive programming 的"放弃完美解、追求 80% 解"哲学反而要忘掉一部分**。GPU 编程里最后那 20% 性能(从 cuBLAS 60% 到 80%)往往是工业价值最大的——因为公司花几百万 GPU,每 1% 就是几万美元。所以优化要做透,不要在 70% 就收手。

**第四,做笔记时把 ICPC 的类比写下来**。这是你独有的视角,后续回头看自己受益,公开发出来对其他 ICPC 转 AI 的人也有价值——这条路径目前在中文社区几乎没有系统材料。

## 十、一句话总结

**先建 GPU 硬件心智模型(Ch1-2,2 周),主修 Triton 写 10 个由易到难的 kernel(Ch3-7,2-3 个月),最后接回 Infra 做 profile 与生产集成(Ch8,持续)**。CUDA 当成"读懂别人代码"的语言,Ch9 按需深入。整个过程和 AI Infra 主线交替推进,两边互相催化。

完成 9 章之后,你已经能在大模型工程的两条主线——分布式系统 + GPU 算子——之间自由切换。这是 2026 年 AI 工程师的稀缺能力。
