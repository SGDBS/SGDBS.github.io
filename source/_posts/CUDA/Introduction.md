CUDA 编程不要按“语法课”学，要按 **GPU 执行模型 → 内存层次 → 性能建模 → Kernel 优化 → AI 算子实现** 这条线学。你以后做 AI Infra / 推理优化，核心不是“会写 CUDA”，而是能判断：

> 一个算子为什么慢？瓶颈在访存、计算、同步、调度、还是数据布局？
> 怎么改 kernel / 改 layout / 改 fusion / 改 quantization 才能变快？

---

# 1. CUDA 编程的总知识框架

可以分成 8 层。

```text
CUDA 编程
├── 1. GPU 硬件模型
│   ├── SM / CUDA Core / Tensor Core
│   ├── Warp / Thread / Block / Grid
│   ├── SIMT 执行模型
│   └── Occupancy / latency hiding
│
├── 2. CUDA 编程模型
│   ├── __global__ / __device__ / __host__
│   ├── kernel launch
│   ├── threadIdx / blockIdx / blockDim / gridDim
│   ├── memory allocation
│   └── stream / event
│
├── 3. GPU 内存层次
│   ├── global memory
│   ├── shared memory
│   ├── register
│   ├── local memory
│   ├── constant memory
│   ├── texture memory
│   └── L1 / L2 cache
│
├── 4. 性能优化基础
│   ├── coalesced memory access
│   ├── memory bandwidth
│   ├── arithmetic intensity
│   ├── roofline model
│   ├── occupancy
│   ├── bank conflict
│   └── warp divergence
│
├── 5. 并行算法模式
│   ├── elementwise
│   ├── reduction
│   ├── scan / prefix sum
│   ├── transpose
│   ├── histogram
│   ├── sort
│   └── GEMM
│
├── 6. AI 核心算子
│   ├── LayerNorm / RMSNorm
│   ├── Softmax
│   ├── MatMul / GEMM
│   ├── Attention
│   ├── FlashAttention
│   ├── RoPE
│   ├── Quantized Linear
│   ├── KV Cache 操作
│   └── Sampling
│
├── 7. 高性能 CUDA 工程
│   ├── cuBLAS / cuDNN / CUTLASS
│   ├── Tensor Core / WMMA / MMA
│   ├── CUDA Graph
│   ├── Stream 并发
│   ├── Host-device pipeline
│   ├── Nsight Compute / Nsight Systems
│   └── PyTorch extension
│
└── 8. AI Infra 结合
    ├── PyTorch 自定义算子
    ├── torch.compile / torch.export
    ├── ONNX Runtime custom op
    ├── TensorRT plugin
    ├── vLLM / SGLang kernel
    ├── quantization kernel
    └── inference serving optimization
```

---

# 2. 最先要理解：GPU 和 CPU 的差异

CPU 适合：

```text
复杂控制流
低延迟
少量线程
大缓存
强单核性能
```

GPU 适合：

```text
大量相似计算
高吞吐
上万线程
高带宽
弱控制流
```

CUDA 编程的本质是：

> 把一个大任务切成很多小任务，让 GPU 上成千上万个线程并行执行。

比如向量加法：

```cpp
C[i] = A[i] + B[i]
```

CPU 可能是一个 for 循环：

```cpp
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```

CUDA 是让很多线程同时算：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N) {
    C[i] = A[i] + B[i];
}
```

每个线程负责一个 `i`。

---

# 3. CUDA 最核心的执行层级

你必须熟悉这几个概念：

```text
Grid
 └── Block
      └── Thread
```

实际执行时还有一个更底层的概念：

```text
Warp = 32 个线程
```

NVIDIA GPU 上，warp 是调度的基本单位。也就是说，不是一个 thread 一个 thread 单独跑，而是 32 个线程一组执行。

所以 CUDA 性能优化经常围绕这些问题：

| 问题                  | 含义                  |
| ------------------- | ------------------- |
| thread 怎么映射数据？      | 每个线程负责哪个元素          |
| block 多大？           | 每个 block 有多少线程      |
| grid 多大？            | 总共 launch 多少 block  |
| warp 是否分歧？          | 同一个 warp 内线程是否走不同分支 |
| 访存是否合并？             | 相邻线程是否访问连续地址        |
| shared memory 是否冲突？ | 是否发生 bank conflict  |
| register 是否太多？      | 是否影响 occupancy      |

---

# 4. 内存层次是 CUDA 的核心

CUDA 优化大部分时候不是“算得不够快”，而是“数据搬得太慢”。

你要记住这个层次：

```text
register       最快，线程私有
shared memory  很快，block 内共享
L1 cache
L2 cache
global memory  慢，但容量大
host memory    更慢，CPU 内存
disk/network   最慢
```

粗略理解：

| 内存            | 作用          | 优化重点                       |
| ------------- | ----------- | -------------------------- |
| Register      | 每个线程自己的变量   | 减少 register pressure       |
| Shared Memory | block 内线程共享 | tile、复用数据、避免 bank conflict |
| Global Memory | 显存 HBM      | 合并访存、减少访存次数                |
| L2 Cache      | 全 GPU 共享缓存  | 数据复用、cache locality        |
| Host Memory   | CPU 内存      | 减少 CPU-GPU 拷贝              |

AI Infra 里很多优化本质上是：

> 减少 global memory 读写，把数据尽量放在 register / shared memory 中复用。

例如 FlashAttention 的核心思想之一就是避免把完整 attention matrix 写回 HBM。

---

# 5. 你应该按什么顺序学

## 第一阶段：CUDA 基础语法

目标：能写简单 kernel。

需要掌握：

```text
__global__ kernel
cudaMalloc
cudaMemcpy
cudaFree
threadIdx
blockIdx
blockDim
gridDim
cudaDeviceSynchronize
```

练习：

| 练习              | 目的                    |
| --------------- | --------------------- |
| vector add      | 理解 thread indexing    |
| scalar multiply | 理解 elementwise        |
| ReLU            | 对应深度学习 elementwise op |
| sigmoid         | 理解数学函数                |
| matrix add      | 理解 2D indexing        |

这一阶段不要急着优化，先把 CUDA execution model 搞清楚。

---

## 第二阶段：内存访问和性能基础

目标：知道为什么 kernel 慢。

重点学：

```text
coalesced memory access
global memory bandwidth
shared memory
register
warp divergence
occupancy
```

你要重点理解：

### Coalesced Access

好访问：

```text
thread 0 -> A[0]
thread 1 -> A[1]
thread 2 -> A[2]
thread 3 -> A[3]
...
```

坏访问：

```text
thread 0 -> A[0]
thread 1 -> A[1024]
thread 2 -> A[2048]
thread 3 -> A[3072]
...
```

GPU 喜欢相邻线程访问连续地址。

这对 AI 里的 layout 很重要，比如：

```text
[N, C]
[B, H, S, D]
[B, S, H, D]
```

不同 layout 会影响 attention / KV cache / quantized linear 的访存效率。

---

## 第三阶段：经典并行模式

目标：能写常见并行算法。

重点练：

| 模式           | 代表算子                |
| ------------ | ------------------- |
| elementwise  | ReLU, GELU, RoPE    |
| reduction    | sum, max, LayerNorm |
| scan         | prefix sum          |
| transpose    | layout transform    |
| tiled matmul | GEMM                |
| histogram    | sampling / token 统计 |
| top-k        | decoding / sampling |

其中对 LLM 推理最重要的是：

```text
reduction
softmax
matmul
transpose
layout transform
top-k
```

---

## 第四阶段：写 AI 算子

这阶段开始进入你的主线：AI Infra。

建议按这个顺序实现：

```text
1. ReLU / GELU
2. RMSNorm
3. LayerNorm
4. Softmax
5. RoPE
6. Naive MatMul
7. Tiled MatMul
8. Quantized Linear
9. Attention
10. FlashAttention 简化版
```

每个算子都要问 5 个问题：

```text
输入 shape 是什么？
每个 thread / block 负责什么？
global memory 读写多少次？
有没有 shared memory 复用？
瓶颈是 memory-bound 还是 compute-bound？
```

---

# 6. 对 AI Infra 最重要的 CUDA 算子

## 1. RMSNorm / LayerNorm

LLM 中大量出现。

形式大概是：

```text
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
```

核心是 reduction：

```text
sum(x_i^2)
```

难点：

```text
一个 token 的 hidden_dim 通常很大
需要 block 内 reduction
需要避免多次读写 global memory
需要处理 fp16 / bf16 / fp32 accumulation
```

你可以先写：

```text
每个 block 处理一个 token
block 内线程处理 hidden_dim
shared memory 做 reduction
最后归一化
```

---

## 2. Softmax

Attention 中核心算子。

公式：

```text
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

需要两个 reduction：

```text
max reduction
sum reduction
```

优化重点：

```text
数值稳定性
warp-level reduction
block-level reduction
shared memory
vectorized load
```

---

## 3. RoPE

RoPE 本质是对 Q/K 做旋转位置编码。

简化理解：

```text
x1' = x1 * cos - x2 * sin
x2' = x1 * sin + x2 * cos
```

这个算子偏 elementwise，瓶颈通常是访存。

重点：

```text
连续访存
half2 / vectorized load
in-place or out-of-place
layout: [B, S, H, D]
```

---

## 4. Quantized Linear

这是你做推理优化会非常常见的东西。

例如 W8A16：

```text
activation: fp16 / bf16
weight: int8
output: fp16 / bf16
```

计算逻辑：

```text
Y = X @ dequant(W)
```

或者：

```text
Y = X @ W_int8 * scale
```

重点问题：

```text
scale 是 per-tensor / per-channel / per-group？
weight 怎么 pack？
反量化在什么时候做？
是否能 fuse dequant + matmul？
是否走 Tensor Core？
```

这块和你之前学的量化、ONNX、推理导出关系很大。

---

## 5. Attention / FlashAttention

普通 attention：

```text
QK^T -> softmax -> P V
```

问题是 `QK^T` 很大。

例如 sequence length = 8192：

```text
attention matrix = 8192 × 8192
```

非常吃显存带宽。

FlashAttention 的思想是：

```text
不要把完整 attention matrix 写回 HBM
分块计算
在 SRAM/shared memory 中做在线 softmax
减少 HBM 读写
```

这就是 CUDA 编程和 AI Infra 的典型交叉点。

---

# 7. 你应该掌握的 CUDA 性能分析工具

只会写 kernel 不够，必须会 profile。

## 必学工具

| 工具             | 用途                                   |
| -------------- | ------------------------------------ |
| Nsight Systems | 看整体时间线、CPU-GPU overlap、kernel launch |
| Nsight Compute | 看单个 kernel 的性能指标                     |
| nvprof         | 老工具，了解即可                             |
| ncu            | Nsight Compute CLI                   |
| nsys           | Nsight Systems CLI                   |

你需要会看这些指标：

```text
kernel duration
achieved occupancy
memory throughput
SM utilization
warp stall reasons
register usage
shared memory usage
L2 hit rate
DRAM throughput
Tensor Core utilization
```

尤其是：

```text
Memory Bound?
Compute Bound?
Launch overhead bound?
Synchronization bound?
```

AI Infra 里优化时经常要判断：

```text
这个 op 值不值得手写 CUDA？
能不能用 Triton？
能不能用 cuBLAS / CUTLASS？
能不能 fuse 掉？
是不是被 memory bandwidth 卡住？
是不是 batch 太小导致 GPU 吃不满？
```

---

# 8. CUDA 和 PyTorch / ONNX / TensorRT 的关系

你现在学的东西可以这样串起来：

```text
PyTorch Model
   ↓
torch.export / torch.compile
   ↓
Graph IR
   ↓
ONNX / FX / TorchInductor
   ↓
Runtime
   ↓
CUDA Kernel / Triton Kernel / Vendor Kernel
   ↓
GPU / NPU
```

CUDA 在最底层。

比如一个 PyTorch op：

```python
y = torch.nn.functional.layer_norm(x)
```

底层可能变成：

```text
PyTorch dispatcher
  -> CUDA backend
    -> layernorm CUDA kernel
```

如果你写自定义算子：

```text
Python API
  -> C++ extension
    -> CUDA kernel
```

如果你导出到 ONNX：

```text
ONNX op
  -> ONNX Runtime CUDA EP
    -> CUDA kernel
```

如果你做 TensorRT plugin：

```text
TensorRT engine
  -> custom plugin
    -> enqueue()
      -> CUDA kernel
```

所以你的目标不是单独学 CUDA，而是学：

> CUDA kernel 如何嵌入到 AI 推理系统里。

---

# 9. 建议学习路线：按 10 周推进

## 第 1-2 周：CUDA 基础

目标：

```text
能写、编译、运行简单 CUDA kernel
```

任务：

```text
vector add
matrix add
ReLU
GELU
simple reduce sum
```

掌握：

```text
threadIdx / blockIdx
cudaMalloc / cudaMemcpy
grid/block 配置
错误检查
```

---

## 第 3-4 周：内存与优化

目标：

```text
理解为什么 CUDA kernel 快或慢
```

任务：

```text
连续访存 vs stride 访存 benchmark
shared memory transpose
reduction 优化
warp-level reduction
```

掌握：

```text
coalescing
shared memory
bank conflict
occupancy
warp divergence
```

---

## 第 5-6 周：矩阵乘法

目标：

```text
理解 GEMM 是如何优化的
```

任务：

```text
naive matmul
tiled matmul
shared memory matmul
register blocking matmul
调用 cuBLAS 对比
```

需要理解：

```text
M, N, K
tile size
data reuse
arithmetic intensity
memory-bound vs compute-bound
```

不要一开始就追求写出 cuBLAS 级别 GEMM。这个很难。目标是理解它为什么需要 tiling。

---

## 第 7-8 周：LLM 算子

目标：

```text
能实现几个小型 LLM CUDA kernel
```

任务：

```text
RMSNorm
LayerNorm
Softmax
RoPE
KV cache append
sampling top-k 简化版
```

重点：

```text
reduction
vectorized load
half / half2
fp32 accumulation
layout transform
```

---

## 第 9 周：PyTorch Extension

目标：

```text
把 CUDA kernel 接进 PyTorch
```

学习：

```text
torch.utils.cpp_extension
pybind11
C++ extension
ATen Tensor
CUDA stream
contiguous check
dtype check
shape check
```

你要能写这种接口：

```python
import my_cuda_ops

y = my_cuda_ops.rmsnorm(x, weight, eps)
```

然后和 PyTorch 原生实现比较：

```python
torch.cuda.Event benchmark
torch.allclose correctness check
```

---

## 第 10 周：推理系统结合

目标：

```text
理解 CUDA kernel 在推理框架中的位置
```

练习：

```text
给一个 Qwen / LLaMA block 替换 RMSNorm kernel
写一个 quantized linear demo
写一个 ONNX Runtime custom op 雏形
看 vLLM / SGLang / TensorRT-LLM 的 kernel 目录
```

这时你就能把 CUDA 和你正在学的：

```text
ONNX
quantization
KV cache
PyTorch export
custom op
inference runtime
```

串起来。

---

# 10. 最推荐的练习项目

按价值排序：

## Project 1：手写 RMSNorm CUDA Kernel

难度适中，和 LLM 强相关。

功能：

```python
y = rmsnorm(x, weight, eps)
```

输入：

```text
x: [batch, seq_len, hidden_dim]
weight: [hidden_dim]
```

优化版本：

```text
v1: naive
v2: block reduction
v3: warp reduction
v4: vectorized load
v5: half2
```

你可以和 PyTorch 版本 benchmark。

---

## Project 2：手写 Softmax Kernel

功能：

```python
y = softmax(x, dim=-1)
```

重点：

```text
max reduction
sum reduction
numerical stability
block-level parallelism
```

这是理解 attention kernel 的前置知识。

---

## Project 3：手写 Tiled MatMul

功能：

```text
C = A @ B
```

版本：

```text
v1: naive global memory
v2: shared memory tiling
v3: register blocking
v4: vectorized load
v5: 对比 cuBLAS
```

这能帮你理解为什么 GEMM 优化复杂。

---

## Project 4：简化版 Attention

功能：

```text
Q, K, V -> Attention Output
```

先不要直接写 FlashAttention。

先写：

```text
QK^T
softmax
PV
```

然后思考为什么它慢：

```text
中间矩阵太大
HBM 读写太多
softmax 需要 reduction
sequence length 增大后显存爆炸
```

再学 FlashAttention 才能真正理解。

---

## Project 5：W8A16 Quantized Linear

功能：

```text
Y = X @ W_int8 * scale
```

重点：

```text
int8 weight
fp16 activation
scale
dequant fusion
per-channel scale
```

这和你实习方向非常贴合。

---

# 11. 学 CUDA 时最容易踩的坑

## 坑 1：只学语法，不学性能模型

会写：

```cpp
__global__ void kernel(...)
```

不代表会 CUDA。

真正重要的是：

```text
这个 kernel 为什么快？
为什么慢？
瓶颈在哪？
理论带宽多少？
实际带宽多少？
```

---

## 坑 2：一上来就学 Tensor Core

Tensor Core 很重要，但不适合最开始学。

顺序应该是：

```text
普通 CUDA thread
→ shared memory tiling
→ warp-level primitive
→ CUTLASS
→ WMMA / MMA / Tensor Core
```

否则容易只记 API，不理解为什么这么设计。

---

## 坑 3：忽略数据 layout

AI kernel 很多性能问题来自 layout。

比如 KV cache：

```text
[B, S, H, D]
[B, H, S, D]
```

不同 layout 对 decode 阶段的访问效率差异很大。

你要经常问：

```text
连续维度是谁？
每个 warp 访问的是不是连续地址？
head_dim 是否适合 vectorized load？
cache locality 怎么样？
```

---

## 坑 4：不做 benchmark

CUDA 学习必须伴随 benchmark。

每写一个 kernel，都要比较：

```text
PyTorch baseline
cuBLAS / cuDNN baseline
Triton baseline
自己的 CUDA kernel
```

同时检查：

```text
正确性
耗时
带宽
吞吐
数值误差
```

---

# 12. 和 Triton 的关系

你之后大概率也要学 Triton。

可以这样理解：

| CUDA          | Triton           |
| ------------- | ---------------- |
| 更底层           | 更高层              |
| 控制力强          | 开发效率高            |
| 学习成本高         | 更适合快速写 AI kernel |
| 适合极致优化 / 特殊场景 | 适合大多数深度学习算子      |
| 必须理解硬件细节      | 仍然需要性能模型         |

建议顺序：

```text
先学 CUDA 基础
再学 Triton
最后看 CUTLASS / TensorRT-LLM / vLLM kernel
```

原因是：

> Triton 把很多底层细节封装了，但你不懂 CUDA 性能模型，就不知道 Triton block size、num_warps、num_stages 为什么影响性能。

---

# 13. 你要形成的最终能力

学完 CUDA 后，你应该能做到：

```text
1. 看懂一个 CUDA kernel 在干什么
2. 判断线程/block/grid 映射是否合理
3. 判断内存访问是否 coalesced
4. 判断 shared memory 是否有 bank conflict
5. 判断 kernel 是 memory-bound 还是 compute-bound
6. 用 Nsight Compute 找瓶颈
7. 写 PyTorch CUDA extension
8. 实现 RMSNorm / Softmax / RoPE 等 LLM 算子
9. 理解 FlashAttention / CUTLASS / TensorRT plugin 的基本思路
10. 把 CUDA kernel 接进推理系统
```

对 AI Infra 岗位来说，最关键的是这 4 项：

```text
性能建模
LLM 算子理解
profile 能力
工程接入能力
```

---

# 14. 最简学习路径总结

你可以按这个顺序学：

```text
CUDA 基础语法
→ GPU 执行模型
→ 内存层次
→ coalesced access
→ shared memory
→ reduction
→ tiled matmul
→ softmax
→ RMSNorm
→ RoPE
→ quantized linear
→ PyTorch extension
→ Nsight profiling
→ FlashAttention / CUTLASS / TensorRT plugin
```

更具体一点：

```text
第一步：写 vector add / ReLU
第二步：写 reduction
第三步：写 LayerNorm / RMSNorm
第四步：写 tiled matmul
第五步：写 softmax
第六步：写 PyTorch extension
第七步：profile 每个 kernel
第八步：读 vLLM / TensorRT-LLM / CUTLASS kernel
```

---

# 15. 对你当前方向的建议

你现在在学：

```text
ONNX
PyTorch export
custom op
quantization
KV cache
QwenVL
AI infra
inference optimization
```

所以 CUDA 不要学成“图形学 CUDA”或者“通用 HPC CUDA”。你应该直接围绕 LLM 推理学。

优先级应该是：

```text
RMSNorm > Softmax > RoPE > KV Cache > Quantized Linear > Attention > GEMM
```

GEMM 很重要，但工业界通常优先用：

```text
cuBLAS
CUTLASS
TensorRT
vendor library
```

你短期内更值得手写的是：

```text
小算子
fused op
layout transform
quant/dequant
KV cache 操作
sampling
custom op glue code
```

因为这些更贴近推理优化岗位里的实际工作。
