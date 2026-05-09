---
title: 1. Vector Addition
categories: 学习笔记- AI Infra 
date: 2026-05-09 22:30:00
mathjax: true
tags:
    - AI
    - AI Infra
---


https://leetgpu.com/challenges/vector-addition


## 一、题目分析：这是什么类型的问题

向量加法属于 **embarrassingly parallel**（极易并行）问题——每个输出元素的计算完全独立，没有跨元素的依赖、没有通信、没有归约。这是 GPU 编程里最简单的一类。

但"简单"不等于"快"。这道题虽然写起来一行就够，分析起来却能引出 GPU 编程里最重要的概念：**memory-bound 计算**和**合并访存**。它是讲这些概念最干净的载体，因为没有任何其他因素干扰。

形式化一点：
- 输入：两个长度为 $N$ 的向量 $A, B \in \mathbb{R}^N$, $N <= 25000000 $
- 输出：$C \in \mathbb{R}^N$，其中 $C_i = A_i + B_i$
- 计算量：$N$ 次浮点加法，即 $N$ FLOPs
- 访存量：读 $A$、读 $B$、写 $C$，共 $3N$ 个 float = $12N$ 字节

每个向量 25M 个 float = 100 MB。三个向量 A、B、C 总共 300 MB。

这个量级的特征：
- **完全装不进任何级别的 cache**。L1/L2 cache 都是 KB ~ MB 级。所以每个元素必然要从 global memory 走一遍。
- **能装进显存**。现代 GPU 显存 8GB ~ 80GB，300 MB 是小菜。
- 数据已经在 GPU 上（LeetGPU 给定），所以不涉及 PCIe 传输。

## 二、代码

```cuda
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

## 三、逐行解析

### `__global__ void vector_add(...)`

`__global__` 表示这是 kernel 函数：CPU 调用、GPU 执行。返回值必须是 `void`——kernel 不能用 `return` 返回值给 CPU，所有结果只能通过指针参数写到显存里。

参数里 `const float*` 加了 const 是正确习惯。这不仅是软件工程上的好习惯，对编译器也有意义：编译器可以推断出 A、B 不会被这个 kernel 修改，从而做更激进的优化（比如把读出来的值缓存到寄存器里复用）。后面遇到更复杂的 kernel 时还会看到 `__restrict__` 关键字，作用类似但更强。

### `int idx = blockIdx.x * blockDim.x + threadIdx.x;`

这一行是 CUDA 编程的"通用前奏"，几乎每个 kernel 第一行都长这样。

它把三层结构（grid → block → thread）映射成一维全局线程编号。回忆一下：
- `blockIdx.x` 我是第几个 block（0 ~ gridDim.x - 1）
- `blockDim.x` 每个 block 多少线程（这里 = 256）
- `threadIdx.x` 我在 block 内是第几个（0 ~ blockDim.x - 1）

公式直观理解：前面 `blockIdx.x` 个 block 已经占了 `blockIdx.x * blockDim.x` 个线程编号，再加上我在自己 block 里的位置 `threadIdx.x`，就是我在所有线程里排第几。

**计算机系统知识：寄存器变量**

`idx` 是函数内的局部变量，CUDA 会把它放在**寄存器**里。GPU 的寄存器和 CPU 不太一样——每个 SM（Streaming Multiprocessor）有一个巨大的寄存器文件（典型值：65536 个 32 位寄存器，即 256 KB），这些寄存器在该 SM 上的所有活跃线程之间**静态分配**。如果你的 kernel 每个线程用了太多寄存器，能同时驻留在 SM 上的线程就会变少，影响 **occupancy**（占用率，后面解释）。向量加法里每个线程只用几个寄存器，没这个烦恼。

### `if (idx < N)`

边界检查。因为 `blocksPerGrid` 是上取整的，最后一个 block 可能有"多余"的线程。比如 N = 1000、threadsPerBlock = 256，那么 blocksPerGrid = 4，总共启动 1024 个线程，最后 24 个线程的 idx 在 [1000, 1024) 范围，必须丢弃。

**GPU 硬件知识：warp divergence**

这个 `if` 会引起一种叫 **warp divergence** 的现象，需要好好讲讲。

GPU 上 32 个线程组成一个 **warp**，warp 内所有线程在硬件上**共用一个程序计数器**（PC）——它们必须执行同一条指令。这叫 **SIMT** 模型（Single Instruction Multiple Threads）。

那 if-else 怎么办？硬件用一个**活跃掩码**（active mask）来标记 warp 里哪些线程要执行当前指令。当出现分支：

```
if (cond) { A; }
else      { B; }
```

执行流程是：
1. 先执行 A 分支：mask 设成"满足 cond 的线程为 1，其他为 0"，所有 32 个线程一起走 A 的指令，但 mask=0 的线程的执行结果被丢弃。
2. 再执行 B 分支：mask 反过来。
3. 两个分支汇合后 mask 恢复全 1。

**关键点：A 和 B 是串行执行的，时间是两者之和。** 这就是 warp divergence 带来的性能损失。

向量加法里：N=1000 时只有最后一个 warp（thread 992-1023）会发生 divergence——里面 8 个线程要执行加法、24 个不执行。其他 31 个 warp 都全员执行同一条路径，没有 divergence。占总工作量极小，可以忽略。但在更复杂的 kernel 里，divergence 是头号性能杀手之一。

> 注：从 Volta 架构（2017）开始，NVIDIA 引入了 Independent Thread Scheduling，每个线程有自己的 PC，divergence 处理更灵活。但代价模型基本没变——分支两边还是要串行执行。

### `C[idx] = A[idx] + B[idx];`

核心计算。一次 global memory 读 A、一次读 B、一次浮点加法、一次写 C。

下面是这道题最值得展开的两个知识点：**合并访存** 和 **memory-bound 分析**。

## 四、GPU 硬件知识：合并访存（Coalesced Memory Access）

这是 GPU 性能优化的**第一原则**。理解了它你就明白为什么 GPU 算法的核心目标常常是"让访存模式规则化"。

### 4.1 GPU 显存的物理特性

GPU 的 global memory（GDDR6 / HBM）和 CPU 的 DRAM 一样，**不是按字节寻址高效的**。物理上，DRAM 一次读出一整行（一个"突发", burst）的数据——典型是 32 字节或 128 字节。读 1 字节和读 32 字节的代价几乎一样。

CPU 处理这个问题的方法是 **cache line**：一次从内存读 64 字节进 L1 缓存，假设你接下来会访问附近的字节（局部性原理）。

GPU 的处理方法不一样，因为 GPU 一次有 32 个线程要同时访存：与其每个线程独立去读，不如**把它们拼起来一次读**。这就是合并访存。

### 4.2 合并访存的规则

硬件会检查一个 warp（32 个线程）在同一条 load 指令里访问的地址：

- **理想情况**：32 个线程访问的地址是**连续的、对齐的 128 字节**（即一个 32×4 字节的块）。硬件用 1 次 128 字节的内存事务就能搞定，每个线程拿到自己那 4 字节。
- **不理想情况**：地址散乱，硬件需要多次内存事务。最坏情况 32 个线程访问 32 个不同的 128 字节区域，要发 32 次事务，**带宽利用率降到 1/32**。

### 4.3 向量加法的访存模式

回到 `C[idx] = A[idx] + B[idx]`，假设当前 warp 里的 32 个线程的 `idx` 是 0, 1, 2, ..., 31。

- 它们读 `A[0..31]`：地址是 A、A+4、A+8、...、A+124，正好是连续的 128 字节，**完美合并**。
- 读 B 同理。
- 写 C 也同理。

所以向量加法天然满足合并访存——这是它能跑满显存带宽的前提。

### 4.4 反面教材：什么样的写法不合并

如果 kernel 改成（错误示例）：

```cuda
int idx = threadIdx.x * gridDim.x + blockIdx.x;  // 故意写错的索引
C[idx] = A[idx] + B[idx];
```

那么 warp 里的线程 0、1、2 访问的地址不连续，跨度很大，每次访存都要发独立的事务，性能可能掉 10 倍以上。

二维数据更要小心。比如 `M[row][col]` 按行存储（C 风格），如果让 `threadIdx.x` 对应 row、`threadIdx.y` 对应 col，那同一个 warp 里的线程访问 `M[0][0], M[1][0], M[2][0], ...`——跨行访问，地址间隔是 col 维度的大小，完全不合并。正确做法是让 `threadIdx.x` 对应 col。这是矩阵乘法等题目里反复要注意的。

### 4.5 一个常被问的细节：地址对齐

要达到 1 次事务，warp 访问的 128 字节块必须**地址对齐到 128 字节边界**。`cudaMalloc` 返回的指针是对齐到 256 字节的，所以从指针开头访问肯定对齐。但如果你写 `A + 1` 然后做合并访存，会跨两个 128 字节块，需要 2 次事务（这个开销其实不大，但概念上要清楚）。

## 五、计算机系统知识：IEEE 754 浮点加法

虽然这题不会出精度问题，但既然遇到了浮点加法，顺手讲一下。

`float` 是 32 位 IEEE 754 单精度浮点数：
- 1 位符号
- 8 位指数（biased，偏移 127）
- 23 位尾数（隐含的前导 1，所以有效精度是 24 位 ≈ 7 位十进制）

浮点加法 `a + b` 的硬件实现大致是：
1. 比较两数指数，把指数小的那个尾数右移直到指数对齐。
2. 尾数相加。
3. 规格化结果（左移或右移使前导位为 1）。
4. 按舍入模式舍入（默认 round-to-nearest-even）。

**重要陷阱：浮点加法不结合**。即 `(a + b) + c ≠ a + (b + c)` 一般成立。例如 `(1e20 + (-1e20)) + 1 = 1`，但 `1e20 + (-1e20 + 1) = 0`（因为 -1e20+1 的 1 被舍入丢失了）。

这个性质对向量加法本身没影响（每个位置独立计算），但对**归约求和**（下一题或下下题会遇到）影响很大——并行求和的顺序和串行不同，结果会有微小差异。这不是 bug，是浮点的本质。

GPU 浮点行为：默认情况下 NVIDIA GPU 的浮点是 IEEE 754 兼容的。但 nvcc 有一些 fast-math 选项（`-use_fast_math`、`--fmad=true`）会牺牲一点精度换速度，比如把 `a*b+c` 融合成一条 FMA 指令（这其实更精确），或者用近似版的 sin/cos/exp。

## 六、性能分析：Roofline 模型

这是分析 GPU 程序性能上限的标准工具，必须掌握。

### 6.1 算术强度（Arithmetic Intensity）

定义：

$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes from memory}}$$

它衡量"每从内存读 1 字节，能做多少次浮点运算"。这是程序本身的属性，与硬件无关。

向量加法的算术强度：

$$\text{AI}_{\text{vector\_add}} = \frac{N \text{ FLOPs}}{12N \text{ bytes}} = \frac{1}{12} \approx 0.083 \text{ FLOP/byte}$$

（每个元素 1 次加法，访存 = 读 A + 读 B + 写 C = 12 字节）

这是**极低**的算术强度。作为对比，矩阵乘法的算术强度可以达到 $O(N)$（N 是矩阵边长），高了好几个数量级。

### 6.2 Roofline 模型

每个硬件有两个关键指标：
- **峰值算力** $P_{\max}$（FLOPs/s）：核心算得多快
- **峰值带宽** $B_{\max}$（bytes/s）：显存读得多快

它们的比值定义了一个**临界算术强度**：

$$\text{AI}_{\text{crit}} = \frac{P_{\max}}{B_{\max}}$$

对一个程序，可达到的最大性能是：

$$P_{\text{achievable}} = \min(P_{\max}, \text{AI} \times B_{\max})$$

画在图上是个屋顶形状（log-log 坐标），所以叫 Roofline。

- 如果 $\text{AI} < \text{AI}_{\text{crit}}$：程序是 **memory-bound**，性能上限由带宽决定。
- 如果 $\text{AI} > \text{AI}_{\text{crit}}$：程序是 **compute-bound**，性能上限由算力决定。

### 6.3 具体数字

以 NVIDIA A100 为例（数据可能因型号微调，量级正确）：
- FP32 峰值算力 ≈ 19.5 TFLOPs/s
- HBM2e 峰值带宽 ≈ 1555 GB/s
- 临界 AI ≈ 19500 / 1555 ≈ 12.5 FLOP/byte

向量加法的 AI = 0.083，远小于 12.5——**严重 memory-bound**。它的实际性能上限是：

$$P = 0.083 \times 1555 \text{ GB/s} \approx 130 \text{ GFLOPs/s}$$

只有峰值算力的 0.7%。换个角度看：完成 N 个元素加法需要的最短时间是

$$t_{\min} = \frac{12N \text{ bytes}}{1555 \text{ GB/s}}$$

无论你 kernel 写得多花哨都不可能更快。**向量加法的优化目标不是"算得更快"，而是"打满显存带宽"**。

### 6.4 这意味着什么

对向量加法这种 memory-bound 任务：
- ✅ **合并访存**至关重要——不合并就直接掉到峰值带宽的一小部分。
- ✅ **数据搬运越少越好**——这就是为什么 fused kernel（融合算子）很重要：连续做几次操作时，避免中间结果写回 global memory。
- ❌ 用 shared memory **没用**——shared memory 是为了减少重复访问 global memory，但向量加法每个元素只读一次，没有重复。
- ❌ 用更复杂的算法**没用**——计算本身已经是最少的了。

## 七、常见陷阱

**1. 忘了边界检查**

把 `if (idx < N)` 去掉、N 不能整除 256 的话，最后一个 block 的多余线程会越界访问。可能会读到别人的数据（如果别的 cudaMalloc 紧挨着）、可能段错误、可能默默写坏其他变量。CUDA 的越界检测不像 CPU 那么强，这种 bug 隐蔽得很。

**2. 启动配置算错**

`(N + threadsPerBlock - 1) / threadsPerBlock` 这个上取整公式，写错成 `N / threadsPerBlock` 就漏掉一部分元素，N 不能整除时输出末尾全是 0（或者未初始化的脏数据，取决于 C 初始化情况）。

**3. 忘了同步**

LeetGPU 的模板里加了 `cudaDeviceSynchronize()`。在自己写的完整程序里，如果忘记同步就直接读 device 上的 C，会读到还没算完的脏数据。`cudaMemcpy`（同步版）会隐式同步，但纯设备内的连续 kernel 之间默认是按顺序执行的（同一个 stream），不需要显式同步。

**4. threadsPerBlock 选了奇怪的数**

比如 100、200。GPU 还是按 32 一组调度 warp，会有"死线程"占着位置不干活，浪费 occupancy。永远用 32 的倍数，常见 128、256、512。

## 八、优化

### 1. 常见优化
下面这些不是这道题要求的，但都是同类问题的常见优化模式，提一下让你心里有个数：

**Grid-stride loop**：当 N 大到 grid 都装不下时（比如 N = 10 亿），用循环让每个线程处理多个元素：

```cuda
for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
     i < N; 
     i += gridDim.x * blockDim.x) {
    C[i] = A[i] + B[i];
}
```

这样可以用一个固定大小的 grid（比如 grid = SM 数 × 4）处理任意大的 N，启动开销也更小。

**向量化访存**：用 `float4` 一次读 16 字节，可以减少访存指令数量、提高带宽利用率：

```cuda
float4 a = reinterpret_cast<const float4*>(A)[idx];
float4 b = reinterpret_cast<const float4*>(B)[idx];
float4 c;
c.x = a.x + b.x; c.y = a.y + b.y; c.z = a.z + b.z; c.w = a.w + b.w;
reinterpret_cast<float4*>(C)[idx] = c;
```

这样总线程数变成 N/4，但每个线程做 4 个元素。要求地址对齐到 16 字节、N 是 4 的倍数。

每个线程处理 1 个元素。能跑，合并访存也满足。带宽利用率通常 70~85%，看 GPU 和驱动版本。

### 2 优化 1：向量化访存（float4）

**核心思想**：让一条 load 指令搬运 16 字节而不是 4 字节，减少访存指令总数，更好地利用 LSU（Load/Store Unit）和内存控制器。

```cuda
__global__ void vector_add_vec4(const float4* A, const float4* B, float4* C, int N4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N4) {
        float4 a = A[idx];
        float4 b = B[idx];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        C[idx] = c;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int N4 = N / 4;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N4 + threadsPerBlock - 1) / threadsPerBlock;
    vector_add_vec4<<<blocksPerGrid, threadsPerBlock>>>(
        reinterpret_cast<const float4*>(A),
        reinterpret_cast<const float4*>(B),
        reinterpret_cast<float4*>(C),
        N4
    );
    
    // 处理 N 不能被 4 整除的尾部
    int remainder = N % 4;
    if (remainder > 0) {
        int tail_start = N4 * 4;
        // 启动一个小 kernel 处理剩下的 0~3 个元素
        // 此处省略，N=25,000,000 = 4×6,250,000 正好整除
    }
    cudaDeviceSynchronize();
}
```

**为什么 N=25,000,000 能整除 4**：25M = 4 × 6.25M，正好。所以这道题直接用 float4 不用处理尾部，干净。

#### 向量化访存的硬件原理

`float4` 是 CUDA 内置的结构体，包含 4 个 float，**在内存中连续布局，且要求 16 字节对齐**。`cudaMalloc` 返回的指针对齐到 256 字节，所以从头访问 float4 一定是对齐的。

编译器看到 `A[idx]`（A 是 float4*）时，会生成一条 **`LD.E.128`** 指令（PTX 层面是 `ld.global.v4.f32`），一次读 128 位 = 16 字节。

**关键概念：MLP（Memory-Level Parallelism）**

GPU 显存延迟很高（几百个时钟周期），但 GPU 通过两种方式隐藏延迟：
1. **TLP（Thread-Level Parallelism）**：一个 warp 在等内存时，SM 切换到另一个 warp 干活。这是"用并发数掩盖延迟"。
2. **MLP（Memory-Level Parallelism）**：单个线程同时发出多个独立的访存请求，让它们在内存系统里 pipeline 起来。

float4 同时增加了两者的效率：每个线程每次发出的"访存请求"携带 4 倍数据，等价于给每个 warp 提供了 4 倍的访存带宽利用率（在事务数有限的情况下）。

#### 为什么向量化有效

GPU 的内存控制器有事务调度的开销。一次 128 字节事务 vs 四次 32 字节事务，前者效率高得多。每个 warp（32 线程）的访存请求：

- 标量版：32 线程 × 4 字节 = 128 字节 → 1 次事务
- float4 版：32 线程 × 16 字节 = 512 字节 → 4 次事务，但 **访存指令只发 1 次**

这里有个微妙之处。看起来事务数变多了，但有几个好处：
1. **指令数减少 4 倍**：每个线程只执行 N/4 次 load 而不是 N 次。指令吞吐也是有限资源。
2. **每条指令发起的内存请求更"重"**：内存系统更容易并行化、流水化。
3. **每个线程做更多工作**：launch overhead 摊薄。

实测向量化通常带来 5~15% 的提升，对 memory-bound kernel 算可观了。

### 3 优化 2：Grid-Stride Loop

**核心思想**：不要让 block 数和 N 一一对应，而是用一个**固定大小**的 grid 反复处理多个元素。

```cuda
__global__ void vector_add_grid_stride(
    const float4* A, const float4* B, float4* C, int N4) 
{
    int stride = gridDim.x * blockDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < N4; 
         idx += stride) 
    {
        float4 a = A[idx];
        float4 b = B[idx];
        float4 c;
        c.x = a.x + b.x; c.y = a.y + b.y; 
        c.z = a.z + b.z; c.w = a.w + b.w;
        C[idx] = c;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int N4 = N / 4;
    int threadsPerBlock = 256;
    
    // 用一个相对小的、与 SM 数挂钩的 grid
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int blocksPerGrid = numSMs * 32;  // 每个 SM 32 个 block，经验值
    
    vector_add_grid_stride<<<blocksPerGrid, threadsPerBlock>>>(
        reinterpret_cast<const float4*>(A),
        reinterpret_cast<const float4*>(B),
        reinterpret_cast<float4*>(C),
        N4
    );
    cudaDeviceSynchronize();
}
```

#### 为什么 grid-stride loop 有用

**第一个理由：launch overhead**

启动一个有 10 万个 block 的 grid，CUDA driver 需要做一些 bookkeeping（虽然每个 block 启动很快，但累积起来有几十微秒）。用 grid-stride loop，grid 只有几百个 block，启动几乎是即时的。

对 N = 25M、运行时间 ~0.2 ms 的任务，几十微秒的 launch overhead 占比就有 10~20%，不可忽视。

**第二个理由：编译器循环优化**

grid-stride loop 让编译器看到一个明显的循环，可以做：
- **指令调度**：把 load 指令提前发，让多次迭代的访存重叠（隐藏延迟）。
- **寄存器复用**：循环变量、stride 等放寄存器里复用。
- **软件流水（software pipelining）**：把当前迭代的计算和下一迭代的 load 重叠。

第一个版本里，每个线程只做一次访存，编译器没办法做这种优化。

**第三个理由：与 SM 数对齐**

GPU 的工作分配是按 block 为单位调度到 SM 上的。SM 数（A100 是 108，H100 是 132）决定了"硬件的并行宽度"。如果 grid 远大于 SM 数 × 每 SM 最大常驻 block 数，多余的 block 只能排队等待——它们提供不了额外的并发收益，只是单纯增加调度开销。

经验法则：grid 大小 = SM 数 × 每 SM 期望常驻 block 数（典型 8~32 之间，看 occupancy）。这是 NVIDIA 自己在 cuBLAS、cuDNN 里反复用的模式。

### 4 综合：向量化 + grid-stride + 每线程多元素

最激进的版本是每个线程一次处理多个 float4：

```cuda
__global__ void vector_add_ultra(
    const float4* A, const float4* B, float4* C, int N4) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // 每个线程一次处理 4 个 float4 = 16 个 float
    for (int idx = tid; idx < N4; idx += stride) {
        float4 a = A[idx];
        float4 b = B[idx];
        float4 c;
        c.x = a.x + b.x; c.y = a.y + b.y;
        c.z = a.z + b.z; c.w = a.w + b.w;
        C[idx] = c;
    }
}
```

实际上 grid-stride 本身已经让每个线程处理多个元素，再嵌套展开收益递减。对纯加法这种 memory-bound 任务，到此为止基本就贴近带宽峰值了。


### 5、新引出的概念，简单点一下

- **Occupancy（占用率）**：每个 SM 上同时活跃的 warp 数 / 最大可能 warp 数。决定了 SM 隐藏延迟的能力。受寄存器数、shared memory 用量、block 大小三者制约。
- **MLP / TLP**：内存级并行 vs 线程级并行，两种隐藏延迟的机制。
- **PTX vs SASS**：CUDA 的两层"汇编"——PTX 是虚拟 ISA（跨架构），SASS 是真实硬件指令。`cuobjdump` / `nvdisasm` 可以查看。
- **LSU（Load/Store Unit）**：SM 里负责访存的硬件单元，数量有限，是访存吞吐的瓶颈之一。


## 三、性能数字预估

按 N = 25,000,000、A100 大致推算（实测会有偏差）：

| 版本 | 大致带宽利用率 | 时间 |
|------|--------------|------|
| 基础版（每线程 1 个 float） | ~75% | ~0.26 ms |
| float4 向量化 | ~85% | ~0.23 ms |
| float4 + grid-stride | ~90% | ~0.21 ms |
| 理论下界（100% 带宽） | 100% | 0.19 ms |

差距看起来不大（毫秒级），但相对幅度可观。LeetGPU 应该是按总执行时间排名，这里能挤出 10~20% 就能上一个台阶。

## 四、对这道题的实战建议

**第一步**：先提交基础版，确保正确性。LeetGPU 通常会显示你的运行时间。

**第二步**：换 float4 版本。N = 25M 整除 4，没有尾部处理的麻烦。这一步通常涨分明显。

**第三步**：加 grid-stride。这一步看具体硬件和驱动，有时收益小（编译器已经做了类似优化），有时收益大（特别是新硬件 + 老编译器组合）。

**第四步**：调 threadsPerBlock。试 128、256、512，看哪个快。这影响 occupancy，没有理论最优解，只能实测。

## 九、小结

这道题的核心知识点：
- **执行模型**：grid / block / thread / warp 的层次结构，SIMT 执行，warp divergence 的代价。
- **内存访问**：合并访存原理，对齐要求，warp 整体作为访存单位。
- **性能模型**：算术强度 AI = 1/12，memory-bound，性能上限由显存带宽决定。
- **数学背景**：浮点数 IEEE 754 的基本结构，加法不结合（这道题不影响，下题归约会影响）。
