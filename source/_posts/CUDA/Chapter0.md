---
title: 0. 从C++到CUDA
categories: LeetGPU 
date: 2026-05-09 22:30:00
mathjax: true
tags:
    - AI
    - AI Infra
---


# 从 C++ 到 CUDA：完整入门笔记

## 第一部分：核心思维转变

### 1.1 设计哲学的差异

| | CPU | GPU |
|---|---|---|
| 核心数量 | 少（8~32） | 多（数千） |
| 单核能力 | 强（缓存、分支预测、乱序执行） | 弱（砍掉了大部分控制逻辑） |
| 擅长 | 复杂的串行逻辑 | 对海量数据做同样的简单操作 |
| 设计目标 | 低延迟 | 高吞吐 |

### 1.2 四个关键思维转变

**从"循环"到"线程网格"**

```cpp
// CPU 视角：我循环 N 次
for (int i = 0; i < N; ++i) C[i] = A[i] + B[i];

// GPU 视角：启动 N 个线程，每个线程做一件事
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) C[idx] = A[idx] + B[idx];
```

每个线程相当于循环的一次迭代被"展开"。

**从"我"到"我们"**

写 kernel 时，代码会被几万个线程同时执行。要时刻想"这一万个'我'各自在做什么，会不会撞车"。读写同一地址要用原子操作，block 内同步用 `__syncthreads()`。

**从"算得快"到"喂得饱"**

GPU 编程中算力几乎从来不是瓶颈，瓶颈是**内存带宽**——能不能让那几千个核心都有活干，不是在等数据。这是和 ICPC/CPU 优化最大的不同。

**从"按需分配"到"批量并行"**

GPU 不擅长动态内存分配、复杂分支、不规则访问。算法常常被重新设计成**规则的、数组化的**形式。

---

## 第二部分：CUDA 的两套内存空间

### 2.1 物理上分离的两套内存

```
CPU 主内存 (Host)              GPU 显存 (Device)
┌──────────────┐               ┌──────────────────┐
│ malloc 分配  │  PCIe / NVLink │ cudaMalloc 分配  │
│ h_ 前缀指针  │ ◄────────────► │ d_ 前缀指针      │
└──────────────┘   cudaMemcpy   └──────────────────┘
```

**关键约束**：
- Host 指针不能在 GPU kernel 里解引用
- Device 指针不能在 CPU 代码里解引用
- 编译器**不会**帮你检查，错用会段错误或更怪异的行为
- 数据必须用 `cudaMemcpy` 显式搬运

### 2.2 PCIe 是性能瓶颈

- PCIe 4.0 x16：~32 GB/s
- GPU 显存带宽：~1000+ GB/s（差几十倍）

**反模式**：拷数据上去 → 跑简单 kernel → 拷回来——传输时间可能比计算时间还长。
**最佳实践**：数据一次上传后，留在显存上多次复用。

LeetGPU 的题目数据已经在 device 上，不涉及这层。

### 2.3 指针变量 vs 指向的内存

```cpp
float *d_A;                  // (1) 栈上的指针变量，初值随机（脏数据）
cudaMalloc(&d_A, bytes);     // (2) 把 GPU 显存地址写进 d_A，覆盖随机值
```

```
CPU 栈                          GPU 显存
┌─────────────┐                 ┌──────────────────────┐
│ d_A: 0x7f.. │ ─────────────▶ │ 100MB 的连续区域      │
└─────────────┘                 └──────────────────────┘
   8 字节                       cudaMalloc 保证虚拟连续
```

**两件事独立**：
- 指针变量本身的初值（随机）→ 被 cudaMalloc 覆盖
- 分配的空间连续性 → cudaMalloc 的契约保证（虚拟地址连续，物理上由 GPU MMU 处理）

---

## 第三部分：CUDA 程序的七步骤模板

```cpp
#include <cuda_runtime.h>

__global__ void kernel(const float* A, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] * 2;
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    // 1. Host 准备数据
    float* h_A = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) h_A[i] = 1.0f * i;

    // 2. cudaMalloc 分配显存
    float *d_A, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_C, bytes);

    // 3. cudaMemcpy 上传数据
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    // 4. 启动 kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);

    // 5. cudaMemcpy 下载结果（隐式同步）
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 6. 使用结果
    printf("h_C[100] = %f\n", h_C[100]);

    // 7. 释放资源
    cudaFree(d_A); cudaFree(d_C);
    free(h_A); free(h_C);
    return 0;
}
```

LeetGPU 帮你做了 1、2、3、5、6、7，你只写第 4 步（kernel + 启动）。

### 3.1 cudaMemcpy 方向参数

- `cudaMemcpyHostToDevice`：H → D
- `cudaMemcpyDeviceToHost`：D → H
- `cudaMemcpyDeviceToDevice`：D 内部
- `cudaMemcpyHostToHost`：基本用不到

`cudaMemcpy`（同步版）会**隐式同步**——会等队列里所有 kernel 跑完。这就是为什么很多简单程序不需要显式 `cudaDeviceSynchronize()`。

---

## 第四部分：CUDA 语法层面的差异

### 4.1 编译

| | C++ | CUDA |
|---|---|---|
| 文件后缀 | `.cpp` | `.cu` |
| 编译器 | `g++` / `clang++` | `nvcc` |

`nvcc` 是驱动程序：把 `.cu` 拆成 host 部分（丢给 g++）和 device 部分（自己编译成 PTX），最后链接。可以在 `.cu` 里自由混写 C++ 和 CUDA。

### 4.2 函数修饰符

| 修饰符 | 在哪执行 | 从哪调用 |
|--------|---------|---------|
| `__host__`（默认） | CPU | CPU |
| `__global__` | GPU | CPU |
| `__device__` | GPU | GPU |

`__host__ __device__` 可叠加，让函数两边都能编译：

```cpp
__host__ __device__ float square(float x) { return x * x; }
```

### 4.3 kernel 调用语法

```cpp
kernel<<<grid, block>>>(args);
kernel<<<grid, block, sharedMemBytes, stream>>>(args);  // 完整形式
```

`<<<...>>>` 是 nvcc 特有的语法扩展。

### 4.4 GPU 代码的限制

- 不能 `throw` / `try-catch`
- 不能用 `std::vector`、`std::string`
- 标准库基本不能用（有 GPU 版的 Thrust 库）
- 错误检查靠返回值

**推荐宏**：

```cpp
#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err));  \
        exit(1);                                               \
    }                                                          \
} while (0)
```

CUDA 错误经常是悄无声息的，不查返回值会让你 debug 时怀疑人生。

### 4.5 异步执行模型

```cpp
kernel<<<...>>>(...);          // 不阻塞，CPU 立即继续
cpu_do_something();            // 可能在 kernel 还没跑完时就执行
cudaDeviceSynchronize();       // 显式等 GPU 跑完
```

**陷阱：错误也是异步报告的**。kernel 内越界访问，`<<<...>>>` 调用本身**不会**报错；要等下一次 CUDA API 调用才能拿到错误。

debug 时的标准做法：

```cpp
kernel<<<...>>>(...);
CUDA_CHECK(cudaGetLastError());        // 检查启动配置错误
CUDA_CHECK(cudaDeviceSynchronize());   // 等 kernel 跑完，捕获运行时错误
```

---

## 第五部分：执行模型——grid / block / thread / warp

### 5.1 三层结构

```
Grid (网格)
  └── Block (线程块) × blocksPerGrid 个
        └── Thread (线程) × threadsPerBlock 个
              └── 32 个组成一个 Warp（硬件层面）
```

类比：
- **thread** = 一个工人
- **block** = 一个工程队（共用工棚 = shared memory，可互相喊话 = `__syncthreads()`）
- **grid** = 整个工地

### 5.2 Block 和 SM 的关系

GPU 硬件层面的执行单元叫 **SM (Streaming Multiprocessor)**：
- 一个 block 整体调度到某个 SM 上执行
- block 内线程共享该 SM 上的 shared memory
- block 之间可能在不同 SM 上，互相隔离不能直接通信

### 5.3 Warp：硬件真实调度单位

block 里的线程在硬件上**32 个一组**（一个 warp）调度，整组一起执行同一条指令——**SIMT** (Single Instruction Multiple Threads)。

**推论**：
- threadsPerBlock 必须是 **32 的倍数**，否则有"死线程"占位
- 常用值：128、256、512
- 256 是不错的默认值

### 5.4 Warp Divergence

warp 里所有线程**共用一个 PC（程序计数器）**，必须执行同一条指令。遇到 if-else：

```
if (cond) { A; }
else      { B; }
```

执行流程：
1. mask = "满足 cond 的线程"，所有线程一起走 A，不满足的结果被丢弃
2. mask 反过来，再走一遍 B
3. **A 和 B 串行执行，时间是两者之和**

这就是 warp divergence 的代价。GPU 不擅长复杂分支逻辑的根本原因。

> Volta（2017）后引入 Independent Thread Scheduling，每线程独立 PC，但代价模型基本不变。

---

## 第六部分：内置变量——CUDA 最重要的疑问

### 6.1 kernel 调用的两组信息

```cpp
vector_add  <<<grid, block>>>  (A, B, C, N);
            ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^
            执行配置             函数参数
            传给硬件调度器       和普通 C++ 一样
            告诉启动多少线程     所有线程收到同样的参数
```

### 6.2 内置变量从哪来

`blockIdx`、`threadIdx` 等**不是函数参数**，是**硬件给每个线程提供的"内置变量"**——本质是特殊寄存器。

| 内置变量 | 含义 | 类型 |
|---------|------|-----|
| `threadIdx` | 我在 block 内的坐标 | dim3 |
| `blockIdx` | 我在 grid 内的坐标 | dim3 |
| `blockDim` | block 的尺寸 | dim3 |
| `gridDim` | grid 的尺寸 | dim3 |
| `warpSize` | warp 大小（永远 32） | int |

PTX 层面：

```ptx
mov.u32  %r1, %ctaid.x;     // blockIdx.x
mov.u32  %r2, %ntid.x;      // blockDim.x
mov.u32  %r3, %tid.x;       // threadIdx.x
mad.lo.s32 %r4, %r1, %r2, %r3;
```

`%ctaid`、`%ntid`、`%tid` 是 PTX 的**特殊寄存器**——硬件保证读它们时返回当前线程的身份。

类比 CPU 的 TLS、ARM 的 MPIDR——硬件给执行单元注入身份信息。

### 6.3 为什么这种设计

如果让 CUDA 把 idx 作为参数传进去，就要预先生成 25M 组参数、启动 25M 次函数调用——CPU 多线程模型，开销巨大。

GPU 的做法：你描述启动多少线程，硬件**自己**给每个线程编号，每个线程**自己**查自己的编号。
- 启动开销极低
- 编号天然分布式
- 代码可以写成"我是谁，我做什么"的视角

### 6.4 心智模型：CUDA 的"两个世界"

**逻辑世界**（业务）：
- 数据：A、B、C 是什么
- 参数：N 是多少
- 通过 `(A, B, C, N)` 函数参数传

**硬件世界**（GPU 调度）：
- 启动多少线程：`<<<grid, block>>>`
- 我是谁：`threadIdx`、`blockIdx`（硬件注入）
- 总共多少人：`blockDim`、`gridDim`（硬件注入）

这两个世界通过 `idx = blockIdx.x * blockDim.x + threadIdx.x` **桥接**——把"我是哪个线程"翻译成"我处理哪份数据"。

---

## 第七部分：线程到数据的映射

### 7.1 重要澄清

CUDA **不会自动**把数组元素分配给线程。它做的事：
1. 把参数 `(d_A, d_B, d_C, N)` 复制给每个线程——**所有线程看到相同的参数**
2. 启动 `blocksPerGrid * threadsPerBlock` 个线程
3. 给每个线程分配唯一的 `(blockIdx, threadIdx)`
4. 让每个线程跑同一份 kernel 代码

**"线程 i 处理数据 i" 这种映射是你在 kernel 里写的**：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) C[idx] = A[idx] + B[idx];
```

如果你写 `int idx = 0`，所有 25M 线程会争抢写 `C[0]`——CUDA 不报错，结果就是错的。

### 7.2 各种映射模式

不同问题有不同的映射规则。这是 CUDA 编程的核心设计决策。

**1D 一对一**（向量加法）：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

**Grid-stride loop**（一个线程处理多个元素）：

```cpp
int stride = gridDim.x * blockDim.x;
for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
     idx < N; 
     idx += stride) { ... }
```

**关键点**：用 `idx += stride` 而非 `idx += 1`，是为了保证同一时刻 warp 里 32 个线程访问连续地址（合并访存）。

**2D 网格**（图像、矩阵）：

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int idx = row * W + col;
```

**关键点**：让 `threadIdx.x` 对应 col（连续维度），保证合并访存。

**Block 协作**（reduction、矩阵乘法）：

每个 block 负责一段数据，block 内线程通过 shared memory 协作。

### 7.3 设计选择遵循的原则

不只是"逻辑上对"，还要：
- **合并访存**（warp 内 32 线程访问连续地址）
- **避免 divergence**（warp 内尽量走同一分支）
- **充分并行**（避免大量线程闲置）
- **数据复用**（用 shared memory 减少 global memory 访问）

---

## 第八部分：合并访存（Coalesced Access）—— GPU 优化第一原则

### 8.1 GPU 显存的特性

GPU 显存（GDDR/HBM）按 burst 读取，一次最少读一行（32~128 字节）。

CPU 用 cache line 处理（64 字节自动缓存附近内容）。

GPU 用 **coalescing 处理**：硬件检查 warp 内 32 个线程的访存地址，**拼起来一次读**。

### 8.2 合并访存的规则

- **理想**：32 线程访问连续的、对齐的 128 字节 → 1 次内存事务
- **最坏**：32 线程访问 32 个不同的 128 字节区域 → 32 次事务，**带宽利用率 1/32**

### 8.3 例子

```cpp
// 好：warp 内线程的 idx 是 0,1,2,...,31，地址连续
int idx = blockIdx.x * blockDim.x + threadIdx.x;
C[idx] = A[idx] + B[idx];

// 坏：warp 内线程访问跨度大
int idx = threadIdx.x * gridDim.x + blockIdx.x;
C[idx] = A[idx] + B[idx];
```

### 8.4 二维数据的陷阱

```cpp
// 错（按 C 风格行主序存储时）：threadIdx.x 跨行访问
int row = threadIdx.x;
int col = threadIdx.y;
M[row * W + col]  // warp 内线程访问 M[0*W], M[1*W], M[2*W], ... 跨度巨大

// 对：threadIdx.x 沿行内访问
int col = threadIdx.x;
int row = threadIdx.y;
M[row * W + col]  // warp 内线程访问连续地址
```

### 8.5 对齐要求

`cudaMalloc` 返回的指针对齐到 256 字节，从开头访问肯定对齐。但 `A + 1` 这种偏移可能跨两个 128 字节块，需要 2 次事务。

---

## 第九部分：性能模型——Roofline

### 9.1 算术强度（Arithmetic Intensity）

$$\text{AI} = \frac{\text{FLOPs}}{\text{从内存读的字节数}}$$

每读 1 字节能做多少次浮点运算。**这是程序本身的属性，与硬件无关**。

向量加法：$\text{AI} = \frac{N}{12N} = \frac{1}{12} \approx 0.083$ FLOP/byte（极低）

矩阵乘法：$\text{AI} = O(N)$（很高）

### 9.2 Roofline 模型

每个硬件有两个指标：
- 峰值算力 $P_{\max}$（FLOPs/s）
- 峰值带宽 $B_{\max}$（bytes/s）

$$\text{AI}_{\text{crit}} = \frac{P_{\max}}{B_{\max}}$$

可达最大性能：

$$P_{\text{achievable}} = \min(P_{\max}, \text{AI} \times B_{\max})$$

- $\text{AI} < \text{AI}_{\text{crit}}$ → **memory-bound**，性能上限由带宽决定
- $\text{AI} > \text{AI}_{\text{crit}}$ → **compute-bound**，性能上限由算力决定

### 9.3 A100 实例

- FP32 峰值算力 ≈ 19.5 TFLOPs/s
- HBM2e 带宽 ≈ 1555 GB/s
- 临界 AI ≈ 12.5 FLOP/byte

向量加法 AI = 0.083 → memory-bound，性能上限 ≈ 130 GFLOPs/s（峰值算力的 0.7%）。

### 9.4 优化导向

对 memory-bound 任务：
- ✅ 合并访存
- ✅ 减少数据搬运（kernel fusion）
- ❌ shared memory（每元素只读一次，无复用）
- ❌ 复杂算法（计算已最少）

---

## 第十部分：内存管理细节

### 10.1 malloc 的底层

```cpp
float* h_A = (float*)malloc(bytes);
```

- **小请求**：从 glibc 维护的堆里切（用 brk 系统调用扩展）
- **大请求（≥128KB）**：走 `mmap` 直接向内核要匿名映射

**懒分配**：`malloc` 只是登记 VMA，**不分配物理内存**。第一次写入触发**缺页中断**才分配物理页。

```cpp
auto t1 = clock();
float* p = (float*)malloc(100*MB);  // 几微秒，不分配物理内存
auto t2 = clock();
for (int i=0; i<N; ++i) p[i] = 0;   // 真正的"分配"开销在这里（缺页）
auto t3 = clock();
```

### 10.2 cudaMalloc vs malloc

| | `malloc` | `cudaMalloc` |
|---|---|---|
| 内存位置 | CPU DRAM | GPU 显存 |
| 谁能访问 | CPU | GPU |
| 大请求路径 | mmap | 驱动管理的显存分配器 |
| 分配时机 | 懒分配（首次访问） | 立即分配 |
| 速度 | 几微秒 | 几十~几百微秒 |
| 是否清零 | 不 | 不 |

### 10.3 Pinned Memory（性能优化）

`cudaMallocHost` / `cudaHostAlloc` 分配**页锁定内存**：
- 不会被 swap
- `cudaMemcpy` 更快（PCIe 4.0 上 ~25 GB/s vs ~12 GB/s）
- 支持 `cudaMemcpyAsync` 异步传输

代价：占真实物理内存，分配慢，分配太多影响系统。

---

## 第十一部分：指针与索引

### 11.1 `A[idx] ≡ *(A + idx)`

**步长由指针类型决定，不是索引类型**：

```cpp
float* A;  int i = 3;
A + i  →  地址 = A + 3 × sizeof(float) = A + 12

double* B;  int i = 3;
B + i  →  地址 = B + 3 × sizeof(double) = B + 24
```

x86-64 汇编层面：

```asm
movss xmm0, [rax + rcx*4]   ; *4 是编译器根据 float* 编进指令的
```

### 11.2 索引类型只关心溢出

```cpp
// N 较小，int 够用
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// N 超过 21 亿，要警惕
size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
//           ↑ 必须在乘法前转换，否则按 32 位算先溢出再转
```

CUDA 的 `threadIdx.x` 等都是 `unsigned int`（32 位）。

---

## 第十二部分：浮点数 IEEE 754

### 12.1 float 结构

32 位 = 1 符号 + 8 指数（偏移 127）+ 23 尾数（隐含前导 1，有效 24 位 ≈ 7 位十进制）

### 12.2 浮点加法不结合

`(a + b) + c ≠ a + (b + c)` 一般成立。

例：`(1e20 + (-1e20)) + 1 = 1`，但 `1e20 + (-1e20 + 1) = 0`

**对向量加法无影响**（每位置独立）。**对并行归约有影响**——并行求和顺序与串行不同，结果会有微小差异，不是 bug。

### 12.3 GPU 浮点

NVIDIA GPU 默认 IEEE 754 兼容。`-use_fast_math` 选项会牺牲精度换速度（用近似 sin/cos/exp、融合 FMA 等）。

---

## 第十三部分：常见陷阱清单

1. **混用 host/device 指针**：`*d_A` 在 CPU 代码里、`*h_A` 在 kernel 里都会崩
2. **忘了边界检查 `if (idx < N)`**：最后一个 block 的多余线程越界
3. **启动配置算错**：用 `(N + tpb - 1) / tpb` 上取整，不是 `N / tpb`
4. **threadsPerBlock 不是 32 的倍数**：有死线程，浪费 occupancy
5. **忘了同步**：异步 kernel 还没跑完就读结果
6. **不检查 CUDA 错误**：错误悄无声息，用 `CUDA_CHECK` 宏
7. **整数索引溢出**：N 大于 21 亿要用 `size_t`
8. **二维访存方向错**：`threadIdx.x` 应对应连续维度
9. **依赖多次 cudaMalloc 返回的地址相邻**：实际不保证

---

## 第十四部分：心智模型总结

每写一个 kernel 前，先回答四个问题：

1. **数据怎么分？** 多少 block、多少线程？每个线程负责哪些数据？
2. **数据放哪？** Global memory 够，还是要搬到 shared memory？
3. **线程要不要协作？** 独立干活（向量加法）还是协作（归约、矩阵乘法）？
4. **数据怎么进出 GPU？** 一次上传多次算，还是反复来回？

这四个问题贯穿所有 CUDA 题目。每道题的不同就在于答案不同。