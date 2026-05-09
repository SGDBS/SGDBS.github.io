---
title: 0. 从C++到CUDA
categories: LeetGPU 
date: 2026-05-09 22:30:00
mathjax: true
tags:
    - AI
    - AI Infra
---


## 一、先看一个完整的 CUDA 程序

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ========== GPU 端代码 ==========
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// ========== CPU 端代码 ==========
int main() {
    const int N = 1 << 20;              // 1M 个元素
    const size_t bytes = N * sizeof(float);

    // 1. 在 CPU (host) 上分配并初始化数据
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f * i;
        h_B[i] = 2.0f * i;
    }

    // 2. 在 GPU (device) 上分配显存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 3. 把输入数据从 CPU 拷到 GPU
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 4. 启动 kernel（在 GPU 上执行计算）
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 5. 把结果从 GPU 拷回 CPU
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 6. 验证 & 使用结果
    printf("C[100] = %f (expected %f)\n", h_C[100], 3.0f * 100);

    // 7. 释放资源
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
```

编译：`nvcc vector_add.cu -o vector_add`，运行：`./vector_add`。

这个程序展示了 CUDA 的所有核心环节。下面拆开讲。

## 二、最重要的概念转变：两套内存空间

C++ 程序员眼里只有一片内存——你 `malloc` 一块，`*p` 就能读写。CUDA 里有**两套独立的内存**，物理上是分开的：

- **Host memory（主机内存）**：CPU 旁边的 DRAM 内存条，C++ 的 `malloc/new` 分配的就是这里。
- **Device memory（设备内存）**：GPU 显卡上的显存（GDDR / HBM），由 `cudaMalloc` 分配。

它们之间隔着 **PCIe 总线**（或者 NVLink），数据传输有明显延迟和带宽限制。这是 CUDA 编程**最关键的认知差异**。

后果：

**指针不能乱用。** `cudaMalloc` 给你的指针指向显存，CPU 代码 `*d_A` 解引用会段错误（或者更难调的怪异行为）。反过来 GPU kernel 解引用 host 指针也会崩。习惯上用 `h_` 前缀表示 host 指针、`d_` 前缀表示 device 指针，就是为了肉眼区分。编译器**不会**帮你检查。

**数据必须显式搬运。** 这就是 `cudaMemcpy` 的作用，它的方向参数：
- `cudaMemcpyHostToDevice`：CPU → GPU
- `cudaMemcpyDeviceToHost`：GPU → CPU
- `cudaMemcpyDeviceToDevice`：GPU 内部拷贝
- `cudaMemcpyHostToHost`：CPU 内部（基本用不到）

**这次拷贝往往是性能瓶颈。** PCIe 4.0 x16 大约 32 GB/s，而现代 GPU 显存带宽轻松上 1000 GB/s。所以一个反模式是"拷数据上去 → 跑一个简单 kernel → 拷回来"——传输时间可能比计算时间还长。GPU 的优势要在数据**留在显存上多次复用**时才能体现。这也是为什么 LeetGPU 的题目都假设数据**已经**在 device 上：它要测的是你 kernel 写得好不好，不是 PCIe 速度。

## 三、CUDA 程序的七步骤模板

把上面的代码再抽象一下，几乎所有 CUDA 程序都是这个流程：

1. **Host 准备数据**：和普通 C++ 一样。
2. **`cudaMalloc` 分配显存**：参数顺序有点反人类，是 `cudaMalloc(&d_ptr, bytes)`，传指针的指针，因为它要修改 `d_ptr` 本身。
3. **`cudaMemcpy` 上传数据**：H2D 方向。
4. **启动 kernel**：`kernel<<<grid, block>>>(args...)`。
5. **`cudaMemcpy` 下载结果**：D2H 方向。这一步**隐式同步**了——它会等 kernel 跑完才开始拷，所以你常常不需要单独写 `cudaDeviceSynchronize()`。
6. **使用结果**：在 host 上做你想做的事。
7. **释放资源**：`cudaFree` 显存，`free` 主机内存。

LeetGPU 帮你做了 1、2、3、5、6、7，把数据已经放在显存上、结果留在显存上让评测器读。你只写 4。

## 四、和 C++ 的语法层面差异

### 4.1 文件后缀和编译器

- C++：`.cpp` 文件，`g++` / `clang++` 编译。
- CUDA：`.cu` 文件，`nvcc` 编译。

`nvcc` 不是从头到尾自己编译的，它是个**驱动程序**：把 `.cu` 文件拆成 host 部分和 device 部分。host 部分丢给系统的 C++ 编译器（Linux 下 g++、Windows 下 MSVC）；device 部分自己编译成 PTX（一种 GPU 汇编中间语言），最后链接到一起。所以你能在 `.cu` 文件里自由混写 C++ 和 CUDA。

### 4.2 函数修饰符

C++ 没有这些，CUDA 多了三个：

| 修饰符 | 在哪执行 | 从哪调用 |
|--------|---------|---------|
| `__host__`（默认） | CPU | CPU |
| `__global__` | GPU | CPU（或 GPU 上的另一个 kernel，叫 dynamic parallelism） |
| `__device__` | GPU | GPU |

`__host__` 和 `__device__` 可以**叠加**，让同一个函数在两边都能编译。这对一些简单工具函数很有用，比如：

```cpp
__host__ __device__ float square(float x) { return x * x; }
```

### 4.3 kernel 调用语法

```cpp
kernel<<<grid, block>>>(args);
kernel<<<grid, block, sharedMemBytes, stream>>>(args);  // 完整形式
```

那个 `<<<...>>>` 是 nvcc 特有的语法扩展，普通 C++ 编译器看不懂。后两个参数（动态共享内存大小、stream）以后用到再讲。

### 4.4 没有异常、没有 RAII（在 device 代码里）

GPU 代码里：
- 不能 `throw` / `try-catch`。
- 不能用 `std::vector`、`std::string` 这些。
- C++ 标准库基本都不能用，但有个叫 **Thrust** 的库提供了 GPU 版的 vector、algorithm。
- 错误检查全靠返回值，`cudaMalloc` 等函数返回 `cudaError_t`，你应该检查它。

实战中通常会包一个宏：

```cpp
#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err));  \
        exit(1);                                               \
    }                                                          \
} while (0)

CUDA_CHECK(cudaMalloc(&d_A, bytes));
```

**强烈建议养成这个习惯。** CUDA 错误经常是悄无声息的，不查返回值会让你 debug 时怀疑人生。

### 4.5 异步执行模型

这是和 C++ 最不一样的运行时行为：

```cpp
kernel<<<...>>>(...);   // 不阻塞，CPU 立即继续
cpu_do_something();     // 这一行可能在 kernel 还没跑完时就执行了
cudaDeviceSynchronize();// 显式等 GPU 跑完
```

C++ 里你调一个函数，它返回时函数就执行完了。CUDA 里 **kernel 启动是异步的**——`<<<...>>>` 只是把 kernel 加进 GPU 的执行队列，CPU 立刻拿回控制权。

这有两个意味：

**好处**：CPU 可以一边等 GPU 算、一边干别的（准备下一批数据、做 IO）。这是性能优化的重要手段。

**陷阱**：错误也是异步报告的。kernel 里越界访问，`<<<...>>>` 调用本身**不会**报错；要等下一次 CUDA API 调用（通常是 `cudaMemcpy` 或 `cudaDeviceSynchronize`）才能拿到错误。debug 时这点很坑，所以会写：

```cpp
kernel<<<...>>>(...);
CUDA_CHECK(cudaGetLastError());        // 检查启动配置错误（如线程数过多）
CUDA_CHECK(cudaDeviceSynchronize());   // 等 kernel 跑完，捕获运行时错误
```

`cudaMemcpy`（普通版本，不是 `Async` 那个）是同步的，它会自动等队列里所有东西跑完。这就是为什么很多简单程序不写 `cudaDeviceSynchronize()` 也能跑——`cudaMemcpy` 已经隐式同步了。

## 五、思维方式的转变

语法是表层的，更难适应的是**思考问题的方式**。从 C++ 到 CUDA 你需要做这几个转变：

**从"循环"到"线程网格"。** C++ 你想"我循环 N 次，每次做一件事"。CUDA 你想"我启动 N 个线程，每个线程做一件事，编号是 idx"。`for (int i = 0; i < N; ++i) C[i] = A[i] + B[i];` 变成 `int idx = ...; if (idx < N) C[idx] = A[idx] + B[idx];`。每个线程是一次循环迭代的"展开"。

**从"我"到"我们"。** C++ 你写代码的视角是"我"——单个执行流。CUDA 写 kernel 时你的代码会被几万个线程同时执行，要时刻想"这一万个'我'各自在做什么，会不会撞车"。读写同一个地址要用原子操作（`atomicAdd` 等），block 内同步用 `__syncthreads()`。

**从"算得快"到"喂得饱"。** ICPC 里你优化常数、剪枝、降复杂度。GPU 编程里**算力几乎从来不是瓶颈**，瓶颈是**内存带宽**——能不能让那几千个核心都有活干，不是在等数据。这就引出了"**合并访问 (coalesced access)**"、"**共享内存**"、"**bank conflict**"等概念。这些在向量加法里看不出来，矩阵乘法以后会大量出现。

**从"按需分配"到"批量并行"。** C++ 里树、链表、动态分配都很自然。GPU 上动态内存分配很慢，分支多的算法跑不快（warp divergence），不规则访问很伤性能。所以 GPU 算法常常被重新设计成**规则的、数组化的**形式。比如 GPU 上的图算法会用 CSR 这种压缩格式而不是邻接表里的指针。

## 六、一个有用的心智模型

每次写 CUDA 程序，先在脑子里回答四个问题：

1. **数据怎么分？** N 个元素分给多少个 block、每个 block 多少线程？每个线程负责哪些数据？
2. **数据放哪？** 全在 global memory（慢但容量大），还是有部分需要搬到 shared memory（快但小、block 内共享）？
3. **线程要不要协作？** 如果每个线程独立干活（如向量加法），简单。如果要协作（如归约求和、矩阵乘法），需要 `__syncthreads()` 和 shared memory。
4. **数据怎么进出 GPU？** 上传一次算多次，还是反复来回？后者通常意味着算法设计有问题。