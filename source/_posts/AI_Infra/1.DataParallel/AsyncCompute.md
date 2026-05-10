---
title: 3. GPU 异步计算
categories: 学习笔记- AI Infra
date: 2026-05-08 22:30:00
mathjax: true
tags:
    - AI
    - AI Infra
---


写 PyTorch 训练代码时,你会发现一些奇怪的现象:`loss.backward()` 后面紧跟一行 `print(loss.item())`,这一行突然变得很慢;明明用了 `torch.cuda.synchronize()` 测时间,profiler 里看到的耗时却完全不一样;DDP 训练里通信看似在反向之后才发生,实际却和反向"同时"在跑。

这些都是同一个东西的影子——**异步计算 (asynchronous computation)**。它不是某个高级 API,而是 GPU 编程模型的底层默认行为。理解它能解开很多"为什么训练比想象中快/慢"的谜团,也是看懂 DDP 计算-通信重叠、ZeRO-3 prefetch、Pipeline 调度这些大模型工程优化的前置知识。

本文按这个顺序展开:先讲清楚同步 / 异步的本质区别,再讲 CUDA 的 stream 模型,然后用 PyTorch 代码看实际怎么用,最后看三个典型场景(DDP / ZeRO-3 / Pipeline)是怎么把异步玩到极致的。

## 一、为什么需要异步:硬件本来就是并发的

### 1.1 一台机器里有几个"工人"

跑深度学习的服务器,**至少**有这些彼此独立的执行单元:

| 工人 | 干的事 | 谁指挥它 |
|---|---|---|
| **CPU 核心** | Python 代码、数据预处理、kernel launch | 你写的 Python |
| **GPU SM(流处理器)** | 矩阵乘、conv、激活函数等计算 kernel | CUDA driver |
| **GPU DMA / Copy 引擎** | Host↔Device、Device↔Device 数据搬运 | CUDA stream |
| **NVLink / PCIe / IB 网卡** | 跨 GPU、跨机通信 | NCCL |

**关键事实**:这些工人**物理上是并行的**——CPU 在跑 Python 时 GPU 不一定闲,GPU 在算 kernel 时网卡完全可以同时传梯度,DMA 在搬数据时 SM 也能继续算别的。

如果代码写得"全同步"(每一步都等上一步彻底干完才发下一条指令),就等于让其他工人都干等。**异步就是为了让所有工人尽可能同时忙起来**。

### 1.2 同步 vs 异步:一段代码看本质

考虑这样一个流程:**算梯度 → 把梯度传给其他卡 → 用平均梯度更新参数**。

**同步版本**(每一步都阻塞):

```
时间 →
CPU/GPU :  [算梯度          ][        ][用平均梯度更新参数]
网卡    :                    [传输/平均]
                                 ↑ GPU 此时干等
总耗时  :  T_compute + T_comm + T_update
```

**异步版本**(算下一批的同时把上一批的梯度传出去):

```
时间 →
CPU/GPU :  [算梯度       ][算下一批梯度       ][用平均梯度更新]
网卡    :                 [传输/平均上一批的梯度]
                          ↑ 同时进行,互不阻塞
总耗时  :  max(T_compute, T_comm) + T_update
```

如果 `T_compute ≈ T_comm`,异步版本几乎**省掉一整段通信时间**。这就是 DDP 跑得近似线性加速的根本原因——**不是通信变快了,而是通信被计算掩盖了**。

### 1.3 但异步不是免费的

异步带来三个新问题,后面每一节都会反复碰到:

1. **依赖管理**:某个 kernel 需要前一个 kernel 的结果,异步发射时怎么保证顺序?
2. **资源竞争**:两个 stream 都在用同一份显存怎么办?
3. **可观测性**:CPU 早就跑过那行代码了,GPU 实际还没开始,你怎么测时间、怎么处理报错?

CUDA 的 stream 模型就是为了解决这三件事而设计的。

## 二、CUDA 编程模型:Stream 与异步执行

### 2.1 两个时间轴:Host 与 Device

在 CUDA 里,CPU(host)和 GPU(device)有**各自独立的指令队列**。你在 Python 里调用一个 GPU 操作,实际上分两步:

1. **Host 提交 (launch)**:把"做什么"写一条指令丢进 GPU 的命令队列,**立即返回**
2. **Device 执行**:GPU 在某个时刻按队列顺序实际做这件事

```
                    时间 →
Host (CPU)  :  [launch K1][launch K2][launch K3][launch K4][... 继续 Python ...]
Device (GPU):              ↓ 排队
                          [    K1 执行    ][    K2 执行    ][    K3 执行    ][K4 ...]
                            ↑ 这里 host 早跑过 launch K1 那行代码了
```

**Host 的时间和 Device 的时间是错开的**——这是 CUDA 异步模型的最核心事实。

代码层面看:

```python
import torch

x = torch.randn(4096, 4096, device="cuda")
y = torch.randn(4096, 4096, device="cuda")

t0 = time.time()
z = x @ y                      # ← 这一行立刻返回!GPU 还没算完
t1 = time.time()
print(t1 - t0)                 # 通常 < 1ms,这是 launch 时间
```

`x @ y` 看起来像在算矩阵乘,实际只是把矩阵乘**任务**提交给 GPU。如果你想测真实计算时间,必须强制等 GPU 干完:

```python
torch.cuda.synchronize()       # 阻塞 host 直到 GPU 队列清空
t0 = time.time()
z = x @ y
torch.cuda.synchronize()       # 再阻塞一次,确保 z 算好
t1 = time.time()
print(t1 - t0)                 # 现在才是真实计算时间
```

**这是 99% 的"GPU 测时间错"的根源**——很多人以为代码慢的地方,其实 host 早就跑过了,真实瓶颈在另一个地方。

### 2.2 Stream:GPU 的指令队列

每张 GPU 上不只有一条命令队列,而是有多个**独立的队列**,每个叫一个 **stream**。

- **同一个 stream 内**:任务严格按提交顺序串行执行(有依赖关系)
- **不同 stream 之间**:任务可以**并发**执行(相互独立)

PyTorch 默认所有操作都跑在一个"默认 stream"上,所以你写的代码看起来是顺序的。但只要你把不同任务派到不同 stream,GPU 的硬件就能让它们并行起来。

```
默认 stream     :  [matmul][add ][relu][matmul][...]    ← 所有计算排成一队
自定义 stream 1 :        [memcpy h2d][memcpy h2d]       ← 数据搬运并发进行
自定义 stream 2 :              [NCCL AllReduce]         ← 通信并发进行
                Time →
```

GPU 内部:SM 跑计算 kernel,Copy 引擎跑 memcpy,NVLink 引擎跑通信——三组硬件资源,三个 stream 各占一个,**真的同时在跑**。

### 2.3 不同 stream 之间的并发示意

举一个具体例子:数据搬运 + 计算重叠。

```
时间轴      0    1    2    3    4    5    6    7
默认 stream :           [matmul A    ][matmul B    ]
copy stream :  [H2D x→A][H2D y→B   ]
                   ↑ 这两件事在 GPU 上同时进行
                   一个用 SM,一个用 DMA,不抢资源
```

如果你**不开 copy stream**,数据搬运和计算就会串行:

```
时间轴      0    1    2    3    4    5    6    7    8    9
默认 stream :  [H2D x→A][matmul A    ][H2D y→B][matmul B    ]
                                        ↑ 此时 SM 闲着等 DMA
```

直观就能看到差距——多 stream 是在用**已经存在但没被利用**的硬件并行性。

### 2.4 同步原语:让乱序的世界保持正确

有了多 stream,就要回答"怎么保证依赖顺序"。CUDA 提供几把锁:

**`torch.cuda.synchronize()`**:阻塞 host,直到 device 上**所有** stream 都干完。最暴力,主要用来测时间和 debug。

**`stream.synchronize()`**:阻塞 host,直到这一个 stream 干完。

**`stream.wait_stream(other)`**:让 `stream` 在 GPU 内部等 `other` 干完才继续——**不阻塞 host**,host 该干嘛干嘛。这是异步重叠的关键 API。

**`Event`**:更细粒度的标记。在某个 stream 上 `event.record()` 打个标记,另一个 stream `event.wait()` 等这个标记被达到。常用于建跨 stream 的依赖图。

```python
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

with torch.cuda.stream(s1):
    a = compute_A()           # 在 s1 上算

ev = torch.cuda.Event()
ev.record(s1)                  # 在 s1 上打标记

with torch.cuda.stream(s2):
    ev.wait()                  # s2 等 s1 的标记
    b = use(a)                 # 现在 s2 才用 a,保证 a 算完
```

整个过程**host 不阻塞**,只是在 GPU 那边建了"s2 的某个 kernel 必须等 s1 那个 event 才能跑"的依赖。这就是异步重叠的标准写法。

## 三、PyTorch 里怎么用异步

### 3.1 你早就在用异步了

只要你在 PyTorch 里跑 `.cuda()` 张量做运算,**默认就是异步的**。下面这段代码看起来"按顺序"运行:

```python
y = model(x)        # forward
loss = criterion(y, target)
loss.backward()     # backward
optimizer.step()    # update
```

实际上,这四行代码执行完时,GPU 上的工作可能**一个都还没真正做完**——它们都被排进了默认 stream 的队列里,host 已经跑到下一行了。直到你做"必须看到结果"的事(打印、保存、转 CPU),host 才会被迫等 GPU。

### 3.2 哪些操作会"强制同步"

这些是 host 与 device 的同步点,会让 host 阻塞等 GPU:

| 操作 | 为什么阻塞 |
|---|---|
| `tensor.item()` | 必须把数取回 CPU 内存 |
| `tensor.cpu()` / `.numpy()` | 数据搬到 CPU,要等 GPU 写完 |
| `print(tensor)` | 打印要看到值 |
| `tensor.tolist()` | 同上 |
| `if tensor > 0:` | 条件判断需要值 |
| `torch.cuda.synchronize()` | 显式阻塞 |

**最常见的 footgun**:训练循环里手贱写一个 `print(loss.item())`,每个 iteration 都会逼 host 等 GPU 队列清空,本来异步重叠的计算-通信瞬间变同步。生产代码里这种打印要么定期(每 100 步)做,要么累积成 tensor 最后再 `.item()`。

### 3.3 数据搬运:`non_blocking=True`

最容易拿到的"免费"异步收益是数据加载。默认情况下 `tensor.to('cuda')` 是同步的(host 等 H2D 拷贝完),但加一个参数就异步了:

```python
# DataLoader 用 pin_memory=True 才能开 non_blocking
loader = DataLoader(dataset, ..., pin_memory=True)

for x, y in loader:
    x = x.to(device, non_blocking=True)   # 异步 H2D
    y = y.to(device, non_blocking=True)
    
    # 此时 H2D 还在 copy stream 跑,但下面的 forward 在默认 stream
    # 默认 stream 看到 x,y 就会自动等 copy 完成(PyTorch 帮你管依赖)
    pred = model(x)
    loss = criterion(pred, y)
```

这样下个 batch 的搬运可以和当前 batch 的计算重叠,DataLoader 不再是瓶颈。

### 3.4 测时间的正确姿势

既然 `time.time()` 测的是 host 时间,不是 GPU 时间,那怎么准确测?三种方法:

```python
# 方法 1:最简单,粗粒度
torch.cuda.synchronize()
t0 = time.time()
heavy_gpu_work()
torch.cuda.synchronize()
t1 = time.time()
print(f"{t1 - t0:.3f}s")
```

```python
# 方法 2:CUDA Event,精度高,不阻塞 host
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
heavy_gpu_work()
end.record()

end.synchronize()              # 等 end 这个事件被达到
print(f"{start.elapsed_time(end):.3f} ms")
```

```python
# 方法 3:torch.profiler,看清楚每个 kernel 在哪个 stream 跑多久
with torch.profiler.profile() as prof:
    heavy_gpu_work()
print(prof.key_averages().table())
```

第二种方法是 production 测速首选——它在 GPU 自己的时钟上打时间戳,不受 host 异步行为干扰。

## 四、典型场景一:DDP 的计算-通信重叠

把前面的概念套到一个实际场景。DDP 反向时要做的事:**算各层的梯度 → AllReduce 跨卡同步 → 优化器更新**。

### 4.1 同步版的反向(基线)

如果直接写"反向算完→AllReduce→更新":

```
时间 →
反向计算  :  [L_N][L_{N-1}][L_{N-2}]...[L_2][L_1]
NCCL 流   :                                    [AR L1][AR L2]...[AR LN]
更新      :                                                            [step]

总时间    =  T_backward + T_allreduce + T_step
```

通信和计算完全串行——通信时间一秒不少地加在 wall-clock 上。

### 4.2 DDP 实际怎么做(异步重叠)

DDP 想做的事一句话:**反向一边算梯度,一边把已经算好的梯度发出去同步,不要等所有梯度算完才动手**。要做到这点需要回答五个问题——一个梯度算好了怎么知道?发太碎了怎么办?通信和反向都用 GPU 怎么不抢资源?怎么让通信尽量提前发射?最后怎么收尾?

下面挨个拆。

#### 1) Hook:梯度就绪的"通知机制"

`backward()` 从 loss 出发沿计算图回溯,算完哪一层的 `dL/dW_i`,autograd 引擎就立刻往那个 `W_i.grad` 写值。**写完那一刻**就是 bucket 该被触发的时机。

DDP 在每个参数的 grad accumulator 上挂了个 hook,语义如下(简化):

```python
for p in model.parameters():
    if not p.requires_grad: continue
    grad_acc = p.expand_as(p).grad_fn.next_functions[0][0]
    grad_acc.register_hook(lambda *_: ddp.mark_param_ready(p))
# 真实 DDP 用的是 C++ 的 Reducer,语义完全一致
```

只要 autograd 写完 `p.grad`,`mark_param_ready(p)` 就会被同步调用——**不需要等 backward 整体结束**。这是后面所有异步动作的触发信号。

#### 2) Bucket:把小梯度凑成大包再发

模型有几百万参数,如果一个梯度就发一次 AllReduce,会被 NCCL 的 launch overhead 和带宽爬坡淹没(小消息根本打不满 NVLink)。DDP 把参数预先分成若干个 **bucket**,默认 25MB 一个,每个 bucket 是一段**连续显存** + 一个待办计数器:

```
参数到 bucket 的映射:
  bucket 0 (25MB):  W_N, W_{N-1}, W_{N-2}, ... pending=3
  bucket 1 (25MB):  W_{N-3}, W_{N-4}, ...     pending=2
  bucket 2 (25MB):  ...                       pending=...
```

`mark_param_ready` 干两件事:把 `p.grad` 拷进 bucket 的 flat buffer,pending 减 1。**减到 0 的那一瞬间整个 bucket 立刻整体发出去**:

```python
def mark_param_ready(self, p):
    bk = self.param_to_bucket[p]
    bk.flat_grad[bk.offset[p] : bk.offset[p]+p.numel()].copy_(p.grad.flatten())
    bk.pending -= 1
    if bk.pending == 0:
        bk.handle = dist.all_reduce(           # 异步发射
            bk.flat_grad,
            op=ReduceOp.SUM,
            async_op=True,
        )
```

`async_op=True` 的语义:把 NCCL 任务排进 NCCL stream 的队列,**host 立刻返回**继续 backward。返回的 `handle` 不是数据,是一张"将来再来收"的票。

#### 3) NCCL stream:通信走专属车道

NCCL 在每张卡上有自己的 CUDA stream。AllReduce 任务排进去之后:

- **NCCL stream** 上,GPU 用 NVLink / IB 引擎搬数据、跑 ring/tree reduce,**几乎不占 SM**
- **默认 stream** 上,backward 的 matmul / conv 继续在 SM 上跑

两条 stream 各用各的硬件资源(SM vs 通信引擎),硬件层面真的并发。"NCCL stream 读 bucket buffer 时默认 stream 不能正在改它"这种依赖,CUDA 通过 event 自动建——使用者不用手写。

#### 4) 反向"逆序" + bucket 反编号:让通信尽早开跑

backward 从最后一层往前算:`L_N → L_{N-1} → ... → L_1`。

DDP 给 bucket 编号时**故意反过来**:bucket 0 装最后几层(`W_N` 附近)的参数,bucket K 装第一层(`W_1` 附近)的参数。这样反向才刚开始没多久,bucket 0 就能凑齐发出第一波 AllReduce——后面**所有** bucket 的通信都和正在进行的反向重叠。

如果不反过来,反向跑了 95% 才凑齐第一个 bucket,根本没多少时间留给重叠。

#### 5) 把五件事拼起来的时序图

设 4 层模型、3 个 bucket,反向耗时 4 个时间单元、单个 bucket 通信耗时 1.5 个:

```
时间 →                  0       1       2       3       4       5
默认 stream (反向)   :  [bk_L4 ][bk_L3 ][bk_L2 ][bk_L1 ]
                              ↓ L4,L3 凑齐 bk0    ↓ L2 凑齐 bk1   ↓ L1 凑齐 bk2
NCCL stream (通信)   :         [AR bk0   ]      [AR bk1   ]     [AR bk2   ]
                                                                          ↑ tail,反向已结束
host 在做啥          :  [launch L4 + hook fires, fire AR bk0]
                                [launch L3 + hook]
                                       [launch L2 + hook,fire AR bk1]
                                              [launch L1 + hook,fire AR bk2]
                                                     [wait_all][step]
                          ↑ host 一路向前提交,完全不阻塞       ↑ 唯一的真同步点
```

读出来三件事:

- **AR bk0 在反向只跑了 25% 时就已经在 NCCL stream 上跑了**——通信被尽早提前
- **反向计算和 NCCL 通信用不同硬件**——同时跑、互不干扰
- **整个 backward 过程只有 wait_all 是真同步**——前面所有 hook、launch、async_op 都不阻塞 host

总耗时从 `T_backward + T_allreduce` 变成 `max(T_backward, T_allreduce) + T_tail`。`T_tail` 是最后一个 bucket 没法被反向覆盖掉的那一小段——它的长度等于"最后一个 bucket 的通信耗时"。如果反向比通信慢,tail 几乎为零(通信"白送");如果通信比反向慢(模型小、跨节点带宽差、bucket 太大导致最后一个还在传),tail 就是 DDP 的真实瓶颈。

#### 6) 收尾:wait 和除以 world_size

backward 末尾(autograd 的 final hook 里),DDP 把所有 bucket 同步掉、求平均、拆回各参数的 `.grad`:

```python
for bk in self.buckets:
    bk.handle.wait()                  # 这里 host 阻塞等 NCCL 完成
    bk.flat_grad.div_(self.world_size)
    for p in bk.params:
        p.grad.copy_(bk.flat_grad[bk.offset[p] : bk.offset[p]+p.numel()].view_as(p))

# optimizer.step() 看到的就是平均后的梯度
```

`bk.handle.wait()` 是整个 backward 唯一的真同步点。前面 hook、launch、async_op 一路狂奔,通信和计算在 GPU 里疯狂重叠;到这里要做参数更新前,host 必须确认通信真的结束了——这是异步重叠的边界,也是 wall-clock 上 DDP 反向"剩下的那一小段时间"的来源。

## 五、典型场景二:ZeRO-3 / FSDP 的参数预取

ZeRO-3 把参数本身分片,每张卡只持有 1/N。前向时每用到一层都要先 AllGather 一下把这层参数凑齐。如果同步做:

```
时间 →
计算 :  [---等---][L1 fwd][---等---][L2 fwd][---等---][L3 fwd]
通信 :  [AG L1  ]         [AG L2  ]         [AG L3  ]
                ↑ GPU SM 干等
```

每次 AllGather 都让计算干等,**相当于把通信完全暴露在 wall-clock 里**,慢得离谱。

FSDP 的解法是 **prefetch**:计算第 i 层的同时,在另一个 stream 上**异步发起**第 i+1 层的 AllGather:

```
时间 →
默认 stream (计算)  :  [L1 fwd][L2 fwd][L3 fwd][L4 fwd]
NCCL stream (通信)  :  [AG L1]
                            [AG L2]    ← 在 L1 计算时已发起
                                  [AG L3]
                                        [AG L4]

整体时间 ≈ max(计算, 通信),通信被掩盖
```

实现关键:用 stream + event 把"L_{i+1} 的 AllGather 必须在 L_i 计算开始之后"这条依赖建好,然后让 NCCL stream 自己往前跑。FSDP 的 `forward_prefetch=True` 和 `backward_prefetch=BACKWARD_PRE` 就是开/关这个机制的开关。

**这是异步重叠在 ZeRO-3 里的应用**——和 DDP 的思路一样(把通信丢到独立 stream + 提前发射),只是触发时机从"反向梯度就绪"换成"前向到达某层之前"。

## 六、典型场景三:Pipeline 并行的 1F1B

异步的另一种形态——不是计算和通信重叠,而是**多个 micro-batch 在不同 stage 上同时流动**。

### 6.1 朴素调度的"气泡"

设 4 个 stage、4 个 micro-batch。如果一个 micro-batch 跑完前向再跑反向才换下一个:

```
时间 →
Stage 0:  [F1][F2][F3][F4]                              [B4][B3][B2][B1]
Stage 1:      [F1][F2][F3][F4]                      [B4][B3][B2][B1]
Stage 2:          [F1][F2][F3][F4]              [B4][B3][B2][B1]
Stage 3:              [F1][F2][F3][F4][B4][B3][B2][B1]
          ←warmup→                              ←cooldown→
            ↑ 大量 stage 干等              ↑ 大量 stage 干等
```

气泡时间正比于 stage 数,stage 越多浪费越大。

### 6.2 1F1B(One Forward One Backward)

Megatron-LM 等用的调度:每个 stage 在 warmup 后,**轮流交替**跑一次前向和一次反向,把流水线塞满:

```
时间 →
Stage 0:  [F1][F2][F3][F4][B1][F5][B2][F6][B3][F7][B4][F8][B5][B6][B7][B8]
Stage 1:      [F1][F2][F3][B1][F4][B2][F5][B3][F6][B4][F7][B5][F8][B6][B7][B8]
Stage 2:          [F1][F2][B1][F3][B2][F4][B3][F5][B4][F6][B5][F7][B6][F8][B7][B8]
Stage 3:              [F1][B1][F2][B2][F3][B3][F4][B4][F5][B5][F6][B6][F7][B7][F8][B8]
                       ↑ warmup 后立刻交替 F/B,各 stage 几乎不空闲
```

气泡只剩下 warmup 和 cooldown 这两段,占比 (stage_count - 1) / num_microbatches,micro-batch 越多气泡越被稀释。

底层用的还是同样的异步思想:**stage 之间用通信传 activation/gradient,通信走独立 stream,计算和通信重叠**。1F1B 调度只是规定了"每个 stage 该按什么顺序往队列塞 F 和 B kernel",硬件层面的并发还是 stream 模型在管。

## 七、易踩的坑

异步带来效率,也带来一堆"看起来不科学"的现象。常见几条:

### 7.1 用 `time.time()` 测出来的 GPU 耗时是错的

`time.time()` 测的是 host 走过那行代码的时间,kernel launch 本身只要几 μs。除非 host 撞到强制同步点,否则你测出来的是"提交时间",不是"GPU 干完时间"。**永远用 `torch.cuda.synchronize()` 包裹或 CUDA Event**,前面 §3.4 讲过。

### 7.2 错误暴露得很晚

GPU kernel 执行错(比如越界、NaN 触发某些 assert),host 不会立刻收到——它已经跑过那行代码了。错误要等到下一次同步点(下一个 `.item()`、`synchronize()`、甚至下一个 iteration 才暴露。看到报错栈指向某行 PyTorch 代码,真凶可能在很多行之前。

调试这种问题的招:加环境变量 `CUDA_LAUNCH_BLOCKING=1`,强制每次 kernel launch 都等 GPU 跑完,把异步关掉。错误就会指向真正出错的那行,代价是训练慢几倍——只用于调试。

### 7.3 OOM 也会延迟

显存不够爆 OOM 同样不一定在你期待的那行报。如果某次显存分配在另一个 stream 上 lazy 触发,栈可能完全是误导的。同样可以用 `CUDA_LAUNCH_BLOCKING=1` 看真实位置,或者用 `torch.cuda.memory._record_memory_history()` 抓显存事件回放。

### 7.4 多 stream 的依赖必须显式建

PyTorch 默认 stream 上的操作彼此自动建依赖(因为是同一个队列)。但**你自己开 stream 后,跨 stream 的依赖必须自己用 event 建**。忘了建依赖就等于在程序里悄悄写了个 race condition,出来的数据可能是上一个 batch 的、可能是部分写好的。

```python
# 错的:s2 用 a,但 s1 还没算完
with torch.cuda.stream(s1):
    a = compute()
with torch.cuda.stream(s2):
    use(a)            # ⚠️ undefined behavior

# 对的:用 event 建依赖
with torch.cuda.stream(s1):
    a = compute()
ev = torch.cuda.Event(); ev.record(s1)
with torch.cuda.stream(s2):
    ev.wait()
    use(a)
```

PyTorch 的 `tensor.record_stream(stream)` 还能告诉显存分配器"这块 tensor 还在被 stream 用着,别回收",防止显存被提前释放。这些细节是手写多 stream 代码时绕不开的。

## 八、一句话总结

GPU 是个**异步加速器**——你写的每行 `.cuda()` 操作几乎都不阻塞 host,只是把任务提交到某个 stream 的队列。Stream 让多种硬件资源(SM / DMA / NVLink)同时忙起来,event 负责跨 stream 建依赖。DDP 的计算-通信重叠、ZeRO-3 的参数预取、Pipeline 的 1F1B,都是同一套"独立 stream + 提前发射 + 显式同步"思想的具体应用。会用异步,大模型训练的吞吐才能逼近硬件理论上限;不理解异步,测出来的时间和报错的位置都会让人摸不着头脑。
