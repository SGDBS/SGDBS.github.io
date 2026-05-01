---
title: DDP(DistributedDataParallel) 从入门到入土
categories: 学习笔记- AI Infra
date: 2026-04-28 22:48:00
mathjax: true
tags:
    - AI
    - AI Infra
---


## 一、数学基础:为什么数据并行是正确的

### 1.1 SGD 的可分解性

考虑 batch size 为 B 的损失函数:

$$L(\theta) = \frac{1}{B}\sum_{i=1}^{B} \ell(x_i, \theta)$$

梯度:

$$\nabla L(\theta) = \frac{1}{B}\sum_{i=1}^{B} \nabla \ell(x_i, \theta)$$

把 B 个样本平均切给 N 张 GPU,每张卡处理 B/N 个样本。第 k 张卡上的局部梯度是:

$$g_k = \frac{1}{B/N}\sum_{i \in S_k} \nabla \ell(x_i, \theta) = \frac{N}{B}\sum_{i \in S_k} \nabla \ell(x_i, \theta)$$

那么 N 张卡的局部梯度的平均是:

$$\frac{1}{N}\sum_{k=1}^{N} g_k = \frac{1}{N}\sum_{k=1}^{N} \frac{N}{B}\sum_{i \in S_k} \nabla \ell(x_i, \theta) = \frac{1}{B}\sum_{i=1}^{B} \nabla \ell(x_i, \theta) = \nabla L(\theta)$$

**结论**:N 张卡分别在 B/N 个样本上算梯度,然后做平均,数学上完全等价于单卡上 batch size = B 的梯度。这就是 DDP 正确性的根基。

### 1.2 关键前提

这个等价的成立依赖三个条件:**所有卡的初始参数相同**、**每步梯度同步保持参数一致**、**优化器是确定性的**(给定相同的参数、梯度、状态,产出相同的更新)。DDP 的所有工程设计都在保障这三点。

### 1.3 一个常被忽视的细节:BatchNorm 不可分解

BN 的均值方差是在 batch 内统计的,所以 N 张卡各算各的 BN 等价于 batch size = B/N(不是 B)。这就是为什么需要 SyncBatchNorm。LayerNorm、RMSNorm 不存在这个问题,因为它们在样本内部归一化,与 batch 无关。Transformer/GPT 用 LayerNorm,所以 DDP 训练 LLM 不用担心这一点。

## 二、Ring AllReduce 的数学与通信复杂度

### 2.1 朴素方案:为什么不用 Parameter Server

朴素做法是所有 worker 把梯度发给一个 master,master 求和后广播回来。设梯度大小为 V,worker 数为 N,master 的接收带宽需求是 N·V,广播又是 N·V,master 是瓶颈,**通信量随 N 线性增长**,无法扩展。

### 2.2 Ring AllReduce

把 N 张卡排成环,每张卡上的梯度切成 N 块。

**Phase 1: Reduce-Scatter** (N-1 轮)

第 k 轮中,卡 i 把自己持有的"第 (i-k) mod N 块"发给卡 (i+1),从卡 (i-1) 接收"第 (i-k-1) mod N 块"并累加到自己对应的位置。

经过 N-1 轮后,卡 i 持有的"第 (i+1) mod N 块"是所有卡上该块的累加和。

**Phase 2: AllGather** (N-1 轮)

每张卡把自己累加好的那块沿环传一圈,N-1 轮后所有卡都拿到所有块的完整累加结果。

### 2.3 通信量推导

每张卡每轮发送 V/N 数据。两个阶段共 2(N-1) 轮。每张卡的总发送量:

$$T_{\text{send}} = 2(N-1) \cdot \frac{V}{N} = 2V \cdot \frac{N-1}{N}$$

当 N 很大时,**每张卡发送量趋近 2V,与 N 无关**。这就是 Ring AllReduce 能扩展到几千卡的关键。延迟是 O(N)(轮数),但每轮数据量小,带宽利用率高。

实际 NCCL 还会用 Tree AllReduce、双二叉树等更复杂的拓扑,延迟更低,但 Ring 是基础。

## 三、DDP 的工作流程

### 3.1 完整生命周期

第一步是**初始化进程组**:每张 GPU 一个进程,通过 NCCL 通信后端组成进程组,每个进程有 rank 和 world_size。

第二步是**模型构造时的状态同步**:DDP 把 rank 0 的模型参数和 buffers 通过 broadcast 发给所有其他 rank,保证所有进程从相同状态开始。

第三步是**反向传播中的梯度同步**:每个 step 各进程独立做前向(无通信),反向时通过 autograd hook 触发 bucket 级的异步 AllReduce,通信和计算重叠。

第四步是**参数更新**:各进程独立调用 optimizer.step(),由于参数、梯度、优化器状态完全一致,更新结果也完全一致。

### 3.2 关键工程优化:Gradient Bucketing

如果每个参数算完梯度就发一次 AllReduce,小消息太多,通信效率低(NCCL 的小消息延迟开销大)。DDP 把参数按一定大小(默认 25MB)打包成 bucket,一个 bucket 内所有参数的梯度都算好后,一次性发起 AllReduce。

更精妙的是 bucket 划分按"反向传播顺序"逆序排列。反向是从最后一层往前算,所以**最后一层的参数最先算完梯度**,DDP 把最后一层放到第一个 bucket,前面的反向还在进行时,这个 bucket 已经可以异步发起 AllReduce,实现**计算与通信重叠**。

## 四、关键代码:从最小可用版本理解

### 4.1 最小可运行的 DDP 训练脚本

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

def setup():
    # torchrun 会自动设置 RANK, LOCAL_RANK, WORLD_SIZE 环境变量
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

class ToyDataset(Dataset):
    def __init__(self, n=10000):
        self.x = torch.randn(n, 128)
        self.y = torch.randint(0, 10, (n,))
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

def main():
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
    # 关键:用 DDP 包装模型
    model = DDP(model, device_ids=[local_rank])

    dataset = ToyDataset()
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        sampler.set_epoch(epoch)  # 关键:让 shuffle 每个 epoch 不同
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()           # 反向时 DDP 自动同步梯度
            optimizer.step()

        if dist.get_rank() == 0:
            print(f"epoch {epoch} loss {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    main()
```

启动命令:

```bash
torchrun --nproc_per_node=4 train.py
```

### 4.2 DDP 内部做了什么:伪代码版

下面是 DDP 关键逻辑的简化伪代码,帮你理解其内部实现:

```python
class DistributedDataParallel:
    def __init__(self, module, device_ids):
        self.module = module

        # 1. 广播参数,保证所有 rank 起点一致
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

        # 2. 把参数划分到 buckets(按反向顺序的逆序)
        params_in_reverse = list(self.module.parameters())[::-1]
        self.buckets = self._build_buckets(params_in_reverse, bucket_size_mb=25)

        # 3. 给每个参数注册 grad hook
        for bucket in self.buckets:
            for param in bucket.params:
                param.register_hook(lambda grad, p=param, b=bucket:
                                     self._on_grad_ready(p, b, grad))

    def _on_grad_ready(self, param, bucket, grad):
        bucket.mark_ready(param)
        # bucket 内所有参数梯度都准备好了,异步发起 AllReduce
        if bucket.all_ready():
            bucket.handle = dist.all_reduce(
                bucket.flat_grad,
                op=dist.ReduceOp.SUM,
                async_op=True
            )

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _wait_and_average(self):
        # backward 结束后,等所有 bucket 的 AllReduce 完成,然后除以 world_size
        for bucket in self.buckets:
            bucket.handle.wait()
            bucket.flat_grad.div_(dist.get_world_size())
```

几个关键点:

**hook 机制**:`param.register_hook` 在该参数的梯度被计算出来时触发回调。DDP 利用这一点知道"哪个参数的梯度已经算好了"。

**bucket.flat_grad**:bucket 内所有参数的梯度被拼成一个连续的 flat tensor 再做 AllReduce,避免多次小通信。AllReduce 完成后再 view 回各参数的 .grad。

**async_op=True**:AllReduce 异步发起,不阻塞反向传播。前面层的反向计算可以与后面层的通信并行。

**除以 world_size**:NCCL 的 AllReduce 是求和,DDP 在最后除以 N 得到平均梯度,保证语义和单卡 batch size = B 一致。

### 4.3 几个工程上必须知道的 API

```python
# 屏障同步,所有 rank 卡到这里再继续
dist.barrier()

# 跨 rank 求和,常用于汇总 loss/accuracy 用于打印
loss_tensor = torch.tensor(loss.item(), device=device)
dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
avg_loss = loss_tensor.item() / dist.get_world_size()

# 只在 rank 0 保存 checkpoint
if dist.get_rank() == 0:
    torch.save(model.module.state_dict(), "ckpt.pt")  # 注意 model.module
dist.barrier()  # 其他 rank 等保存完再继续

# SyncBatchNorm:把所有 BN 替换成跨卡同步版本
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[local_rank])
```

注意 `model.module` 这个细节:DDP 包装后,原模型在 `.module` 属性里,保存 state_dict 通常用 `model.module.state_dict()` 而不是 `model.state_dict()`,这样保存的是不带 DDP 前缀的干净权重。

## 五、有效 batch size 与学习率

DDP 下的 effective batch size:

$$B_{\text{effective}} = B_{\text{per\_gpu}} \times N_{\text{gpus}} \times N_{\text{accum}}$$

其中 N_accum 是梯度累积步数。学习率按 **Linear Scaling Rule** 缩放:相对于基准 batch size,batch 扩大 k 倍,学习率也扩大 k 倍。这个 rule 在 ResNet 训练上验证有效,LLM 训练有时用平方根缩放或者实际调试。

## 六、梯度累积与 no_sync

如果想做梯度累积,naive 写法每个 micro-batch 都触发 AllReduce,通信浪费。正确做法:

```python
for i, (x, y) in enumerate(loader):
    is_last = (i + 1) % accum_steps == 0
    # 非最后一个 micro-batch 时,禁用梯度同步
    ctx = model.no_sync() if not is_last else nullcontext()
    with ctx:
        loss = criterion(model(x), y) / accum_steps
        loss.backward()
    if is_last:
        optimizer.step()
        optimizer.zero_grad()
```

`model.no_sync()` 是 DDP 提供的上下文管理器,在它作用域内 backward 不触发 AllReduce,只在最后一步累积完所有梯度后做一次同步,通信量减少 accum_steps 倍。

## 七、面试中的高频追问及标准答案

**Q1: DDP 的 AllReduce 在反向传播的哪个时机触发?**

不是反向结束后统一触发,而是在反向过程中,当某个 bucket 内所有参数的梯度都算完时,立刻异步触发该 bucket 的 AllReduce。这样通信和计算重叠。

**Q2: 为什么 bucket 按反向顺序的逆序排列?**

因为反向是从最后一层算到第一层,最后一层的参数最先有梯度。把最后一层的参数放第一个 bucket,可以让通信尽早开始,与前面层的反向计算重叠。

**Q3: DDP 怎么保证所有 rank 参数始终一致?**

三个保证:初始化时 broadcast rank 0 的参数;每个 step 反向后 AllReduce 让梯度一致;优化器是确定性的,相同输入产出相同输出。三者结合,参数始终一致(忽略浮点累加顺序导致的极小数值差异)。

**Q4: 如果某些参数在某次 forward 中没有参与计算会怎样?**

DDP 等不到这些参数的 grad hook,bucket 永远不会 ready,AllReduce 不触发,程序卡住或报错。解决办法是 `DDP(model, find_unused_parameters=True)`,它会在反向开始时遍历计算图标记未使用的参数,代价是有一定开销。更好的做法是重构模型避免动态分支。

**Q5: DDP 解决了什么问题,没解决什么问题?**

解决:算力扩展(N 张卡的算力被利用)、效率(通信计算重叠)、扩展性(Ring AllReduce 的通信量与 N 几乎无关)。

没解决:**显存扩展**。每张卡仍然需要存完整的模型参数、梯度、optimizer state。对于 7B 以上模型,单卡显存放不下,必须用 ZeRO/FSDP 把这些状态分片,或者结合 TP/PP。

**Q6: NCCL 为什么比 Gloo 快?**

NCCL 是 NVIDIA 为 GPU 优化的集合通信库,直接走 NVLink/PCIe/InfiniBand,绕过 CPU 内存拷贝,支持 Ring/Tree 等高效拓扑。Gloo 是通用 CPU 后端,GPU 训练场景下性能差很多。GPU 训练默认用 NCCL。

**Q7: DDP 训练时 random seed 怎么处理?**

各 rank 应该用不同的 seed,否则数据增强、dropout 等随机操作在所有 rank 上完全一样,失去了多 rank 看不同样本的意义。常见做法是 `seed = base_seed + rank`。但 DistributedSampler 内部用的是相同 seed + rank shuffle,这是 sampler 自己处理的。

**Q8: 为什么用 model.module.state_dict() 而不是 model.state_dict()?**

DDP 包装后会给参数名加 `module.` 前缀。如果直接保存,加载时需要处理这个前缀,不优雅。保存 `model.module.state_dict()` 得到干净的权重文件,与单卡训练得到的 ckpt 格式一致,加载时不需要特殊处理。
