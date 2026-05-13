---
title: 6. 多机训练与 NCCL 工程
categories: 学习笔记- AI Infra
date: 2026-05-09 12:00:00
mathjax: true
tags:
    - AI
    - AI Infra
---


前面 DDP / ZeRO / 异步计算 几篇都默认了一个很 naive 的设定:**`torchrun --nproc_per_node=4` 就跑起来了**。这在单机调试时够用,但只要你打算上 16 卡以上的训练——也就是必须多机——那一行命令就会给你扔出一堆稀奇古怪的 NCCL 报错、卡死、超时,而错误信息几乎没有一个直接告诉你哪里错了。

这一篇把多机训练绕不开的工程基础讲透:**进程组怎么建立**、**`RANK / LOCAL_RANK / WORLD_SIZE` 各自是什么**、**NCCL 用什么网卡**、**拓扑感知怎么设计** 到**容错和 elastic 训练**。读完之后,8 卡到 1024 卡之间的调度、debug、性能调优,你心里都该有谱。

## 一、起因:单机不够之后

### 1.1 一台机器有几张卡

主流的训练机器(通常叫 DGX node、HGX node 或 8-GPU 服务器)是 **8 张 GPU 一台**:H100、A100、H200 都是这个标配。这个数字不是巧合——它是**主板 PCIe 通道 + NVSwitch 拓扑**能支持的上限。NVIDIA 的 NVSwitch 让 8 张卡两两之间都有约 900 GB/s 的双向 NVLink 带宽,**节点内全互连**;9 张以上需要更复杂的硬件。

所以"一台机器"在 AI Infra 的语境里几乎等于 8 张 GPU。再多就要靠**节点间网络**:InfiniBand(IB)或者 RDMA over Converged Ethernet(RoCE)。带宽通常是单卡 200-400 Gbps,看起来不小,但相比节点内 NVLink **慢 10 倍以上**。

```
节点内(8 张 GPU):
   GPU0 ━━━━━ NVSwitch ━━━━━ GPU1
              ↕
   GPU2 ━━━━━ NVSwitch ━━━━━ GPU3
              ↕
   GPU4 ━━━━━ NVSwitch ━━━━━ GPU5
              ↕
   GPU6 ━━━━━ NVSwitch ━━━━━ GPU7
   
   两两之间 ~900 GB/s 双向带宽

跨节点:
   Node 0 ─── IB switch ─── Node 1
                ↕
              Node 2
                ↕
              Node 3
   
   每节点对外 8×200 Gbps = 200 GB/s aggregate(理想情况)
```

**这条带宽差**是后续所有"为什么要节点内 TP、节点间 DP"、"为什么 ZeRO-3 跨节点会瓶颈"、"为什么有 HYBRID_SHARD"的根本原因。整章讲的所有工程细节都是为了**让通信尽可能贴节点内 NVLink、跨节点能少跑就少跑**。

### 1.2 多机训练的三个基础问题

任何多机训练系统都要回答三个问题:

1. **谁是谁**——每个进程怎么知道自己是 N 个进程里的第几个,在哪台机器上?
2. **怎么找到对方**——所有进程怎么互相发现并建立通信通道?
3. **网卡走哪根**——一台多网卡的机器,NCCL 怎么知道用哪张做 GPU 间通信?

这三问对应 PyTorch 的 RANK/WORLD_SIZE 体系、`init_process_group` 的 `init_method`、NCCL 的 `NCCL_SOCKET_IFNAME` 等环境变量。

## 二、进程组:谁是谁,以及怎么找到彼此

### 2.1 五个核心环境变量

PyTorch 分布式训练全靠这五个环境变量串起来,**没有人会显式 import 它们,但每行 NCCL 都依赖它们**:

| 变量 | 含义 | 谁设的 |
| --- | --- | --- |
| `RANK` | 全局进程编号(0 ~ WORLD_SIZE-1) | torchrun |
| `WORLD_SIZE` | 总进程数 = 节点数 × 每节点进程数 | torchrun |
| `LOCAL_RANK` | 本节点内的进程编号(0 ~ LOCAL_WORLD_SIZE-1) | torchrun |
| `LOCAL_WORLD_SIZE` | 本节点的进程数(= GPU 数) | torchrun |
| `MASTER_ADDR` / `MASTER_PORT` | rendezvous 协调进程的地址 | 你启动时填 |

举个例子:**2 节点 × 8 卡** 的训练,`WORLD_SIZE = 16`。Node 0 上的 GPU 5:`RANK = 5`、`LOCAL_RANK = 5`。Node 1 上的 GPU 5:`RANK = 13`(8 + 5)、`LOCAL_RANK = 5`。

**关键准则**:`LOCAL_RANK` 用来选 GPU(`torch.cuda.set_device(local_rank)`),`RANK` 用来逻辑分工(数据切分、rank 0 负责 logging 等)。**永远不要用 `RANK` 选 GPU**——多节点上不同 RANK 都可能映射到 GPU 0,会让两个进程抢同一张卡。

### 2.2 `init_process_group` 的三种 rendezvous

进程组建立的核心问题:N 个进程刚启动,**互相不认识对方**,必须有个"集合点"让它们交换 IP 和 rank,然后建立完整的 mesh 通信。这个过程叫 **rendezvous**(法语"集合"),PyTorch 提供三种实现:

```python
import torch.distributed as dist

# 方式 1:env:// —— 现代主流,torchrun 默认
dist.init_process_group(backend="nccl", init_method="env://")
# 从 MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE 环境变量读

# 方式 2:tcp:// —— 显式 IP 和端口
dist.init_process_group(
    backend="nccl",
    init_method="tcp://192.168.1.100:29500",
    rank=rank, world_size=world_size,
)

# 方式 3:file:// —— 共享文件系统协调,适合无网络的本地调试
dist.init_process_group(
    backend="nccl",
    init_method="file:///shared/init_file",
    rank=rank, world_size=world_size,
)
```

实际工业部署里几乎只用 `env://` + `torchrun`。其他两种主要在调试或者老代码里。

### 2.3 torchrun 启动多机训练

单机 8 卡:

```bash
torchrun --nproc_per_node=8 train.py
```

多机:**每台机器各跑一次 `torchrun`**,但要告诉它"我是哪个节点、一共几个节点、master 在哪":

```bash
# Node 0 (master)
torchrun \
    --nnodes=4 \
    --node_rank=0 \
    --nproc_per_node=8 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train.py

# Node 1
torchrun --nnodes=4 --node_rank=1 --nproc_per_node=8 \
    --master_addr=192.168.1.100 --master_port=29500 \
    train.py

# Node 2/3 同理,只改 node_rank
```

注意:`master_addr` 必须是 **Node 0 实际的 IP**,所有节点能 ping 通的那个。**初学者最大的坑是写成 `localhost` 或者 master 自己的 hostname**,然后非 master 节点完全连不上,torchrun 卡 30 秒后超时退出,错误是 `[c10d] connect timeout`。

### 2.4 集群调度器场景:把 torchrun 包进 Slurm/K8s

工业训练通常跑在 Slurm 或 K8s 上,你不会手动开 4 个 SSH 会话。Slurm 里典型脚本:

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1     # 一个 task 启 torchrun,torchrun 内部再 fork 8 进程
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64

# Slurm 的环境变量可以推出 NODE_RANK / MASTER_ADDR
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py
```

K8s 通常用 KubeFlow 的 PyTorchJob CRD,框架会自动注入这些环境变量,你只管在容器里写 `torchrun` 命令。但底层逻辑完全一致——还是那五个变量。

### 2.5 旧 API:为什么不要再用 `mp.spawn` / `launch`

历史上 PyTorch 提供过两套启动方式:

- `torch.multiprocessing.spawn(fn, nprocs=8)`:在 Python 里 fork 子进程。**只能单机**,跨节点没用
- `python -m torch.distributed.launch`:torchrun 的前身,**已 deprecated**,两年前就不再推荐

现代 PyTorch(1.12+)统一用 `torchrun`,它本身就是 `torch.distributed.run` 的快捷方式,后端是 c10d 的 elastic launcher,支持容错重启(后面 §6 会讲)。你看到老代码里这两套 API,直接换成 torchrun 就行。

## 三、NCCL:GPU 通信的实际承载者

### 3.1 NCCL 是什么

NCCL(NVIDIA Collective Communications Library,读作"nickel")是 NVIDIA 为 GPU 优化的集合通信库。它实现了 Ring AllReduce、Tree AllReduce、Broadcast、ReduceScatter、AllGather 等所有原语,直接驱动 NVLink、PCIe 和 InfiniBand 的硬件通道,**绕过 CPU 内存拷贝**,延迟和带宽都比 MPI/Gloo 高一个数量级。

PyTorch 的 `dist.all_reduce(...)` 在 GPU tensor 上的实现就是 NCCL,你只需要传 `backend="nccl"`,后续所有调用都走它。

### 3.2 NCCL 关心的几件事

NCCL 跑起来要决定:

1. **哪些 GPU 在一个 group 里**(world_size)
2. **每张 GPU 走哪条物理链路通信**(NVLink? PCIe? IB? socket?)
3. **是 Ring 拓扑还是 Tree 拓扑**(取决于消息大小和卡数)

第 1 件事 PyTorch 帮你做了。第 3 件事 NCCL 自己根据消息大小和拓扑探测决定,基本无脑。**第 2 件事是多机训练的几乎所有 NCCL bug 的来源**——下一节专门讲。

### 3.3 通信走错网卡 = 训练慢 100 倍

一台多机训练服务器通常有**多张网卡**:

- `eth0` / `enpXsY`:管理网,1 Gbps,用来 SSH 和 K8s 控制
- `ib0`、`ib1` ... :InfiniBand 网卡,200-400 Gbps,**真正用来 NCCL 跨节点通信**
- `bond0`:多网卡 bonding 后的虚拟接口

如果 NCCL 自动探测选错网卡——比如选了 `eth0` 跑 GPU 通信——你的训练速度直接下降 100 倍,从 200 GB/s 掉到 100 MB/s。**最经典的 footgun**。

排查方法,加环境变量看 NCCL 实际选了什么:

```bash
export NCCL_DEBUG=INFO
torchrun ... train.py 2>&1 | grep -i "nccl"
```

输出里会有这样的行:

```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB ...
NCCL INFO NET/Socket : Using [0]eth0:192.168.1.5<0> [1]ib0:10.0.0.5<0> ...
NCCL INFO Selected interface: eth0   ← ⚠️ 选错了!应该选 ib0
```

强制指定:

```bash
# 走 IB
export NCCL_IB_HCA=mlx5_0      # 指定具体的 IB 网卡
export NCCL_IB_DISABLE=0       # 0 = 启用 IB,1 = 禁用

# 或走特定的 socket 网卡
export NCCL_SOCKET_IFNAME=ib0  # 强制走 ib0,不要走 eth0
```

### 3.4 必备的 NCCL 环境变量速查

| 环境变量 | 作用 | 常用值 |
| --- | --- | --- |
| `NCCL_DEBUG` | 打印 NCCL 详细日志 | `INFO`(调试)、`WARN`(默认) |
| `NCCL_DEBUG_SUBSYS` | 控制 INFO 日志范围 | `INIT,NET,GRAPH` |
| `NCCL_SOCKET_IFNAME` | TCP fallback 时用哪张网卡 | `ib0` 或 `^lo,docker` |
| `NCCL_IB_HCA` | 用哪张 InfiniBand HCA | `mlx5_0,mlx5_1` |
| `NCCL_IB_DISABLE` | 是否禁用 IB | `0`(用)或 `1`(禁) |
| `NCCL_P2P_DISABLE` | 是否禁用节点内 P2P (NVLink) | `0`(默认开)、`1`(禁) |
| `NCCL_ALGO` | 强制 AllReduce 算法 | `Ring` / `Tree` / `CollnetDirect` |
| `NCCL_PROTO` | 选择数据包协议 | `Simple` / `LL` / `LL128` |
| `NCCL_NSOCKS_PERTHREAD` | 单线程内 socket 数 | 调优用,默认够 |
| `NCCL_BUFFSIZE` | NCCL 内部环形 buffer 大小 | 调优用,默认 4 MB |
| `NCCL_TIMEOUT` | 集合通信超时(秒) | 默认 30 分钟,卡死时调小可早报错 |

实战配置例子(节点间走 IB,禁用错的网卡):

```bash
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5    # 这台机器实际有 4 个 IB 接口
export NCCL_SOCKET_IFNAME=ib0                       # 实在退到 socket 时走 ib0
export NCCL_IB_GID_INDEX=3                          # RoCE 专用,IB 不需要
export NCCL_DEBUG=WARN                              # 只看异常
```

### 3.5 NCCL 与 NVLink / PCIe / IB 的关系

NCCL 在初始化时探测 GPU 之间的物理拓扑,决定每对 GPU 之间走哪条路。**节点内**:

- 8 卡 NVSwitch 机型(DGX-A100 / DGX-H100 / HGX):任意两卡都直连 NVLink,~900 GB/s
- 4 卡 + 4 卡 PCIe 桥接的廉价机型:跨 4 卡组要走 PCIe,带宽降到 ~50 GB/s

**节点间**:

- IB:NCCL 走 RDMA verbs,直接 GPU memory ↔ GPU memory,bypass CPU
- RoCE(IB over Ethernet):同样走 RDMA,但 fabric 是以太网
- TCP socket fallback:走主机网卡,GPU 数据要先 copy 到 CPU 内存再发,**慢一个数量级**

判断你的集群有没有走 RDMA,看 NCCL_DEBUG=INFO 输出里是 `NET/IB` 还是 `NET/Socket`。**Socket 模式是性能毒药**,确认无 RDMA 就别上规模训练。

### 3.6 跨节点带宽紧张时:用 BF16 做梯度通信

§3.3-3.5 讲的都是**硬件层**的优化——走对网卡、走对 RDMA、走对 NVLink。但应用层还有一类常用优化:当跨节点带宽真的紧张(100Gbps Ethernet 而不是 200/400Gbps IB),即使硬件都对齐了,梯度 AllReduce 仍然可能盖不住反向计算,DDP §3.2 那张时间轴里"反向计算"和"NCCL stream"的两条线就会拉开差距。这时候**用半精度做梯度通信**(带宽减半,精度几乎无损)是最划算的解法。

DDP 提供了 `register_comm_hook` 接口让你接管梯度通信策略,PyTorch 自带几个常用 hook 可以直接用。

#### 最常用:`bf16_compress_hook`

```python
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

model = DDP(model, device_ids=[local_rank], static_graph=True)
# BF16 做通信(范围广,LLM 推荐)
model.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)
# 或者 FP16(范围窄,需要小心数值范围)
# model.register_comm_hook(state=None, hook=default_hooks.fp16_compress_hook)
```

工作流程:

```
.grad (FP32) ──► cast to BF16 ──► AllReduce (BF16) ──► cast back FP32 ──► .grad
                    ↓                                       ↓
                通信量减半                              精度损失极小(LLM 几乎无影响)
```

注意这只影响**通信中的 dtype**,本地 `.grad` 和 master 参数仍然是 FP32——和《4. 混合精度训练》里讲的"训练用什么精度"是完全独立的设置。

#### `register_comm_hook` 的通用接口

如果想自定义通信逻辑(自己实现压缩、quantization、跨节点协议替换),`register_comm_hook` 接受一个返回 `Future` 的回调:

```python
def my_hook(state, bucket: dist.GradBucket) -> torch.futures.Future:
    """
    输入:bucket 内的梯度(flatten 成一个大 tensor)
    输出:Future,resolve 后 .grad 应该是同步好的最终梯度
    """
    grad = bucket.buffer()
    # ...自定义通信逻辑...
    fut = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True).get_future()
    return fut.then(lambda f: f.value()[0] / dist.get_world_size())

model.register_comm_hook(state=None, hook=my_hook)
```

DDP 内部不再自己做 AllReduce,而是 bucket 凑齐后调用 hook,等返回的 Future resolve 之后认为通信完成。只要 hook 返回的 `.grad` 在所有 rank 上语义一致,DDP 不做额外干预。

#### PowerSGD:更激进但 LLM 不用

PyTorch 还内置一个基于低秩近似的更激进压缩,叫 **PowerSGD**——把每个 bucket 的梯度看成矩阵 $G \in \mathbb{R}^{m \times n}$,用 power iteration 找 $G \approx PQ^T$,只通信小得多的 $P, Q$,压缩比能到 50-100×。代价是精度——低秩近似会丢掉梯度的高频成分。

实战经验:视觉、NLP 小模型上能用,精度损失 < 1%;**LLM 预训练上影响较大,生产里基本不用**;但在**带宽极差又必须分布式**的场景(教育机构、跨地理区域 fine-tuning)能直接拉训练速度 5-10×,是救命稻草。本文不展开,有需要查 `torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook`。

#### 什么时候开 BF16 压缩

简单决策:

- **单节点(≤ 8 卡 NVLink/NVSwitch)**:节点内 ~900 GB/s,带宽极充足,**不需要压缩**
- **多节点 + IB 200Gbps 或更好**:大多数 LLM 训练通信能被反向计算盖住,先 profile 再决定
- **多节点 + 100Gbps Ethernet 或更差**:几乎必开 `bf16_compress_hook`

判断带宽是否成为瓶颈最简单的办法是用 Nsight Systems / PyTorch Profiler 看一次 step 里 NCCL kernel 占的时间——如果 NCCL 时间 > 反向计算时间,DDP 的 bucket 异步重叠就盖不住通信,这时压缩通信就有意义了。

## 四、网络拓扑感知:HYBRID_SHARD 与 3D 并行

### 4.1 拓扑层级

理解了节点内 NVLink 比节点间 IB 快十倍以上,**所有大规模训练的并行策略设计都围绕"让最重的通信走 NVLink、最轻的通信走 IB"**。

具体在 ZeRO/FSDP 里,这就是 `HYBRID_SHARD` 的设计动因:

```
拓扑感知的 ZeRO 分组:

  ┌─────────────────────────────────────────────────────┐
  │ Node 0 (8 GPU, NVLink)                              │
  │   GPU0 ─ GPU1 ─ GPU2 ─ ... ─ GPU7                   │
  │   ↑ 节点内做 ZeRO-3:每张卡只存 1/8 参数            │
  │   AllGather 走 NVLink,飞快                         │
  └─────────────────────────────────────────────────────┘
              ↕ 跨节点做 DDP:广播完整梯度
              ↕ AllReduce 走 IB,只在每个 step 末尾一次
  ┌─────────────────────────────────────────────────────┐
  │ Node 1 (8 GPU, NVLink)                              │
  │   ZeRO-3 分组(独立的 1/8 切分)                    │
  └─────────────────────────────────────────────────────┘
              ↕
            Node 2 ...
```

每个节点内部是一个独立的 ZeRO-3 分组,所有重通信(每层的 AllGather)走 NVLink;节点间只做 DDP 风格的梯度 AllReduce,频率低、容易藏在反向后面。这套拓扑感知配置在跨多机训练里普遍比 FULL_SHARD 快 30%-50%。

### 4.2 3D 并行的拓扑映射

更激进的多机训练用 **3D 并行 = TP × PP × DP**:

- **Tensor Parallel**:层内切分,通信非常重(每层都要 AllReduce),**必须走 NVLink → 节点内**。典型 TP rank 数 = 8(整个节点)
- **Pipeline Parallel**:层间切分,通信只在 stage 边界(P2P send/recv),**可以跨节点**——不同 stage 在不同节点
- **Data Parallel**:不同数据,通信频率最低(每 step 一次),**可以最远——跨集群**

实际拓扑:

```
                        [DP group]
                  ┌───────────┴───────────┐
                  │                       │
            [PP group]                [PP group]
            ┌─────┴─────┐             ┌─────┴─────┐
            │           │             │           │
       Node 0,1     Node 2,3      Node 4,5     Node 6,7
       (PP stage 0) (PP stage 1)  ...           ...
            │           │             │           │
        TP within    TP within    TP within    TP within
        each node    each node    each node    each node
        (8-way)      (8-way)      (8-way)      (8-way)
```

这就是 Megatron-LM、DeepSpeed 大规模训练的标准拓扑——256 卡 = 8(TP) × 4(PP) × 8(DP),每根线的通信负载和它要走的物理介质带宽匹配。

## 五、Elastic 训练与容错

### 5.1 长训练必须容错

训练 70B+ 模型要跑几周,1024 张 GPU 同时跑,任何一张挂掉、任何一根 IB 链路抽风,整个 job 都会卡死。**MTBF 算下来一两天必坏一台**。如果每次 fail 都从头开始训,你永远训不完。

### 5.2 torchrun 的 elastic 模式

torchrun 内置 `--max-restarts`:

```bash
torchrun \
    --nnodes=4 \
    --max-restarts=10 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=192.168.1.100:29500 \
    --rdzv-id=my_job_id \
    train.py
```

工作流程:

1. 某个 worker 崩了 → torchrun 检测到 → 通知其他 worker 暂停
2. 重新做 rendezvous(可能 world_size 变小了)
3. 所有 worker 加载最新 checkpoint
4. 从上次保存的 step 继续训

要让这套机制真正能用,你的训练脚本必须满足:

- **有定期 checkpoint**(比如每 500 步)——后面 `Checkpoint.md` 会专门讲
- **dataloader 状态可恢复**——`DistributedSampler.set_epoch` + 步数偏移
- **优化器和 scheduler 状态都在 checkpoint 里**——光保存 model 是不够的

### 5.3 慢节点(straggler)问题

elastic 解决的是"挂了"的节点。**更阴险的是"慢"的节点**:某张 GPU 因为温度过高降频、某个 IB 链路丢包、某个磁盘 IO 卡——这台机器没坏,只是变慢了 10%。

AllReduce 是同步集合操作,最慢的那台拖整个 group。1024 卡训练里,1 台慢 10% = 整个吞吐降 10%。

排查方法:

- **NCCL_DEBUG=INFO** 看每次 AllReduce 的耗时,有些节点显著比别的慢
- **`torch.cuda.synchronize()` 后打 timestamp**,定位每个 step 哪台先到
- **Slurm/K8s 的 GPU 利用率监控**,找平均利用率显著低的节点

发现 straggler 后通常的处理:把这台机器从 job 里踢掉(`scontrol`),让 elastic 重新 rendezvous,继续训练。生产集群通常有自动 straggler detection 系统。

## 六、生产化的几个细节

### 6.1 Logging 永远 rank 0 一份

多机训练默认所有 rank 都 print,日志会乱套。统一规则:

```python
import logging
import torch.distributed as dist

if dist.get_rank() == 0:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.ERROR)  # 其他 rank 只打 error

def log(msg):
    if dist.get_rank() == 0:
        logging.info(msg)
```

但**异常仍然要让所有 rank 看到**——某个非 0 rank 的 NCCL 报错如果被压住,你只能看到 rank 0 卡死,debug 困难。所以异常用 `print(f"[rank {rank}] {e}")` 而非 `logger`。

### 6.2 时间不同步会让 NCCL 超时

NCCL 集合通信默认 30 分钟超时。如果你的 ckpt 保存只在 rank 0 做且耗时 25 分钟,其他 rank 在 AllReduce 等了 25 分钟,NCCL 监控线程觉得"差不多要超时了"开始打 watchdog 警告。再加个 10 分钟,直接 abort。

解决:**checkpoint 在 rank 0 写完之后必须 `dist.barrier()`**,让其他 rank 显式等;或者用异步 ckpt(后面 Checkpoint.md 讲)。

### 6.3 OS 层面的 ulimit

NCCL 需要打开很多 fd(每个对端通信连接一个),容器或者新装的机器默认 `ulimit -n` 可能只有 1024。1024 卡训练里这远远不够,会看到 `Too many open files` 或者神秘的 NCCL "channels" 报错。

**生产环境标配**:

```bash
ulimit -n 1048576       # file descriptors
ulimit -l unlimited     # locked memory (RDMA 需要)
ulimit -s unlimited     # stack
```

通常写到 `/etc/security/limits.conf` 或者容器启动时设置。

### 6.4 NCCL 死锁的常见模式

NCCL 卡死的两类原因:

**模式 1:某个 rank 没参与集合通信**。比如 forward 里有 `if rank == 0: do_something()`,某个步骤只有 rank 0 做了 AllReduce。其他 rank 不知道这次通信存在,就在下一个集合通信里等永远不会来的 rank 0,死锁。

**模式 2:不同 rank 的集合通信顺序不一致**。所有 rank 必须**严格按相同顺序**调用 AllReduce/AllGather/Broadcast。如果你写了:

```python
if loss > threshold:
    dist.all_reduce(metric_a)
else:
    dist.all_reduce(metric_b)
```

不同 rank 的 loss 不一样,有的进 if 有的进 else,**两边各自的 AllReduce 在等彼此,死锁**。修法:把控制流改成所有 rank 一致,或者把 metric 通信都做。

调试这两类问题的招:`NCCL_DEBUG=INFO` 看每个 rank 卡在哪一次 collective 上,通常一眼就看出谁少了或者顺序对不上。

## 七、最小可用的多机训练 checklist

把整章串起来,验证一个新集群能不能跑多机训练,按这个顺序检查:

1. **网络连通**:Node 0 能 ping 通其他所有 Node 的 IB 接口 IP(`ping ib0_ip`)
2. **`nvidia-smi topo -m`**:确认节点内 GPU 拓扑是 NV-link 全互连
3. **`ibstat`**:确认 IB 接口 active,LinkSpeed 在期望速率
4. **`NCCL_DEBUG=INFO` 跑 2-node × 2-GPU 的 hello world AllReduce**:看到 `NET/IB` 而不是 `NET/Socket`
5. **测一次完整 step 的耗时**,与单机 8 卡的吞吐量做对比,期望 multi-node 的 throughput 至少 80% × `nodes` × 单机吞吐
6. **打开 `--max-restarts=N`**,人工 kill 一台 worker 验证 elastic 恢复
7. **测 ckpt 保存 + 加载流程**,验证从 ckpt resume 后 loss 曲线无跳变

每个项目失败都对应一个具体的工程问题,没有跳过的捷径。**多机训练 80% 的时间在调通信和配置,只有 20% 在跑实际训练**——这是大模型工程师的日常。
