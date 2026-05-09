---
title: 7. Checkpoint 与训练恢复
categories: 学习笔记- AI Infra
date: 2026-05-09 13:00:00
mathjax: true
tags:
    - AI
    - AI Infra
---


长训练必然挂——硬件会坏、网络会抽风、人会按错按钮、kernel 会触发数值 NaN。一个跑两周的 70B 训练 job 中途崩一次,如果没有 checkpoint,**两周白干**。所以 checkpoint 不是"训练好了之后存起来",而是**训练循环里和 forward/backward 同等重要的一环**。

但 checkpoint 远不是 `torch.save(model.state_dict())` 那么简单。在 DDP / FSDP / ZeRO / Pipeline 各种并行下,"完整状态"包含什么、怎么保存、怎么加载、怎么和不同的 world_size 兼容、怎么不让 ckpt I/O 阻塞训练——每个问题都有对应的工程坑。这一篇把这些坑全部串起来讲。

## 一、起因:什么算"训练状态"

### 1.1 错误示范:只存权重

新手最容易犯的错:

```python
# ❌ 这样保存,resume 之后训练动力学被悄悄重置
if rank == 0:
    torch.save(model.state_dict(), "ckpt.pt")
```

这样恢复出来的训练**和原训练不等价**,以下东西全都会被重置或丢失:

- **优化器状态**(Adam 的 m、v):重置后头几百步训练会偏向重新冷启动的方向
- **学习率调度器**:重新从初始 LR 开始 warmup,等价于 LR schedule 整个错位
- **DataLoader 进度**:从 epoch 0 开始,等于把已经训过的数据再训一遍
- **RNG 状态**:dropout / 数据增强的随机性变了,严格意义上不等价
- **GradScaler 状态**(FP16 训练时):loss scale 重置回初始值,前几百步可能 skip 大量更新

如果你的 loss 曲线从 ckpt resume 之后看起来"明显偏离原轨迹",大概率就是这些状态没存全。

### 1.2 完整 checkpoint 包含的内容

**严格意义上能让训练完全恢复**的 ckpt 至少包括:

| 项 | 是什么 | 为什么必须存 |
| --- | --- | --- |
| `model.state_dict()` | 模型参数 | 显然 |
| `optimizer.state_dict()` | Adam m、v、step counter 等 | 优化方向连续性 |
| `scheduler.state_dict()` | 当前 step、LR 计划状态 | LR 不被重置 |
| `step` / `epoch` | 训练进度 | DataLoader 跳到正确位置 |
| `dataloader.state_dict()` | 数据迭代器位置(可选,新 API) | 不重复训练旧数据 |
| `torch.get_rng_state()` 等 | CPU/GPU/NumPy RNG 状态 | 严格可复现 |
| `scaler.state_dict()` | (FP16 时)GradScaler 内部状态 | loss scale 不被重置 |
| 任何自定义 metric tracker | 你的 EMA、moving average 等 | 状态不丢 |

实际工业代码里,通常用一个大 dict 一次性存:

```python
ckpt = {
    "step": global_step,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict() if scaler else None,
    "rng": {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),  # 注意是 all,每张卡一份
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    },
    "config": training_config,  # 训练超参,resume 时用来 sanity check
}
torch.save(ckpt, ckpt_path)
```

以及 resume 时严格按相反顺序加载:

```python
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
scheduler.load_state_dict(ckpt["scheduler"])
if scaler: scaler.load_state_dict(ckpt["scaler"])
torch.set_rng_state(ckpt["rng"]["cpu"])
torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
np.random.set_state(ckpt["rng"]["numpy"])
random.setstate(ckpt["rng"]["python"])
global_step = ckpt["step"]
```

## 二、DDP 的 checkpoint:陷阱与标准写法

### 2.1 `model.module` 这个细节

DDP 包装后,你的原模型在 `.module` 属性下:

```python
model = DDP(my_model, device_ids=[local_rank])
# model.state_dict() 的 key 都带 "module." 前缀
# model.module.state_dict() 才是干净的、能跨 DDP 加载的版本
```

**保存规则**:统一存 `model.module.state_dict()`,不带 `module.` 前缀。这样 ckpt 能被:

- 单卡推理直接加载
- 任意 world_size 的 DDP 加载(只要参数形状一样)
- 别人 fork 出去用任何方式加载

如果你不小心存了 `model.state_dict()`,加载到单卡模型时会报"Unexpected key 'module.xxx'",还得手写代码去 strip 前缀,纯属自找麻烦。

### 2.2 只在 rank 0 写,其他 rank barrier 等

DDP 下所有 rank 的 model.module 是同步的(每步 AllReduce 保证),没必要每个 rank 都写一份 ckpt:

```python
def save_ckpt(state, path):
    if dist.get_rank() == 0:
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)   # 原子替换,避免半截 ckpt
    dist.barrier()  # 其他 rank 等 rank 0 写完
```

`dist.barrier()` 这一行不能少:

- 没有它,非 0 rank 立刻往下跑,可能开始下一次 AllReduce,而 rank 0 还在写 I/O,**两边时序错开**
- 写 ckpt 在大模型上耗时几十秒到几分钟,期间 NCCL 没有任何通信,**watchdog 会触发 30 分钟 timeout 警告**

### 2.3 写 .tmp 再 rename 的原子性

`os.replace(tmp, final)` 在 POSIX 文件系统上是原子操作。这个细节防止"程序在写 ckpt 的过程中又崩了"——如果直接写 `final`,挂掉时 ckpt 是半截的,resume 会读到坏数据。先写 `.tmp`、再 rename,任何时候 `final` 要么是上次的完整 ckpt、要么是这次的完整 ckpt,**永远不会是半截**。

更工业化的做法:保留多份 ckpt 滚动,比如 `ckpt-1000.pt / ckpt-2000.pt / ckpt-3000.pt`,只保留最近 3 份,加上一个软链 `latest.pt -> ckpt-3000.pt`。这样万一最新 ckpt 因某种原因损坏,可以回退到上一份。

## 三、FSDP / ZeRO 的 checkpoint:三种 state_dict 类型

DDP 下所有 rank 的模型完全一样,存哪个 rank 都行。FSDP 下**每张卡只持有 1/N 参数分片**——保存时怎么做就有了选择。FSDP 提供三种 state_dict 类型,各有适用场景:

### 3.1 `FULL_STATE_DICT`:聚合到一份完整 ckpt

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)

cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
    state = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, "ckpt.pt")
dist.barrier()
```

工作流程:

```
GPU0 [P₀] ┐
GPU1 [P₁] ├─ AllGather 到 rank 0 的 CPU 内存 ─► [P₀|P₁|P₂|P₃]
GPU2 [P₂] │                                              ↓
GPU3 [P₃] ┘                                          torch.save
```

**优点**:ckpt 是单一完整文件,可以单卡加载、跨任意 world_size 加载、和 transformers 等推理库直接兼容。

**缺点**:rank 0 的 CPU 必须能装下整个模型——70B 的 ckpt 是 280 GB(FP32)或 140 GB(BF16),需要相应大小的内存。同时聚合本身有 AllGather 开销。**适合中小模型(< 30B)和最终发布**。

### 3.2 `SHARDED_STATE_DICT`:每张卡存自己那一份

```python
from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig
import torch.distributed.checkpoint as dcp

cfg = ShardedStateDictConfig(offload_to_cpu=True)
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg):
    state = {"model": model.state_dict()}
    dcp.save(state, checkpoint_id="ckpt_dir")
```

工作流程:**每张卡把自己持有的 1/N 参数写入磁盘**,文件夹结构类似:

```
ckpt_dir/
  __0_0.distcp     ← rank 0 的分片
  __1_0.distcp     ← rank 1 的分片
  __2_0.distcp     ← rank 2 的分片
  ...
  .metadata        ← 描述每个 tensor 在哪个文件、什么 offset
```

**优点**:写 I/O 完全分布式,**保存速度随 world_size 线性提升**;rank 0 不需要装下整个模型;文件大小均衡。**适合大模型训练中频繁保存**(每 500 步一次)。

**缺点**:加载时需要相同(或兼容)的并行策略;不能直接喂给单卡推理。

### 3.3 `LOCAL_STATE_DICT`:不带 reshape 信息

更底层、不带 metadata 的版本,几乎只在 PyTorch 内部用,普通用户不直接接触。

### 3.4 三种类型怎么选

| 场景 | 用哪个 |
| --- | --- |
| 训练过程中频繁保存(每 500-2000 步) | SHARDED_STATE_DICT(快、不阻塞) |
| 训练结束的最终 ckpt(要发布、要推理) | FULL_STATE_DICT(单文件、通用) |
| 中途想换卡数继续训练(比如 8 卡训到一半改成 16 卡) | SHARDED + DCP 的 reshard 加载 |

最佳实践:**训练中只存 sharded,定期(比如每 epoch)再存一份 full**——既快又有可移植的 ckpt 备份。

## 四、`torch.distributed.checkpoint`:跨 world_size 加载

### 4.1 重要的能力:resharding

实际训练里经常碰到这种需求——8 卡训到一半,扩容到 32 卡继续训。如果 ckpt 是按 8 卡 shard 存的,32 卡能加载吗?

PyTorch 2.x 的 `torch.distributed.checkpoint`(简称 DCP)支持**自动 resharding**:加载时根据当前 world_size 自动把每个 tensor 重新切分到对应 rank 上。

```python
import torch.distributed.checkpoint as dcp

# 8 卡时保存
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    state = {"model": model.state_dict(), "optim": optimizer.state_dict()}
    dcp.save(state, checkpoint_id="ckpt_dir")

# 32 卡时加载——自动 reshard,代码完全一样
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    state = {"model": model.state_dict(), "optim": optimizer.state_dict()}
    dcp.load(state, checkpoint_id="ckpt_dir")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optim"])
```

DCP 内部的工作:读 `.metadata` 知道每个 tensor 完整形状;每个 rank 计算自己在 32 卡下应该持有哪一段;从 8 卡的 .distcp 文件里**只读自己需要的那部分**,组装成本地 shard。

**这是大模型训练扩缩容的关键基础设施**——没有它,改 world_size 等于重新训。

### 4.2 优化器状态的分片

DDP 下优化器状态和参数同步,直接 `optimizer.state_dict()` 就是完整的。FSDP 下优化器状态本身就是分片的(每卡只管自己那 1/N 参数的 m、v),保存时直接每卡存自己的就行——**FSDP + DCP 内部已经处理好了**。

但有个重要细节:`optimizer.state_dict()` 返回的状态用 `state[param_id]` 索引,**`param_id` 是参数在创建顺序里的位置**,不是参数名。如果模型结构变了(添加/删除某层),resume 时 param_id 就对不上。所以**ckpt 必须用相同的模型结构加载**——sanity check 这一点放进 resume 代码里:

```python
# resume 时验证 config 一致
assert ckpt["config"]["hidden_size"] == current_config["hidden_size"]
assert ckpt["config"]["num_layers"] == current_config["num_layers"]
# ... 其他关键超参
```

## 五、DataLoader 状态:被忽视的"半个 ckpt"

### 5.1 不存 dataloader 状态会怎样

假设你训了 10000 步,在第 7000 步保存 ckpt。每步 batch_size = 256,数据集有 100M 样本——你已经过了 7000 × 256 = 1.79M 个样本,大约 1.8% 的数据集。

resume 时,默认 `DataLoader` 会**从头开始迭代**:你用 ckpt 的模型权重,但又从样本 0 开始重新喂数据。等价于让模型多看了一遍那些已经训过的数据,**LR scheduler 已经走到 7000 步对应的低 LR,但训练分布偏移了**——loss 曲线会出现一段诡异的"先降后升"。

这个问题在小训练上不明显(数据循环很快),但在大模型训练上极其重要——预训练通常**只过一遍数据**,重复样本意味着浪费配额。

### 5.2 DistributedSampler 的 set_epoch + step offset

最简单的方案:把"这是第几个 epoch、epoch 内已经走了多少步"存进 ckpt:

```python
# 保存时
ckpt["epoch"] = current_epoch
ckpt["step_in_epoch"] = step_in_epoch

# 加载时
sampler.set_epoch(ckpt["epoch"])
loader_iter = iter(loader)
for _ in range(ckpt["step_in_epoch"]):
    next(loader_iter)  # 跳过已经训过的样本
```

这个方案简单但 ugly——`next` 跳过会触发实际的 IO 和数据增强,resume 慢且浪费。

### 5.3 PyTorch 2.x 的 stateful DataLoader

PyTorch 2.x 起的 `StatefulDataLoader`(在 `torchdata` 包)直接支持 state_dict:

```python
from torchdata.stateful_dataloader import StatefulDataLoader

loader = StatefulDataLoader(dataset, batch_size=256, ...)

# 保存
ckpt["dataloader"] = loader.state_dict()

# 加载
loader.load_state_dict(ckpt["dataloader"])
# 接下来 iter(loader) 直接从断点继续,无 skip 开销
```

它内部记录了当前 worker 的位置、shuffle seed 状态、prefetch 队列等,resume 后**精确从中断点继续**,这是当前最干净的做法。

## 六、异步 checkpoint:不让 I/O 阻塞训练

### 6.1 为什么 ckpt 写慢会拖训练

70B 模型 BF16 一份 ckpt 140 GB,即使按 sharded 切到 32 张卡,每张也要写 4.4 GB。本地 NVMe 写带宽通常 3-7 GB/s,**单卡写一份要 1-1.5 秒**。

这 1.5 秒里所有 rank 都被 `dist.barrier()` 阻塞,什么训练都做不了。如果每 500 步存一次,1.5 秒的 ckpt 意味着 **0.3% 的训练时间被 ckpt I/O 占用**——看起来不多,但在 1024 卡训练上 1% 就是几千美元的算力浪费。

更糟糕的是远端存储:很多集群 ckpt 必须写到共享文件系统(如 Lustre、GPFS、S3),网络带宽远低于本地 NVMe,可能拖到 30 秒一份。

### 6.2 异步 ckpt 的思路

Ckpt 数据本身在 GPU 显存里,实际 I/O 阶段 GPU 并不参与——它只需要把数据搬到 CPU pinned memory,然后让 CPU 后台慢慢写磁盘:

```
GPU 训练流:  step N ──► step N+1 ──► step N+2 ──► step N+3 ...
                                                     ↑ 这一步前 ckpt 已经写好

GPU→CPU 拷贝:    [拷贝 N 状态]
                     ↓
CPU 后台 I/O:        [写磁盘 N 状态.....................]
                     ↑ 在 GPU 跑 step N+1, N+2 期间慢慢写
```

PyTorch 2.x 的 DCP 提供 `async_save`:

```python
import torch.distributed.checkpoint as dcp

# 异步保存,立刻返回一个 future
future = dcp.async_save(state, checkpoint_id="ckpt_dir")

# 训练继续,GPU 不阻塞
for step in range(...):
    train_step()

# 在下一次 ckpt 之前确认上一次写完了
future.result()
```

**关键约束**:`async_save` 拷贝 state 到 CPU 后,你不能继续修改原 state(否则数据竞态)。所以异步 ckpt 通常配合 sharded state_dict 用——CPU 收到的是 tensor 引用,后续训练修改的是 GPU 上的原 tensor,引用本身不变,但内容会被覆盖。要么 DCP 内部拷贝一份 CPU 副本(显存换时间),要么你保证 ckpt 拷贝完 N 步以内 GPU 不会修改这些 tensor。

### 6.3 直觉:把 ckpt 当后台 logging

异步 ckpt 之后,保存 ckpt 在心理上和 print logging 一样——主线训练完全不感知它在干什么,只在很久之后某次 future.result() 才确认完成。这才是工业级长训练的标准模式,不是"训练 → 等 30 秒写 ckpt → 继续训练"。

## 七、几个实战 checklist

### 7.1 启动训练前

- 把 ckpt 路径配好,**先建空目录、确认有写权限**——别等训练 1000 步要保存第一个 ckpt 时才发现写不进
- 设置 ckpt 保留策略,**最少保留 3 份**(防止最新一份损坏)
- 决定 ckpt 频率:训练总长 / 每个 ckpt 时间 / 可接受的 fail 重训成本——典型设置是每 30-60 分钟一份

### 7.2 训练中

- **每个 ckpt 都验证一遍**:加载回来跑一个 micro-step,验证 loss 与保存时一致
- 监控 ckpt 写入耗时,**异常变长**通常意味着存储系统拥堵或者 NIC 问题
- 异步 ckpt 时,确保 ckpt 队列没有挤压(否则 CPU 内存会涨)

### 7.3 Resume 时

- **首先验证 config 一致性**,模型结构变了直接报错
- 加载顺序:`model → optimizer → scheduler → scaler → dataloader → rng`
- **加载完后跑一个 forward + loss,与 ckpt 保存时记录的 loss 比较**——偏差超过 1% 说明哪里没恢复对
- LR 应该**正好等于** scheduler 在该 step 的 LR,不要自己手动改 LR

### 7.4 发布最终模型时

- 训练用 sharded ckpt,发布前用 FULL_STATE_DICT 转一份完整 ckpt
- **去掉所有训练专用状态**(optimizer、scheduler、scaler、rng、dataloader),只留 model state_dict
- 文件命名带 step / epoch / loss 信息,方便后续 ablation
- 最好附一个 `model_card.md` 说明训练数据、硬件、超参——不属于 infra 范畴但相当重要

## 八、一句话总结

Checkpoint 是大模型训练的**保险机制**——理解它的不只是"`torch.save` 几行代码",而是**完整状态包含什么、并行策略下怎么切、I/O 怎么不阻塞训练、resume 怎么严格无偏**。这套东西做不好,你训练的每一步都在赌运气;做好之后,长训练才会有真正的稳定可恢复性。
