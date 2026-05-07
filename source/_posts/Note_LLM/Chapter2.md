---
title: Chapter2 视觉对比学习：InfoNCE 与 SimCLR/MoCo
categories: 学习笔记-大模型
date: 2026-03-27 16:03:34
mathjax: true
tags:
    - AI
    - AI面试知识
---

> **本章定位**：从 Ch1 的"点之间的相似度"过渡到"用相似度训练表征模型"。SimCLR/MoCo 是对比学习的两个奠基性工作，**它们的损失函数（InfoNCE）和训练范式直接影响了后来的 CLIP（Ch3）和所有现代 embedding 模型**。

> **承上**：基于 Ch1 §3 的点积相似度和 §6 的交叉熵。
> **启下**：本章只讲视觉范式；CLIP / SimCSE / BGE 等多模态与文本对比学习见 Ch3；BYOL/SimSiam/DINO 等无负样本路线见 Ch4。

---

# §A 数学原理

## 1. 对比学习的核心思想

**Instance Discrimination**：通过构造正负样本对，在无标注数据下学习"物以类聚，人以群分"的特征表示。

具体到视觉自监督：对同一张图 $x$ 应用两次随机增强 $t, t' \sim \mathcal{T}$，生成正样本对 $(x_i, x_j)$。模型必须把这两个增强视图在特征空间拉近，把它们与 batch 内其他样本拉远。

## 2. 视觉数据增强 —— 对比学习的"数据语义"

| 增强 | 作用 |
| :--- | :--- |
| **Random Resized Crop**（最关键） | 强迫模型学习局部与整体、多尺度语义一致性 |
| **Color Jitter / Grayscale** | 防止模型用颜色直方图作弊 |
| **Gaussian Blur** | 模糊纹理细节，迫使模型关注高层轮廓 |
| **Horizontal Flip** | 引入左右镜像不变性 |

> 这些增强本质上是**人工注入的不变性先验**：什么样的变化"语义不变"是由我们选择的。

## 3. InfoNCE 损失函数

$$\boxed{\mathcal{L}_{q, k_+} = -\log \frac{\exp(q \cdot k_+ / \tau)}{\exp(q \cdot k_+ / \tau) + \sum_{i=1}^{K} \exp(q \cdot k_i / \tau)}}$$

形式上就是一个 $K+1$ 路 softmax 交叉熵：正类是 $k_+$，负类是 $K$ 个负样本。

### 3.1 与互信息（MI）的关系

InfoNCE 是互信息 $I(q; k_+)$ 的**下界**：
$$I(q; k_+) \geq \log K - \mathcal{L}_{\text{InfoNCE}}$$

**推导直觉**：把 InfoNCE 看作"在 $K+1$ 个候选里挑出正样本"的 $\log K$-bit 分类任务。分类越准（loss 越低），$q$ 包含越多关于 $k_+$ 的信息。

**结论**：负样本数 $K$ 越大，下界越紧，特征越好。这是 SimCLR 需要大 batch（4096–8192）和 MoCo 需要队列（65536）的根本原因。

### 3.2 温度参数 $\tau$ 的双面性

$\tau$ 控制 softmax 分布的**尖锐度**：

| $\tau$ 大小 | softmax 形状 | 效应 |
| :--- | :--- | :--- |
| **大 $\tau$** (如 1.0) | 平滑、接近均匀 | 各样本权重接近，训练稳定但难以聚焦 hard negatives |
| **小 $\tau$** (如 0.05) | 尖锐 | 极大放大"长得像但不是"的困难负样本的 loss，学到细致判别特征；但太小会导致梯度集中在少数样本上 |

经验值：SimCLR $\tau = 0.1\sim 0.5$，MoCo $\tau = 0.07$。

---

# §B 模型结构（PyTorch 实现）

## B.1 SimCLR：端到端对称对比

**核心逻辑**：在大 Batch 中寻找自己。

```
原图 x  ──┐
          ├── 增强 t  ──→ x_i  ──┐
          │                       ├── Encoder f ──→ h ── Projector g ──→ z
          └── 增强 t' ──→ x_j  ──┘
所有 2N 个 z 进入 InfoNCE 计算
```

### B.1.1 SimCLR 损失（NT-Xent）的 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """SimCLR 的 InfoNCE 实现：2N 个样本两两计算相似度"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1, z2: [N, D] 同一 batch 的两个增强视图（已经过 projector）
        """
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                     # [2N, D]
        z = F.normalize(z, dim=-1)                         # L2 归一化

        # 相似度矩阵：所有对所有
        sim = z @ z.T / self.temperature                   # [2N, 2N]

        # 屏蔽对角线（自己与自己的相似度）
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))

        # 正样本位置：z[i] 的正样本是 z[i+N]（i < N）或 z[i-N]（i >= N）
        targets = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)

        return F.cross_entropy(sim, targets)
```

**关键点**：
- 把 batch 内所有 $2N$ 个样本展平做相似度矩阵，**正样本由 `targets` 显式索引**
- 对角线（自己 × 自己）必须屏蔽，否则它会成为最强的"正样本"
- L2 归一化后点积 = 余弦相似度（呼应 Ch1 §2）

### B.1.2 完整 SimCLR 模型

```python
class SimCLR(nn.Module):
    def __init__(self, encoder, proj_dim=128):
        super().__init__()
        self.encoder = encoder                              # ResNet-50 等
        feat_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()                          # 移除原 FC
        # Projection Head：2 层 MLP
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x)                                 # 表征：用于下游任务
        z = self.projector(h)                               # 投影：用于对比
        return h, z
```

> **面试 Tip：为什么需要 Projection Head？** 投影头作为"防火墙"，让信息损失发生在 $z$ 层，保护 $h$ 层保留更多下游任务有用信息。下游任务用 $h$，而非 $z$。

## B.2 MoCo：动量字典查询

**核心逻辑**：维护一个平滑演变的负样本队列。

```
x ── 增强 ──┬── x_q ── Query Encoder θ_q ──→ q  (梯度更新)
            └── x_k ── Key   Encoder θ_k ──→ k+ (动量更新)
                                              │
                          Queue ←─ 入队 ──── k+
                                              │
                负样本：q · k_i（i=1...K，来自 queue）
```

### B.2.1 MoCo 关键机制 PyTorch

```python
class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07):
        super().__init__()
        self.K, self.m, self.T = K, m, T                    # 队列大小 / 动量 / 温度

        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        # Key encoder 初始化 = Query encoder
        for p_q, p_k in zip(self.encoder_q.parameters(),
                            self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False                       # ⭐ 不通过梯度更新

        # 负样本队列：FIFO，存历史 batch 的 key
        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for p_q, p_k in zip(self.encoder_q.parameters(),
                            self.encoder_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr+batch_size] = keys.T          # 入队
        self.queue_ptr[0] = (ptr + batch_size) % self.K     # 环形缓冲

    def forward(self, im_q, im_k):
        q = F.normalize(self.encoder_q(im_q), dim=1)        # [N, D]

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.encoder_k(im_k), dim=1)    # [N, D]

        # 正对 logits: [N, 1]
        l_pos = (q * k).sum(dim=1, keepdim=True)
        # 负对 logits: [N, K] —— q 与 queue 中所有历史 key 的点积
        l_neg = q @ self.queue.clone().detach()

        logits = torch.cat([l_pos, l_neg], dim=1) / self.T  # [N, K+1]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=q.device)

        loss = F.cross_entropy(logits, labels)              # 正样本永远在第 0 位
        self._dequeue_and_enqueue(k)
        return loss
```

**MoCo 三个关键设计**：
1. **动量更新 Key Encoder**：$\theta_k \leftarrow m\theta_k + (1-m)\theta_q$，$m=0.999$，保证 queue 中负样本的"特征一致性"
2. **队列 FIFO**：负样本数 $K$ 与 batch size 解耦，单卡也能用 65536 个负样本
3. **不对称编码**：Query 走梯度，Key 走动量。这一思想直接通往 Ch4 BYOL 的 EMA Target

---

# §C 训练与推理

## C.1 训练流程：SimCLR 完整循环

```python
def simclr_train_step(model, criterion, x, optimizer, augment):
    # 1. 双路增强
    x1 = augment(x)                                         # [B, 3, 224, 224]
    x2 = augment(x)

    # 2. 前向：得到表征 h 和投影 z
    _, z1 = model(x1)
    _, z2 = model(x2)

    # 3. NT-Xent loss
    loss = criterion(z1, z2)

    # 4. 反向 + 更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

**关键超参（SimCLR 论文）**：
- Batch size：**4096–8192**（这是最大瓶颈）
- Optimizer：LARS (大 batch 友好)
- Learning rate：`0.3 × batch_size / 256`，cosine schedule
- Epochs：800–1000（自监督收敛慢）
- 投影头：2 层 MLP，输出 128 维

## C.2 训练流程：MoCo 完整循环

MoCo 单卡可训，loss 形式与上面 forward 已包含。要点：

| 配置 | SimCLR | MoCo v2 |
| :--- | :--- | :--- |
| Batch size | 4096–8192 | 256 即可 |
| 负样本数 | 2N - 2 | K = 65536（队列） |
| 硬件 | TPU 集群 | 8×V100 |
| Loss 形式 | 对称 NT-Xent | 单向 InfoNCE |

## C.3 推理视角一：Linear Probe 评测协议

自监督模型训练完后**怎么评估其表征质量**？标准协议是 **Linear Probe**：
1. 冻结 encoder $f$
2. 在其输出 $h$ 上加一个 **线性分类头**
3. 用 ImageNet 标签做监督训练（只训练这个线性层）
4. 看 top-1 accuracy

```python
# 冻结 backbone，只训练线性分类器
for p in model.encoder.parameters():
    p.requires_grad = False

classifier = nn.Linear(feat_dim, num_classes)
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1)

for x, y in loader:
    with torch.no_grad():
        h = model.encoder(x)                                # 冻结特征
    logits = classifier(h)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
```

**关键数字**（ImageNet linear probe）：
- 监督 ResNet-50：76.5%
- SimCLR：69.3%（v2 76.5%）
- MoCo v2：71.1%

> Linear probe 高 = 表征"线性可分性"好 = encoder 学到了有意义的语义。

## C.4 推理视角二：视觉骨干在 LLM 中的应用

SimCLR/MoCo 训出的 ResNet 主要用作**下游分类/检测**的预训练初始化。但在多模态 LLM 时代，更常用的视觉骨干来自**对比学习 + 图文对**（CLIP，详见 Ch3）和**自蒸馏**（DINO，详见 Ch4）。

| 视觉骨干路线 | 代表 | 主要用途 |
| :--- | :--- | :--- |
| 视觉对比（本章） | SimCLR / MoCo | 早期下游任务初始化 |
| 图文对比 | **CLIP**（Ch3） | LLaVA、GPT-4V、Qwen-VL 默认骨干 |
| 自蒸馏 | **DINOv2**（Ch4） | 密集预测任务（分割、深度）更优 |

---

# §D 负样本：到底为什么重要？

## D.1 数学本质

回到 InfoNCE 的 MI 下界 $I(q; k_+) \geq \log K - \mathcal{L}_{\text{InfoNCE}}$：
- **没有负样本时**：模型最简单的"偷懒"是输出常数 → loss = 0 但什么都没学到（**塌缩**）
- **负样本数 $K$ 越大**：MI 下界越紧 → 特征空间分布刻画越精准

## D.2 解决负样本成本过高的四条路线

| 路线 | 核心思路 | 代表方法 | 后续章节 |
| :--- | :--- | :--- | :--- |
| **A. 缓存机制** | 队列 + 动量 → 与 batch size 解耦 | **MoCo**（本章） | — |
| **B. Hard Negative Mining** | 只挖"难"的几个负样本，权重高 | MoCHi、ANCE、RocketQA | Ch3 §C |
| **C. 不对称结构** | 完全去掉负样本 | **BYOL / SimSiam** | **Ch4** |
| **D. 特征去相关** | 协方差矩阵 → 单位矩阵 | Barlow Twins | Ch4 §B 简介 |

---

## 承上启下

本章为对比学习打下了**数学（InfoNCE + MI 下界）+ 工程（队列 + 动量）**的底子。

下一章 **Ch3** 会把这套机制推广到两个新场景：
- **跨模态**：CLIP 用图文对训练，4 亿对，一举成为多模态 LLM 的视觉骨干
- **文本**：SimCSE 用 dropout 当增强、BGE/E5 用 Hard Negative Mining 训出现代 RAG 的 retriever

随后 **Ch4** 会进入完全不同的路线：**没有负样本也能训表征模型**（BYOL/SimSiam/DINO）。
