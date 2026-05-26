---
title: Chapter4 自监督新范式：BYOL、SimSiam、DINO 与 EM 视角
categories: 学习笔记-大模型
date: 2026-03-29 16:03:34
mathjax: true
tags:
    - AI
    - AI面试知识
---

> **本章定位**：完全不同于 Ch2/Ch3 的"有负样本"路线，本章探讨**完全不需要负样本**的自监督方法。BYOL/SimSiam/DINO 给出的答案：**只要打破对称性（Stop-gradient + Predictor），模型就不会塌缩**。这套机制是 Ch6 RLHF Reference Policy 设计的精神来源。

> **承上**：Ch2 §D.2 提到的"路线 C：不对称结构"和"路线 D：特征去相关"。
> **启下**：Ch6 的 Reference Policy = 本章的 Target Network。

---

# §A 数学原理

## 1. BYOL 的非对称设计

BYOL（Bootstrap Your Own Latent，DeepMind 2020）首次证明：**在没有负样本的情况下也能训出 SOTA 表征**。

> **关键实验数字**：BYOL 用 ResNet-50 在 ImageNet linear probe 达 **74.3%**，首次实现"无负样本 > SimCLR"。

### 1.1 网络架构

* **Online Network**（参数 $\theta$）：Encoder $f_\theta$ + Projector $g_\theta$ + **Predictor $q_\theta$**（核心）
* **Target Network**（参数 $\xi$）：Encoder $f_\xi$ + Projector $g_\xi$（**没有 Predictor**）

### 1.2 损失函数

视图 $v$ 经过 Online，$v'$ 经过 Target。L2 归一化后计算 MSE：
$$L = \|\bar{q_\theta(g_\theta(f_\theta(v)))} - \bar{g_\xi(f_\xi(v'))}\|_2^2$$

**MSE 与余弦的等价性**（呼应 Ch1 §2）：
$$\|\bar{a} - \bar{b}\|^2 = 2 - 2\bar{a}^T\bar{b} = 2(1 - \cos\theta)$$

最小化归一化 MSE = 最大化余弦相似度。

### 1.3 参数更新

* **Online**：标准梯度下降 $\theta \leftarrow \theta - \eta \nabla_\theta L$
* **Target**：动量更新（不传梯度）
$$\xi \leftarrow m \xi + (1-m) \theta, \quad m \approx 0.99\sim 0.999$$

### 1.4 为什么不会塌缩？

设想塌缩解：所有输入 $\to$ 同一向量 $c$。则 $L = 0$，但模型无意义。BYOL 用三重机制避免此解：

1. **Predictor $q_\theta$ 引入非线性**：Online 必须"预测"Target 的输出，而非简单复制
2. **Stop-gradient 切断 Target 梯度**：Target 不会主动向常数解漂移
3. **EMA 滞后性**：Target 是 Online 的"过去自己"，时间不一致性打破同步塌缩

---

## 2. SimSiam 的极简化：连 EMA 都不需要

SimSiam（何恺明 2021）证明：**Stop-gradient + Predictor 已是充要条件，EMA 可去掉**。

> **关键实验数字**：SimSiam ImageNet linear probe **71.3%**，仅略低于 BYOL，但训练成本远低（无 EMA 双网络）。

### 2.1 架构与流程

完全共享权重的孪生网络：
- Encoder $f$、Projector $g$、Predictor $h$ 都共享
- 两路输入 $x_1, x_2$ 都经过 $f$ 和 $g$，得到 $z_1, z_2$
- **只对 $z_1$ 用 Predictor**：$p_1 = h(z_1)$
- **对称损失**：
$$L = \frac{1}{2} D(p_1, \text{stop\_grad}(z_2)) + \frac{1}{2} D(p_2, \text{stop\_grad}(z_1))$$
其中 $D(p, z) = -\frac{p^T z}{\|p\| \|z\|}$ 是负余弦相似度。

### 2.2 EM 推导（紧凑版）

何恺明在论文中给出"为什么不塌缩"的数学解释：SimSiam 实际是 **EM 算法**。

**目标函数**：假设每张图 $x$ 有"理想表示" $\eta_x$（隐变量），损失为
$$E(\theta, \eta) = \mathbb{E}_{x, T}\left[\|\mathcal{F}_\theta(T(x)) - \eta_x\|^2\right]$$

直接同时优化 $\theta$ 和 $\eta$ → 必然塌缩（$\eta_x \equiv c$）。SimSiam 通过**交替优化**避免：

| 步骤 | 操作 | SimSiam 中的实现 |
| :--- | :--- | :--- |
| **E-step**（固定 $\theta$，求 $\eta$） | $\eta_x^* = \mathbb{E}_T[\mathcal{F}_\theta(T(x))]$ | 期望不可算 → **用单样本 $\mathcal{F}_\theta(T'(x))$ 近似**（即另一路 + stop-grad） |
| **M-step**（固定 $\eta$，更新 $\theta$） | $\theta \leftarrow \arg\min_\theta L(\theta, \eta)$ | 标准梯度下降 |

### 2.3 Predictor 的真正作用 = 学习条件期望

E-step 用单样本替代真期望，引入巨大噪声。Predictor $h$ 的作用是**为这种噪声"去噪"**：

$$h^*(z_1) = \mathbb{E}_{T_2}[z_2 \mid z_1]$$

即 Predictor 学到的是**条件期望**——给定 $z_1$，最优预测是 $z_2$ 的期望。这就是为什么：
- ❌ 没有 Predictor → 模型直接对齐两个噪声样本 → 塌缩
- ✅ 有 Predictor → 模型对齐"去噪后的 $z_1$"和"原始 $z_2$" → 隐式估计条件期望，避免塌缩

---

## 3. DINO：自蒸馏 + 中心化 + 锐化

DINO（**Di**stillation with **NO** labels，Caron et al. 2021）：BYOL 的"亲表兄"，但用了**软标签蒸馏 + 防塌缩 trick**而非 MSE。

### 3.1 架构

Student $f_{\theta_s}$ + Teacher $f_{\theta_t}$（EMA）。两者都输出 $K$ 维 logits（不是 d 维向量），然后 softmax 得到分布。

### 3.2 损失：交叉熵（而非 MSE）

$$L = -\sum_K p_t(x)^{(K)} \log p_s(x)^{(K)}$$

其中 $p_s, p_t$ 是 student/teacher 的 softmax 输出。

### 3.3 防塌缩两件套

DINO 不靠 Predictor，而是靠两个 trick：

**Centering**：对 teacher 输出维护一个 EMA 中心 $C$，每次输出前减去：
$$p_t = \text{softmax}((g_{\theta_t}(x) - C) / \tau_t)$$
$$C \leftarrow m_C \cdot C + (1 - m_C) \cdot \frac{1}{B}\sum_b g_{\theta_t}(x_b)$$

**Sharpening**：teacher 用更小的温度 $\tau_t$（如 0.04），student 用更大的温度 $\tau_s$（如 0.1）。
- Sharpening：让 teacher 输出尖锐（接近 one-hot），强迫 student 学到"决断的"分类
- Centering：防止 teacher 总是偏向某些维度（避免另一种塌缩形式）

> 两者**互相牵制**：Sharpening 倾向于让某些维度主导（一种塌缩），Centering 把它们拉平。组合在一起达成动态平衡。

---

# §B 模型结构（PyTorch 实现）

## B.1 BYOL 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class BYOL(nn.Module):
    def __init__(self, encoder, hidden_dim=4096, proj_dim=256, m=0.996):
        super().__init__()
        feat_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()

        # Online: Encoder + Projector + Predictor
        self.online_encoder = encoder
        self.online_projector = self._make_mlp(feat_dim, hidden_dim, proj_dim)
        self.online_predictor = self._make_mlp(proj_dim, hidden_dim, proj_dim)

        # Target: 深拷贝 Online（无 Predictor）
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False                                  # ⭐ 不参与反传
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.m = m

    def _make_mlp(self, in_dim, hidden_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    @torch.no_grad()
    def update_target(self):
        """EMA 更新 Target 网络"""
        for online_p, target_p in zip(
            list(self.online_encoder.parameters()) + list(self.online_projector.parameters()),
            list(self.target_encoder.parameters()) + list(self.target_projector.parameters()),
        ):
            target_p.data = self.m * target_p.data + (1 - self.m) * online_p.data

    def forward(self, v1, v2):
        # Online 路径：含 Predictor
        p1 = self.online_predictor(self.online_projector(self.online_encoder(v1)))
        p2 = self.online_predictor(self.online_projector(self.online_encoder(v2)))

        # Target 路径：无 Predictor，无梯度
        with torch.no_grad():
            z1 = self.target_projector(self.target_encoder(v1))
            z2 = self.target_projector(self.target_encoder(v2))

        # 对称损失：归一化 MSE = -2·cos
        loss = byol_loss(p1, z2.detach()) + byol_loss(p2, z1.detach())
        return loss.mean()


def byol_loss(p, z):
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1)                               # 即 ||p-z||^2
```

## B.2 SimSiam 完整实现

```python
class SimSiam(nn.Module):
    """SimSiam = BYOL 去掉 EMA 和 Target 网络"""
    def __init__(self, encoder, proj_dim=2048, pred_dim=512):
        super().__init__()
        feat_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()
        self.encoder = encoder
        self.projector = self._make_proj(feat_dim, proj_dim)
        self.predictor = self._make_pred(proj_dim, pred_dim)

    def _make_proj(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False), nn.BatchNorm1d(in_dim), nn.ReLU(),
            nn.Linear(in_dim, in_dim, bias=False), nn.BatchNorm1d(in_dim), nn.ReLU(),
            nn.Linear(in_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim, affine=False),
        )

    def _make_pred(self, in_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # ⭐ 关键：z 上加 detach() 实现 stop-gradient
        loss = -(F.cosine_similarity(p1, z2.detach(), dim=-1).mean() +
                 F.cosine_similarity(p2, z1.detach(), dim=-1).mean()) * 0.5
        return loss
```

> **面试考点**：去掉 `z.detach()` 立即塌缩。原因见 §A.2 EM 推导——E-step 必须固定 $\eta$ 才能避免 $\eta = $ 常数的退化解。

## B.3 DINO 关键模块

```python
class DINOLoss(nn.Module):
    """DINO 的 centering + sharpening loss"""
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))      # ⭐ 中心向量

    def forward(self, student_output, teacher_output):
        student_out = student_output / self.student_temp
        # ⭐ Teacher 输出：先减 center，再除小温度（更尖锐）
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()                           # stop-grad

        # 交叉熵
        loss = -(teacher_out * F.log_softmax(student_out, dim=-1)).sum(dim=-1).mean()

        # ⭐ EMA 更新 center
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
```

---

# §C 训练与推理

## C.1 训练循环：通用模板

无负样本自监督的训练循环都长这样（以 BYOL 为例）：

```python
def train_byol(model, loader, optimizer, augment, epochs):
    for epoch in range(epochs):
        for x in loader:
            # 1. 双路增强
            v1 = augment(x)
            v2 = augment(x)

            # 2. 前向 + 损失
            loss = model(v1, v2)

            # 3. 反向 + Online 网络更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 4. ⭐ EMA 更新 Target（关键步骤）
            model.update_target()
```

**SimSiam** 没有第 4 步（不需要 EMA），其他完全一致。
**DINO** 第 4 步包含 Teacher EMA 更新和 Center EMA 更新。

## C.2 训练 vs 推理：模型状态的差异

| 阶段 | Online / Student | Target / Teacher | Predictor |
| :--- | :--- | :--- | :--- |
| **训练** | 梯度更新 | EMA 跟随（或 detach 共享） | 训练 |
| **推理** | **只用 Encoder $f_\theta$** | 丢弃 | 丢弃 |

> **关键**：所有 Projector / Predictor / Target 网络在推理时都**丢弃**——只保留主干 Encoder $f_\theta$ 用于下游任务。

## C.3 评测协议：Linear Probe / KNN

无监督方法训完怎么评估？两个标准协议：

**Linear Probe**（详见 Ch2 §C.3）：冻结 encoder，加线性头做监督训练。

**KNN Classification**：
```python
# 1. 全训练集 forward 得到特征库
features_train = model.encoder(x_train)                              # [N_train, D]
features_train = F.normalize(features_train, dim=-1)

# 2. 测试样本检索 K 个最近邻
features_test = F.normalize(model.encoder(x_test), dim=-1)
sim = features_test @ features_train.T                                # [N_test, N_train]
topk_indices = sim.topk(k=20, dim=-1).indices

# 3. 投票
predictions = labels_train[topk_indices].mode(dim=-1).values
```

> **关键数字**（ImageNet linear probe / KNN top-1）：
> - SimCLR：69.3% / 64.5%
> - MoCo v2：71.1% / 67.5%
> - **BYOL：74.3% / 69.6%**
> - **SimSiam：71.3% / 67.0%**
> - **DINO (ViT-B/16)：78.2% / 76.1%** —— ViT 上效果惊人

## C.4 推理视角：DINOv2 在多模态 LLM 中的应用

DINOv2（Meta 2023）是 DINO 在 ViT-g 上的工程加强版，已成为 **CLIP 之外的另一主流视觉骨干**：

| 视觉骨干 | 代表 VLM | 优势 |
| :--- | :--- | :--- |
| **CLIP**（Ch3） | LLaVA, Qwen-VL, GPT-4V | 与语言对齐，"看图说话"自然 |
| **DINOv2** | Llama 3.2 Vision, 部分 Qwen-VL 变体 | **密集预测任务（分割、深度）更优** |
| **混合**（CLIP + DINOv2） | 一些 SOTA VLM | 两者特征拼接，兼顾 |

**为什么 DINOv2 在密集任务上更强？**
- DINOv2 用纯视觉自监督，特征更"忠实于像素"
- CLIP 用图文对，倾向于"对象级别语义"，对像素级细节没那么敏感

---

# §D 落地 LLM：Stop-gradient + EMA 在大模型训练中的真实身影

这是本章最重要的一节。BYOL/SimSiam 看似 CV 技术，但其**核心机制在 LLM 训练流程中无处不在**：

| BYOL/SimSiam 概念 | LLM 训练中的对应 | 详见 |
| :--- | :--- | :--- |
| **Target Network** | **Reference Policy $\pi_{\text{ref}}$**（PPO/DPO/GRPO 的参考模型） | Ch6, Ch7 |
| **Stop-gradient on Target** | $\pi_{\text{ref}}$ 不参与反向传播 | Ch6 §B |
| **EMA 更新 Target** | **Online DPO / SPIN / Self-Rewarding LM** 中的周期性 reference 更新 | Ch7 §C |
| **Predictor 打破对称** | 知识蒸馏中 student 用额外结构匹配 teacher | Ch5 |
| **避免塌缩** | RLHF KL 惩罚防止策略塌缩到 reward-hacking 单点 | Ch6 §C |

### 案例预告：DPO 损失中的 Stop-gradient（详见 Ch7）

DPO 损失：
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

* $\pi_\theta$（policy）= **Online network**：被梯度更新
* $\pi_{\text{ref}}$（reference）= **Target network**：通常是 SFT 模型的冻结快照

这就是 BYOL 的精神在 LLM 上的直接应用。

---

# §E 横向对比：四大对比学习方法

| 方法 | 负样本 | EMA Target | Predictor | Stop-grad | 关键贡献 | 章节 |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **SimCLR** | ✓ | ✗ | ✗ | ✗ | 大 batch + 投影头 | Ch2 |
| **MoCo** | ✓ (队列) | ✓ | ✗ | ✓ | 队列解耦显存与负样本数 | Ch2 |
| **BYOL** | ✗ | ✓ | ✓ | ✓ | 证明无负样本可行 | **Ch4** |
| **SimSiam** | ✗ | ✗ | ✓ | ✓ | 证明 EMA 也非必需 | **Ch4** |
| **DINO** | ✗ | ✓ | ✗ | ✓ | Centering + Sharpening 防塌缩 | **Ch4** |

> **演进逻辑**：从"显式负样本排斥" → "时间不对称（EMA）" → "结构不对称（Predictor + Stop-grad）" → "概率分布动态平衡（Centering + Sharpening）"。每一步都在剥离防塌缩的依赖项。

---

## 承上启下

至此（Ch1–Ch4），笔记的"**表示学习**"部分完整结束：
- Ch1：数学工具（点积、KL、CE）
- Ch2：视觉对比学习（SimCLR/MoCo）
- Ch3：跨模态/文本对比（CLIP/SimCSE/BGE）
- Ch4：无负样本自监督（BYOL/SimSiam/DINO）

从下一章 **Ch5** 起，笔记进入"**生成模型对齐**"部分：
- **Ch5**：SFT + LoRA（让 Base Model 学会"听话"）
- **Ch6**：经典 RLHF（RM + PPO，把 Ch4 的 Stop-grad + EMA 思想用到对齐上）
- **Ch7**：DPO 家族（用闭式解砍掉 RM 和 Critic）
- **Ch8**：推理时代（GRPO + PRM + RLAIF）
