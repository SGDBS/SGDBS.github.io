---
title: Chapter3 对比学习BYOL， SimSiam
categories: 学习笔记-大模型
date: 2026-03-28 16:03:34
mathjax: true
tags:
    - AI
    - AI面试知识
---

> Chapter2 已概述了 BYOL/SimSiam 的非对称设计，本章深入其数学机理，并最终落到 **LLM 训练流程中 Stop-gradient + EMA 的真实对应**（RLHF / DPO 的 reference policy）。

---

## BYOL (Bootstrap Your Own Latent)

### 1. 核心理念
BYOL 证明了在**完全没有负样本（Negative Pairs）**的情况下，通过构建**非对称结构（Asymmetric Architecture）**和**预测机制**，也能实现有效的自监督学习，并成功避免模型塌缩（Collapse）。

> **关键实验数字**：BYOL 用 ResNet-50 在 ImageNet 上 linear probe 准确率 **74.3%**，首次实现"无负样本方法超越 SimCLR"。

### 2. 网络架构
BYOL 由两个互动的网络分支组成：
* **Online Network（在线网络）**: 参数为 $\theta$。
    * 组成：
        * Encoder $f_\theta$：通常是 ResNet-50。它将增强后的图像 $v$ 映射为高维特征向量 $h_\theta$。
        * Projector $g_\theta$：一个多层感知机（MLP）。它将 $h_\theta$ 投影到一个更紧凑的空间 $z_\theta$。
        * Predictor $q_\theta$（核心层）：又一个 MLP。它试图将 $z_\theta$ 映射到 Target 网络的表示空间，输出 $\hat{z}_\theta$。
    * 更新方式：通过梯度下降（SGD/Adam）实时更新。
* **Target Network（目标网络）**: 参数为 $\xi$。
    * 组成：
        * Encoder $f_\xi$：结构与 $f_\theta$ 完全一致。
        * Projector $g_\xi$：结构与 $g_\theta$ 完全一致。
    * 更新方式：**动量更新（EMA）**，不计算梯度。其参数是 Online 参数的历史加权平均。

### 3. 具体算法流程 (Step-by-Step)
1. **视图生成**：对原始图片 $x$ 进行两种随机数据增强，得到视图 $v$ 和 $v'$。
2. **Online 前向传播**：视图 $v$ 经过编码器、投影层和**预测层**，输出
$$\hat{z}_\theta = q_\theta(g_\theta(f_\theta(v)))$$
3. **Target 前向传播**：视图 $v'$ 经过编码器和投影层，输出目标表示
$$z_\xi = g_\xi(f_\xi(v'))$$
4. **损失计算**：对 $\hat{z}_\theta$ 和 $z_\xi$ 进行 $L_2$ 归一化后计算均方误差：
$$L = \|\bar{\hat{z}}_\theta - \bar{z}_\xi\|_2^2$$
其中 $\bar{\cdot}$ 表示 L2 归一化向量。

   **MSE 与余弦相似度的等价性**：当 $\|\bar{a}\| = \|\bar{b}\| = 1$ 时，
   $$\|\bar{a} - \bar{b}\|_2^2 = \|\bar{a}\|^2 + \|\bar{b}\|^2 - 2\bar{a}^T\bar{b} = 2 - 2\cos\theta$$
   所以**最小化归一化 MSE = 最大化余弦相似度**（与 Chapter1 §2 的转换公式完全对应）。

5. **梯度与参数更新**：
    * **Online**：计算 $L$ 对 $\theta$ 的梯度并执行更新。
    * **Target**：不传梯度，执行动量平滑更新：$\xi \leftarrow m\xi + (1-m)\theta$，$m$ 通常取 0.99~0.999。

### 4. 关键机制深度解析

#### A. 为什么会陷入"平凡解"（塌缩）？
**数学本质**：在没有负样本时，模型为了最小化 Loss，最简单的"捷径"是将所有输入映射为同一个常数向量。此时正样本对相似度为 1，Loss 为 0，但模型失去了特征区分能力。

假设模型将所有输入 $x$ 都映射成单位向量 $c$（例如 $[1, 0, 0, \dots]$）：
- 对于任意正样本 $(v, v')$，输出分别是 $z = c$ 和 $z' = c$。
- 余弦相似度 $\cos(z, z') = 1$，损失 $L = 0$。

#### B. 为什么 BYOL 能防止塌缩？
* **Predictor 的预测作用**：Predictor 引入了非线性变换，使得 Online 端必须去"预测" Target 的特征，而不仅仅是简单的恒等拷贝。
* **Stop-gradient（停止梯度）**：关键在于梯度不流向 Target 分支。这使得 Target 在优化过程中是一个"被动观察者"，不会为了减小 Loss 而主动向常数解靠拢。
* **动量滞后性**：Target 网络是 Online 网络的一个"缓慢移动的影子"。这种时间上的滞后和不一致性打破了坍缩所需的同步性。

#### C. 动量更新（EMA）的具体作用
* **提供稳定目标**：由于 Target 更新极慢，它为 Online 提供了一个连续且平滑的回归目标，起到了正则化作用。
* **信息集成**：Target 实际上是 Online 历史多个版本的集成（Ensemble），包含了更丰富的特征空间信息。

### 5. 面试考点总结
* **对比 MoCo**：MoCo 的动量是为了维持负样本队列的一致性；BYOL 的动量是为了在无负样本时稳定目标、防止塌缩。
* **Predictor 必要性**：若去掉预测层，双路结构完全对称，模型会立即陷入平凡解。
* **核心结论**：BYOL 证明了"非对称性"是自监督学习中除了"负样本约束"外的另一种有效的防塌缩手段。

---

## SimSiam (Simple Siamese)

SimSiam 是何恺明团队对 BYOL 的极大简化。它证明了：**甚至不需要动量更新（Momentum），只要有 Stop-gradient 就能训练。**

> **关键实验数字**：SimSiam 在 ImageNet 上 linear probe **71.3%**，仅略低于 BYOL，但**训练成本远低**（无 EMA 双网络、无需大 batch）。

### 1. 网络结构
SimSiam 采用完全共享权重的**孪生网络（Siamese Network）**：
* Encoder $f$：通常是 ResNet。两路输入共享同一套参数 $\theta$。
* Projector $g$：一个 MLP，将高维特征映射到中间空间。
* Predictor $p$：仅在其中一路使用的非线性 MLP，将一路的输出匹配到另一路的表示。

### 2. 具体算法流程

假设输入一张图像 $x$：
1. **数据增强**：对 $x$ 进行两次随机增强，得到视图 $x_1$ 和 $x_2$。
2. **提取表示**：两张图都经过相同的 Encoder $f$ 和 Projector $g$，得到 $z_1 = g(f(x_1))$ 和 $z_2 = g(f(x_2))$。
3. **预测映射**：$p_1 = p(z_1)$，$p_2 = p(z_2)$。
4. **计算对称损失** (Symmetrized Loss)：
    * $D(p_1, \text{stop\_grad}(z_2))$：$p_1$ 去追 $z_2$，但 $z_2$ 不准产生梯度。
    * $D(p_2, \text{stop\_grad}(z_1))$：$p_2$ 去追 $z_1$，但 $z_1$ 不准产生梯度。
    * 距离函数 $D$：负余弦相似度：$D(p, z) = -\frac{p \cdot z}{\|p\|_2 \|z\|_2}$。
5. **更新参数**：
$$L = \frac{1}{2}(L_1 + L_2), \quad \nabla \theta = \frac{1}{2} \frac{\partial D(p_1, z_2)}{\partial \theta} + \frac{1}{2} \frac{\partial D(p_2, z_1)}{\partial \theta}$$

```python
import torch.nn.functional as F

def D(p, z):
    return -F.cosine_similarity(p, z, dim=-1).mean()

# 1. 得到表示 (z) 和预测 (p)
z1, z2 = model.backbone(x1), model.backbone(x2)  # backbone = encoder + projector
p1, p2 = model.predictor(z1), model.predictor(z2)

# 2. 计算损失：detach() 即 stop-gradient
loss = (D(p1, z2.detach()) + D(p2, z1.detach())) * 0.5

# 3. 更新参数：梯度只从 p 路径回传，不从 z.detach() 回传
loss.backward()
optimizer.step()
```

### 3. 为什么有效？—— EM 视角下的紧凑推导

何恺明在论文中实验证明：去掉 Stop-gradient，模型瞬间塌缩。SimSiam 的优化本质是 **EM 算法**。

**目标函数**：假设每张图 $x$ 有"理想表示" $\eta_x$（隐变量），损失为
$$E(\theta, \eta) = \mathbb{E}_{x, T} \left[ \| \mathcal{F}_{\theta}(T(x)) - \eta_x \|^2 \right]$$

如果同时优化 $\theta$ 和 $\eta$，最优捷径是 $\eta_x \equiv C$ 且 $\mathcal{F}_\theta \equiv C$，即塌缩。SimSiam 通过交替优化避免这一陷阱：

| 步骤 | 操作 | 实现方式 |
| :--- | :--- | :--- |
| **E-step**（固定 $\theta$，更新 $\eta$） | $\eta_x^* = \mathbb{E}_T[\mathcal{F}_\theta(T(x))]$ | 期望不可算 → 用单样本 $\mathcal{F}_\theta(T'(x))$ 近似（即另一路 + stop-gradient） |
| **M-step**（固定 $\eta$，更新 $\theta$） | $\theta \leftarrow \arg\min_\theta L(\theta, \eta)$ | 标准梯度下降 |

**Predictor 的真正作用 = 学习条件期望**

由于 E-step 用单样本 $\mathcal{F}_\theta(T'(x))$ 替代了真期望，引入巨大噪声。Predictor $h$ 的优化目标可写成
$$h^*(z_1) = \mathbb{E}_{T_2}[z_2 \mid z_1]$$
即 Predictor 学到的是 $z_1$ 条件下 $z_2$ 的**条件期望**——它在"为单样本噪声去噪"，让模型朝真正的 $\eta$ 演进。这就是 Predictor 不可省略的根本原因。

---

## 横向对比：四大对比学习方法

| 方法 | 负样本 | EMA Target | Predictor | Stop-grad | 关键贡献 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SimCLR** | ✓ | ✗ | ✗ | ✗ | 大 batch + 投影头 |
| **MoCo** | ✓ (队列) | ✓ | ✗ | ✓ | 队列解耦显存与负样本数 |
| **BYOL** | ✗ | ✓ | ✓ | ✓ | 证明无负样本可行 |
| **SimSiam** | ✗ | ✗ | ✓ | ✓ | 证明 EMA 也非必需 |

> **演进逻辑**：从"显式排斥"（负样本）→ "时间不对称"（EMA）→ "结构不对称"（Predictor + Stop-grad）。每一步都在**剥离防塌缩的依赖项**，最终发现 Stop-gradient + Predictor 这两个最小条件就够了。

---

## 延伸：DINO / DINOv2 —— BYOL 的精神继承者

* **DINO**（Caron et al., 2021）：自蒸馏 + 中心化 + 锐化（centering & sharpening），架构上与 BYOL 极相似（Student-Teacher EMA + Stop-gradient）。
* **DINOv2**（Meta, 2023）：DINO 在 ViT-g 上的工程加强版，已成为 **CLIP 之外另一个主流视觉编码器选择**。
* **在多模态 LLM 中的角色**：部分 VLM（如 Llama 3.2 Vision、某些 Qwen-VL 变体）采用 DINOv2 作为视觉骨干，因其在密集预测任务（分割、深度）上优于 CLIP。

---

## 落地 LLM：Stop-gradient + EMA 在大模型训练中的真实身影

这是本章最该记住的部分。BYOL/SimSiam 看起来是 CV 技术，但其**核心机制在 LLM 训练流程中无处不在**：

| BYOL/SimSiam 概念 | LLM 训练中的对应 |
| :--- | :--- |
| **Target Network** | **Reference Policy $\pi_{\text{ref}}$**（DPO/PPO/GRPO 中的参考模型） |
| **Stop-gradient on Target** | $\pi_{\text{ref}}$ 不参与反向传播，只用于计算 KL/对数比 |
| **EMA 更新 Target** | **Online DPO / SPIN / Self-Rewarding LM** 中用 EMA 缓慢更新 reference policy |
| **Predictor 打破对称** | 知识蒸馏中 student 用额外结构匹配 teacher 软标签 |
| **避免塌缩** | RLHF KL 惩罚防止策略塌缩到 reward-hacking 单点解 |

### 案例：DPO 损失中的 Stop-gradient

DPO 损失函数：
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

* $\pi_\theta$（policy）= **Online network**：被梯度更新
* $\pi_{\text{ref}}$（reference）= **Target network**：通常是 SFT 模型的冻结快照，**梯度被 stop**
* $\beta$ 项 = **KL 隐式约束**：防止 $\pi_\theta$ 跑得太远（类似 BYOL 中"不让 Target 变化太快"）

### 案例：Self-Rewarding / Online DPO 的 EMA 思想

最新的迭代式对齐方法（如 SPIN、Self-Rewarding LM）会**周期性地**用当前 policy 替换 reference，本质上就是离散版的 EMA：
$$\pi_{\text{ref}}^{(t+1)} = \pi_\theta^{(t)} \quad \text{(每 N 步替换一次)}$$

> **统一视角**：无论是视觉自监督还是 LLM 对齐，"**用一个慢速演化的 Target 网络作为锚点，让 Online 网络去追赶**" 都是防止训练崩溃的通用工程范式。理解了 BYOL，就理解了 RLHF 中 reference policy 为什么必须存在、为什么必须冻结。
