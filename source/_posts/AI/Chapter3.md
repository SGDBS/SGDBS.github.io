---
title: Chapter3 对比学习BYOL， SimSiam
date: 2026-03-28 16:03:34
categories: AI大学习
mathjax: true
tags:
    - AI
    - AI面试知识
---

## BYOL (Bootstrap Your Own Latent)

### 1. 核心理念
BYOL 证明了在**完全没有负样本（Negative Pairs）**的情况下，通过构建**非对称结构（Asymmetric Architecture）**和**预测机制**，也能实现有效的自监督学习，并成功避免模型塌缩（Collapse）。

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
1.  **视图生成**: 对原始图片 $x$ 进行两种随机数据增强，得到视图 $v$ 和 $v'$。（例如一张裁剪，一张变色）
2.  **Online 前向传播**: 视图 $v$ 经过编码器、投影层和**预测层**，输出 $$\hat{z}_\theta = q_\theta(g_\theta(f_\theta(v)))$$
3.  **Target 前向传播**: 视图 $v'$ 经过编码器和投影层，输出目标表示 $$z_\xi = g_\xi(f_\xi(v'))$$
4.  **损失计算**: 
    * 对 $\hat{z}_\theta$ 和 $z_\xi$ 进行 $L_2$ 归一化。
    * 计算均方误差（MSE）：$L = \|\bar{q}_\theta(z_\theta) - \bar{z}_\xi\|_2^2$（本质上是最大化余弦相似度）。
5.  **梯度与参数更新**:
    * **Online**: 计算 $L$ 对 $\theta$ 的梯度并执行更新。
    * **Target**: 不传梯度，执行动量平滑更新：$\xi \leftarrow m\xi + (1-m)\theta$。

### 4. 关键机制深度解析

#### A. 为什么会陷入“平凡解”（塌缩）？
* **数学本质**: 在没有负样本时，模型为了最小化 Loss，最简单的“捷径”是将所有输入映射为同一个常数向量（常数映射）。此时正样本对相似度为 1，Loss 为 0，但模型失去了特征区分能力。
> 假设模型 $f_\theta$ 将所有输入 $x$ 都映射成一个单位向量 $c$（例如 $[1, 0, 0, \dots]$）。
    >* 对于任意一对正样本 $(v, v')$，它们的输出分别是 $z = c$ 和 $z' = c$。
    >* 此时，余弦相似度 $\cos(z, z') = 1$，损失函数 $L = 1 - \cos(z, z') = 0$。

#### B. 为什么 BYOL 能防止塌缩？
* **Predictor 的预测作用**: Predictor 引入了非线性变换，使得 Online 端必须去“预测” Target 的特征，而不仅仅是简单的恒等拷贝。
* **Stop-gradient（停止梯度）**: 关键在于梯度不流向 Target 分支。这使得 Target 在优化过程中是一个“被动观察者”，不会为了减小 Loss 而主动向常数解靠拢。
* **动量滞后性**: Target 网络是 Online 网络的一个“缓慢移动的影子”。这种时间上的滞后和不一致性打破了坍缩所需的同步性。



#### C. 动量更新（EMA）的具体作用
* **提供稳定目标**: 由于 Target 更新极慢，它为 Online 提供了一个连续且平滑的回归目标，起到了正则化作用。
* **信息集成**: Target 实际上是 Online 历史多个版本的集成（Ensemble），包含了更丰富的特征空间信息。

### 5. 面试考点总结
* **对比 MoCo**: MoCo 的动量是为了维持负样本队列的一致性；BYOL 的动量是为了在无负样本时稳定目标、防止塌缩。
* **Predictor 必要性**: 若去掉预测层，双路结构完全对称，模型会立即陷入平凡解。
* **核心结论**: BYOL 证明了“非对称性”是自监督学习中除了“负样本约束”外的另一种有效的防塌缩手段。


## SimSiam (Simple Siamese)

SimSiam 是何恺明团队对 BYOL 的极大简化。它证明了：**甚至不需要动量更新（Momentum），只要有 Stop-gradient 就能训练。**

### 1 SimSiam 的网络结构
SimSiam 采用的是完全共享权重的**孪生网络（Siamese Network）**结构：
* Encoder (编码器 $f$)：通常是 ResNet。两路输入共享同一套参数 $\theta$。
* Projector (投影层 $g$)：一个 MLP，将高维特征映射到中间空间,参数 $\theta$。
* Predictor (预测层 $p$)：仅在其中一路使用。它是一个非线性 MLP，用于将一路的输出匹配到另一路的表示,参数 $\theta$。

### 2 具体算法流程

假设输入一张图像 $x$，流程如下：
1. **数据增强**：对 $x$ 进行两次随机增强，得到两个视图 $x_1$ 和 $x_2$。
2. **提取表示**：
    * 两张图都经过相同的 Encoder $f$ 和 Projector $g$。
    * 得到两个向量：$z_1 = g(f(x_1))$ 和 $z_2 = g(f(x_2))$。
3. **预测映射**：
    * 将 $z_1$ 喂入 Predictor $p$，得到预测向量 $p_1 = p(z_1)$。
    * 同理，将 $z_2$ 喂入 Predictor $p$，得到预测向量 $p_2 = p(z_2)$。
4. **计算对称损失** (Symmetrized Loss)：
    SimSiam 计算两次损失并取平均。
    * 第一部分：$D(p_1, \text{stop\_grad}(z_2))$。这里 $p_1$ 去追 $z_2$，但 $z_2$ 不准产生梯度。
    * 第二部分：$D(p_2, \text{stop\_grad}(z_1))$。这里 $p_2$ 去追 $z_1$，但 $z_1$ 不准产生梯度。
    * 距离函数 $D$：通常使用负余弦相似度：$D(p, z) = -\frac{p \cdot z}{\|p\|_2 \|z\|_2}$。
5. **更新参数**：根据总损失 $L = \frac{1}{2}(L_1 + L_2)$，通过反向传播更新所有共享参数 $\theta$：
$$\nabla \theta = \frac{1}{2} \frac{\partial D(p_1, z_2)}{\partial \theta} + \frac{1}{2} \frac{\partial D(p_2, z_1)}{\partial \theta}$$

```py
# 1. 得到表示 (z) 和 预测 (p)
z1, z2 = model.backbone(x1), model.backbone(x2) # backbone 包含 encoder 和 projector
p1, p2 = model.predictor(z1), model.predictor(z2)

# 2. 计算损失：注意 detach() 就是 stop-gradient 的实现
# D 是余弦相似度
loss = (D(p1, z2.detach()) + D(p2, z1.detach())) * 0.5

# 3. 统一更新参数
loss.backward() # 梯度只从 p1 传回，不从 z2 传回
optimizer.step()
```

### 3 为什么它有效？
何恺明在论文中做了一个实验：一旦去掉 Stop-gradient，模型会瞬间塌缩。
数学上，SimSiam 的优化可以看作是**交替优化（Alternating Optimization）**。
如果不加 Stop-gradient，模型是在同时优化 $\theta$ 和表示 $z$，这会产生退化解。加上之后，它变成了一个类似 EM 算法 的过程：

1. 定义目标函数我们假设图像 
    $x$ 有一个“理想的特征表示” $\eta_x$。

    我们的目标是训练一个网络 $\mathcal{F}_\theta$（包含 Encoder, Projector 和 Predictor），使得它能预测出这个 $\eta_x$。
    
    定义损失函数 $E(\theta, \eta)$：

    $$E(\theta, \eta) = \mathbb{E}_{x, T} \left[ \| \mathcal{F}_{\theta}(T(x)) - \eta_x \|^2 \right]$$
    
    * 这里有两个待优化的变量：$\theta$：网络的参数（我们想要学习的模型）。
    * $\eta$：图像的表示（隐变量，我们并不知道每张图最完美的向量是什么）。

2. 为什么直接优化会“塌缩”？
    如果像普通的端到端训练那样同时对 $\theta$ 和 $\eta$ 求导：
    $$\min_{\theta, \eta} E(\theta, \eta)$$
    模型会发现一个超级捷径：让所有的 $\eta_x$ 都等于一个常数 $C$，同时让网络 $\mathcal{F}_\theta$ 对任何输入也输出 $C$。
    此时误差为 0，但模型什么也没学到。这就是**塌缩（Collapse）**。

3. 引入 EM 思想：交替优化
    
    为了防止上述情况，SimSiam（以及 BYOL）实际上将优化拆成了两步，这与 EM 算法 的逻辑完美对应：
    
    **第一步：E-step（固定 $\theta$，优化 $\eta$）**
    $\eta_x$（样本分配）：就像 K-Means 里的分配标签。我们更新标签，把样本分配给最近的中心。
    在这一步，我们不更新网络，而是问：“**基于现在的模型，哪种特征表示 $\eta$ 最能代表这张图？**”
    $$\eta^t \leftarrow \arg\min_{\eta} L(\theta^t, \eta)$$
    为了让 MSE 最小，$\eta_x$ 的最优解应该是该图片在所有可能的数据增强 $T$ 下的平均表示：
    $$\eta_x^t \leftarrow \mathbb{E}_T [\mathcal{F}_{\theta^t}(T(x))]$$
    实际上$T$的平均表示很难获取，于是我们用
    $$\eta_x \leftarrow \mathcal{F}_{\theta^t}(T'(x))$$
    代替。
    
    **第二步：M-step（固定 $\eta$，优化 $\theta$）**
    $\theta$（聚类中心）：就像 K-Means 里的中心点。我们更新中心点，让它靠近分配给它的样本。
    $$\theta^t \leftarrow \arg\min_{\theta} L(\theta, \eta^{t-1})$$

4. Predictor
    * 理想状态：$\eta_x$ 应该是所有增强视图的期望 $\mathbb{E}_T [\mathcal{F}(T(x))]$。
    * 由于计算期望太慢，我们每次只随机采样一个视图 $\mathcal{F}_{\theta^t}(T'(x))$ 
    来代替期望$\mathbb{E}_T [\mathcal{F}_{\theta^t}(T(x))]$:
        * 因为我们用“单次采样”代替了“期望”，这引入了巨大的噪声。
        * 它的目标就是最小化 $\|h(z_1) - z_2\|^2$。
        * $h(z_1)$ 的最优解在数学上恰好就是 $z_2$ 的期望 $\mathbb{E}[z_2]$
        * 所以，Predictor 实际上是在“**预判期望**”。它弥补了由于单次采样带来的信息缺失，让模型能够朝着真正的 $\eta$（期望表示）演进。