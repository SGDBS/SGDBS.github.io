---
title: RL Chapter10 探索：从 ε-greedy 到 Curiosity
categories: 学习笔记-强化学习
date: 2026-05-10 10:00:00
mathjax: true
tags:
    - AI
    - 强化学习
    - AI面试知识
---

> **本章定位**：所有 RL 算法都隐式依赖"探索"才能成立——没有探索，agent 永远学不到更好的策略。本章系统讲解从经典多臂老虎机到现代 deep RL 的探索方法。

> **承上**：Ch4 ε-greedy + Ch9 高斯噪声 + Ch5/8 熵正则。
> **启下**：Ch11 offline RL（不需要探索的极端案例）+ Ch13 LLM 中的探索（temperature/top-p）。

---

# §A 数学原理

## 1. 探索-利用窘境 (Exploration-Exploitation Dilemma)

### 1.1 多臂老虎机 (Multi-Armed Bandit)

最简单的 RL 设置：$K$ 个老虎机臂，每个臂有未知奖励分布 $\mathcal{N}(\mu_k, \sigma_k^2)$。每次拉一个臂，获得 reward。

**目标**：$T$ 步内最大化总 reward。

**窘境**：
- **利用**（Exploitation）：选当前 $\hat{\mu}_k$ 最大的臂
- **探索**（Exploration）：选还不确定的臂，试试它真实的 $\mu$

> 这是 RL 最根本的张力。每个算法都在"如何平衡 E/E"上做不同选择。

### 1.2 Regret（懊悔）：评估指标

$$R(T) = T \mu^* - \mathbb{E}\left[\sum_{t=1}^T r_t\right]$$

即"如果一直选最优臂能得多少 reward" 减去 "实际得了多少"。好算法应有 **次线性 regret**：$R(T) = o(T)$。

## 2. 多臂老虎机的经典策略

### 2.1 ε-greedy

$$\pi(a) = \begin{cases} \arg\max_k \hat{\mu}_k & \text{w.p. } 1 - \epsilon \\ \text{uniform} & \text{w.p. } \epsilon \end{cases}$$

**Regret**：$O(T)$（线性！— 因为 ε 永远在浪费）。除非 ε 衰减。

### 2.2 UCB (Upper Confidence Bound)

**核心思想**：**乐观面对不确定性 (Optimism in the Face of Uncertainty)**。给每个臂一个"上置信界"：

$$\text{UCB}_k(t) = \hat{\mu}_k + c \sqrt{\frac{\log t}{N_k(t)}}$$

其中 $N_k(t)$ 是臂 $k$ 被拉的次数。**总是选 UCB 最大的臂**。

**直觉**：
- $\hat{\mu}_k$ 大 → 利用
- $N_k$ 小（不确定性大）→ 探索
- $c$ 是探索系数（通常 $c = \sqrt{2}$）

**Regret**：$O(\log T)$ —— **次线性**，理论最优级别。

### 2.3 Thompson Sampling

贝叶斯版：维护每个臂奖励的后验分布，每次**从后验采样**一个 $\mu_k$，选最大的拉。

```python
# Beta 后验（Bernoulli reward）
for arm in arms:
    sample[arm] = Beta(alpha[arm], beta[arm]).sample()
chosen = argmax(sample)
# 观察 reward r ∈ {0, 1}
alpha[chosen] += r
beta[chosen] += 1 - r
```

**优势**：实证表现往往优于 UCB，尤其在大规模推荐场景（Yahoo!、Microsoft 用过）。

## 3. RL 中的探索（远比老虎机难）

老虎机是无状态的。RL 中状态是序列演化的，"探索"变得复杂得多：
- **稀疏奖励**：Montezuma's Revenge 中走几百步才能拿到 reward
- **deep exploration**：需要"提前规划"才能到达远处状态
- **大状态空间**：UCB 风格"看每个状态的访问次数"不可行（状态太多）

下面介绍现代 deep RL 中的探索方法。

## 4. Count-Based 探索

### 4.1 思想

把 UCB 的"访问次数"扩展到状态：在 reward 上加一个**探索奖励** (intrinsic reward)：
$$r_t^+ = r_t + \beta \cdot \frac{1}{\sqrt{N(s_t)}}$$

新颖状态访问次数少，奖励高 → 鼓励 agent 去新地方。

### 4.2 大状态空间下的"近似计数"

直接计数不可行（状态太多）。两个工程做法：

**Pseudo-counts** (Bellemare 2016)：用密度模型 $\rho(s)$ 估计访问频率：
$$\hat{N}(s) = \frac{\rho(s)(1 - \rho'(s))}{\rho'(s) - \rho(s)}$$

**Hash-based**：把状态哈希到离散桶，按桶计数。

## 5. Curiosity-Driven 探索

### 5.1 ICM (Intrinsic Curiosity Module, 2017)

**思想**：让 agent 对"模型预测错误"的状态感兴趣。

构造一个**前向模型** $\hat{f}(s_t, a_t) \to \hat{s}_{t+1}$。如果 $\hat{s}_{t+1}$ 与真实 $s_{t+1}$ 差很多 → 这个状态"难以预测"→ 给 intrinsic reward。

$$r^+_t = r_t + \beta \|\hat{f}(s_t, a_t) - s_{t+1}\|^2$$

但直接在像素空间预测有问题——agent 可能"造电视雪花"骗模型说"我不知道"。ICM 用一个**编码器**把状态压到特征空间预测，避免低层噪声干扰。

### 5.2 RND (Random Network Distillation, 2018)

**更巧妙的方案**：训练一个网络去预测**另一个固定的随机网络**的输出。

- $f^*$：随机初始化、永远不动
- $\hat{f}_\theta$：训练去预测 $f^*$
- 预测误差 $\|\hat{f}_\theta(s) - f^*(s)\|^2$ 就是新颖度

**关键洞察**：
- 已访问过的状态：$\hat{f}_\theta$ 训得好 → 预测误差低
- 新状态：$\hat{f}_\theta$ 没见过 → 预测误差高

RND 工程上极简单，效果惊人——OpenAI 用 RND 第一次在 Montezuma's Revenge 上达到人类水平。

## 6. 噪声扰动探索

### 6.1 Action-space noise

DDPG 的高斯噪声、ε-greedy 都是这类。简单但效果有限。

### 6.2 Parameter-space noise (NoisyNet)

直接给**参数**加噪声：
$$\theta = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

$\mu, \sigma$ 都是可训练参数。$\sigma$ 大 → 探索强。

**优势**：策略的随机性是"结构性"的（连续多步保持一致），比随机抖动更有效率。

### 6.3 Bootstrapped DQN

维护 $K$ 个独立 Q 网络头，每个 episode 随机选一个用。$K$ 个网络初始化不同 → 不同的"假说"，agent 在不同假说下行动 → **结构化探索**。

---

# §B 模型架构

## B.1 RND 的实现（最简洁的现代探索方法）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNDModule(nn.Module):
    def __init__(self, obs_dim, feat_dim=128):
        super().__init__()
        # ⭐ 固定的随机网络（不训练）
        self.target_net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, feat_dim),
        )
        for p in self.target_net.parameters():
            p.requires_grad = False                       # ⭐ 永远冻结

        # 可训练的预测网络
        self.predictor_net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, feat_dim),
        )

    def forward(self, obs):
        with torch.no_grad():
            target_feat = self.target_net(obs)            # 不参与训练
        pred_feat = self.predictor_net(obs)
        return pred_feat, target_feat

    def intrinsic_reward(self, obs):
        """计算 RND 探索奖励"""
        with torch.no_grad():
            pred, target = self.forward(obs)
            return ((pred - target) ** 2).sum(dim=-1)     # [B]

    def update(self, obs, optimizer):
        """训练 predictor"""
        pred, target = self.forward(obs)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
```

## B.2 RND + PPO 的组合

```python
# 在 PPO（Ch8）的 reward 计算中加入 RND
def compute_combined_reward(rewards, obs, rnd, beta=0.5):
    intrinsic = rnd.intrinsic_reward(obs).cpu().numpy()
    return rewards + beta * intrinsic

# 训练循环（修改 Ch8 的 PPO）
for step in range(...):
    # ... 原 PPO 流程
    extrinsic = env.step(a)
    intrinsic = rnd.intrinsic_reward(obs)
    total_reward = extrinsic + 0.5 * intrinsic            # ⭐ 合并

    # 同时更新 RND predictor
    rnd.update(obs, rnd_optim)
```

## B.3 NoisyNet 实现

```python
class NoisyLinear(nn.Module):
    """参数空间噪声层"""
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # ⭐ μ + σ⊙ε，μ 和 σ 都可训练
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(torch.full((out_features,), sigma_init))

        nn.init.uniform_(self.weight_mu, -1/in_features**0.5, 1/in_features**0.5)
        nn.init.uniform_(self.bias_mu, -1/in_features**0.5, 1/in_features**0.5)

    def forward(self, x):
        if self.training:
            # ⭐ 训练时加噪声
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            weight = self.weight_mu + self.weight_sigma * weight_eps
            bias = self.bias_mu + self.bias_sigma * bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
```

把 DQN 中的 `nn.Linear` 替换为 `NoisyLinear` 即可——**不再需要 ε-greedy**。

---

# §C 训练与推理

## C.1 实验：Montezuma's Revenge

经典稀疏奖励难题。Vanilla DQN: reward ≈ 0；DQN + RND: reward > 8000。

```python
import gymnasium as gym
env = gym.make("MontezumaRevengeNoFrameskip-v4")
# ... PPO + RND 训练
```

实测 RND 的"探索曲线"（房间数量随时间变化）：
- 不加 RND：永远停在第 1 间房
- 加 RND：1 千万步后探索到 14+ 间房

## C.2 各种探索方法的性能对比

| 方法 | 复杂度 | 稀疏 reward | Atari 稠密 reward |
| :--- | :---: | :---: | :---: |
| ε-greedy | ★ | ✗ | ★★ |
| 高斯噪声 (DDPG) | ★ | ✗ | ★★ |
| 熵正则 (PPO/SAC) | ★★ | ★ | ★★★ |
| Count-based | ★★ | ★★ | ★★ |
| ICM | ★★★ | ★★★ | ★★★ |
| **RND** | ★★ | ★★★★ | ★★★ |
| NoisyNet | ★★★ | ★★ | ★★★ |
| Bootstrapped DQN | ★★★ | ★★★ | ★★★ |

> **2026 实践**：RND 是稀疏 reward 任务的默认探索方法（实现简单 + 效果好）。

## C.3 推理视角：探索是否还需要？

| 阶段 | 是否需要探索 |
| :--- | :--- |
| **训练** | **必须**（否则学不到更好策略） |
| **推理** | **不需要**（已学好，直接利用） |

部署时把所有探索机制关掉：
- DQN: ε = 0
- SAC: 用均值（不采样）
- NoisyNet: `model.eval()` 自动关闭噪声

## C.4 LLM 中的探索

LLM 推理时也要"探索"——但完全是另一种语境：

| LLM 中的探索 | RL 类比 |
| :--- | :--- |
| **temperature** | DDPG 的高斯噪声幅度 |
| **top-p / top-k** | 截断分布的探索 |
| **best-of-N** | UCB 的采样多次取最优 |
| **MCTS（如 o1/R1）** | RL 中的 model-based 搜索 |

详见 Ch13 RL × LLM。

---

# §D 章末速查

## D.1 主要方法速记

| 类别 | 方法 | 核心 |
| :--- | :--- | :--- |
| **多臂老虎机** | UCB / Thompson Sampling | 次线性 regret |
| **结构噪声** | NoisyNet / Bootstrapped DQN | 给参数 / 多个头加噪声 |
| **新颖度** | Count-based / RND / ICM | "去没去过的地方" |
| **熵正则** | SAC / PPO entropy bonus | 鼓励策略保持随机性 |
| **思路全局** | MCTS / planning | 提前规划 |

## D.2 常见面试题

**Q1：UCB 为什么是次线性 regret？**
- 关键：探索项 $\sqrt{\log t / N_k}$ 让"被冷落的臂"逐渐被试到
- 一旦试出真值低，就不再选它（探索奖励降不下来 → 让位）
- 数学上严格证明 regret = $O(\log T)$（Auer 2002）

**Q2：RND 为什么能避免 ICM 的"造电视雪花"问题？**
- ICM 直接预测下一帧 → agent 可以创造"无法预测的噪声"骗模型
- RND 预测的是**固定随机网络的输出**，agent 改变环境也无法改变"预测目标"
- 因此预测误差只反映"该状态被访问的次数"，不会被骗

**Q3：熵正则与 RND 探索的差异？**
- 熵正则：让策略保持**局部**随机性（每步抖一抖）
- RND：让策略追求**全局**新颖性（去没去过的地方）
- 互补使用：SAC + RND 在多个 benchmark 上是 SOTA

**Q4：探索-利用 trade-off 在 LLM 时代有什么新形式？**
- 推理时：temperature、top-p
- 训练时：RLHF 的 KL penalty 限制策略 vs 探索 reward
- Best-of-N + reranker：批量探索 + 选最优
- MCTS（o1/R1）：树搜索式的结构化探索

**Q5：NoisyNet 与 ε-greedy 的关键差异？**
- ε-greedy：每步独立随机 → "白噪声"
- NoisyNet：参数级噪声 → 多步策略保持一致 → "结构化"
- NoisyNet 在 Atari 上 + 50% 提升

---

## 承上启下

探索是 RL 的"水下冰山"——所有算法都隐式依赖它。下一章 **Ch11 离线 RL** 走完全相反的路：**完全不探索**，只用预先收集的固定数据集训练。

这看似奇怪，但其实是 LLM 时代最重要的范式：
- **DPO** 本质就是离线 RL
- **RLAIF** 用 AI 生成的固定偏好数据训练
- **数据效率第一** 才是工业落地的关键
