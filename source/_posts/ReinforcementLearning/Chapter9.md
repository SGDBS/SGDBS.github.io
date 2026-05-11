---
title: RL Chapter9 DDPG 与 SAC：连续控制的两个里程碑
categories: 学习笔记-强化学习
date: 2026-05-10 10:00:00
mathjax: true
tags:
    - AI
    - 强化学习
    - AI面试知识
---

> **本章定位**：DDPG 把 DQN 思想扩展到连续动作；SAC 在 DDPG 基础上加上"最大熵 RL"框架，成为连续控制的当前 SOTA（机器人、自动驾驶首选）。

> **承上**：Ch7 DQN（Replay Buffer + Target Network）+ Ch8 PPO（连续动作但 on-policy）。
> **启下**：Ch10 探索机制 + Ch11 离线 RL。

---

# §A 数学原理

## 1. 连续动作空间的挑战

回忆：
- DQN 用 $\arg\max_a Q(s, a)$ → 连续 $a$ 时是优化问题，不可行
- PG/PPO 用 $\pi_\theta(a \mid s)$ → 可以处理连续，但 on-policy 数据效率低

**目标**：能否用 off-policy + Replay Buffer（DQN 的优势）+ 连续动作（PG 的优势）？

答案：**确定性策略梯度（DPG）**+**Actor-Critic** = DDPG。

## 2. 确定性策略梯度 (DPG)

### 2.1 确定性策略 vs 随机策略

- 随机策略：$a \sim \pi_\theta(\cdot \mid s)$
- 确定性策略：$a = \mu_\theta(s)$（直接输出唯一的最优动作）

### 2.2 DPG 定理（Silver 2014）

对确定性策略 $\mu_\theta$，policy gradient 简化为：
$$\nabla_\theta J(\mu_\theta) = \mathbb{E}_{s \sim d^\mu}\left[\nabla_\theta \mu_\theta(s) \cdot \nabla_a Q^\mu(s, a)\big|_{a=\mu_\theta(s)}\right]$$

**关键差别**：随机 PG 需要对动作求**期望**（用 score function trick 化简），DPG 直接用 **链式法则**——因为策略是确定性的，没有"动作分布"概念。

### 2.3 直觉

DPG 的梯度可以解读为：
$$\nabla_\theta J = \nabla_\theta \mu_\theta(s) \cdot \nabla_a Q(s, a)$$

即 **"调整 $\mu_\theta$ 让输出的 action 朝 $\nabla_a Q$ 的方向移动"**。$Q$ 在 $a$ 上更高的方向，就是 actor 该走的方向。

## 3. DDPG (Deep Deterministic Policy Gradient)

### 3.1 算法结构

DDPG = DPG + DQN 工程：
- **Actor** $\mu_\theta(s)$：神经网络，输出确定性动作
- **Critic** $Q_\phi(s, a)$：神经网络，估计 Q
- **Replay Buffer**：off-policy
- **Target Networks** $\mu_{\theta^-}, Q_{\phi^-}$：软更新

### 3.2 损失函数

**Critic（Q 网络）**：标准 TD 损失
$$\mathcal{L}_\phi = \mathbb{E}_{(s, a, r, s')}\left[\left(Q_\phi(s, a) - y\right)^2\right]$$
$$y = r + \gamma Q_{\phi^-}(s', \mu_{\theta^-}(s'))$$

**Actor（策略网络）**：最大化 Q
$$\mathcal{L}_\theta = -\mathbb{E}_{s}\left[Q_\phi(s, \mu_\theta(s))\right]$$

注意：$\mathcal{L}_\theta$ 中梯度会通过 $Q_\phi$ 反传到 $\mu_\theta$（链式法则），实现 DPG。

### 3.3 探索：动作空间噪声

确定性策略没有内在随机性，必须**人工加噪声**才能探索：
$$a = \mu_\theta(s) + \mathcal{N}(0, \sigma^2)$$

或更精细的 OU 噪声（Ornstein-Uhlenbeck，时间相关）。$\sigma$ 通常 0.1-0.3。

### 3.4 DDPG 的痛点

- 对超参敏感
- Q 函数过估计严重（DQN 的 max bias 在 DDPG 中变成"actor 滥用 Q 的不准确性"）
- 训练不稳定，常崩

→ TD3 和 SAC 是其后继改进。

## 4. TD3 (Twin Delayed DDPG)：DDPG 的修复版

TD3 (Fujimoto 2018) 提出三个改进：

### 4.1 双 Q 网络（Twin）

维护两个 Q 网络 $Q_{\phi_1}, Q_{\phi_2}$，target 取较小的：
$$y = r + \gamma \min_{i=1,2} Q_{\phi_i^-}(s', \mu_{\theta^-}(s'))$$

**为什么取 min？** 抑制过估计（max 偏差的"反向"）。

### 4.2 延迟更新（Delayed）

每更新 Critic 两次才更新 Actor 一次。让 Critic 先稳下来。

### 4.3 目标策略平滑（Target Policy Smoothing）

对 target 中的 action 加噪声：
$$y = r + \gamma \min_{i} Q_{\phi_i^-}(s', \mu_{\theta^-}(s') + \text{clip}(\mathcal{N}(0, \sigma), -c, c))$$

**直觉**：让 Q 函数对 action 的细微变化更鲁棒。

## 5. SAC (Soft Actor-Critic)：最大熵 RL

SAC（Haarnoja 2018）是现代连续控制的金标准。核心创新：**目标函数中加入熵奖励**。

### 5.1 最大熵 RL 目标

$$J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t) \sim \pi}\left[r_t + \alpha H(\pi(\cdot \mid s_t))\right]$$

其中 $H(\pi(\cdot \mid s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s)]$ 是熵，$\alpha$ 是温度系数。

**直觉**：奖励 agent **既要拿高 reward，又要保持随机性**。这给两个好处：
1. **更好的探索**：随机性让 agent 不陷入局部最优
2. **更鲁棒的策略**：学到的策略能应对状态扰动

### 5.2 Soft Bellman 方程

最大熵框架下的 Bellman 方程：
$$Q^\pi(s, a) = r(s, a) + \gamma \mathbb{E}_{s'}\left[V^\pi(s')\right]$$
$$V^\pi(s) = \mathbb{E}_{a \sim \pi}\left[Q^\pi(s, a) - \alpha \log \pi(a \mid s)\right]$$

注意第二个公式中的 $-\alpha \log \pi$——多了熵项。

### 5.3 SAC 的三个网络

| 网络 | 参数 | 用途 |
| :--- | :--- | :--- |
| **Actor** $\pi_\theta(a \mid s)$ | $\theta$ | 随机策略（高斯）|
| **双 Q-Critic** $Q_{\phi_1}, Q_{\phi_2}$ | $\phi_1, \phi_2$ | 估计 $Q^\pi$，取 min 防过估计 |
| **Target Critic** $Q_{\phi_i^-}$ | EMA | 提供稳定 target |

> **SAC 不显式维护 V 网络**——通过 $V = \mathbb{E}_{a \sim \pi}[Q - \alpha \log \pi]$ 间接计算。

### 5.4 SAC 损失函数

**Critic 损失**：
$$\mathcal{L}_{\phi_i} = \mathbb{E}\left[(Q_{\phi_i}(s, a) - y)^2\right]$$
$$y = r + \gamma \mathbb{E}_{a' \sim \pi_\theta}\left[\min_i Q_{\phi_i^-}(s', a') - \alpha \log \pi_\theta(a' \mid s')\right]$$

**Actor 损失**：最小化 KL 到 "Boltzmann 策略"，等价于
$$\mathcal{L}_\theta = \mathbb{E}_{s, a \sim \pi_\theta}\left[\alpha \log \pi_\theta(a \mid s) - \min_i Q_{\phi_i}(s, a)\right]$$

**Reparameterization trick**：$a = \mu_\theta(s) + \sigma_\theta(s) \cdot \epsilon, \epsilon \sim \mathcal{N}(0, 1)$，让梯度可以从 $a$ 反传到 $\theta$。

### 5.5 自动温度调节

$\alpha$ 难调，SAC 提出**自动调节**：
$$\mathcal{L}_\alpha = -\alpha \cdot \mathbb{E}\left[\log \pi_\theta(a \mid s) + \bar{H}\right]$$

其中 $\bar{H}$ 是目标熵（如 $-|\mathcal{A}|$）。这相当于："如果策略熵低于目标，增大 $\alpha$ 鼓励探索；高于目标则减小 $\alpha$"。

实践中通常学 $\log \alpha$（保证 $\alpha > 0$）。

---

# §B 模型架构

## B.1 SAC 数据流

```
                            ┌────────────────┐
              ┌──────────── │ Replay Buffer  │ ←──── (s, a, r, s')
              │             └────────────────┘       (来自 Actor 探索)
              │                     │
              │ sample                ↓
              │             ┌────────────────┐
              │             │   minibatch    │
              │             └────────────────┘
              │                     │
       ┌──────┴──────┬──────────────┴────────────┐
       ▼             ▼                            ▼
  ┌─────────┐  ┌─────────┐                ┌─────────────┐
  │ Critic  │  │ Critic  │                │   Actor     │
  │   Q₁    │  │   Q₂    │                │ π(a|s) (μ,σ)│
  └────┬────┘  └────┬────┘                └──────┬──────┘
       │             │                            │
       └──────┬──────┘                            │
              ▼                                    │
        min(Q₁, Q₂) ─────────────────────────────┐│
              │                                   ▼│
              ▼                          target_a ~ π
       Critic loss                              │
                                                ▼
                                        target Q (with α·log π)
                                                │
                                                ▼
                                          Actor loss
```

## B.2 Gaussian Policy with Reparameterization

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SACGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        h = self.shared(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mu, std

    def sample(self, obs):
        """⭐ Reparameterization trick + tanh 压缩"""
        mu, std = self.forward(obs)
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()                              # ⭐ 重参数化采样
        a = torch.tanh(x)                                 # ⭐ 压缩到 [-1, 1]

        # tanh 压缩后的 log_prob 修正
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6)        # ⭐ 雅可比修正
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return a, log_prob
```

> **tanh + 雅可比修正**：MuJoCo 的动作空间通常是 $[-1, 1]$。把高斯样本经 tanh 压缩后，需要修正 log-prob 的雅可比项 $\log(1 - \tanh^2(x))$。

## B.3 SAC 完整训练循环

```python
import copy
import numpy as np
from torch.optim import Adam

class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


def train_sac(env, n_steps=int(1e6), batch_size=256, gamma=0.99, tau=0.005,
              lr=3e-4, target_entropy=None, buffer_size=int(1e6),
              learning_starts=10000):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = SACGaussianActor(obs_dim, act_dim)
    q1 = QNet(obs_dim, act_dim)
    q2 = QNet(obs_dim, act_dim)
    q1_target = copy.deepcopy(q1)
    q2_target = copy.deepcopy(q2)
    for p in q1_target.parameters(): p.requires_grad = False
    for p in q2_target.parameters(): p.requires_grad = False

    actor_optim = Adam(actor.parameters(), lr=lr)
    q1_optim = Adam(q1.parameters(), lr=lr)
    q2_optim = Adam(q2.parameters(), lr=lr)

    # ⭐ 自动温度调节
    if target_entropy is None:
        target_entropy = -act_dim                          # 标准默认
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_optim = Adam([log_alpha], lr=lr)

    buffer = ReplayBuffer(buffer_size)                      # 见 Ch7 §B.3
    obs, _ = env.reset()

    for step in range(n_steps):
        # ============ 与环境交互 ============
        if step < learning_starts:
            a = env.action_space.sample()                  # 完全随机探索
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                a, _ = actor.sample(obs_t)
                a = a.squeeze(0).numpy()

        obs_new, r, terminated, truncated, _ = env.step(a)
        buffer.push(obs, a, r, obs_new, float(terminated))
        obs = obs_new if not (terminated or truncated) else env.reset()[0]

        # ============ 训练 ============
        if step > learning_starts and len(buffer) > batch_size:
            s, action, reward, s_new, done = buffer.sample(batch_size)
            alpha = log_alpha.exp().item()

            # --- Critic 更新 ---
            with torch.no_grad():
                a_new, log_prob_new = actor.sample(s_new)
                q1_target_val = q1_target(s_new, a_new)
                q2_target_val = q2_target(s_new, a_new)
                q_target = torch.min(q1_target_val, q2_target_val)
                # ⭐ Soft Bellman target（含熵项）
                y = reward + gamma * (1 - done) * (q_target - alpha * log_prob_new.squeeze(-1))

            q1_loss = (q1(s, action) - y).pow(2).mean()
            q2_loss = (q2(s, action) - y).pow(2).mean()

            q1_optim.zero_grad(); q1_loss.backward(); q1_optim.step()
            q2_optim.zero_grad(); q2_loss.backward(); q2_optim.step()

            # --- Actor 更新 ---
            a_pi, log_prob_pi = actor.sample(s)
            q1_val = q1(s, a_pi)
            q2_val = q2(s, a_pi)
            q_min = torch.min(q1_val, q2_val)
            # ⭐ Actor loss: minimize KL to softmax(Q/α)
            actor_loss = (alpha * log_prob_pi.squeeze(-1) - q_min).mean()

            actor_optim.zero_grad(); actor_loss.backward(); actor_optim.step()

            # --- 温度更新 ---
            alpha_loss = -(log_alpha * (log_prob_pi.squeeze(-1) + target_entropy).detach()).mean()
            alpha_optim.zero_grad(); alpha_loss.backward(); alpha_optim.step()

            # --- 软更新 Target ---
            with torch.no_grad():
                for p, p_target in zip(q1.parameters(), q1_target.parameters()):
                    p_target.data.mul_(1 - tau).add_(p.data, alpha=tau)
                for p, p_target in zip(q2.parameters(), q2_target.parameters()):
                    p_target.data.mul_(1 - tau).add_(p.data, alpha=tau)

    return actor, q1, q2
```

---

# §C 训练与推理

## C.1 实验：MuJoCo HalfCheetah

```python
import gymnasium as gym
env = gym.make("HalfCheetah-v4")
actor, _, _ = train_sac(env, n_steps=int(1e6))
```

典型结果（reward 范围）：
- 10万步：~2000
- 50万步：~8000
- 100万步：~12000（接近 SOTA）

## C.2 SAC vs PPO（连续控制对比）

| 维度 | PPO（Ch8） | SAC（本章） |
| :--- | :--- | :--- |
| **采样方式** | On-policy | Off-policy |
| **数据效率** | 中 | **高** |
| **超参敏感度** | 高 | 低（自动温度） |
| **收敛速度（机器人）** | 慢 | 快 |
| **稳定性** | 高 | 高 |
| **调参难度** | 中 | 低 |
| **推荐场景** | 仿真够便宜 | 真机训练（需高数据效率） |

> **业界经验**：MuJoCo benchmark 上 SAC > PPO；但 Atari 上 PPO/Rainbow > SAC。
> 现代框架（Stable-Baselines3、Tianshou）默认 SAC 处理连续控制，PPO 处理离散。

## C.3 推理

```python
def inference(actor, obs):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mu, _ = actor.forward(obs_t)
        a = torch.tanh(mu)                                # ⭐ 推理时用均值（确定性）
    return a.squeeze(0).numpy()
```

**注意**：SAC 训练时**采样**（`sample()`），推理时**取均值**（`mu`）——把随机性去掉。

## C.4 工程经验

| 问题 | 解决 |
| :--- | :--- |
| Q 值发散 | 减小 lr / 增大 batch / 加 reward scaling |
| Actor 不学习 | 检查 entropy target / α 是否合理 |
| 早期发散 | learning_starts 设大些（10K-50K） |
| 训练慢 | 增加 critic 更新频率（如 update_every=2） |

---

# §D 章末速查

## D.1 三大算法对比

| 算法 | 策略类型 | 主要创新 | 当前地位 |
| :--- | :--- | :--- | :--- |
| **DDPG** | 确定性 | DPG + DQN | 已被 TD3/SAC 取代 |
| **TD3** | 确定性 | Twin Q + Delayed + Smoothing | 仍常用 |
| **SAC** | 随机（最大熵） | Soft Bellman + 自动温度 | **连续控制 SOTA** |

## D.2 核心公式

| # | 公式 | 含义 |
| :---: | :--- | :--- |
| 1 | $\nabla_\theta J = \nabla_\theta \mu_\theta \cdot \nabla_a Q$ | DPG 定理 |
| 2 | $J = \mathbb{E}[\sum r + \alpha H(\pi)]$ | 最大熵 RL 目标 |
| 3 | $V^\pi(s) = \mathbb{E}_a[Q - \alpha \log \pi]$ | Soft V |
| 4 | $\mathcal{L}_\theta = \mathbb{E}[\alpha \log \pi - Q]$ | SAC actor loss |
| 5 | $\mathcal{L}_\alpha = -\alpha (\log \pi + \bar{H})$ | 自动温度 |

## D.3 常见面试题

**Q1：SAC 为什么要最大化熵？**
- 鼓励探索（保持随机性，避免局部最优）
- 鲁棒性（能应对状态扰动）
- 数学上对应 "soft Bellman 方程"，提供更平滑的优化目标

**Q2：DDPG 与 SAC 的核心差别？**
- DDPG：确定性策略 + 人工噪声
- SAC：随机策略 + 熵正则
- SAC 把"探索"内化到目标函数里，而 DDPG 只能事后加噪声

**Q3：为什么 SAC 用双 Q 网络？**
- 抑制 max 偏差（target 取 min）
- DDPG 在 Atari 上 Q 经常爆炸，TD3/SAC 双 Q 是关键修复

**Q4：自动温度调节背后的数学是什么？**
- 把熵约束 $H(\pi) \geq \bar{H}$ 用拉格朗日松弛化
- $\alpha$ 是拉格朗日乘数
- 通过梯度上升让约束自动满足

**Q5：SAC 中的 reparameterization trick 为什么必要？**
- Actor 损失里有 $a \sim \pi_\theta$，但 $a$ 是采样得到——不可微
- 重参数化把采样从 "$a \sim \mathcal{N}(\mu, \sigma)$" 改为 "$a = \mu + \sigma \epsilon, \epsilon \sim \mathcal{N}(0, 1)$"
- 这样 $a$ 对 $\mu, \sigma$ 可微，梯度可以反传

---

## 承上启下

到目前为止我们已经掌握了 RL 的几乎所有"主流路线"：
- Value-based: Q-Learning (Ch4) → DQN (Ch7)
- Policy-based: REINFORCE (Ch5) → A2C (Ch6) → PPO (Ch8)
- Hybrid: DDPG / SAC (Ch9)

但还有几个**横切关注点**：
- **探索**：所有方法都依赖探索，但前几章只用了简单的 ε-greedy / 高斯噪声。**Ch10 介绍现代探索方法**（UCB、Curiosity、RND）。
- **离线学习**：所有方法都需要 online 采样。**Ch11 介绍 offline RL**——只用预先收集的数据训练。这条路线最终通向 **DPO**。
- **模仿学习**：从专家演示中学习。**Ch12 介绍 BC / DAgger / GAIL / IRL**。这是 **SFT** 的 RL 视角。
