---
title: RL Chapter6 Actor-Critic 与 GAE：Policy Gradient 的工业化
categories: 学习笔记-强化学习
date: 2026-05-14 10:00:00
mathjax: true
tags:
    - AI
    - 强化学习
    - AI面试知识
---

> **本章定位**：把 Policy Gradient（Ch5）从 "MC 风格"升级为 "TD 风格"——引入 Critic 网络作为可学习的 baseline + 用 bootstrap 估计优势。最终诞生的 **GAE**（Generalized Advantage Estimation）是 PPO 的核心。

> **承上**：Ch5 §A.5 baseline + Ch3 §A.4 TD(λ)。
> **启下**：Ch8 在 AC 之上加 trust region 得到 TRPO/PPO。

---

# §A 数学原理

## 1. 核心思想：用 Critic 学习 baseline

回忆 Ch5 §A.5：减 baseline 不改变期望但降方差。最佳 baseline 是 $V^\pi(s_t)$。

**Actor-Critic 核心**：
- **Actor** = 策略网络 $\pi_\theta(a \mid s)$，做动作
- **Critic** = 价值网络 $V_\phi(s)$，评估状态
- Actor 用 $A_t = G_t - V_\phi(s_t)$ 作梯度信号
- Critic 用回归损失 $(V_\phi(s_t) - G_t)^2$ 学习

## 2. 优势函数 (Advantage Function)

定义：
$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

**含义**：在状态 $s$ 选 action $a$ 比"按策略 $\pi$ 平均水平"好多少。

**关键性质**：
- 若 $A^\pi(s, a) > 0$：动作 $a$ 比平均好 → 提高 $\pi(a \mid s)$
- 若 $A^\pi(s, a) < 0$：动作 $a$ 比平均差 → 降低 $\pi(a \mid s)$
- $\mathbb{E}_{a \sim \pi}[A^\pi(s, a)] = 0$（按定义）

**PG 定理优势函数版**：
$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) A^\pi(s_t, a_t)\right]$$

## 3. 优势函数的多种估计方式

回到 Ch3 的"价值估计"工具箱，可用于估计 $A_t$：

### 3.1 单步 TD（最简单）
$$A_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) = \delta_t$$

即 TD-error。优点：方差最小；缺点：偏差最大（依赖 $V_\phi$ 准确）。

### 3.2 n-step
$$A_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V_\phi(s_{t+n}) - V_\phi(s_t)$$

### 3.3 蒙特卡洛
$$A_t^{(\infty)} = G_t - V_\phi(s_t) = \sum_{k=0}^\infty \gamma^k r_{t+k} - V_\phi(s_t)$$

偏差最小（$G_t$ 无偏），方差最大。

### 3.4 GAE：连接两端的"指数加权"

借鉴 Ch3 的 TD(λ)，定义**广义优势估计**：

$$\boxed{\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}}$$

其中 $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ 是 TD-error。

**关键参数**：
- $\lambda = 0$：$\hat{A}_t = \delta_t$（单步 TD）
- $\lambda = 1$：$\hat{A}_t = \sum_l \gamma^l \delta_{t+l}$，可证等于 MC（带 V baseline）
- $\lambda \in (0, 1)$：bias-variance trade-off 上的中间地带

**实践常用**：$\lambda = 0.95$，$\gamma = 0.99$。

### 3.5 GAE 的递推形式（实现关键）

从定义可以推出递推公式：
$$\hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}$$

这意味着可以**从后往前一次扫描计算所有 $\hat{A}_t$**，复杂度 $O(T)$。

## 4. Actor-Critic 完整算法

### 4.1 损失函数

**Actor 损失**：
$$\mathcal{L}^{\text{actor}} = -\sum_t \log \pi_\theta(a_t \mid s_t) \hat{A}_t \quad (\text{stop-grad on } \hat{A}_t)$$

**Critic 损失**：
$$\mathcal{L}^{\text{critic}} = \sum_t \left( V_\phi(s_t) - V_t^{\text{target}} \right)^2$$

其中 $V_t^{\text{target}} = \hat{A}_t + V_\phi(s_t) = $ "GAE-corrected return"。

**完整目标**（含熵正则）：
$$\mathcal{L} = \mathcal{L}^{\text{actor}} + c_v \mathcal{L}^{\text{critic}} - c_h \mathbb{E}[H(\pi_\theta)]$$

经验值：$c_v = 0.5$，$c_h = 0.01$。

### 4.2 A2C vs A3C

| 算法 | 特点 |
| :--- | :--- |
| **A2C** (Advantage AC) | 同步版本，多个 worker 同时跑收集数据，集中更新 |
| **A3C** (Asynchronous AC) | 异步，每个 worker 独立更新（OpenAI 早期用这个加速 Atari 训练） |

实践中 A2C 简单且效果不差，A3C 现已较少用。

## 5. On-policy vs Off-policy

AC 是 **on-policy**：
- 每次更新需要新采样
- 数据用过即丢
- 数据效率低

PPO（Ch8）通过 importance sampling 让 AC 可以**在 minibatch 内重复使用数据**，把 on-policy 数据效率拉到接近 off-policy。

---

# §B 模型架构

## B.1 数据流：Actor-Critic 双头网络

```
            obs (s_t)
              │
              ▼
    ┌────────────────────┐
    │  Shared Encoder    │   (可选共享)
    └─────────┬──────────┘
              │
        ┌─────┴─────┐
        ▼           ▼
   ┌─────────┐ ┌─────────┐
   │ Actor   │ │ Critic  │
   │ Head    │ │ Head    │
   └────┬────┘ └────┬────┘
        ▼           ▼
   π(a|s)        V(s)
```

## B.2 Actor-Critic 网络的 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        # 共享底层
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # Actor head
        self.actor_head = nn.Linear(hidden, n_actions)
        # Critic head
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        """
        obs: [B, obs_dim]
        返回: (action distribution, state value)
        """
        h = self.shared(obs)
        logits = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)           # [B]
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value

    def act(self, obs):
        """采样 + 返回 log_prob, value"""
        dist, value = self.forward(obs)
        a = dist.sample()
        log_prob = dist.log_prob(a)
        return a, log_prob, value
```

> **要不要共享 backbone？** 
> - 共享：参数少，policy 和 value 可以互相规约
> - 不共享：训练更稳，policy / value 各自学习曲线不互相干扰
> - 实践：CartPole / 简单环境共享，复杂环境（Atari、MuJoCo）不共享

## B.3 GAE 计算（最关键的代码段）

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    GAE 的反向递推实现
    rewards: [T] 每步 reward
    values:  [T+1] 每步 V(s)，最后一个是 V(s_T)
    dones:   [T] 每步是否 episode 结束（1 = 结束）
    返回:
        advantages: [T]
        returns:    [T] = advantages + values[:-1]
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0
    for t in reversed(range(T)):
        # ⭐ 如果 t 步是 episode 末尾，下一状态价值为 0
        next_value = values[t + 1] * (1 - dones[t])
        delta = rewards[t] + gamma * next_value - values[t]      # TD-error
        # ⭐ GAE 递推
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:-1]
    return advantages, returns
```

**理解 dones 的作用**：episode 边界处不能让 GAE "跨 episode" 累加，所以遇到 done=1 时 GAE 重置。

## B.4 完整 A2C 训练循环

```python
import gymnasium as gym
from torch.optim import Adam

def a2c(env_name="CartPole-v1", n_steps=100000, n_envs=8, n_step_rollout=16,
        gamma=0.99, lam=0.95, lr=3e-4, c_v=0.5, c_h=0.01):
    """
    Synchronous Advantage Actor-Critic
    """
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(n_envs)])
    obs_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n

    model = ActorCritic(obs_dim, n_actions)
    optim = Adam(model.parameters(), lr=lr)

    obs, _ = envs.reset()
    total_steps = 0

    while total_steps < n_steps:
        # ============ Rollout：收集 n_step_rollout 步数据 ============
        log_probs, values, rewards, dones, entropies = [], [], [], [], []

        for _ in range(n_step_rollout):
            obs_t = torch.tensor(obs, dtype=torch.float32)
            dist, v = model.forward(obs_t)
            a = dist.sample()
            log_prob = dist.log_prob(a)
            entropy = dist.entropy()

            obs_new, r, terminated, truncated, _ = envs.step(a.numpy())
            done = np.logical_or(terminated, truncated).astype(np.float32)

            log_probs.append(log_prob)
            values.append(v)
            rewards.append(torch.tensor(r, dtype=torch.float32))
            dones.append(torch.tensor(done))
            entropies.append(entropy)

            obs = obs_new
            total_steps += n_envs

        # ============ 计算 GAE ============
        # 末尾状态的 V (用于 bootstrap)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32)
            _, last_v = model.forward(obs_t)

        # 拼接成 [T+1, n_envs] 形状的 values
        values_full = torch.stack(values + [last_v])     # [T+1, n_envs]
        rewards = torch.stack(rewards)                    # [T, n_envs]
        dones = torch.stack(dones)                        # [T, n_envs]

        # 对每个 env 独立计算 GAE
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_v = values_full[t + 1] * (1 - dones[t])
            delta = rewards[t] + gamma * next_v - values_full[t]
            last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + values_full[:-1]

        # 标准化优势（重要 trick）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ============ Loss ============
        log_probs = torch.stack(log_probs)               # [T, n_envs]
        entropies = torch.stack(entropies)
        values_t = values_full[:-1]                       # [T, n_envs]

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = (values_t - returns.detach()).pow(2).mean()
        entropy_loss = -entropies.mean()                  # 负号：最大化熵
        loss = actor_loss + c_v * critic_loss + c_h * entropy_loss

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)   # ⭐ 梯度裁剪
        optim.step()

    return model
```

**几个关键工程细节**：
1. `n_envs` 个并行环境提供 batch（大 batch 更稳定）
2. `n_step_rollout` 是 rollout 长度（典型 16-128）
3. `advantages.detach()`：防止梯度从 actor loss 流回 critic
4. `returns.detach()`：critic 用 TD target，target 不应反传
5. **梯度裁剪** 是 PG 系列调参的"必备"（PPO 也用）

---

# §C 训练与推理

## C.1 实战：CartPole 上的 A2C

```python
model = a2c("CartPole-v1", n_steps=50000, n_envs=8)
```

典型结果：
- 5000 步：reward ~ 50
- 10000 步：reward ~ 200
- 20000 步：reward ~ 500（满分）

对比 Ch5 的 REINFORCE：A2C 收敛速度通常快 3-5 倍。

## C.2 不同 λ 的影响（GAE 调参）

```python
for lam in [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]:
    model = a2c("CartPole-v1", n_steps=30000, lam=lam)
    # 评测
```

经验：
- $\lambda = 0$（纯 TD）：偏差大，CartPole 很难收敛
- $\lambda = 0.9 \sim 0.95$：实战甜点
- $\lambda = 1$（纯 MC）：方差大，收敛慢但最终可以

## C.3 推理视角：A2C 训练完后

推理时只用 Actor，丢弃 Critic：
```python
def inference(model, obs):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        dist, _ = model(obs_t)
        a = dist.probs.argmax(dim=-1)                     # greedy
    return a.item()
```

> **关键观察**：所有 PG 系列（包括 PPO、DPO）训练时都需要 Critic 或 reference，**推理时只保留 Actor**。

## C.4 工程经验

| 问题 | 解决 |
| :--- | :--- |
| 训练发散 | 学习率过大 / 梯度未裁剪 |
| Critic 不学习 | $c_v$ 过小 / Critic 学习率单独调 |
| 收敛后又爆炸 | 学习率衰减 / `polyak averaging` |
| 早期就熵塌缩 | $c_h$ 调大到 0.05+ |
| 长 episode 稀疏 reward | 上 PPO（Ch8）+ reward shaping |

---

# §D 章末速查

## D.1 核心公式速记

| # | 公式 | 含义 |
| :---: | :--- | :--- |
| 1 | $A^\pi(s, a) = Q^\pi(s,a) - V^\pi(s)$ | 优势函数 |
| 2 | $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | TD-error |
| 3 | $\hat{A}_t^{\text{GAE}} = \sum_l (\gamma\lambda)^l \delta_{t+l}$ | GAE |
| 4 | $\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$ | GAE 递推 |
| 5 | $V_t^{\text{target}} = \hat{A}_t + V(s_t)$ | Critic 目标 |

## D.2 常见面试题

**Q1：什么是优势函数？为什么用它而不直接用 Q？**
- $A = Q - V$，衡量"action 比平均好/差多少"
- 用 A 比 Q 更稳：$A$ 期望为 0，避免梯度估计被 Q 的"基线水平"干扰
- 标准化 $A$ 后梯度更小、更稳定

**Q2：GAE 的两个超参 $\gamma, \lambda$ 各起什么作用？**
- $\gamma$：折扣因子（与环境本身的"远视程度"相关）
- $\lambda$：偏差-方差权衡（与 $V$ 估计的准确性相关）
- $\gamma$ 大、$\lambda$ 大 → 方差大、偏差小

**Q3：Actor 和 Critic 的更新可以共享 backbone 吗？**
- 可以，参数少
- 但 actor 和 critic 的"梯度尺度"不同，共享时需要小心 $c_v$ 调整
- 复杂任务通常分开（独立 backbone）

**Q4：A2C 是 on-policy 还是 off-policy？**
- On-policy：每次更新需要新采样
- 用过即丢，数据效率低
- PPO（Ch8）通过 importance sampling 在 minibatch 内复用数据

**Q5：GAE 与 TD(λ) 的关系？**
- TD(λ)：用于估计 V（基于价值的指数加权）
- GAE：用于估计 A（基于优势的指数加权）
- 数学上几乎一样：GAE = TD(λ) - V(s)

---

## 承上启下

我们现在有了 PG 的"工业级"实现：A2C + GAE。但还有两大痛点：
1. **数据效率低**（on-policy）
2. **训练步长不稳**：策略更新太激进会崩

下一章 **Ch7 DQN** 走完 value-based 路线（看 Q-Learning 如何在大状态空间下工作），然后 **Ch8 TRPO/PPO** 解决 PG 的两大痛点：
- **重要性采样** 让数据可以复用 → 数据效率
- **Trust Region** 让步长不会过大 → 训练稳定

PPO 是 RLHF 的核心算法，是连接经典 RL 与 LLM 时代的关键桥梁。
