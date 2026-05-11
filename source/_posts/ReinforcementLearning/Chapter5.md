---
title: RL Chapter5 Policy Gradient：直接对策略求梯度
categories: 学习笔记-强化学习
date: 2026-05-10 10:00:00
mathjax: true
tags:
    - AI
    - 强化学习
    - AI面试知识
---

> **本章定位**：RL 的另一条主线——**直接参数化策略并求梯度上升**。Policy Gradient 是 PPO（Ch8）、DPO（Ch11/Ch13）、GRPO 的共同根基。本章包含整套 RL 中**最重要的一个证明**：策略梯度定理。

> **承上**：Ch1 价值函数 + Ch3 期望估计。
> **启下**：Ch6 把 PG + 价值函数 = Actor-Critic；Ch8 把 AC 加上 trust region = PPO。

---

# §A 数学原理

## 1. 为什么需要 Policy Gradient？

Q-Learning（Ch4）有三个无法逾越的限制：

1. **离散动作空间**：$\arg\max_a Q$ 在连续动作下是优化问题（机器人关节角度怎么 max？）
2. **确定性策略**：纯 greedy 没有随机性，部分可观测环境下不够
3. **策略改进的间接性**：先估 $Q$ 再贪心，两步走

**Policy Gradient 的解决方案**：
- 直接参数化策略 $\pi_\theta(a \mid s)$（神经网络输出动作分布）
- 对参数 $\theta$ 做梯度上升
- **目标**：$\max_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$

## 2. 策略参数化

### 2.1 离散动作：Categorical 分布

神经网络输出 $|\mathcal{A}|$ 维 logits，softmax 后得到概率：
$$\pi_\theta(a \mid s) = \frac{\exp(f_\theta(s)_a)}{\sum_{a'} \exp(f_\theta(s)_{a'})}$$

### 2.2 连续动作：Gaussian 分布

神经网络输出均值 $\mu_\theta(s)$ 和（可选的）方差 $\sigma_\theta(s)$：
$$\pi_\theta(a \mid s) = \mathcal{N}(a; \mu_\theta(s), \sigma_\theta(s)^2)$$

实践中 $\log\sigma$ 通常作为独立可训练参数（与 state 无关）。

## 3. Policy Gradient 定理（**最重要的证明**）

### 3.1 目标函数

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)], \quad R(\tau) = \sum_{t=0}^T \gamma^t r_t$$

我们要算 $\nabla_\theta J(\theta)$。

### 3.2 朴素尝试遇到的困难

$$\nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau) R(\tau) d\tau$$

直接求导有问题：$R(\tau)$ 不依赖 $\theta$，但 $p_\theta(\tau)$ 是关于 $\theta$ 的复杂函数，**无法直接微分一个分布**。

### 3.3 Score Function Trick（log-derivative trick）

利用恒等式：
$$\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)$$

代回：
$$\nabla_\theta J(\theta) = \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) d\tau = \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log p_\theta(\tau) R(\tau)]$$

**这一步把"对分布求导"变成"对 log-概率求导"**——后者就是 supervised learning 中常见的对 NLL 求导！

### 3.4 展开 $\log p_\theta(\tau)$

轨迹概率：
$$p_\theta(\tau) = p(s_0) \prod_{t=0}^T \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)$$

取 log：
$$\log p_\theta(\tau) = \log p(s_0) + \sum_t \log \pi_\theta(a_t \mid s_t) + \sum_t \log P(s_{t+1} \mid s_t, a_t)$$

求 $\nabla_\theta$ 时，**$p(s_0)$ 和 $P$ 都不依赖 $\theta$**，全部消去！

$$\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

### 3.5 最终公式：Policy Gradient 定理

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R(\tau) \right]}$$

> **三个值得反复琢磨的特性**：
> 1. **不需要环境模型 $P$**——它在 log-derivative 后自动消失
> 2. **样本估计**：用 $N$ 条轨迹的均值替代期望
> 3. **形式直观**：把 supervised learning 的 cross-entropy 损失乘以一个 reward 系数

## 4. REINFORCE 算法

把策略梯度定理写成实用算法：

```
for episode = 1, 2, ...:
    用 π_θ 采样轨迹 τ = (s_0, a_0, r_0, ..., s_T)
    计算累积回报 G_0, G_1, ..., G_T

    for t = 0, ..., T:
        梯度 g = ∇_θ log π_θ(a_t | s_t) · G_t
        θ ← θ + α g
```

REINFORCE（Williams 1992）是 PG 第一个实用算法。注意公式里的 $R(\tau)$ 应该是从 $t$ 开始的累积回报 $G_t$（"未来 reward 影响当前 action 的好坏"），不应包括 $t$ 之前的 reward——这叫 **causality** trick。

## 5. Baseline 减方差

### 5.1 问题

REINFORCE 的方差极大：每条轨迹 $G_t$ 在数百到数千的范围波动，梯度估计噪声严重。

### 5.2 关键观察：减一个 baseline 不影响期望

**定理**：对任意只依赖 $s$（不依赖 $a$）的函数 $b(s)$，
$$\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a \mid s) \cdot b(s)] = 0$$

**证明**：
$$\mathbb{E}_{a}[\nabla_\theta \log \pi_\theta(a \mid s) b(s)] = b(s) \int \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s) da$$
$$= b(s) \int \nabla_\theta \pi_\theta(a \mid s) da = b(s) \nabla_\theta \int \pi_\theta(a \mid s) da = b(s) \nabla_\theta 1 = 0 \quad \square$$

### 5.3 用法

把 $G_t$ 替换为 $G_t - b(s_t)$，期望不变但方差通常更小：
$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) (G_t - b(s_t))\right]$$

**最佳 baseline**：$b(s_t) = V^\pi(s_t)$（状态价值函数）。这时
$$G_t - V^\pi(s_t) \approx Q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)$$
即**优势函数**（Advantage）。这就是下一章 Actor-Critic 的起点。

### 5.4 直觉

REINFORCE 用 $G_t$ 衡量"action 的绝对好坏"——但即使 $G_t > 0$ 也未必意味着 action 好（可能整条轨迹都好）。

减去 baseline 后，衡量的是"**这个 action 比平均水平好/差多少**"，更精准。

## 6. 方差分析（深入）

REINFORCE 的方差为什么这么大？关键在于 $G_t$ 是"长期累积"。

设每步 reward 方差为 $\sigma^2$，$G_t$ 的方差大致是 $T \sigma^2$（独立时）。乘以梯度后，方差进一步放大。

减 baseline 后，方差降到 $\text{Var}(A_t) \cdot \|\nabla \log \pi\|^2$，通常小一两个数量级。

> **核心思想**：PG 有正确的 expectation，但实战中关键是控制 variance。所有后续算法（Actor-Critic、TRPO、PPO）都在做"如何在保持低 bias 的前提下降 variance"。

## 7. 重要性采样：On-policy → Off-policy 的桥梁

PG 是 on-policy（每次更新需要新采样），数据效率低。**重要性采样** 把 off-policy 数据"修正"成 on-policy：

$$\mathbb{E}_{\tau \sim \pi_\theta}[f(\tau)] = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}}\left[\frac{p_\theta(\tau)}{p_{\theta_{\text{old}}}(\tau)} f(\tau)\right]$$

这就是后续 PPO 和 TRPO 中 **importance ratio** $r_t(\theta) = \pi_\theta / \pi_{\theta_{\text{old}}}$ 的来源。

---

# §B 模型架构

## B.1 Categorical Policy 的 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoricalPolicy(nn.Module):
    """离散动作策略（如 CartPole）"""
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),                # 输出 logits
        )

    def forward(self, obs):
        """
        obs: [B, obs_dim]
        返回: Categorical 分布对象
        """
        logits = self.net(obs)                            # [B, n_actions]
        return torch.distributions.Categorical(logits=logits)

    def sample(self, obs):
        """采样 + 返回 log_prob"""
        dist = self.forward(obs)
        a = dist.sample()                                 # [B]
        log_prob = dist.log_prob(a)                       # [B]
        return a, log_prob
```

## B.2 Gaussian Policy（连续动作）

```python
class GaussianPolicy(nn.Module):
    """连续动作策略（如 MuJoCo）"""
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)        # 均值头
        self.log_std = nn.Parameter(torch.zeros(act_dim))# log(σ)，独立参数

    def forward(self, obs):
        h = self.shared(obs)
        mu = self.mu_head(h)
        std = self.log_std.exp().expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def sample(self, obs):
        dist = self.forward(obs)
        a = dist.sample()                                 # [B, act_dim]
        log_prob = dist.log_prob(a).sum(dim=-1)           # 多维独立 → 求和
        return a, log_prob
```

> **为什么 log_std 不依赖 state？** 实验经验：让 std 随 state 变化反而更难训。固定 log_std 是 OpenAI baselines 的默认做法。

## B.3 完整 REINFORCE 实现

```python
import gymnasium as gym
from torch.optim import Adam

def reinforce(env_name="CartPole-v1", n_episodes=1000, gamma=0.99, lr=1e-3,
              use_baseline=False):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = CategoricalPolicy(obs_dim, n_actions)
    optim = Adam(policy.parameters(), lr=lr)

    if use_baseline:
        # 用一个简单的 V 网络作为 baseline
        value_net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        v_optim = Adam(value_net.parameters(), lr=lr)

    rewards_history = []
    for ep in range(n_episodes):
        # ============ 采样 episode ============
        obs, _ = env.reset()
        log_probs, rewards, states = [], [], []

        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            a, log_prob = policy.sample(obs_t.unsqueeze(0))
            obs_new, r, terminated, truncated, _ = env.step(a.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(r)
            states.append(obs_t)
            obs = obs_new

        rewards_history.append(sum(rewards))

        # ============ 计算累积回报 G_t ============
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # ⭐ 减 baseline（如果启用）
        if use_baseline:
            states_tensor = torch.stack(states)
            values = value_net(states_tensor).squeeze(-1)
            advantages = returns - values.detach()
            # 同时训练 V 网络
            v_loss = ((values - returns) ** 2).mean()
            v_optim.zero_grad(); v_loss.backward(); v_optim.step()
        else:
            advantages = returns

        # 标准化（trick：进一步降方差）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ============ Policy Gradient Loss ============
        log_probs = torch.stack(log_probs).squeeze()
        loss = -(log_probs * advantages).sum()           # ⭐ 注意是负号

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (ep + 1) % 50 == 0:
            avg = sum(rewards_history[-50:]) / 50
            print(f"Episode {ep+1}: avg reward = {avg:.1f}")

    return policy, rewards_history
```

**代码关键点**：
1. `log_prob = dist.log_prob(a)`：score function trick 的核心
2. `loss = -(log_probs * advantages).sum()`：最大化 J = 最小化 -J
3. `advantages.detach()`：baseline 网络的输出不能反向传到 policy
4. `(adv - mean) / std`：标准化是 PG 调参的"金标准 trick"

## B.4 数据流总览

```
Obs (s_t)
   │
   ▼
┌──────────────────┐
│  Policy Network  │ → logits (or μ, σ)
└──────────────────┘
   │
   ▼
┌──────────────────┐
│ Categorical /    │ → 采样 a_t, 计算 log π(a_t|s_t)
│ Normal Dist      │
└──────────────────┘
   │           │
   ▼           ▼
执行 a_t   存储 log_prob
   │
   ▼
观察 r_t, s_{t+1}
   │
   ▼
[episode 结束后]
   │
   ▼
计算 G_t (从后往前累加)
   │
   ▼
loss = -Σ log π(a_t|s_t) · (G_t - b(s_t))
   │
   ▼
loss.backward() → 更新 θ
```

---

# §C 训练与推理

## C.1 在 CartPole 上的训练曲线

```python
policy, rewards = reinforce("CartPole-v1", n_episodes=1000, use_baseline=True)
```

**典型结果**：
- 不带 baseline：~500 episode 收敛到 reward=200（CartPole-v1 上限是 500，难收敛）
- 带 baseline：~200 episode 收敛到 reward=500
- 带 baseline + advantage normalization：~150 episode 收敛

## C.2 推理：评估学到的策略

```python
def evaluate(policy, env_name="CartPole-v1", n_eval=10):
    env = gym.make(env_name)
    rewards = []
    for _ in range(n_eval):
        obs, _ = env.reset()
        total_r = 0
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist = policy(obs_t)
                a = dist.probs.argmax(dim=-1).item()    # ⭐ greedy 推理
            obs, r, terminated, truncated, _ = env.step(a)
            total_r += r
            done = terminated or truncated
        rewards.append(total_r)
    return sum(rewards) / n_eval

print(f"Eval reward: {evaluate(policy):.1f}")
```

**注意**：
- 训练时：`dist.sample()` 探索
- 推理时：`dist.probs.argmax()` 利用（greedy）
- 也可以保留随机性（ensemble 风格）：`dist.sample()`

## C.3 工程经验

| 问题 | 解决 |
| :--- | :--- |
| 收敛慢 / 奖励震荡 | 加 advantage normalization |
| 梯度爆炸 | 梯度裁剪 `nn.utils.clip_grad_norm_(policy.parameters(), 0.5)` |
| 收敛到次优 | 加熵正则 `loss -= 0.01 * dist.entropy()` |
| 大方差 | 用 baseline / Actor-Critic（Ch6） |
| 数据效率低 | 用 PPO / DDPG 等 off-policy 方法（Ch8/Ch9） |

## C.4 熵正则（Entropy Regularization）

为防止策略过早确定（mode collapse），常加一个熵奖励项：
$$\mathcal{L} = -\sum_t \log \pi_\theta(a_t \mid s_t) A_t - \beta \cdot H(\pi_\theta(\cdot \mid s_t))$$

其中 $H(\pi) = -\sum_a \pi(a) \log \pi(a)$ 是熵。$\beta$ 通常取 0.001-0.01。

**直觉**：策略越随机熵越大，鼓励熵 = 鼓励探索。

---

# §D 章末速查

## D.1 关键公式速记

| # | 公式 | 含义 |
| :---: | :--- | :--- |
| 1 | $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$ | 目标函数 |
| 2 | $\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ | log-derivative 化简 |
| 3 | $\nabla_\theta J = \mathbb{E}[\sum_t \nabla \log \pi_\theta(a_t \mid s_t) \cdot G_t]$ | PG 定理 |
| 4 | $\mathbb{E}_a[\nabla \log \pi(a \mid s) b(s)] = 0$ | baseline 减方差 |
| 5 | $A^\pi = Q^\pi - V^\pi$ | 优势函数（Ch6 起核心） |

## D.2 常见面试题

**Q1：策略梯度定理的核心 trick 是什么？**
- log-derivative trick（score function trick）
- $\nabla p = p \nabla \log p$，把"对分布求导"变成"对 log-概率求导"
- 关键好处：$\log p_\theta(\tau)$ 中环境项 $P, p(s_0)$ 自动消失，**不需要 model**

**Q2：为什么减 baseline 能降方差但不改变期望？**
- $\mathbb{E}_a[\nabla \log \pi(a \mid s) b(s)] = 0$（积分中 $b(s)$ 是常数）
- baseline 与 reward 越相关（如 $V^\pi$），方差降得越多

**Q3：REINFORCE 与 supervised learning 的关系？**
- 形式上几乎一样：$\sum_t \log \pi_\theta(a_t \mid s_t)$ 就是 NLL
- 区别：监督学习每个样本权重为 1，REINFORCE 权重为 $G_t$（reward 加权 NLL）
- 这是为什么 RLHF/DPO 看起来"像监督学习"——本质是带权重的 MLE

**Q4：什么时候 PG 比 Q-Learning 好？**
- 连续动作空间（PG 直接输出动作）
- 需要随机策略（部分可观测、博弈论场景）
- 大动作空间（不需要遍历所有 a）

**Q5：PG 与监督学习的关键差异？**
- 数据分布依赖于 θ（policy 一变，数据分布就变）
- 这导致 PG 是 **non-stationary optimization**——优化目标本身随训练变化
- 这也是 PG 难训的根本原因

---

## 承上启下

REINFORCE 给出了"对策略求梯度"的范式，但实战缺陷明显：
- 必须等 episode 结束（与 Ch3 MC 的痛点完全相同）
- 即使有 baseline，方差仍较大

下一章 **Ch6 Actor-Critic** 解决这两个问题：
- 用 Critic（V 网络）作为可学习的 baseline
- 用 TD-style bootstrap 替代 MC-style $G_t$
- 把 reward 信号"局部化"到每一步

这套思想最终发展成 **GAE**（Generalized Advantage Estimation），是 PPO 的核心组件。
