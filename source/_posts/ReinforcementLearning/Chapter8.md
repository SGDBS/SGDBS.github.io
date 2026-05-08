---
title: RL Chapter8 TRPO 与 PPO：信赖域与策略梯度的工程化
categories: 学习笔记-强化学习
date: 2026-05-16 10:00:00
mathjax: true
tags:
    - AI
    - 强化学习
    - AI面试知识
---

> **本章定位**：本系列最关键的一章。TRPO 用 trust region 思想第一次稳定了 policy gradient 训练；PPO 把 trust region 工程化为简单的 clip 操作，成为 RLHF 时代的"瑞士军刀"。**理解 PPO 就理解了《学习笔记-大模型》Ch6 的 RLHF 核心**。

> **承上**：Ch5 PG 定理 + Ch6 GAE。
> **启下**：Ch9 SAC 给连续控制最强基线；Ch11 引出 offline RL 与 DPO 的关系。

---

# §A 数学原理

## 1. 为什么 PG 训练这么不稳？

回忆 Ch5/Ch6 的痛点：策略梯度方向是对的，但**步长无法控制**。
- 步长太小 → 学得慢
- 步长太大 → 策略一步走偏，后续轨迹质量崩坏
- **崩了之后无法恢复**——因为坏策略采样到的全是坏数据

> **核心问题**：PG 是 non-stationary optimization——当前梯度只在当前策略附近"近似有效"，远离当前策略后，梯度估计全部失效。

**解法思路**：限制每次更新的"策略变化幅度"，确保新策略 $\pi_{\theta_{\text{new}}}$ 与旧策略 $\pi_{\theta_{\text{old}}}$ 不要差太远。这就是 **Trust Region**（信赖域）思想。

## 2. TRPO：单调改进定理

### 2.1 期望回报的差分表达

**关键引理**（Kakade & Langford 2002）：
$$J(\pi') - J(\pi) = \mathbb{E}_{s \sim d^{\pi'}, a \sim \pi'}[A^\pi(s, a)]$$

其中 $d^{\pi'}$ 是新策略下的状态访问分布。

**直观**：新策略的提升 = 在新策略访问的状态分布下，新策略选的 action 在**旧策略 advantage** 上的期望。

### 2.2 局部近似（关键近似）

上式中 $d^{\pi'}$ 难以计算（依赖未知的 $\pi'$）。TRPO 用**旧策略的状态分布** $d^\pi$ 近似：

$$\tilde{J}(\pi') = J(\pi) + \mathbb{E}_{s \sim d^\pi, a \sim \pi'}[A^\pi(s, a)]$$

接着用重要性采样把 $a \sim \pi'$ 转回 $a \sim \pi$：

$$\tilde{J}(\pi') = J(\pi) + \mathbb{E}_{s, a \sim \pi}\left[\frac{\pi'(a \mid s)}{\pi(a \mid s)} A^\pi(s, a)\right]$$

记 $r(s, a) = \pi'(a \mid s) / \pi(a \mid s)$ 为 **importance ratio**。

### 2.3 单调改进定理

**定理**（Schulman et al. 2015）：
$$J(\pi') \geq \tilde{J}(\pi') - C \cdot D_{KL}^{\max}(\pi \| \pi')$$

其中 $C$ 是与 $\gamma$ 和 reward 范围相关的常数，$D_{KL}^{\max}$ 是 $\pi, \pi'$ 在所有状态下 KL 散度的最大值。

**含义**：只要新策略 $\pi'$ 在 KL 意义下离 $\pi$ 不远，**$\tilde{J}(\pi')$ 是 $J(\pi')$ 的下界**。优化下界 → 保证真实 $J$ 也在变好（**单调改进**）。

### 2.4 TRPO 的优化形式

把上面变成约束优化：
$$\max_{\pi'} \mathbb{E}_{s, a \sim \pi}\left[r(s, a) A^\pi(s, a)\right] \quad \text{s.t.} \quad \mathbb{E}_{s \sim \pi}[D_{KL}(\pi(\cdot \mid s) \| \pi'(\cdot \mid s))] \leq \delta$$

实际中用 mean KL（而非 max KL）作为约束，$\delta \approx 0.01$。

### 2.5 TRPO 的工程难点

求解上面的约束优化需要：
1. 计算 KL 约束的 Hessian（Fisher 信息矩阵 $F$）
2. 用共轭梯度求解 $F^{-1} g$（natural gradient）
3. 用 line search 确保 KL 约束满足

→ **二阶优化**，实现复杂、训练慢。

## 3. PPO：把 Trust Region 一阶化

PPO（Schulman 2017）的目标：保留 trust region 思想，但只用一阶优化器（Adam）。

### 3.1 PPO-Penalty（早期版本，已少用）

把 KL 约束变成软惩罚：
$$\mathcal{L}^{\text{KLPEN}} = \mathbb{E}\left[\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} A^{\pi_{\text{old}}} - \beta D_{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta)\right]$$

$\beta$ 自适应调整（KL 大就增大 β）。但效果不稳定。

### 3.2 PPO-Clip（主流版本）

**核心思路**：直接限制 importance ratio 的幅度。

$$\boxed{\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left( r_t(\theta) \hat{A}_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)\right]}$$

其中：
- $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$
- $\epsilon = 0.1$ 或 $0.2$
- $\hat{A}_t$ 用 GAE 估计（Ch6 §A.3.4）

### 3.3 PPO-Clip 的几何理解

考虑两种情况：

**情况 1：$\hat{A}_t > 0$（动作好，想增大概率）**
- $r_t < 1 + \epsilon$：自由更新
- $r_t \geq 1 + \epsilon$：clip 截断梯度，**不再奖励"大幅推高"**

**情况 2：$\hat{A}_t < 0$（动作差，想降低概率）**
- $r_t > 1 - \epsilon$：自由更新
- $r_t \leq 1 - \epsilon$：clip 截断梯度，**不再惩罚"大幅压低"**

**核心保护**：每次更新只允许 $\pi_\theta$ 在 ratio $\in [1-\epsilon, 1+\epsilon]$ 内变化。

### 3.4 完整 PPO 损失

$$\mathcal{L}^{\text{PPO}} = \mathcal{L}^{\text{CLIP}} - c_v \mathcal{L}^{\text{VF}} + c_h \mathbb{E}[H(\pi_\theta)]$$

其中：
- $\mathcal{L}^{\text{VF}} = (V_\phi(s_t) - V_t^{\text{target}})^2$（Critic 损失）
- $\mathbb{E}[H(\pi_\theta)]$（熵正则，鼓励探索）
- $c_v = 0.5$，$c_h = 0.01$

### 3.5 PPO 的核心 trick：minibatch 多轮更新

每次收集一批 rollout 数据后，**用同一批数据训练 K 轮**（通常 K=4），每轮内部分成多个 minibatch。

为什么可以？因为：
- $\theta_{\text{old}}$ 固定（rollout 时的策略）
- ratio $r_t$ 自然把"采样分布的差异"修正回来
- clip 保证 $\theta$ 不会跑太远

**这让 PPO 数据效率显著优于 vanilla PG**。

## 4. PPO 在 LLM 中的应用（RLHF）

把 PPO 用在 LLM 上有几个特殊性：

| 特性 | LLM 与经典 RL 的差异 |
| :--- | :--- |
| **状态** | 完整 prompt + 已生成 token |
| **动作** | 词表中的下一个 token（动作空间几万-几十万） |
| **轨迹长度** | 几十到几千 token |
| **奖励** | 仅在 EOS 时由 RM 给一个标量 |
| **per-token KL** | KL penalty 加在每个 token 上，作为 reward shaping |

详见《学习笔记-大模型》Ch6（PPO RLHF）。

---

# §B 模型架构

## B.1 PPO 数据流（重要！）

```
┌──────────────────────────────────────────────────────────┐
│ Phase 1: Rollout（采样数据，无梯度）                      │
│   for t = 1...T:                                          │
│     a_t ~ π_θ_old(·|s_t)                                  │
│     log_prob_old, value_old = policy_old.act(s_t)         │
│     s_{t+1}, r_t = env.step(a_t)                          │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│ Phase 2: 计算 GAE                                         │
│   advantages, returns = compute_gae(rewards, values, ...) │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│ Phase 3: K 轮 minibatch 更新                              │
│   for epoch = 1...K:                                      │
│     for batch in shuffle(dataset):                        │
│       log_prob_new = π_θ.log_prob(a_t)                    │
│       ratio = exp(log_prob_new - log_prob_old)            │
│       loss = -min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)       │
│           + c_v · (V(s_t) - returns)²                     │
│           - c_h · entropy                                 │
│       loss.backward(); optim.step()                       │
└──────────────────────────────────────────────────────────┘
```

## B.2 PPO 完整 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from torch.optim import Adam

class ActorCritic(nn.Module):
    """Categorical Actor + V Critic"""
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.shared(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, obs):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        log_prob = dist.log_prob(a)
        return a, log_prob, value

    def evaluate(self, obs, action):
        """给定 obs 和 action，返回 (log_prob, value, entropy)"""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value, entropy


def ppo_train(env_name="CartPole-v1", n_steps=int(2e5), n_envs=8,
              rollout_len=128, n_epochs=4, n_minibatches=4,
              gamma=0.99, lam=0.95, eps_clip=0.2,
              lr=3e-4, c_v=0.5, c_h=0.01, max_grad_norm=0.5):

    envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(n_envs)])
    obs_dim = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n

    model = ActorCritic(obs_dim, n_actions)
    optim = Adam(model.parameters(), lr=lr)

    obs, _ = envs.reset()
    total_steps = 0

    while total_steps < n_steps:
        # ============ Phase 1: Rollout ============
        # 存 [T, n_envs, ...] 的数据
        obs_buf = torch.zeros(rollout_len, n_envs, obs_dim)
        act_buf = torch.zeros(rollout_len, n_envs, dtype=torch.long)
        logp_buf = torch.zeros(rollout_len, n_envs)
        rew_buf = torch.zeros(rollout_len, n_envs)
        val_buf = torch.zeros(rollout_len, n_envs)
        done_buf = torch.zeros(rollout_len, n_envs)

        for t in range(rollout_len):
            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                a, log_prob, value = model.act(obs_t)
            obs_new, r, terminated, truncated, _ = envs.step(a.numpy())
            done = np.logical_or(terminated, truncated).astype(np.float32)

            obs_buf[t] = obs_t
            act_buf[t] = a
            logp_buf[t] = log_prob
            val_buf[t] = value
            rew_buf[t] = torch.tensor(r, dtype=torch.float32)
            done_buf[t] = torch.tensor(done)

            obs = obs_new
            total_steps += n_envs

        # ============ Phase 2: GAE ============
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32)
            _, last_val = model.forward(obs_t)
            last_val = last_val.detach()

        advantages = torch.zeros(rollout_len, n_envs)
        last_gae = torch.zeros(n_envs)
        for t in reversed(range(rollout_len)):
            next_val = last_val if t == rollout_len - 1 else val_buf[t + 1]
            delta = rew_buf[t] + gamma * next_val * (1 - done_buf[t]) - val_buf[t]
            last_gae = delta + gamma * lam * (1 - done_buf[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf

        # 展平 [T, n_envs, ...] → [T*n_envs, ...]
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)

        # ============ Phase 3: K 轮 minibatch 更新 ============
        batch_size = rollout_len * n_envs
        mb_size = batch_size // n_minibatches
        idx = np.arange(batch_size)

        for epoch in range(n_epochs):
            np.random.shuffle(idx)
            for start in range(0, batch_size, mb_size):
                mb_idx = idx[start: start + mb_size]

                # 当前 policy 的 logp 和 value
                new_logp, new_val, entropy = model.evaluate(b_obs[mb_idx], b_act[mb_idx])

                # ⭐ Importance ratio
                ratio = torch.exp(new_logp - b_logp[mb_idx])

                # 标准化 advantage
                adv = b_adv[mb_idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # ⭐ Clipped surrogate
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss（带 clip 的版本，可选）
                v_loss = (new_val - b_ret[mb_idx]).pow(2).mean()

                # 熵正则
                entropy_loss = -entropy.mean()

                loss = actor_loss + c_v * v_loss + c_h * entropy_loss

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optim.step()

    return model
```

> 这段代码是 PPO 的"生产级最简"实现，约 100 行。CleanRL 的实现与之高度一致。

## B.3 PPO 在 LLM 上的对应（呼应 LLM 笔记 Ch6）

LLM RLHF 中 PPO 的差别（参见之前的《学习笔记-大模型》Ch6）：

```python
# 经典 PPO（本章）的 reward
rew_buf[t] = env.step(a)              # 来自环境

# LLM PPO 的 reward（per-token KL + 末尾 RM）
kl = log_prob_new - log_prob_ref       # per-token KL
reward = -beta * kl                    # 每 token 一个负 KL
reward[-1] += rm_score                 # 末尾加 RM 分
```

其余（GAE、clip、minibatch 多轮更新）完全一致。

---

# §C 训练与推理

## C.1 PPO 调参经验

| 参数 | 推荐值 | 备注 |
| :--- | :--- | :--- |
| `eps_clip` | 0.1-0.2 | 0.2 标配；连续控制可降到 0.1 |
| `n_epochs` | 4-10 | 多了会过拟合"旧分布" |
| `lam` | 0.95 | GAE 的 sweet spot |
| `gamma` | 0.99 | 视任务远视程度调 |
| `lr` | 3e-4 | Adam 默认 |
| `c_v` | 0.5 | 太大 critic 主导 |
| `c_h` | 0.01 | 复杂任务调到 0.001-0.05 |
| `max_grad_norm` | 0.5 | 必须有，否则会爆 |
| `rollout_len × n_envs` | ~2048 总样本 | 太小方差大、太大数据陈旧 |

## C.2 实验：CartPole + PPO

```python
model = ppo_train("CartPole-v1", n_steps=int(1e5))
```

典型结果：
- 1万步：reward = 50
- 3万步：reward = 200
- 5万步：reward = 500（满分）

PPO 在 CartPole 上的收敛速度通常比 A2C 快 1.5-2 倍，且更稳定。

## C.3 PPO 推理

PPO 推理时只用 Actor（与 A2C 一致），完全去掉 Critic：

```python
def inference(model, obs):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model.forward(obs_t)
        a = logits.argmax(dim=-1).item()                 # greedy
    return a
```

## C.4 工程经验

**1. Reward normalization**：复杂任务（MuJoCo）reward 尺度差异大，必须用 running mean/std 归一化。

**2. Observation normalization**：连续状态空间也用类似归一化。

**3. Clip 不只是 actor**：PPO2 还可以 clip critic 的 value 更新（防止 critic 学过头）。

**4. Hyper-parameter 互相影响**：`eps_clip` 大 → `n_epochs` 要小；反之亦然。

**5. KL 监控**：训练时打印 mean KL，若 KL 突然飙升（> 0.05），说明 clip 失效，需调小 lr。

---

# §D 章末速查

## D.1 关键公式

| # | 公式 | 含义 |
| :---: | :--- | :--- |
| 1 | $r_t(\theta) = \pi_\theta(a_t\|s_t) / \pi_{\theta_{\text{old}}}(a_t\|s_t)$ | importance ratio |
| 2 | $\mathcal{L}^{\text{CLIP}} = \min(r_t \hat{A}_t, \text{clip}(r_t, 1\pm\epsilon)\hat{A}_t)$ | PPO 目标 |
| 3 | $J(\pi') - J(\pi) \approx \mathbb{E}_{s, a \sim \pi}[r(s,a) A^\pi(s,a)]$ | 单调改进引理 |
| 4 | $J(\pi') \geq \tilde{J}(\pi') - C \cdot D_{KL}^{\max}$ | TRPO 下界 |
| 5 | $\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$ | GAE 递推（Ch6） |

## D.2 常见面试题

**Q1：PPO 的 clip 在数学上等价于 trust region 吗？**
- **不严格等价**，只是工程化的近似
- TRPO 用 KL 硬约束 + 二阶优化
- PPO 用 ratio clip + 一阶优化
- 但在大多数实验上 PPO ≥ TRPO，且简单很多

**Q2：为什么 PPO 能用同一批数据训 K 轮？**
- $\theta_{\text{old}}$ 固定时，importance ratio $r_t$ 自然修正分布偏差
- clip 保证 $\theta$ 不会跑远
- K 轮内策略仍在"trust region"内，数据有效
- K 太大（>10）会过拟合旧分布——KL 飙升

**Q3：PPO 与 A2C 的关系？**
- PPO = A2C + importance ratio + clip + 多轮更新
- 数据效率：PPO ≫ A2C
- 稳定性：PPO ≫ A2C（clip 保护）
- 实现复杂度：PPO 略高

**Q4：PPO 为什么在 LLM RLHF 中是首选？**
- 离散动作空间（token），PPO 天然支持
- 训练稳定（KL clip 防策略崩坏）
- 数据效率（rollout 一次更新多轮）
- 工程社区成熟（TRL/DeepSpeed 等都有现成实现）

**Q5：PPO 的"Proximal"在哪？**
- $\epsilon$-clip 限制策略变化幅度
- 让 $\pi_\theta$ 始终"接近 (proximal to)" $\pi_{\theta_{\text{old}}}$
- 这是 trust region 思想的本质

---

## 承上启下

PPO 是 RL 与 LLM 时代的桥梁：
- 经典 RL 中用它处理离散/连续控制
- LLM RLHF 中用它把模型对齐到人类偏好

下一章 **Ch9** 探讨**连续控制的另一条路**——DDPG 与 SAC。在机器人、自动驾驶等连续动作场景，SAC 是当前 SOTA。

理解了 PPO + SAC，你就掌握了现代 RL 的两大主流路径。
