---
title: RL Chapter12 模仿学习与逆 RL：BC、DAgger、GAIL、IRL
categories: 学习笔记-强化学习
date: 2026-05-20 10:00:00
mathjax: true
tags:
    - AI
    - 强化学习
    - AI面试知识
---

> **本章定位**：模仿学习从专家演示中学策略，**SFT 的 RL 视角即 BC**。本章串起 BC → DAgger → GAIL → IRL，最后回到 LLM：为什么 SFT 不够（BC 的分布偏移），为什么需要 RLHF（DPO 是 GAIL 风格的偏好学习）。

> **承上**：Ch11 offline RL 的极端案例（连 reward 都没有，只有专家轨迹）。
> **启下**：Ch13 RL × LLM 的整体回顾。

---

# §A 数学原理

## 1. 问题设定

给定专家演示数据集：
$$\mathcal{D}^E = \{\tau_i\}_{i=1}^N, \quad \tau_i = (s_0, a_0, s_1, a_1, \dots)$$

**目标**：学一个 $\pi_\theta$ 模仿专家。

**关键差异 vs RL**：
- 没有 reward signal（只有 (s, a) 对）
- 不能与环境交互（在最严格的设定下）

## 2. Behavior Cloning (BC)：最简单的模仿

### 2.1 公式

把模仿当作监督学习：
$$\mathcal{L}_{\text{BC}} = -\mathbb{E}_{(s, a) \sim \mathcal{D}^E}\left[\log \pi_\theta(a \mid s)\right]$$

直接在专家数据上做 NLL。这就是 **SFT 的本质**。

### 2.2 BC 的致命问题：分布偏移 (Distribution Shift / Compounding Errors)

**问题描述**：BC 在专家访问的状态分布 $d^E$ 上训练。但部署时：
- $\pi_\theta$ 不完美 → 第一步会有小偏差
- 偏差让 agent 进入 $d^E$ 没覆盖的状态
- 在新状态上 $\pi_\theta$ 表现更差 → 偏差累积
- **指数级误差累积**

### 2.3 数学分析（DAgger 论文，Ross 2011）

设 $\pi_\theta$ 在每个状态的错误率为 $\epsilon$，则在长度 $T$ 的轨迹上总误差：
$$\mathbb{E}[\text{total errors}] = O(T^2 \epsilon)$$

**注意是 $T^2$ 不是 $T$**——这就是 compounding errors。

直觉：每犯一次错就跳到一个更陌生的状态分布，后面错得更多。

### 2.4 应对 BC 的工程做法

| 方法 | 思路 |
| :--- | :--- |
| **数据增强** | 加噪声、扰动专家轨迹 |
| **更多专家数据** | 简单粗暴 |
| **DAgger** | 在线收集额外数据修正 |
| **GAIL** | 用对抗学习匹配状态分布 |

## 3. DAgger (Dataset Aggregation, 2011)

### 3.1 思想

让专家"伴飞"：
1. 用当前 $\pi_\theta$ rollout 收集状态
2. 在这些状态上**让专家给出 action** $a^E$
3. 把 $(s, a^E)$ 加到训练集
4. 重训 $\pi_\theta$
5. 反复

**关键**：训练分布逐渐覆盖 $\pi_\theta$ 实际访问的状态。

### 3.2 优劣

- ✅ 解决 BC 分布偏移
- ✅ 总误差从 $O(T^2 \epsilon)$ 降到 $O(T \epsilon)$
- ❌ 需要专家全程在线（昂贵 / 不可行）

LLM 类比：DAgger 不能直接用，但 **RLHF 的 iterative DPO**（呼应《学习笔记-大模型》Ch7 §C.5）就是 DAgger 思想——周期性地让模型自己生成 + 重新标注。

## 4. GAIL (Generative Adversarial Imitation Learning, 2016)

### 4.1 思想

不直接模仿 action，而是**让 $\pi_\theta$ 生成的 (s, a) 分布与专家分布一致**。借鉴 GAN：

- **Generator** = $\pi_\theta$
- **Discriminator** $D_\phi(s, a)$：判别 (s, a) 来自专家还是 $\pi_\theta$
- 训练目标：minimax

$$\min_\theta \max_\phi \mathbb{E}_{\pi^E}[\log D_\phi(s, a)] + \mathbb{E}_{\pi_\theta}[\log(1 - D_\phi(s, a))]$$

### 4.2 GAIL 的训练循环

```
for iter:
    1. Rollout: 用 π_θ 采样 (s, a) 对
    2. 训练 D：区分 expert vs π_θ
    3. 用 -log(1 - D) 作为 reward signal，PPO 更新 π_θ
    4. 重复
```

**$-\log(1 - D)$ 当 reward**：被判别为"像专家"的 (s, a) → reward 高 → policy 更倾向产生这种分布。

### 4.3 GAIL vs BC

| 维度 | BC | GAIL |
| :--- | :--- | :--- |
| 数据 | 仅专家 (s, a) | 专家数据 + 自己 rollout |
| 分布偏移 | 严重 | 自动校正 |
| 是否需要环境 | 不需要 | 需要 online 交互 |
| 是否需要 reward | 不需要 | 不需要 |
| 训练稳定性 | 高 | **低**（GAN 不稳） |

## 5. Inverse RL (IRL)：从演示反推 reward

### 5.1 问题

GAIL 跳过了 reward 学习。**IRL 直接学 reward 函数**：

> 给定专家演示，找一个 reward $r(s, a)$，使得在它下专家是最优策略。

### 5.2 数学

**Maximum Entropy IRL**（Ziebart 2008）：
$$P(\tau \mid r) = \frac{1}{Z(r)} \exp\left(\sum_t r(s_t, a_t)\right)$$

最大化专家轨迹的概率：
$$\max_r \mathbb{E}_{\tau \sim \mathcal{D}^E}[\log P(\tau \mid r)]$$

这等价于 RL 中的"最大熵 RL"（Ch9 SAC 的母体框架）。

### 5.3 与 RLHF 的关系

**关键观察**：
- RLHF 的 RM = IRL 学到的 reward
- 但 RLHF 用**偏好对**（"哪个更好"）学 RM
- IRL 用**完整专家轨迹**学 reward

偏好对的好处：
- 人类更擅长比较而非打分
- 更鲁棒（绝对 reward 容易标注不一致）

## 6. 现代统一视角

### 6.1 模仿学习的统一框架

把所有方法看作**最小化某种"分布距离"**：

| 方法 | 距离 | 工具 |
| :--- | :--- | :--- |
| **BC** | $D_{KL}(\pi^E(\cdot \mid s) \| \pi_\theta(\cdot \mid s))$ | 监督学习 |
| **DAgger** | KL on $\pi_\theta$ 访问的状态 | online 修正 |
| **GAIL** | JS divergence on (s, a) 分布 | GAN |
| **AIRL** | KL via reward + soft Bellman | GAN + IRL |

### 6.2 与 LLM 对齐的对应

| RL 模仿学习 | LLM 对齐 |
| :--- | :--- |
| **BC** | **SFT** |
| **DAgger** | **Iterative DPO / Self-rewarding** |
| **GAIL** | （间接对应）某些 RLHF 设计 |
| **IRL** | **RM 训练**（学 reward） |
| **MaxEnt IRL** | **DPO**（最大熵 + KL 约束） |

> **RLHF 的完整 pipeline 在数学上是 IRL + RL 的二阶段实现**：先学 reward（RM），再优化策略（PPO/DPO）。

---

# §B 模型架构

## B.1 BC 的 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BCAgent(nn.Module):
    """连续动作 BC，输出 Gaussian 策略"""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim * 2),               # μ + log_std
        )

    def forward(self, obs):
        out = self.net(obs)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-20, 2)
        return torch.distributions.Normal(mu, log_std.exp())


def train_bc(expert_data, n_epochs=100, batch_size=256, lr=1e-3):
    """
    expert_data: dict {observations, actions}
    """
    obs_dim = expert_data['observations'].shape[1]
    act_dim = expert_data['actions'].shape[1]

    agent = BCAgent(obs_dim, act_dim)
    optim = torch.optim.Adam(agent.parameters(), lr=lr)

    obs_data = torch.tensor(expert_data['observations'], dtype=torch.float32)
    act_data = torch.tensor(expert_data['actions'], dtype=torch.float32)
    n = len(obs_data)

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            obs = obs_data[idx]
            act = act_data[idx]

            dist = agent(obs)
            log_prob = dist.log_prob(act).sum(dim=-1)
            loss = -log_prob.mean()                       # ⭐ NLL

            optim.zero_grad()
            loss.backward()
            optim.step()

    return agent
```

> **BC = SFT**：把 obs 换成 prompt token、act 换成 next token，这就是《学习笔记-大模型》Ch5 的 SFT。

## B.2 DAgger 的关键代码

```python
def train_dagger(expert_policy, env, n_iters=20, n_rollouts_per_iter=10):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = BCAgent(obs_dim, act_dim)
    optim = torch.optim.Adam(agent.parameters(), lr=1e-3)

    # 初始数据集 = 专家直接 rollout
    obs_dataset, act_dataset = collect_expert_rollouts(expert_policy, env)

    for it in range(n_iters):
        # ============ 标准 BC 训练 ============
        train_bc({
            'observations': obs_dataset,
            'actions': act_dataset,
        })

        # ============ ⭐ DAgger 关键步骤 ============
        # 用当前 agent rollout
        new_obs = []
        for _ in range(n_rollouts_per_iter):
            obs, _ = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    a = agent(torch.tensor(obs).float()).mean
                obs_new, r, terminated, truncated, _ = env.step(a.numpy())
                new_obs.append(obs)
                obs = obs_new
                done = terminated or truncated

        # ⭐ 让专家给这些 obs 标注 action
        new_acts = [expert_policy(o) for o in new_obs]

        # ⭐ 数据聚合
        obs_dataset = np.concatenate([obs_dataset, np.array(new_obs)])
        act_dataset = np.concatenate([act_dataset, np.array(new_acts)])

    return agent
```

## B.3 GAIL 的关键架构

```python
class Discriminator(nn.Module):
    """判别 (s, a) 是 expert 还是 policy"""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return torch.sigmoid(self.net(x))                 # 概率


def gail_train_step(agent, discriminator, expert_data, env_data,
                    d_optim, agent_optim):
    # ============ Step 1: 训练 Discriminator ============
    expert_logits = discriminator(expert_data['obs'], expert_data['act'])
    policy_logits = discriminator(env_data['obs'], env_data['act'])

    d_loss = -(torch.log(expert_logits + 1e-8).mean() +
               torch.log(1 - policy_logits + 1e-8).mean())
    d_optim.zero_grad(); d_loss.backward(); d_optim.step()

    # ============ Step 2: 用 -log(1 - D) 作 reward ============
    with torch.no_grad():
        d_score = discriminator(env_data['obs'], env_data['act'])
        reward = -torch.log(1 - d_score + 1e-8).squeeze(-1)

    # 把 reward 注入 PPO/SAC 训练 agent
    # ... (调用 PPO/SAC 训练，把 env reward 替换为 GAIL reward)
```

---

# §C 训练与推理

## C.1 实验：MuJoCo 上的 BC vs DAgger vs GAIL

```python
# 假设 expert_policy 是预训练好的 SAC agent
expert_data = collect_rollouts(expert_policy, env, n_episodes=100)

# 方法 1: BC（baseline）
bc_agent = train_bc(expert_data, n_epochs=100)

# 方法 2: DAgger
dagger_agent = train_dagger(expert_policy, env, n_iters=20)

# 方法 3: GAIL
gail_agent = train_gail(expert_data, env, n_steps=int(1e6))
```

**典型结果**（HalfCheetah，专家 reward = 12000）：

| 方法 | 100 episodes | 1000 episodes | 10000 episodes |
| :--- | :---: | :---: | :---: |
| BC | 4000 | 8000 | 11000 |
| DAgger | 6000 | 11500 | 11900 |
| GAIL | 5000 | 10000 | 12000 |

> **观察**：
> - 数据少时 DAgger 更好（专家在线修正）
> - 数据足时三者差不多
> - **GAIL 最终效果最佳**（学到与专家同分布的策略）

## C.2 推理视角：模仿学习 vs RL

| 维度 | 模仿学习 | RL |
| :--- | :--- | :--- |
| **是否需要 reward** | 不需要 | 需要 |
| **是否需要专家** | 需要 | 不需要 |
| **超过专家** | **难** | 可以 |
| **训练数据** | 专家轨迹 | 自己 rollout |

模仿学习的天花板 = 专家水平。**想超过专家必须走 RL**——这是为什么 SFT 之后还要 RLHF/DPO。

## C.3 LLM 中的对应关系（关键）

| LLM 步骤 | 对应模仿学习 |
| :--- | :--- |
| **预训练** | 不属于（自监督） |
| **SFT** | **BC** |
| **RM 训练** | **IRL**（从偏好反推 reward） |
| **PPO RLHF** | RM-based RL |
| **DPO** | **AWR / GAIL 风格**（直接从偏好对优化策略） |
| **RLAIF** | DAgger 风格（AI 在线生成额外标注） |

理解这张表，整个 LLM 训练流程就在 RL 框架里清晰了。

---

# §D 章末速查

## D.1 主要方法速记

| 方法 | 核心 | 关键限制 |
| :--- | :--- | :--- |
| **BC** | 监督学习 NLL | 分布偏移（compounding errors） |
| **DAgger** | 在线收集 + 专家修正 | 需要专家全程在线 |
| **GAIL** | GAN 风格分布匹配 | GAN 训练不稳 |
| **IRL** | 反推 reward + RL | 计算昂贵 |
| **MaxEnt IRL** | 加熵正则的 IRL | DPO 的数学根基 |

## D.2 关键公式

| # | 公式 | 含义 |
| :---: | :--- | :--- |
| 1 | $\mathcal{L}_{\text{BC}} = -\mathbb{E}_{\mathcal{D}^E}[\log \pi_\theta(a\|s)]$ | BC = SFT 损失 |
| 2 | $\mathbb{E}[\text{errors}] = O(T^2 \epsilon)$ | BC 分布偏移 |
| 3 | $\min_\theta \max_\phi \mathbb{E}_{\pi^E}[\log D] + \mathbb{E}_{\pi_\theta}[\log(1-D)]$ | GAIL 目标 |
| 4 | $P(\tau\|r) \propto \exp(\sum r)$ | MaxEnt IRL |

## D.3 常见面试题

**Q1：BC 的 compounding error 在数学上为什么是 $O(T^2)$？**
- 每一步错误率 $\epsilon$
- 一旦走出专家分布，下一步误差至少 $\epsilon$（可能更大）
- $T$ 步累积：$\sum_t (T - t) \epsilon = O(T^2 \epsilon)$

**Q2：DAgger 为什么能把误差降到 $O(T)$？**
- 每轮 DAgger 把 $\pi_\theta$ 实际访问的状态加入训练
- 训练分布逐渐覆盖部署分布
- 单步误差仍是 $\epsilon$，但累积不再爆炸

**Q3：GAIL 与 BC 的本质区别？**
- BC：匹配 $\pi_\theta(a\|s) \approx \pi^E(a\|s)$（条件分布）
- GAIL：匹配 $d^{\pi_\theta}(s, a) \approx d^{\pi^E}(s, a)$（联合分布）
- GAIL 更鲁棒（看分布而非单点）

**Q4：DPO 在哪个意义上是 GAIL 的近亲？**
- 都是 "不显式估 reward 的对抗式学习"
- DPO 用偏好对（GAIL 用 expert vs policy）
- DPO 用 BT 模型 + log-ratio（GAIL 用 GAN discriminator）

**Q5：为什么 SFT 不够，必须 RLHF？**
- SFT = BC → 有 compounding errors
- SFT 学不到"什么是更好的"（只学到"标注者怎么写"）
- SFT 的天花板 = 标注质量；RLHF 可超过标注水平

---

## 承上启下

到此 RL 的所有主要范式已经讲完：
- **Value-based**：Q-Learning (Ch4) → DQN (Ch7)
- **Policy-based**：REINFORCE (Ch5) → AC (Ch6) → PPO (Ch8)
- **Hybrid**：DDPG / SAC (Ch9)
- **Offline**：BCQ / CQL / IQL (Ch11)
- **模仿**：BC / DAgger / GAIL / IRL (Ch12)
- **横切**：探索 (Ch10)

下一章 **Ch13** 是这套笔记的"压轴"——把所有 RL 概念在 LLM 时代的对应找回来，详细对比 PPO、DPO、GRPO 的数学渊源，理清 R1 / o1 / Claude 的算法选择。
