---
title: RL Chapter11 离线强化学习：BCQ、CQL、IQL 与 DPO 的渊源
categories: 学习笔记-强化学习
date: 2026-05-19 10:00:00
mathjax: true
tags:
    - AI
    - 强化学习
    - AI面试知识
---

> **本章定位**：和前面所有章节（online RL）正相反——**完全不与环境交互，只用预先收集的固定数据集训练**。这是医疗、自动驾驶、推荐系统的现实约束，也是 **DPO 的数学根基**。理解 offline RL，就理解了为什么 DPO 能存在。

> **承上**：Ch4 Q-Learning + Ch9 DDPG/SAC（online 的对照组）。
> **启下**：Ch13 直接展开 DPO 与 GRPO。

---

# §A 数学原理

## 1. Offline RL 问题设定

给定一个**固定的数据集** $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N$（来自某个**未知**的 behavior policy $\pi_\beta$），目标：在不与环境交互的前提下，从 $\mathcal{D}$ 学一个尽量好的 $\pi_\theta$。

> **关键差异**：
> - **Online RL**：策略可以采样新数据
> - **Off-policy RL**（DQN/SAC）：用 replay buffer 但仍 online 采样
> - **Offline RL**：**永远不能采样新数据**

## 2. 核心难题：分布外 (OOD) 动作高估

### 2.1 问题诊断

朴素地用 Q-Learning（Ch4 §A.3）on offline data 会**剧烈失败**。原因：
- TD target 中 $\max_{a'} Q(s', a')$ 选了一个"分布外"的动作
- $Q(s', a_{\text{OOD}})$ 是 Q 网络的"幻想估计"——没有数据支持
- 这个 $Q_{\text{OOD}}$ 通常被显著**高估**
- TD 把这个高估反馈到 $Q(s, a)$ → 越学越离谱
- 没有 online 采样的"现实校正"，错误持续累积

### 2.2 BCQ (Batch-Constrained Q-Learning, 2019)

**第一个 offline RL 算法**。核心思想：**只考虑数据集中出现过的动作**。

构造一个生成模型 $G_\omega(s)$ 拟合 $\pi_\beta$（VAE 或 GAN），TD target 改为：
$$y = r + \gamma \max_{a' \in G_\omega(s')} Q(s', a')$$

即只在"看似来自数据"的动作上取 max。强制 policy 不能太离谱。

## 3. CQL (Conservative Q-Learning, 2020)

### 3.1 思想

不是约束 policy，而是**在 Q 函数训练时压低 OOD 动作的 Q 值**。

修改 Q 损失：
$$\mathcal{L}_Q = \underbrace{\mathbb{E}_{(s, a, r, s') \sim \mathcal{D}}\left[(Q(s, a) - y)^2\right]}_{\text{标准 TD}} + \alpha \underbrace{\left( \mathbb{E}_{s \sim \mathcal{D}, a \sim \mu}[Q(s, a)] - \mathbb{E}_{(s, a) \sim \mathcal{D}}[Q(s, a)] \right)}_{\text{保守正则}}$$

**直觉**：
- 第一项：标准 TD 损失（让 Q 准确）
- 第二项：让"任意分布 $\mu$ 下"的 Q 比"数据中"的 Q 低
- 实际中 $\mu$ 选 uniform 或 softmax(Q)，强制压低未见过动作的 Q

### 3.2 优势

- 不需要训生成模型（比 BCQ 简单）
- 理论保证：学到的 Q 是真 $Q^\pi$ 的下界

## 4. IQL (Implicit Q-Learning, 2021)

### 4.1 一个革命性洞察

CQL 和 BCQ 都还在用 $\max_a$ —— 只是想办法绕过它的危害。**能否完全不用 max？**

IQL 的答案：**用 expectile 回归代替 max**。

### 4.2 Expectile 回归

定义 expectile loss（$\tau \in (0, 1)$ 是分位）：
$$L^\tau(u) = \begin{cases} (1 - \tau) u^2 & u \leq 0 \\ \tau u^2 & u > 0 \end{cases}$$

- $\tau = 0.5$：标准 MSE，回归均值
- $\tau \to 1$：回归"上分位数"，越接近 max

### 4.3 IQL 的三步

**Step 1**：训练 V 网络拟合 Q 的高分位（替代 max）
$$\mathcal{L}_V = \mathbb{E}_{(s, a) \sim \mathcal{D}}\left[L^\tau\left(Q_{\hat\phi}(s, a) - V_\psi(s)\right)\right]$$
$\tau$ 通常取 0.7-0.9。

**Step 2**：训练 Q 网络（用 V 当 target）
$$\mathcal{L}_Q = \mathbb{E}\left[(Q_\phi(s, a) - (r + \gamma V_\psi(s')))^2\right]$$
**注意：完全没有 $\max$！**

**Step 3**：从 Q, V 提取 policy（advantage-weighted 行为克隆）
$$\mathcal{L}_\pi = -\mathbb{E}\left[\exp\left(\beta \cdot (Q(s, a) - V(s))\right) \log \pi_\theta(a \mid s)\right]$$

直觉：模仿数据中 $A = Q - V > 0$ 的动作（advantage 高的）。

### 4.4 IQL 为什么这么强？

- 完全 in-sample（不查询任何 OOD action）
- 训练稳定（无 max 的反馈循环）
- D4RL 上 SOTA

> **IQL 的精神** 与 DPO 极其相似：**直接从数据中学，不用 max 操作**。下一节展开。

## 5. Offline RL 与 DPO 的深层联系

### 5.1 KL-约束 RL 的闭式解（再次出现）

回忆《学习笔记-大模型》Ch7 §A.2：KL-约束 RL 目标
$$\max_\pi \mathbb{E}[r] - \beta D_{KL}(\pi \| \pi_{\text{ref}})$$
有闭式解
$$\pi^*(a \mid s) \propto \pi_{\text{ref}}(a \mid s) \exp(r(s, a) / \beta)$$

### 5.2 Advantage-Weighted Regression (AWR)

把闭式解写成"加权 BC"形式（用数据集中的 $a$）：
$$\mathcal{L}_{\text{AWR}} = -\mathbb{E}_{(s, a) \sim \mathcal{D}}\left[\exp(A(s, a) / \beta) \cdot \log \pi_\theta(a \mid s)\right]$$

这就是 IQL Step 3 的精神，也是 **offline RL 的"标准最终步"**。

### 5.3 DPO 的位置

DPO（《学习笔记-大模型》Ch7）的推导：
1. 写出 KL-约束 RL 闭式解 → ✓ AWR 也用了这一步
2. **反解 reward 用 log-ratio 表达**
3. 代入 Bradley-Terry → DPO loss

**关键差异**：
- AWR：需要从数据估计 $A(s, a)$，再做加权 BC
- DPO：用偏好对 $(y_w, y_l)$ **完全跳过 reward 估计**

> **本质**：DPO = "KL-约束 RL 闭式解" + "从偏好对反解 reward" + "BT 模型"。它是 **offline RL 在 LLM 偏好学习场景的特化**。

### 5.4 为什么 DPO 在 LLM 上 work 而 offline RL 在游戏上需要 CQL/IQL 这么麻烦？

- LLM 的"动作空间"是 token，但**reference policy** $\pi_{\text{ref}}$ 是预训练好的、覆盖大部分合理动作 → 没有"OOD"问题
- 经典 offline RL 的 behavior policy 通常很弱 → OOD 问题严重
- DPO 的"用 SFT 当 reference + KL 约束"自然规避了 OOD

---

# §B 模型架构

## B.1 IQL 完整 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

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


class VNet(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class Actor(nn.Module):
    """Gaussian Actor"""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        h = self.shared(obs)
        mu = self.mu(h)
        std = self.log_std.exp().expand_as(mu)
        return torch.distributions.Normal(mu, std)


def expectile_loss(diff, tau=0.7):
    """⭐ Expectile 回归损失"""
    weight = torch.where(diff > 0, tau, 1 - tau)
    return weight * diff.pow(2)


def train_iql(dataset, n_steps=int(1e6), batch_size=256,
              gamma=0.99, tau=0.7, beta=3.0, lr=3e-4, polyak=0.005):
    """
    dataset: D4RL 风格 dict {observations, actions, rewards, next_observations, terminals}
    """
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]

    actor = Actor(obs_dim, act_dim)
    q1 = QNet(obs_dim, act_dim)
    q2 = QNet(obs_dim, act_dim)
    v = VNet(obs_dim)
    q1_target = copy.deepcopy(q1)
    q2_target = copy.deepcopy(q2)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=lr)
    q1_optim = torch.optim.Adam(q1.parameters(), lr=lr)
    q2_optim = torch.optim.Adam(q2.parameters(), lr=lr)
    v_optim = torch.optim.Adam(v.parameters(), lr=lr)

    for step in range(n_steps):
        # ============ 采样 batch ============
        idx = np.random.choice(len(dataset['observations']), batch_size)
        s = torch.tensor(dataset['observations'][idx], dtype=torch.float32)
        a = torch.tensor(dataset['actions'][idx], dtype=torch.float32)
        r = torch.tensor(dataset['rewards'][idx], dtype=torch.float32)
        s_new = torch.tensor(dataset['next_observations'][idx], dtype=torch.float32)
        done = torch.tensor(dataset['terminals'][idx], dtype=torch.float32)

        # ============ Step 1: V 网络（expectile 回归到 Q）============
        with torch.no_grad():
            q_target_val = torch.min(q1_target(s, a), q2_target(s, a))
        v_pred = v(s)
        diff = q_target_val - v_pred
        v_loss = expectile_loss(diff, tau).mean()
        v_optim.zero_grad(); v_loss.backward(); v_optim.step()

        # ============ Step 2: Q 网络（TD with V，无 max！）============
        with torch.no_grad():
            q_target_y = r + gamma * (1 - done) * v(s_new)        # ⭐ 没有 max
        q1_loss = F.mse_loss(q1(s, a), q_target_y)
        q2_loss = F.mse_loss(q2(s, a), q_target_y)
        q1_optim.zero_grad(); q1_loss.backward(); q1_optim.step()
        q2_optim.zero_grad(); q2_loss.backward(); q2_optim.step()

        # ============ Step 3: Actor（advantage-weighted BC）============
        with torch.no_grad():
            q_val = torch.min(q1(s, a), q2(s, a))
            advantage = q_val - v(s)
            weight = torch.exp(beta * advantage).clamp(max=100.0)  # ⭐ 防爆

        # log π(a|s)
        dist = actor(s)
        log_prob = dist.log_prob(a).sum(dim=-1)
        actor_loss = -(weight * log_prob).mean()                   # ⭐ 加权 BC

        actor_optim.zero_grad(); actor_loss.backward(); actor_optim.step()

        # ============ 软更新 Target ============
        with torch.no_grad():
            for p, p_tgt in zip(q1.parameters(), q1_target.parameters()):
                p_tgt.data.mul_(1 - polyak).add_(p.data, alpha=polyak)
            for p, p_tgt in zip(q2.parameters(), q2_target.parameters()):
                p_tgt.data.mul_(1 - polyak).add_(p.data, alpha=polyak)

    return actor, q1, v
```

**理解 IQL 的关键三步**：
1. **V 用 expectile 回归到 Q**（替代 $\max$）
2. **Q 用 V 做 TD target**（完全 in-sample）
3. **Actor 做 advantage-weighted BC**（KL-约束 RL 的闭式解）

## B.2 与 DPO 的对照（呼应 LLM 笔记 Ch7）

```python
# IQL Actor loss
actor_loss = -(exp(beta * advantage) * log_prob).mean()

# DPO Loss（LLM 笔记 Ch7）
dpo_loss = -log_sigmoid(beta * (
    log_pi_chosen - log_pi_ref_chosen
    - log_pi_rejected + log_pi_ref_rejected
)).mean()
```

**结构相似性**：都是"基于偏好/优势的加权 NLL"。DPO 把"advantage"用偏好对的相对 log-ratio 表达，进一步省掉了 V 网络。

---

# §C 训练与推理

## C.1 实验：D4RL Benchmark

D4RL（Datasets for Deep Data-Driven Reinforcement Learning）是 offline RL 的标准 benchmark：

| 数据集 | 描述 |
| :--- | :--- |
| `halfcheetah-medium-v2` | 用半训练好的 SAC 收集的数据 |
| `halfcheetah-medium-replay-v2` | medium 训练过程的所有 replay buffer |
| `halfcheetah-medium-expert-v2` | medium + expert 的混合 |
| `halfcheetah-expert-v2` | 完美 SAC 的 rollout |

**典型结果**（normalized score，越大越好）：

| Algorithm | medium | medium-replay | medium-expert |
| :--- | :---: | :---: | :---: |
| BC | 42 | 36 | 56 |
| BCQ | 47 | 41 | 81 |
| CQL | 49 | 47 | 91 |
| **IQL** | **51** | **48** | **94** |
| Online SAC（参考） | — | — | 100 |

> IQL 在大多数 task 上是 SOTA，且**比 CQL 训练更稳**。

## C.2 推理视角

Offline RL 训完后推理与 online 完全一样：
```python
def inference(actor, obs):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        dist = actor(obs_t)
        a = dist.mean                                     # 取均值（确定性）
    return a.squeeze(0).numpy()
```

但**部署前必须充分评估**——因为没有 online 反馈，offline 模型的失败模式可能很微妙。

## C.3 何时用 Offline RL？

| 场景 | 适合 | 不适合 |
| :--- | :---: | :---: |
| 真机训练成本极高（机器人、自动驾驶） | ✓ | |
| 有大量历史数据（推荐、广告） | ✓ | |
| 需要安全性保证（医疗） | ✓ | |
| Reward 稠密、环境便宜（Atari） | | ✓（直接 online） |
| 数据非常少（< 1万 transition） | | ✓（offline 学不出来） |

## C.4 LLM 视角

LLM 的对齐过程**几乎全是 offline RL**：
- SFT = Behavior Cloning
- DPO = AWR 风格的 offline RL
- 即使 RLHF（PPO）的 reward 来自固定 RM，也带有强烈 offline 色彩

理解了本章，再回头看《学习笔记-大模型》Ch7 的 DPO 推导，会有"原来 DPO 不是凭空冒出来的"的体悟。

---

# §D 章末速查

## D.1 主要算法对比

| 算法 | 核心 | 特点 |
| :--- | :--- | :--- |
| **BC** | 直接模仿 | 最简单，分布偏移问题严重 |
| **BCQ** | 限制 action 在数据分布内 | 需要训生成模型 |
| **CQL** | 在 Q 上加保守正则 | 简单，理论保证 |
| **IQL** | Expectile 回归 + AWR | **完全 in-sample，最稳定** |
| **DPO** | KL-约束 RL 闭式解 + BT | LLM 专用 |

## D.2 关键公式

| # | 公式 | 含义 |
| :---: | :--- | :--- |
| 1 | $L^\tau(u) = \tau u^2 \cdot \mathbb{1}[u > 0] + (1-\tau) u^2 \cdot \mathbb{1}[u \leq 0]$ | Expectile loss |
| 2 | $\pi^*(a\|s) \propto \pi_\beta(a\|s) \exp(r/\beta)$ | KL-RL 闭式解 |
| 3 | $\mathcal{L}_{\text{AWR}} = -\mathbb{E}[\exp(A/\beta) \log \pi(a\|s)]$ | Advantage-weighted BC |
| 4 | $r(x, y) = \beta \log(\pi(y\|x)/\pi_{\text{ref}}(y\|x)) + \beta \log Z$ | DPO 反解 reward |

## D.3 常见面试题

**Q1：Offline RL 的核心难题是什么？**
- OOD action 高估
- 朴素 Q-Learning 会被 max 操作放大对未见过 action 的"幻想估计"
- 解决思路有三：限制 policy（BCQ）、压低 OOD 的 Q（CQL）、不用 max（IQL）

**Q2：IQL 的 expectile 回归在干什么？**
- 用一个连续可导的"近似 max"
- $\tau \to 1$ 越接近 max，但损失曲面更平滑
- 关键：**只在 in-sample 数据上回归**，不查询 OOD action

**Q3：DPO 是 offline RL 吗？**
- 是
- 数学上等价于"用偏好对实现的 AWR"
- 不需要 V 网络（用偏好对 $(y_w, y_l)$ 的相对 log-ratio 替代 advantage）

**Q4：为什么 LLM 上 DPO 比 PPO 更工业化主流？**
- LLM 的 reference policy（SFT 模型）很强 → OOD 问题轻
- 偏好对数据易收集（不需要 reward signal）
- 不需要 4 个模型同时训（PPO 显存压力大）
- 数据效率高（不需要 online 采样）

**Q5：BC vs IQL vs DPO 的关系？**
- BC：纯模仿，所有数据等权
- IQL：advantage-weighted BC（数据中"好"的 action 权重高）
- DPO：用偏好对实现的 IQL（连 advantage 都不用算）

---

## 承上启下

下一章 **Ch12** 走另一条 offline 路线：**模仿学习**。BC 是"无脑模仿"，DAgger 修复 BC 的"分布偏移"，GAIL 用对抗思想学 reward，IRL 反推 reward 函数。

模仿学习的精神 = SFT 的精神。理解模仿学习，就理解了为什么 SFT 是 RLHF 的第一步、它的局限在哪、以及 GAIL 风格的"对抗式偏好学习"如何帮助理解 DPO。
