---
title: RL Chapter7 DQN 家族：深度学习接入 Q-Learning
categories: 学习笔记-强化学习
date: 2026-05-15 10:00:00
mathjax: true
tags:
    - AI
    - 强化学习
    - AI面试知识
---

> **本章定位**：把 Ch4 的 Q-Learning 从表格法升级到神经网络，进入"深度强化学习"时代。DQN（DeepMind 2013/2015）是 RL 史上的里程碑——首次在大状态空间（Atari 像素输入）上达到人类水平。

> **承上**：Ch4 Q-Learning 公式 + Ch1 Bellman 最优方程。
> **启下**：DQN 解决离散动作，Ch8/Ch9 处理连续动作和稳定性。

---

# §A 数学原理

## 1. 函数逼近的挑战

### 1.1 表格法的死路

表格 Q 在 Atari 上不可行：
- 状态 = 84×84 灰度图 = $256^{7056}$ 种可能
- 不可能枚举存储

**自然的想法**：用神经网络 $Q_\theta(s, a)$ 逼近 $Q$。但简单地把 Q-Learning 加上神经网络会发散。三大致命问题：

| 问题 | 原因 | 解决方案 |
| :--- | :--- | :--- |
| **样本高度相关** | 连续轨迹中 $s_t, s_{t+1}$ 高度相似 | **Replay Buffer** |
| **目标非平稳** | $\max_{a'} Q_\theta(s', a')$ 中 $\theta$ 自身在变 | **Target Network** |
| **过估计** | $\max$ 操作放大噪声 | **Double DQN** |

### 1.2 训练目标

DQN 的损失（最简形式）：
$$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s')}\left[\left(Q_\theta(s, a) - \underbrace{(r + \gamma \max_{a'} Q_\theta(s', a'))}_{\text{TD target}}\right)^2\right]$$

这是一个**回归问题**：让 $Q_\theta(s, a)$ 拟合 TD target。

## 2. Replay Buffer：解决样本相关性

### 2.1 思想

存储所有过往交互 $(s, a, r, s', \text{done})$ 到一个固定大小（如 $10^6$）的环形缓冲区。每次更新从中**随机采样 minibatch**。

### 2.2 为什么有效？

1. **打破时序相关性**：连续状态在 minibatch 内被打散，更接近 i.i.d.
2. **重复利用数据**：每个样本被使用多次（数据效率高）
3. **稳定训练分布**：避免"策略一变，训练分布就变"的问题

### 2.3 Q-Learning 的 off-policy 性使其可行

Replay Buffer 中的数据来自旧策略，但 Q-Learning 的目标 $\max_{a'} Q$ 不依赖采样策略——所以历史数据仍可用。

> **这是 Q-Learning off-policy 优势的最重要体现**：PG 系列（Ch5/6）on-policy，没法用 Replay Buffer。

## 3. Target Network：解决目标非平稳

### 3.1 问题诊断

朴素 DQN 的 TD target 中 $\max_{a'} Q_\theta(s', a')$ 用的是当前 $\theta$。问题：
- 我们刚把 $Q_\theta(s, a)$ 推高
- 但同样的 $\theta$ 也让 $Q_\theta(s', a')$ 变高
- 然后 target 跟着升 → $Q_\theta(s, a)$ 又被推得更高
- ... **正反馈循环 → 发散**

### 3.2 解决：用一个"冻结"的 Target Network

引入 $Q_{\theta^-}$，参数 $\theta^-$ 是 $\theta$ 的延迟副本：
$$y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$$

更新规则：
- 每步用 $y$ 训练 $Q_\theta$
- **每 $C$ 步把 $\theta^- \leftarrow \theta$**（如 $C = 10000$）

这样 target 在每 $C$ 步内保持不变，回归目标稳定。

### 3.3 软更新（Polyak Averaging）

变种：每步做指数移动平均
$$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$$

$\tau$ 通常取 0.005。这就是 Ch4 BYOL 的 EMA Target 思想（呼应 Ch4 §D）！

> **跨章联系**：BYOL/SimSiam 中的 EMA Target、DQN 中的 Target Network、PPO/DPO 中的 Reference Policy ——本质都是同一个想法：**用一个慢速演化的目标提供稳定回归方向**。

## 4. Double DQN：解决最大化偏差

### 4.1 问题：DQN 的 max 偏差

延续 Ch4 §A.5：$\max$ 一组带噪 Q 值会过估计真值。在神经网络场景下这个问题更严重——网络初期 Q 估计极不准。

### 4.2 解法（同 Ch4 §A.5.2）

把"选 action"和"评估 Q"解耦：
$$y = r + \gamma Q_{\theta^-}\left(s', \underbrace{\arg\max_{a'} Q_\theta(s', a')}_{\text{用 online 网络选}}\right)$$

- 用 **online 网络** $Q_\theta$ 选最佳 action
- 用 **target 网络** $Q_{\theta^-}$ 评估这个 action 的 Q

仅一行改动（从 Q-target.max() 改为 Q-target.gather(...)），但在 Atari 上提升 10-30%。

## 5. Dueling DQN：分离 V 和 A

### 5.1 思想

把 $Q$ 分解为 $V + A$：
$$Q(s, a) = V(s) + A(s, a)$$

直觉：很多 action 其实差别不大，重要的是状态本身的价值。**分别学 V 和 A 比直接学 Q 更有数据效率**。

### 5.2 网络结构

```
                  ┌──── V_head ─────► V(s)        [B, 1]
共享 backbone ────┤
                  └──── A_head ─────► A(s, a)     [B, n_actions]
```

合并方式（避免 V 和 A 分配不唯一）：
$$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\right)$$

减去 A 的均值是为了"锚定" V 和 A 的分配（数学上严格对称的设计）。

## 6. Rainbow：六大改进的集成

DeepMind 2017 把六种改进打包：
1. Double DQN
2. Dueling DQN
3. Prioritized Experience Replay（按 TD-error 加权采样）
4. Multi-step Returns（n-step）
5. Distributional RL（学 Q 的分布而非均值）
6. Noisy Nets（参数空间噪声替代 ε-greedy）

Rainbow 在 Atari 上比 vanilla DQN 性能高约 200%。但每加一项的边际收益递减——实践中常用 Double + Dueling + Prioritized 三件套。

---

# §B 模型架构

## B.1 数据流总览

```
            Atari 游戏画面 (84×84×4 stack)
                       │
                       ▼
              ┌──────────────────┐
              │  CNN Backbone    │
              │  (3 个 conv 层)  │
              └──────────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Flatten + FC    │
              └──────────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Q Head: [n_a]   │
              └──────────────────┘
                       │
                       ▼
              Q(s, a₀), Q(s, a₁), ..., Q(s, a_{n-1})
```

每一步的 shape：

| 层 | 输入 | 输出 |
| :--- | :--- | :--- |
| 输入 | $[B, 4, 84, 84]$ | — |
| Conv1 (8×8, stride 4, 32 ch) | $[B, 4, 84, 84]$ | $[B, 32, 20, 20]$ |
| Conv2 (4×4, stride 2, 64 ch) | $[B, 32, 20, 20]$ | $[B, 64, 9, 9]$ |
| Conv3 (3×3, stride 1, 64 ch) | $[B, 64, 9, 9]$ | $[B, 64, 7, 7]$ |
| Flatten | $[B, 64, 7, 7]$ | $[B, 3136]$ |
| FC 512 + ReLU | $[B, 3136]$ | $[B, 512]$ |
| Output | $[B, 512]$ | $[B, n\_actions]$ |

## B.2 DQN 网络的 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Atari-style CNN DQN"""
    def __init__(self, n_actions, in_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        """
        x: [B, 4, 84, 84]  浮点 0-1（注意要先 / 255）
        返回: [B, n_actions] Q 值
        """
        h = self.conv(x).reshape(x.size(0), -1)
        return self.fc(h)
```

## B.3 Replay Buffer

```python
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """简单环形缓冲区"""
    def __init__(self, capacity=int(1e6)):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_new, done):
        self.buffer.append((s, a, r, s_new, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_new, done = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.array(s_new), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)
```

## B.4 完整 DQN 训练循环（含 Double DQN）

```python
import copy
from torch.optim import Adam

def train_dqn(env, n_steps=int(1e6), batch_size=32, gamma=0.99,
              lr=1e-4, buffer_size=int(1e6), eps_start=1.0, eps_end=0.05,
              eps_decay_steps=int(1e5), target_update_freq=10000,
              learning_starts=10000, double_dqn=True):
    n_actions = env.action_space.n

    online = DQN(n_actions)
    target = copy.deepcopy(online)                        # ⭐ Target 网络
    for p in target.parameters(): p.requires_grad = False

    optim = Adam(online.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size)

    obs, _ = env.reset()
    for step in range(n_steps):
        # ============ ε 衰减 ============
        eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay_steps)

        # ============ 选 action ============
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) / 255.0
                a = online(obs_t).argmax(dim=-1).item()

        # ============ 环境交互 ============
        obs_new, r, terminated, truncated, _ = env.step(a)
        done = float(terminated)
        buffer.push(obs, a, r, obs_new, done)
        obs = obs_new if not (terminated or truncated) else env.reset()[0]

        # ============ 训练 ============
        if step > learning_starts and len(buffer) > batch_size:
            s, action, reward, s_new, done = buffer.sample(batch_size)
            s = s / 255.0
            s_new = s_new / 255.0

            # 当前 Q
            q = online(s).gather(1, action.unsqueeze(1)).squeeze(1)   # [B]

            # ⭐ Target 计算
            with torch.no_grad():
                if double_dqn:
                    # online 选 action，target 评估
                    a_star = online(s_new).argmax(dim=-1)              # [B]
                    q_next = target(s_new).gather(1, a_star.unsqueeze(1)).squeeze(1)
                else:
                    q_next = target(s_new).max(dim=-1).values
                y = reward + gamma * q_next * (1 - done)

            loss = F.smooth_l1_loss(q, y)                              # ⭐ Huber loss 比 MSE 稳

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optim.step()

        # ============ 定期同步 Target ============
        if step % target_update_freq == 0:
            target.load_state_dict(online.state_dict())                # ⭐

    return online
```

**几个关键工程细节**：
1. **Huber loss (smooth_l1_loss)**：对 outlier 不敏感，比 MSE 稳定
2. **Frame stack 4**：把过去 4 帧拼在一起，弥补 Markov 性不足（单帧无法判断速度）
3. **Reward clipping**：reward 截断到 [-1, 1]，跨游戏统一尺度
4. **ε 线性衰减**：1.0 → 0.05 在前 10% 训练步内
5. **Target update freq**：10000 步（约 4 万环境步）

## B.5 Dueling DQN 的 PyTorch 实现

```python
class DuelingDQN(nn.Module):
    def __init__(self, n_actions, in_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )
        self.fc_v = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512), nn.ReLU(),
            nn.Linear(512, 1),                            # ⭐ V head
        )
        self.fc_a = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512), nn.ReLU(),
            nn.Linear(512, n_actions),                    # ⭐ A head
        )

    def forward(self, x):
        h = self.conv(x).reshape(x.size(0), -1)
        v = self.fc_v(h)                                   # [B, 1]
        a = self.fc_a(h)                                   # [B, n_actions]
        # ⭐ Q = V + (A - mean(A))
        q = v + (a - a.mean(dim=-1, keepdim=True))
        return q
```

---

# §C 训练与推理

## C.1 Atari Pong 训练经验

```python
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

env = gym.make("PongNoFrameskip-v4")
env = AtariPreprocessing(env, grayscale_obs=True, frame_skip=4)
env = FrameStack(env, num_stack=4)                        # ⭐ 帧堆叠

dqn = train_dqn(env, n_steps=int(2e6))                    # 约 2 小时（GPU）
```

**关键预处理**：
1. 灰度化（节省计算）
2. Resize to 84×84
3. Frame skip = 4（每 4 帧只取动作 1 次）
4. Frame stack = 4（把过去 4 帧叠起来作为状态，弥补 Markov 性）

**典型训练曲线**（Pong）：
- 50 万步：reward = -19（开始击球但每球都丢）
- 100 万步：reward = -5
- 200 万步：reward = 18（接近满分 21）

## C.2 推理：DQN 部署

```python
def play(env, dqn):
    obs, _ = env.reset()
    total_r = 0
    while True:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) / 255.0
            a = dqn(obs_t).argmax(dim=-1).item()           # greedy
        obs, r, terminated, truncated, _ = env.step(a)
        total_r += r
        if terminated or truncated: break
    return total_r
```

注意：DQN 推理时**完全去掉 ε**（纯 greedy）。target 网络也不需要——只用 online 网络。

## C.3 DQN 调参 checklist

| 参数 | 推荐值 | 备注 |
| :--- | :--- | :--- |
| learning rate | 1e-4 | 太大会发散 |
| batch size | 32-64 | |
| gamma | 0.99 | 远视 Atari |
| ε 衰减 | 1.0 → 0.05 in 100K-1M 步 | 视任务难度 |
| target update | 10K 步 | 软更新 τ=0.005 也常用 |
| buffer size | 1e6 | Atari 内存吃紧时降到 1e5 |
| learning_starts | 10K-50K | 让 buffer 先填一些数据 |
| Huber δ | 1.0 | smooth_l1_loss 的默认 |

## C.4 DQN 的局限

DQN 的成功仅限于：
- **离散动作**（Atari）
- **马尔可夫近似良好**（4 帧 stack 够用）

不适用：
- **连续动作**（机器人）→ Ch9 DDPG/SAC
- **大动作空间**（围棋 19×19 = 361 个 action 还行，但更大就不行）
- **稀疏 reward**（探索难）→ Ch10 探索方法

---

# §D 章末速查

## D.1 三大改进对照表

| 改进 | 解决问题 | 代码改动 |
| :--- | :--- | :--- |
| **Replay Buffer** | 样本相关 / 数据效率 | 维护 deque，随机 sample |
| **Target Network** | 目标非平稳 | 维护 $\theta^-$，定期同步 |
| **Double DQN** | 最大化偏差 | online 选 action，target 评估 |
| **Dueling DQN** | 数据效率 | $Q = V + (A - \bar{A})$ |
| **Prioritized Replay** | 重要样本欠学习 | 按 TD-error 加权采样 |

## D.2 常见面试题

**Q1：DQN 为什么需要 Replay Buffer？**
- 打破样本时序相关性
- 提高数据利用率
- 稳定训练分布
- 仅 off-policy 算法可用——这是 Q-Learning 的关键优势

**Q2：Target Network 必须每 N 步 hard update 吗？**
- 不一定。软更新 $\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$ 也常用（DDPG 的标配）
- 软更新呼应 Ch4 BYOL 的 EMA Target、Ch6 PPO 的 Reference Policy

**Q3：Double DQN 为什么有效？**
- 解耦"选 action"（online）与"评估 action 的 Q"（target）
- 减轻 max 偏差（Jensen 不等式：$\mathbb{E}[\max] > \max \mathbb{E}$）

**Q4：DQN 与 Q-Learning 算法上有何不同？**
- Q-Learning：tabular，每步更新 1 个 Q
- DQN：神经网络 + Replay Buffer + Target Network
- DQN 在大状态空间下可行，但需要很多 trick 让它稳定

**Q5：Dueling DQN 的 V/A 分解为什么需要 "减均值"？**
- 不减均值会有"分配不唯一"问题：$Q = V + A = (V + c) + (A - c)$
- 减均值锚定 $A$ 的均值为 0，让分解唯一
- 也可以减 max，但减均值的实证效果更好

---

## 承上启下

DQN 系列把 Q-Learning 在视觉领域推上人类水平，但只能处理离散动作。**连续控制**（机器人、自动驾驶）需要不同的算法。

下一章 **Ch8 TRPO/PPO** 走 policy-based 路线，用 trust region 思想稳定 PG 训练。这是 RLHF 的核心，也是从经典 RL 走向 LLM 时代的关键桥梁。
