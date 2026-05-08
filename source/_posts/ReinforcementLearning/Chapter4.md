---
title: RL Chapter4 Q-Learning 与 SARSA：从评估到控制
categories: 学习笔记-强化学习
date: 2026-05-12 10:00:00
mathjax: true
tags:
    - AI
    - 强化学习
    - AI面试知识
---

> **本章定位**：把 Ch3 的 TD 思想从"评估"扩展到"控制"——直接从交互经验中**学最优策略**，不需要已知 MDP。Q-Learning 是 RL 史上最经典的算法，第一个在理论上保证收敛到最优的 model-free 控制方法。

> **承上**：Ch2 §6 GPI 框架 + Ch3 §A.2 TD-error。
> **启下**：Ch7 把 Q-Learning 与神经网络结合 → DQN。

---

# §A 数学原理

## 1. 从估计 V 到估计 Q：为什么必须切换？

Ch3 的 TD/MC 估计的是 $V^\pi(s)$。但要从 $V$ 改进策略需要：
$$\pi'(s) = \arg\max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s') \right]$$

**问题**：这个公式需要 $P, R$！MDP 未知时无法用。

**解决**：直接估计 $Q^\pi$。从 $Q$ 改进策略不需要 $P, R$：
$$\pi'(s) = \arg\max_a Q^\pi(s, a)$$

> **这是 model-free 控制的根本原因**：估计 $Q$ 而非 $V$，让策略改进不再依赖 MDP 模型。

## 2. SARSA：On-policy TD 控制

### 2.1 名字由来

SARSA = 用样本 $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$ 来更新——五个字母的首字母。

### 2.2 更新规则

把 Ch3 §A.2.2 的 TD(0) 公式从 $V$ 移植到 $Q$：

$$\boxed{Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]}$$

**关键**：$a_{t+1}$ 是**根据当前策略 $\pi$ 实际采样到的动作**。这就是 "on-policy" 的含义——更新用的是策略自己的轨迹。

### 2.3 ε-greedy 探索

如果策略一直选 $\arg\max Q$，会陷入局部最优——某些动作永远不被尝试，$Q$ 估计永远更新不到。

**ε-greedy 策略**：
$$\pi(a \mid s) = \begin{cases} 1 - \epsilon + \epsilon/|\mathcal{A}|  & \text{if } a = \arg\max_a Q(s, a) \\ \epsilon/|\mathcal{A}| & \text{otherwise} \end{cases}$$

简言之：以概率 $1-\epsilon$ 选最优，以概率 $\epsilon$ 随机探索。$\epsilon$ 通常从 1.0 衰减到 0.01。

### 2.4 SARSA 算法

```
初始化: Q(s, a) = 0 for all s, a
for episode = 1, 2, ...:
    s ← 环境初始化
    a ← ε-greedy(Q, s)
    while not done:
        执行 a，观察 r, s'
        a' ← ε-greedy(Q, s')
        Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]    # ⭐ SARSA
        s, a ← s', a'
```

## 3. Q-Learning：Off-policy TD 控制

### 3.1 关键改动

把 SARSA 的 $Q(s_{t+1}, a_{t+1})$ 替换为 $\max_{a'} Q(s_{t+1}, a')$：

$$\boxed{Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]}$$

**这一改动的本质**：Q-Learning 的更新目标是 **Bellman 最优方程**（不是期望方程），直接朝 $Q^*$ 靠拢。

### 3.2 Off-policy 的含义

| 方面 | On-policy (SARSA) | Off-policy (Q-Learning) |
| :--- | :--- | :--- |
| 更新使用的 action | 实际策略采样的 $a_{t+1}$ | $\arg\max Q(s_{t+1}, \cdot)$ |
| 行为策略 | 与目标策略相同（都是 ε-greedy） | 行为是 ε-greedy，目标是 greedy |
| 数据来源 | 必须是当前策略生成 | **可以是任何策略生成**（包括历史数据、其他 agent） |

> **为什么 Q-Learning 能用历史数据？** 因为 $\max_{a'}$ 只依赖 Q 函数，不依赖采样动作。这个性质后来被 DQN（Ch7）的经验回放（Experience Replay）大规模利用。

### 3.3 Q-Learning 算法

```
初始化: Q(s, a) = 0
for episode = 1, 2, ...:
    s ← 初始化
    while not done:
        a ← ε-greedy(Q, s)
        执行 a，观察 r, s'
        Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]    # ⭐ Q-Learning
        s ← s'
```

### 3.4 Q-Learning 的收敛性

**定理**（Watkins 1989）：在 tabular 设置下，若：
1. 所有 $(s, a)$ 都被访问无限多次
2. 学习率满足 Robbins-Monro 条件 $\sum \alpha_t = \infty, \sum \alpha_t^2 < \infty$

则 $Q$ 依概率 1 收敛到 $Q^*$。

> **关键**：Q-Learning 即使在使用 ε-greedy 探索（行为策略 ≠ 目标策略）下，仍能收敛到最优。这是 RL 史上的重大结果。

## 4. SARSA vs Q-Learning：经典对比

### 4.1 Cliff Walking 例子（Sutton & Barto 经典）

```
S . . . . . . . . . . G       S 起点, G 终点
T T T T T T T T T T T T       T 是悬崖：reward = -100
                              其他: reward = -1
```

agent 走悬崖边缘可以最快到达，但稍偏一点就摔死。

**SARSA 学到**："小心翼翼"路线，离悬崖远一些（因为 ε-greedy 的随机性会让它偶尔掉下去）。

**Q-Learning 学到**：贴着悬崖走的最优路径（理论最短）。但**实际行动时**用 ε-greedy 偶尔会掉下去。

> **直觉**：
> - SARSA 学的是 "ε-greedy 策略下" 的最优 Q（保守）
> - Q-Learning 学的是 "完全 greedy 策略下" 的最优 Q（理想最优）
> - 实战中如果 ε-greedy 是真实部署策略，SARSA 反而更好

### 4.2 一图总结

| 维度 | SARSA | Q-Learning |
| :--- | :--- | :--- |
| 更新公式 | $r + \gamma Q(s', a')$ | $r + \gamma \max Q(s', \cdot)$ |
| 策略类型 | On-policy | Off-policy |
| 数据要求 | 当前策略生成 | 任意策略生成 |
| 学到的是 | 实际部署策略的 Q | 最优策略的 Q |
| 安全性 | **更保守** | 激进 |
| 经典应用 | 安全攸关任务 | DQN（Ch7）的基础 |

## 5. 最大化偏差 (Maximization Bias)

Q-Learning 的 $\max_{a'}$ 操作引入一个微妙的问题——**对噪声的过估计**。

### 5.1 问题来源

设真实 $Q^*(s, a) = 0$ for all $a$，但因为采样噪声，估计 $\hat{Q}(s, a)$ 有方差。
$$\mathbb{E}[\max_a \hat{Q}(s, a)] > \max_a \mathbb{E}[\hat{Q}(s, a)] = 0$$

即 $\max$ 一组带噪估计，平均下来比真值大。这就是**最大化偏差**。

### 5.2 Double Q-Learning

**思想**：用两组独立的 Q 估计 $Q_A, Q_B$，让"选 action"和"评估 Q"由不同的网络做：

$$Q_A(s, a) \leftarrow Q_A(s, a) + \alpha \left[ r + \gamma Q_B\left(s', \arg\max_{a'} Q_A(s', a')\right) - Q_A(s, a) \right]$$

每步随机选 $A$ 或 $B$ 来更新（另一个固定）。

### 5.3 现代意义

Double Q-Learning 直接启发了 **Double DQN**（Ch7 §B.3）——DQN 时代最常用的改进之一。在 Atari 任务上 Double DQN 比原 DQN 性能高 10-30%。

---

# §B 模型架构

## B.1 数据流：表格法控制循环

```
                    ┌────────────────────┐
                    │ Q[s, a] 表（状态×动作） │
                    └──────────┬─────────┘
                               │ 查询
                               ▼
   ┌──────────┐  s        ┌────────────────┐  a
   │  Env     │─────────▶│ ε-greedy 策略   │────┐
   └──────────┘           └────────────────┘    │
        ▲                                        ▼
        │ s', r                            ┌──────────┐
        └──────────────────────────────────│  执行 a  │
                                            └──────────┘
                                                  │
                                                  ▼
                            ┌────────────────────────────┐
                            │  TD 更新:                  │
                            │  Q[s,a] += α(target - Q)   │
                            │  target = r + γ·next_term  │
                            │   - SARSA:    γ Q(s', a')  │
                            │   - Q-Learn: γ max_a' Q    │
                            └────────────────────────────┘
```

## B.2 Q-Learning 完整 numpy 实现

```python
import numpy as np
import gymnasium as gym

def q_learning(env, n_episodes=2000, alpha=0.1, gamma=0.99,
               eps_start=1.0, eps_end=0.05, eps_decay=0.995):
    """
    Tabular Q-Learning on a discrete state/action env (e.g. FrozenLake)
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))                  # ⭐ Q 表

    eps = eps_start
    rewards_history = []

    for ep in range(n_episodes):
        s, _ = env.reset()
        total_r = 0
        done = False

        while not done:
            # ⭐ ε-greedy 选择动作
            if np.random.random() < eps:
                a = env.action_space.sample()             # 探索
            else:
                a = Q[s].argmax()                         # 利用

            s_new, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            # ⭐ Q-Learning 更新
            target = r + gamma * Q[s_new].max() * (not terminated)
            Q[s, a] += alpha * (target - Q[s, a])

            s = s_new
            total_r += r

        eps = max(eps_end, eps * eps_decay)              # ε 衰减
        rewards_history.append(total_r)

    return Q, rewards_history


# ============ 在 FrozenLake 上跑通 ============
env = gym.make("FrozenLake-v1", is_slippery=False)
Q, rewards = q_learning(env)

# 评估学到的策略
def eval_policy(env, Q, n_eval=100):
    successes = 0
    for _ in range(n_eval):
        s, _ = env.reset()
        done = False
        while not done:
            a = Q[s].argmax()                            # greedy
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            if r > 0: successes += 1
    return successes / n_eval

print(f"Success rate: {eval_policy(env, Q):.2%}")
```

## B.3 SARSA 实现（对比 Q-Learning 的差异）

```python
def sarsa(env, n_episodes=2000, alpha=0.1, gamma=0.99,
          eps_start=1.0, eps_end=0.05, eps_decay=0.995):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    eps = eps_start

    def epsilon_greedy(s):
        if np.random.random() < eps:
            return env.action_space.sample()
        return Q[s].argmax()

    for ep in range(n_episodes):
        s, _ = env.reset()
        a = epsilon_greedy(s)                            # ⭐ 第一个 action
        done = False

        while not done:
            s_new, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            a_new = epsilon_greedy(s_new)                # ⭐ 用 ε-greedy 选下一动作

            # ⭐ SARSA 更新（注意是 Q[s_new, a_new] 而非 max）
            target = r + gamma * Q[s_new, a_new] * (not terminated)
            Q[s, a] += alpha * (target - Q[s, a])

            s, a = s_new, a_new                          # ⭐ 双更新

        eps = max(eps_end, eps * eps_decay)

    return Q
```

**两份代码的关键差异（Q-Learning vs SARSA）**：

| Q-Learning | SARSA |
| :--- | :--- |
| `target = r + γ Q[s_new].max()` | `target = r + γ Q[s_new, a_new]` |
| 不需要预先选 `a_new` | 必须用 ε-greedy 选 `a_new` |
| 内层循环只更新 `s` | 内层循环更新 `s, a` |

## B.4 Double Q-Learning 实现

```python
def double_q_learning(env, n_episodes=2000, alpha=0.1, gamma=0.99):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q_A = np.zeros((n_states, n_actions))
    Q_B = np.zeros((n_states, n_actions))
    eps = 0.1

    for ep in range(n_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            # ε-greedy 用 Q_A + Q_B
            if np.random.random() < eps:
                a = env.action_space.sample()
            else:
                a = (Q_A[s] + Q_B[s]).argmax()

            s_new, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            # ⭐ 一半概率更新 A，一半概率更新 B
            if np.random.random() < 0.5:
                # 用 A 选 action，用 B 评估
                a_star = Q_A[s_new].argmax()
                target = r + gamma * Q_B[s_new, a_star] * (not terminated)
                Q_A[s, a] += alpha * (target - Q_A[s, a])
            else:
                a_star = Q_B[s_new].argmax()
                target = r + gamma * Q_A[s_new, a_star] * (not terminated)
                Q_B[s, a] += alpha * (target - Q_B[s, a])

            s = s_new

    return (Q_A + Q_B) / 2
```

---

# §C 训练与推理

## C.1 实验：Cliff Walking 上的 SARSA vs Q-Learning

```python
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("CliffWalking-v0")

Q_sarsa = sarsa(env, n_episodes=500, eps_start=0.1, eps_end=0.1)
Q_qlearn, rewards_q = q_learning(env, n_episodes=500, eps_start=0.1, eps_end=0.1)

# 评估时关闭探索（greedy）
def visualize_path(env, Q):
    s, _ = env.reset()
    path = [s]
    for _ in range(100):
        a = Q[s].argmax()
        s, r, terminated, truncated, _ = env.step(a)
        path.append(s)
        if terminated or truncated: break
    return path

print("SARSA 路径:", visualize_path(env, Q_sarsa))
print("Q-Learning 路径:", visualize_path(env, Q_qlearn))
```

**经典结论**（与 Sutton & Barto Figure 6.5 一致）：
- **SARSA**：选远离悬崖的"安全路径"
- **Q-Learning**：选贴着悬崖的"最优路径"
- **训练期间 reward**：SARSA 更高（不掉悬崖），Q-Learning 更低（ε 探索导致摔死）

## C.2 推理视角：训练完后

训练完后只用 Q 表的 greedy：
```python
def deploy(env, Q):
    s, _ = env.reset()
    while True:
        a = Q[s].argmax()                                # 关闭探索
        s, r, terminated, truncated, _ = env.step(a)
        if terminated or truncated: break
```

注意：
- 训练时 ε-greedy（探索 + 利用）
- 推理时纯 greedy（只利用）
- ε 衰减策略：通常 1.0 → 0.05 在 ~10% 训练步骤完成

## C.3 工程经验

| 问题 | 解决 |
| :--- | :--- |
| Q 一直不更新 | 学习率太小 / ε 太小（探索不足） |
| Q 震荡 | 学习率太大 / ε 太大 |
| 收敛慢 | 试试 Double Q-Learning / TD(λ) |
| 状态空间过大 | 上 DQN（Ch7） |
| 动作空间是连续的 | 用 Policy Gradient（Ch5） |

---

# §D 章末速查

## D.1 三个核心公式

| 算法 | 更新公式 |
| :--- | :--- |
| **TD(0)** (Ch3) | $V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]$ |
| **SARSA** | $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s', a') - Q(s,a)]$ |
| **Q-Learning** | $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s', a') - Q(s,a)]$ |

## D.2 常见面试题

**Q1：Q-Learning 为什么是 off-policy？**
- 更新目标 $\max_{a'} Q(s', a')$ 不依赖采样的 $a'$
- 因此训练数据可来自任何策略（历史数据、其他 agent）
- 这一性质让 DQN 的经验回放成为可能

**Q2：SARSA 和 Q-Learning 学到的策略一样吗？**
- 不一样
- SARSA 学到"ε-greedy 下的最优"
- Q-Learning 学到"完全 greedy 下的最优"
- 当 $\epsilon \to 0$，两者收敛到同一最优策略

**Q3：为什么 Q-Learning 会有 maximization bias？**
- $\max$ 一组带噪估计 → 偏向高估
- $\mathbb{E}[\max \hat{Q}] > \max \mathbb{E}[\hat{Q}]$（Jensen 不等式反向）
- Double Q-Learning 用两个独立网络解耦"选 action"和"评估"

**Q4：ε-greedy 的 ε 怎么衰减？**
- 经典：$\epsilon_t = \epsilon_0 \cdot \text{decay}^t$，decay=0.99-0.995
- 或线性衰减：$\epsilon = \max(\epsilon_{\min}, 1 - t/T_{\text{anneal}})$
- DQN 标配：1.0 → 0.05 在前 10% 训练步内完成

**Q5：Q-Learning 在大状态空间会怎样？**
- 表格法存不下 → 函数逼近（神经网络）
- 但 Q-Learning + 神经网络容易发散，需要 **Replay Buffer + Target Network**（DQN，Ch7）

---

## 承上启下

到此我们已经掌握 RL 的"前现代"工具：MDP（Ch1）、DP（Ch2）、MC/TD（Ch3）、Q-Learning/SARSA（Ch4）。但表格法只能处理小状态空间。

接下来分两条路：
- **Value-based 路线**：Ch7 把 Q-Learning 扩展到神经网络 → DQN
- **Policy-based 路线**：Ch5 直接对策略求梯度 → REINFORCE

下一章 **Ch5** 走 policy-based 路线。这条路线不需要"先估 Q 再贪心"，而是**直接对策略 $\pi_\theta(a \mid s)$ 求导**，是后续 PPO、DPO 的根基。
