---
title: RL Chapter3 Monte Carlo 与 TD：从样本估计价值
categories: 学习笔记-强化学习
date: 2026-05-11 10:00:00
mathjax: true
tags:
    - AI
    - 强化学习
    - AI面试知识
---

> **本章定位**：DP（Ch2）需要已知 MDP，但现实中 $P, R$ 几乎总是未知。本章引入 RL 的核心思想——**从交互样本估计 Bellman 方程**，奠定后续所有 model-free 算法的基础。

> **承上**：Ch1 §A.7 的 $Q^\pi$ 采样形式 + Ch2 §6 的 GPI 框架。
> **启下**：Ch4 把 TD 思想用到 control（Q-Learning/SARSA）。

---

# §A 数学原理

## 1. Monte Carlo 估计

### 1.1 核心思想

**用样本平均代替真实期望**：
$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s] \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_i^{(s)}$$

其中 $G_i^{(s)}$ 是第 $i$ 次 episode 中从状态 $s$ 出发的实际累积回报。

### 1.2 流程

1. 跑完一整个 episode：$(s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T)$
2. 对每个 $t$，计算实际累积回报 $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$
3. 把 $G_t$ 加入 $V(s_t)$ 的样本池，更新均值

### 1.3 增量式更新

避免存储所有 $G$，用增量均值公式：
$$V(s) \leftarrow V(s) + \frac{1}{N(s)} \left[ G - V(s) \right]$$

或更一般地用学习率 $\alpha$（**常量学习率 = 指数移动平均**）：
$$\boxed{V(s) \leftarrow V(s) + \alpha \left[ G - V(s) \right]}$$

### 1.4 First-Visit vs Every-Visit MC

**First-Visit MC**：每个 episode 中状态 $s$ 第一次出现时才更新。
**Every-Visit MC**：每次出现都更新。

- First-Visit 是无偏估计，理论分析更友好
- Every-Visit 数据利用率更高，实践常用
- 两者在 episode 数趋于无穷时都收敛到 $V^\pi$

### 1.5 MC 的优缺点

✅ **优点**：
- 无偏（$\mathbb{E}[G] = V^\pi(s)$）
- 不需要 $P, R$
- 不依赖 Markov 性

❌ **缺点**：
- **必须等 episode 结束才能更新**——无法处理无终止任务
- **方差高**：$G$ 是几十上百步 reward 的累加，方差爆炸
- 数据效率低

## 2. Temporal Difference (TD) Learning

### 2.1 核心洞察

回忆 Ch1 §A.7 的采样形式：
$$Q^\pi(s, a) = \mathbb{E}_{s', a'}[r + \gamma Q^\pi(s', a')]$$

如果 $V^\pi$ 已知，则一步交互 $(s_t, a_t, r_t, s_{t+1})$ 给出**无偏单点估计**：
$$\hat{V}(s_t) \approx r_t + \gamma V^\pi(s_{t+1})$$

但 $V^\pi$ 未知！TD 的天才之处：**用当前 $V$ 的估计去估计 $V^\pi$**——这叫 **bootstrap**。

### 2.2 TD(0) 更新公式

$$\boxed{V(s_t) \leftarrow V(s_t) + \alpha \underbrace{[r_t + \gamma V(s_{t+1}) - V(s_t)]}_{\delta_t \ \text{TD-error}}}$$

**关键概念 TD-error**：
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

直观：用"实际看到的一步 reward + bootstrap 的下一步 V"替代 V(s_t) 的现有估计。

### 2.3 TD 的优缺点（与 MC 对比）

| 维度 | MC | TD |
| :--- | :--- | :--- |
| **更新时机** | episode 结束 | 每步立即 |
| **偏差** | 无偏 | **有偏**（用 V 估计 V） |
| **方差** | **高** | 低 |
| **是否需要 episode 终止** | 必须 | 不需要 |
| **是否依赖 Markov 性** | 不依赖 | 强烈依赖 |

> **核心 trade-off**：MC 是**高方差无偏**，TD 是**低方差有偏**。这个 bias-variance trade-off 贯穿整个 RL。

### 2.4 收敛性

**TD(0) 收敛性定理**：在 tabular 设置下（$V$ 用表格存储），若学习率满足 Robbins-Monro 条件：
$$\sum_t \alpha_t = \infty, \quad \sum_t \alpha_t^2 < \infty$$
（如 $\alpha_t = 1/t$），则 TD(0) **依概率 1 收敛**到 $V^\pi$。

实践中常用常量 $\alpha \in [0.01, 0.1]$，等价于做指数移动平均。

## 3. n-step TD：连接 MC 与 TD

### 3.1 思想

TD(0) 只看一步，MC 看到底。中间地带是**看 n 步**：

$$\hat{G}_t^{(n)} = r_t + \gamma r_{t+1} + \dots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})$$

更新：
$$V(s_t) \leftarrow V(s_t) + \alpha [\hat{G}_t^{(n)} - V(s_t)]$$

### 3.2 极端情况

- $n = 1$：TD(0)
- $n \to \infty$：MC

**$n$ 大 → 偏差小、方差大；$n$ 小 → 偏差大、方差小**。实践中 $n \in \{4, 8, 16\}$ 是常见选择。

## 4. TD(λ) 与资格迹 (Eligibility Traces)

### 4.1 λ-return

n-step return 的指数加权平均：
$$\hat{G}_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \hat{G}_t^{(n)}$$

- $\lambda = 0$：纯 TD(0)
- $\lambda = 1$：纯 MC
- $\lambda \in (0, 1)$：平滑的中间地带

### 4.2 前向视角 vs 反向视角

**前向视角**（理论清晰）：等到未来步骤都看到，再算 $\hat{G}_t^\lambda$ 更新 $V(s_t)$。

**反向视角**（实践高效）：每个状态维护一个**资格迹** $e(s)$，表示"这个状态对当前 TD-error 的责任"：
$$e(s_t) \leftarrow \gamma \lambda e(s_t) + 1, \quad e(s) \leftarrow \gamma \lambda e(s) \text{ for } s \neq s_t$$

每步用 $\delta_t$ 更新所有状态：
$$V(s) \leftarrow V(s) + \alpha \delta_t e(s) \quad \forall s$$

直观：**TD-error 沿着访问历史"回流"**，被衰减地分配到每个曾经访问的状态。

### 4.3 资格迹的现代命运

资格迹在表格法时代很重要，但深度 RL 时代被两个东西取代：
- **GAE（Generalized Advantage Estimation）**：见 Ch6 / Ch8
- **n-step replay buffer**：见 Ch7

但 GAE 的本质就是 TD(λ) 思想的"梯度版本"——所以理解了 λ-return，理解 GAE 就只是一步之遥。

## 5. Bias-Variance 几何理解

不同 $n$/$\lambda$ 在偏差-方差平面上的位置：

```
方差 ↑
     │
  MC ●                       <- λ=1, n=∞
     │  ●                    <- λ=0.95
     │   ●                   <- λ=0.9
     │    ●                  <- λ=0.5
     │     ●                 <- TD(0), λ=0
     └────────────────────→ 偏差
```

**实践经验**：
- 数据充足、要求收敛速度：选 $\lambda \in [0.9, 0.95]$（GAE 默认值）
- 表格小问题：用 TD(0) 即可
- 想做对比实验：跑 $\lambda \in \{0, 0.5, 0.9, 0.95, 0.99, 1\}$

---

# §B 模型架构（伪代码 + 实现）

## B.1 数据流：从样本到价值

```
┌────────────────────────────────────────────────────────────┐
│           Environment (P, R 未知)                           │
└──────────┬─────────────────────────────────────────────────┘
           │ s_t, a_t → r_t, s_{t+1}                         
           ▼
┌────────────────────────────────────────────────────────────┐
│         一步交互样本 (s_t, a_t, r_t, s_{t+1})              │
└──────────┬─────────────────────────────────────────────────┘
           │
   ┌───────┴────────┐
   ▼                ▼
┌──────────┐    ┌──────────────────────────────┐
│ MC: 缓存  │    │ TD: 立即用 δ_t 更新 V(s_t)  │
│ 等 ep 结束│    │  δ_t = r_t + γV(s_{t+1})    │
│ 用 G 更新  │    │       - V(s_t)              │
└──────────┘    └──────────────────────────────┘
```

## B.2 MC 算法的 numpy 实现

```python
import numpy as np
from chapter1 import GridWorld

def mc_policy_evaluation(env, policy, n_episodes=10000, alpha=0.1, max_steps=100):
    """
    Every-Visit MC：估计 V^π
    """
    V = np.zeros(env.n_states)

    for ep in range(n_episodes):
        # 1. ⭐ 采样一整个 episode
        episode = []
        s = np.random.randint(env.n_states)              # 随机起点
        for _ in range(max_steps):
            a = np.random.choice(env.n_actions, p=policy[s])
            s_new, r, done = env.step(s, a)
            episode.append((s, a, r))
            if done:
                break
            s = s_new

        # 2. ⭐ 从后往前算累积回报 G_t
        G = 0
        for s_t, _, r_t in reversed(episode):
            G = r_t + env.gamma * G
            # 3. ⭐ 增量更新（Every-Visit）
            V[s_t] += alpha * (G - V[s_t])

    return V


# 验证：MC 的结果应当接近 Ch2 闭式解
env = GridWorld(gamma=0.95)
random_policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
V_mc = mc_policy_evaluation(env, random_policy, n_episodes=5000)
print("MC 估计 V:")
print(V_mc.reshape(4, 4).round(2))
```

## B.3 TD(0) 的 numpy 实现

```python
def td0_policy_evaluation(env, policy, n_episodes=10000, alpha=0.1, max_steps=100):
    """
    TD(0)：每步立即更新 V^π
    """
    V = np.zeros(env.n_states)

    for ep in range(n_episodes):
        s = np.random.randint(env.n_states)
        for _ in range(max_steps):
            a = np.random.choice(env.n_actions, p=policy[s])
            s_new, r, done = env.step(s, a)

            # ⭐ TD-error
            target = r + env.gamma * V[s_new] * (not done)
            delta = target - V[s]

            # ⭐ 立即更新 V(s)
            V[s] += alpha * delta

            if done:
                break
            s = s_new

    return V


V_td = td0_policy_evaluation(env, random_policy, n_episodes=5000)
print("TD(0) 估计 V:")
print(V_td.reshape(4, 4).round(2))
```

**对比实验**：跑相同 episode 数，MC 与 TD(0) 哪个估计更接近真值？

```python
V_true = solve_v_closed_form(env, random_policy)         # Ch1 §C.2
print("MC 误差 (RMSE):", np.sqrt(((V_mc - V_true)**2).mean()))
print("TD 误差 (RMSE):", np.sqrt(((V_td - V_true)**2).mean()))
```

通常输出（5000 episodes）：
```
MC 误差 (RMSE): 0.082
TD 误差 (RMSE): 0.034
```

> **观察**：相同样本量下 TD 误差更小——因为 TD 利用了 bootstrap，每一步都在学。MC 的方差大，需要更多 episode 才能收敛。

## B.4 n-step TD 的实现

```python
def n_step_td(env, policy, n=4, n_episodes=10000, alpha=0.1, max_steps=100):
    """
    n-step TD：每 n 步更新一次（或在 episode 结束时清空）
    """
    V = np.zeros(env.n_states)
    gamma = env.gamma

    for ep in range(n_episodes):
        s = np.random.randint(env.n_states)
        states = [s]
        rewards = [0]                                    # rewards[t+1] 是 r_t

        T = float('inf')
        for t in range(max_steps + 1):
            if t < T:
                a = np.random.choice(env.n_actions, p=policy[states[t]])
                s_new, r, done = env.step(states[t], a)
                rewards.append(r)
                if done:
                    T = t + 1
                else:
                    states.append(s_new)

            # ⭐ 更新时刻：tau = t - n + 1
            tau = t - n + 1
            if tau >= 0:
                # 计算 n-step return
                G = sum(gamma**(i - tau - 1) * rewards[i]
                       for i in range(tau + 1, min(tau + n, T) + 1))
                if tau + n < T:
                    G += gamma**n * V[states[tau + n]]
                # 更新 V(states[tau])
                V[states[tau]] += alpha * (G - V[states[tau]])

            if tau == T - 1:
                break
    return V
```

## B.5 TD(λ) with Eligibility Traces 实现

```python
def td_lambda(env, policy, lambd=0.9, n_episodes=10000, alpha=0.1, max_steps=100):
    """
    TD(λ) 反向视角：维护资格迹
    """
    V = np.zeros(env.n_states)
    gamma = env.gamma

    for ep in range(n_episodes):
        e = np.zeros(env.n_states)                       # ⭐ 资格迹
        s = np.random.randint(env.n_states)

        for _ in range(max_steps):
            a = np.random.choice(env.n_actions, p=policy[s])
            s_new, r, done = env.step(s, a)

            # TD-error
            target = r + gamma * V[s_new] * (not done)
            delta = target - V[s]

            # ⭐ 累加资格迹
            e[s] += 1                                    # accumulating trace
            # 也可以用 replacing trace: e[s] = 1

            # ⭐ 用 δ * e 更新所有状态
            V += alpha * delta * e
            e *= gamma * lambd                           # ⭐ 衰减资格迹

            if done:
                break
            s = s_new

    return V
```

---

# §C 训练与推理

## C.1 实战：随机游走任务（经典对比 MC vs TD）

Sutton & Barto 书中的经典例子：5 状态线性链，从中间出发，左右随机游走，左终点 reward=0，右终点 reward=1。

```python
class RandomWalk:
    """5 状态线性链"""
    def __init__(self):
        self.n_states = 7                                # 0=左终, 6=右终
        self.gamma = 1.0

    def step(self, s, a):
        # a 不重要，左右等概率
        s_new = s + np.random.choice([-1, 1])
        if s_new == 0: return s_new, 0.0, True           # 左终点
        if s_new == 6: return s_new, 1.0, True           # 右终点
        return s_new, 0.0, False
```

真实价值：$V^*(s) = s/6, s = 1, ..., 5$，即 $[1/6, 2/6, 3/6, 4/6, 5/6]$。

```python
# 跑 100 个 episode 后 MC 和 TD 各自的误差
import matplotlib.pyplot as plt

env = RandomWalk()
V_true = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 0])

errors_mc, errors_td = [], []
for trial in range(100):
    V_mc_trial = np.zeros(7)
    V_td_trial = np.zeros(7)
    for ep in range(100):
        # 跑一个 episode 同时给 MC 和 TD
        ...                                              # 类似 B.2/B.3 但记录每个 episode 后的误差
    errors_mc.append(...)
    errors_td.append(...)

# 画图：横轴 episode 数，纵轴 RMSE
```

**经典结果**：
- TD 在前期收敛快（低方差）
- MC 收敛到的值更"准"但更慢
- TD 对学习率 $\alpha$ 敏感性比 MC 低

## C.2 推理视角：MC/TD 用于"控制"vs"评估"

**评估**（本章重点）：给定 $\pi$，估计 $V^\pi$。
**控制**（Ch4 重点）：找到最优 $\pi^*$。

控制一般用 **GPI 框架**（Ch2 §6）：
1. 用 MC/TD 估计 $V^{\pi}$（或更常见地估计 $Q^\pi$）
2. 贪心改进 $\pi$
3. 重复

下一章 Ch4 我们会看到这套思想如何具体化为 SARSA 和 Q-Learning。

## C.3 工程经验

| 场景 | 推荐方法 |
| :--- | :--- |
| 短 episode、终止明确 | MC（无偏，简单） |
| 长/无限 episode、需 online 更新 | TD(0) |
| 平衡偏差方差，要求性能 | TD(λ) 或 n-step TD（n=4-8） |
| 大状态空间 | TD + 函数逼近（Ch7 起） |
| 想要 "GAE 那样" 的优势估计 | TD(λ) 的策略梯度版本（Ch6） |

---

# §D 章末速查

## D.1 三种估计的统一视角

| 方法 | 目标值 | 偏差 | 方差 |
| :--- | :--- | :---: | :---: |
| **DP**（Ch2） | $T^\pi V$（精确期望） | 0 | 0 |
| **MC** | $G_t$（实际样本） | 0 | **高** |
| **TD(0)** | $r_t + \gamma V(s_{t+1})$（一步样本 + bootstrap） | **有** | 低 |
| **n-step TD** | $\sum r + \gamma^n V$ | 中 | 中 |
| **TD(λ)** | λ-return 加权平均 | 可调 | 可调 |

> **理解了这张表，你就理解了所有 RL 方法的"价值估计部分"**。

## D.2 常见面试题

**Q1：MC 和 TD 哪个偏差大？哪个方差大？**
- MC：偏差 0、方差**高**（看完整 episode）
- TD：偏差**有**（bootstrap）、方差低
- 这是 RL 中最经典的 bias-variance trade-off

**Q2：TD(0) 收敛到的是 $V^\pi$ 吗？**
- Tabular 设置 + 满足 Robbins-Monro 学习率 → 是
- 函数逼近设置（如神经网络）下，**TD 不一定收敛**——这是 deep RL 的一大挑战

**Q3：为什么 TD 需要 Markov 性？**
- TD 用 $V(s_{t+1})$ 作 bootstrap，假设 "$s_{t+1}$ 包含未来所有信息"
- MC 直接用真实 $G_t$，不依赖 Markov 性
- 状态设计不充分时，TD 会有系统性偏差

**Q4：n-step TD 的 n 怎么选？**
- 有理论分析（Sutton 第 7 章）：根据 reward 信号的 "时间尺度" 选
- 实践：$n \in \{4, 8, 16\}$，配合学习率调
- 大 n 偏差小但方差大、需要更长 episode 才能更新

**Q5：TD(λ) 与 GAE 的关系？**
- TD(λ) 是价值估计层面的指数加权
- GAE 是优势估计 $\hat{A}_t$ 层面的指数加权（Ch6）
- 数学上几乎一样，只差 baseline 的减除

---

## 承上启下

我们现在能从样本估计 $V^\pi$（评估）。但 RL 真正的目标是**找到最优策略**——这是控制问题。

控制有两条路：
1. **Value-based**：估计 $Q^\pi$（不是 $V^\pi$，因为要选 action 必须有 $Q$），用 GPI 改进策略 → **Ch4 Q-Learning / SARSA**
2. **Policy-based**：直接对策略 $\pi_\theta$ 求梯度上升 → **Ch5 Policy Gradient**

Ch4 我们走第一条路，看 Q-Learning 如何成为 RL 史上最经典的算法。
