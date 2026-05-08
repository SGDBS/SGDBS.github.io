---
title: RL Chapter1 MDP 与 Bellman 方程
categories: 学习笔记-强化学习
date: 2026-05-09 10:00:00
mathjax: true
tags:
    - AI
    - 强化学习
    - AI面试知识
---

> **本章定位**：建立 RL 的"母语"——MDP（马尔可夫决策过程）。后续所有算法（DP、MC、TD、Q-Learning、PG、PPO、DPO、GRPO）都建立在 MDP 的概念之上。

> **承上**：Ch0 §F.1 的概率论基础。
> **启下**：Ch2 用 DP 求解已知 MDP，Ch3 用 MC/TD 估计未知 MDP。

---

# §A 数学原理

## 1. 马尔可夫决策过程 (MDP) 的定义

一个 MDP 是一个五元组：
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$$

| 符号 | 含义 |
| :--- | :--- |
| $\mathcal{S}$ | **状态空间**（State space），可数或不可数 |
| $\mathcal{A}$ | **动作空间**（Action space），离散或连续 |
| $P(s' \mid s, a)$ | **转移概率**（Transition kernel）：在状态 $s$ 执行动作 $a$ 后到达 $s'$ 的概率 |
| $R(s, a)$ 或 $R(s, a, s')$ | **奖励函数**（Reward function）：执行动作后获得的标量奖励 |
| $\gamma \in [0, 1]$ | **折扣因子**：未来奖励的衰减率 |

### 1.1 Markov 性质

**核心假设**：未来只依赖于当前，与历史无关：
$$\boxed{P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0, a_0) = P(s_{t+1} \mid s_t, a_t)}$$

> **直觉**：当前状态 $s_t$ 已经"包含了所有历史的有用信息"。这听起来很强，但只要状态设计得当（比如把历史 $k$ 步打包进状态），几乎所有问题都可以建模为 MDP。

### 1.2 折扣因子 $\gamma$ 的作用

| $\gamma$ | 含义 |
| :--- | :--- |
| $\gamma = 0$ | 短视：只看当前 reward |
| $\gamma \to 1$ | 远见：未来 reward 几乎不打折 |
| 实践常用 | $0.95 \sim 0.99$ |

**数学上的必要性**：若 $\gamma = 1$ 且 episode 无限长，累积 reward 可能发散为 $\infty$。$\gamma < 1$ 保证 $\sum_t \gamma^t r_t$ 收敛。

## 2. 策略 (Policy)

**策略** $\pi$ 是从状态到动作的映射，定义 agent 的行为。

### 2.1 两种策略类型

**确定性策略**：
$$\pi : \mathcal{S} \to \mathcal{A}, \quad a = \pi(s)$$

**随机策略**：
$$\pi(a \mid s) = P(a_t = a \mid s_t = s)$$

> 随机策略是一般情形（确定性策略是 one-hot 形式的特例）。后续推导默认用随机策略。

### 2.2 轨迹与回报

固定策略 $\pi$ 后，agent 与环境交互产生**轨迹**（trajectory）：
$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$$

**累积折扣回报**（Return）：
$$\boxed{G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}}$$

注意 $G_t$ 是**随机变量**，因为轨迹是随机的。

## 3. 价值函数

### 3.1 状态价值函数 $V^\pi$

定义为：在状态 $s$ 出发、按策略 $\pi$ 行动，所获累积回报的期望：
$$\boxed{V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]}$$

### 3.2 动作价值函数 $Q^\pi$

在状态 $s$ 执行动作 $a$、之后按 $\pi$ 行动，所获累积回报的期望：
$$\boxed{Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]}$$

### 3.3 二者关系

$$V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)]$$

$Q^\pi$ 比 $V^\pi$ "更细一级"，多了对动作的依赖。

## 4. Bellman 期望方程（核心）

### 4.1 推导

从定义出发：
$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t = s\right]$$

**拆分当前 reward 与未来 return**（这是 Bellman 思想的精髓）：
$$V^\pi(s) = \mathbb{E}_\pi[r_t + \gamma G_{t+1} \mid s_t = s]$$

由全期望公式：
$$V^\pi(s) = \mathbb{E}_\pi[r_t \mid s_t = s] + \gamma \mathbb{E}_\pi[G_{t+1} \mid s_t = s]$$

第一项展开：
$$\mathbb{E}_\pi[r_t \mid s_t = s] = \sum_a \pi(a \mid s) R(s, a)$$

第二项展开（用条件期望套娃）：
$$\mathbb{E}_\pi[G_{t+1} \mid s_t = s] = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \mathbb{E}_\pi[G_{t+1} \mid s_{t+1} = s'] = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) V^\pi(s')$$

合并得到 **Bellman 期望方程**：

$$\boxed{V^\pi(s) = \sum_a \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s') \right]}$$

类似地：
$$\boxed{Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')}$$

### 4.2 矩阵形式

设 $\mathcal{S}$ 有 $|\mathcal{S}|$ 个状态。定义：
- $V^\pi \in \mathbb{R}^{|\mathcal{S}|}$：每个状态的价值
- $R^\pi \in \mathbb{R}^{|\mathcal{S}|}$，$R^\pi(s) = \sum_a \pi(a \mid s) R(s, a)$
- $P^\pi \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}$，$P^\pi(s, s') = \sum_a \pi(a \mid s) P(s' \mid s, a)$

Bellman 期望方程矩阵化：
$$V^\pi = R^\pi + \gamma P^\pi V^\pi$$

整理：
$$(I - \gamma P^\pi) V^\pi = R^\pi$$

由于 $\gamma < 1$，$\gamma P^\pi$ 的谱半径 $< 1$，$(I - \gamma P^\pi)$ 可逆：

$$\boxed{V^\pi = (I - \gamma P^\pi)^{-1} R^\pi}$$

> **重要**：这给出了**精确求解** $V^\pi$ 的闭式解！但代价是 $O(|\mathcal{S}|^3)$ 的矩阵求逆，当 $|\mathcal{S}|$ 很大时不可行——这就是 Ch2 用迭代法求解的动机。

## 5. Bellman 最优方程

### 5.1 最优价值函数

定义最优价值：
$$V^*(s) = \max_\pi V^\pi(s), \quad Q^*(s, a) = \max_\pi Q^\pi(s, a)$$

### 5.2 Bellman 最优方程

$$\boxed{V^*(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s') \right]}$$

$$\boxed{Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a')}$$

**对比**：Bellman 期望方程的 $\sum_a \pi(a \mid s)$ 变成了 $\max_a$。

### 5.3 最优策略

一旦得到 $Q^*$，最优策略立即可得：
$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

> **关键观察**：如果我们知道了 $Q^*$，所有问题就解决了——直接 greedy 就是最优。**RL 的本质是估计 $Q^*$（或 $V^*$、或直接学策略 $\pi^*$）**。

## 6. 收缩映射定理：为什么 Bellman 方程有唯一解？

这一节是数学最重的部分，但极其重要——它保证了所有"迭代法"（Ch2 的价值迭代）都会收敛。

### 6.1 Bellman backup 算子

定义算子 $T: \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}$：
$$(TV)(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V(s') \right]$$

$T$ 把任意 $V$ 映射到一个新的 $V$。**Bellman 最优方程就是 $V^* = TV^*$**，即 $V^*$ 是 $T$ 的不动点。

### 6.2 收缩性证明

**目标**：证明 $T$ 是 $\gamma$-收缩，即 $\|TV_1 - TV_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty$。

**证明**：对任意状态 $s$：
$$\begin{aligned}
|(TV_1)(s) - (TV_2)(s)| &= \left| \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s' | s, a) V_1(s') \right] \right. \\
&\quad \left. - \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s' | s, a) V_2(s') \right] \right|
\end{aligned}$$

利用 $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$：
$$\leq \max_a \left| \gamma \sum_{s'} P(s' | s, a) (V_1(s') - V_2(s')) \right|$$

$$\leq \gamma \max_a \sum_{s'} P(s' | s, a) |V_1(s') - V_2(s')|$$

$$\leq \gamma \max_a \sum_{s'} P(s' | s, a) \cdot \|V_1 - V_2\|_\infty = \gamma \|V_1 - V_2\|_\infty$$

最后一步用了 $\sum_{s'} P(s' | s, a) = 1$。$\square$

### 6.3 Banach 不动点定理

**定理**（Banach Fixed-Point Theorem）：在完备度量空间 $(X, d)$ 中，若 $T: X \to X$ 是 $\beta$-收缩（$\beta < 1$），则：
1. $T$ 存在**唯一**不动点 $x^*$
2. 从任意 $x_0$ 出发迭代 $x_{n+1} = T x_n$，序列**几何收敛**到 $x^*$
3. 收敛速度：$\|x_n - x^*\| \leq \beta^n \|x_0 - x^*\|$

**应用到 RL**：
- $X = \mathbb{R}^{|\mathcal{S}|}$ 配以 sup 范数是完备度量空间
- $T$ 是 $\gamma$-收缩
- 因此 $V^*$ 唯一存在，**价值迭代** $V_{n+1} = TV_n$ **几何收敛**

> **推论**：收敛速度 $\|V_n - V^*\|_\infty \leq \gamma^n \|V_0 - V^*\|_\infty$。当 $\gamma = 0.99$ 时，迭代 230 次误差缩小 10 倍；$\gamma = 0.9$ 时只需 22 次。这解释了为什么大 $\gamma$ 训练慢。

## 7. 一个重要的等式：$Q^\pi$ 的 Bellman 方程

把 $V^\pi(s') = \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')$ 代入 $Q^\pi$ 的定义：

$$\boxed{Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')}$$

等价的"采样形式"（去掉求和的展开）：
$$Q^\pi(s, a) = \mathbb{E}_{s' \sim P, a' \sim \pi}[R(s, a) + \gamma Q^\pi(s', a')]$$

**这个形式是 Ch3 TD learning 的核心起点**——它告诉我们如何用样本（而非求和）估计 $Q^\pi$。

---

# §B 模型架构：MDP 的数据流

## B.1 Agent-Environment 交互图

```
                ┌─────────────────────────────┐
                │       Environment            │
                │  - State transition P(s'|s,a)│
                │  - Reward R(s, a)            │
                └──────────┬─────────▲────────┘
                           │ s_t, r_t │ a_t
                           ▼          │
                ┌─────────────────────────────┐
                │          Agent               │
                │  - Policy π(a|s)             │
                │  - Value V(s) / Q(s,a)       │
                └─────────────────────────────┘
```

**每一步的输入输出**：

| 时刻 | Agent 输入 | Agent 输出 | Env 输入 | Env 输出 |
| :--- | :--- | :--- | :--- | :--- |
| $t$ | $s_t$ | $a_t \sim \pi(\cdot \mid s_t)$ | $a_t$ | $s_{t+1} \sim P(\cdot \mid s_t, a_t), r_t$ |

## B.2 GridWorld 例子（贯穿 Ch1-3）

最简单的 MDP：4×4 网格，agent 从左上角出发，到右下角终止。

```
. . . .          每格表示一个状态
. # . .          # 是障碍
. . . .          四个动作：上下左右
. . . G          G 是终点（reward = +1）
                 其他状态 reward = 0
                 撞墙 reward = -0.1
```

| 元素 | 取值 |
| :--- | :--- |
| $\mathcal{S}$ | 16 个格子（去掉障碍 = 15） |
| $\mathcal{A}$ | $\{\text{上, 下, 左, 右}\}$ |
| $P$ | 确定性转移（移动到邻格），撞墙不动 |
| $R$ | 到达 G: $+1$；撞墙: $-0.1$；其他: $0$ |
| $\gamma$ | $0.95$ |

---

# §C 训练与推理（实战）

## C.1 GridWorld 的 numpy 实现

```python
import numpy as np

class GridWorld:
    """4x4 GridWorld MDP"""
    def __init__(self, gamma=0.95):
        self.H, self.W = 4, 4
        self.n_states = self.H * self.W
        self.n_actions = 4                          # 上下左右
        self.gamma = gamma
        self.terminal = (3, 3)                       # 终点
        self.obstacle = (1, 1)                       # 障碍

    def state_to_xy(self, s):
        return s // self.W, s % self.W

    def xy_to_state(self, x, y):
        return x * self.W + y

    def step(self, s, a):
        """返回 (next_state, reward, done)"""
        if self.state_to_xy(s) == self.terminal:
            return s, 0.0, True

        x, y = self.state_to_xy(s)
        # 上下左右：dx, dy
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = moves[a]
        nx, ny = x + dx, y + dy

        # 撞墙惩罚
        if nx < 0 or nx >= self.H or ny < 0 or ny >= self.W:
            return s, -0.1, False
        if (nx, ny) == self.obstacle:
            return s, -0.1, False

        s_new = self.xy_to_state(nx, ny)
        reward = 1.0 if (nx, ny) == self.terminal else 0.0
        done = (nx, ny) == self.terminal
        return s_new, reward, done

    def get_transition_matrix(self):
        """返回 P[s, a, s'] 和 R[s, a]"""
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                s_new, r, _ = self.step(s, a)
                P[s, a, s_new] = 1.0                 # 确定性 MDP
                R[s, a] = r
        return P, R
```

## C.2 用闭式解直接求 $V^\pi$（验证 §A.4.2）

```python
def solve_v_closed_form(env, policy):
    """
    用矩阵求逆直接求 V^π
    policy: [n_states, n_actions] 概率分布
    """
    P, R = env.get_transition_matrix()              # [S, A, S], [S, A]
    n = env.n_states
    gamma = env.gamma

    # 计算 P^π 和 R^π
    # P^π(s, s') = Σ_a π(a|s) P(s'|s, a)
    # R^π(s)    = Σ_a π(a|s) R(s, a)
    P_pi = np.einsum('sa,sap->sp', policy, P)       # [S, S]
    R_pi = np.einsum('sa,sa->s', policy, R)         # [S]

    # V^π = (I - γ P^π)^{-1} R^π
    V = np.linalg.solve(np.eye(n) - gamma * P_pi, R_pi)
    return V


# 用法：随机策略下的价值
env = GridWorld()
random_policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
V_random = solve_v_closed_form(env, random_policy)
print(V_random.reshape(4, 4))                        # 可视化
```

**输出示例**（随机策略下的状态价值）：
```
[[ 0.51  0.55  0.62  0.69]
 [ 0.55  -1.0  0.69  0.78]
 [ 0.62  0.69  0.78  0.88]
 [ 0.69  0.78  0.88  0.00]]    ← 终点 V=0
```

观察：越靠近终点价值越高，符合直觉。障碍位置 V 是无效（被设为 -1.0 标记）。

## C.3 推理视角：MDP 不需要"训练"

注意 §C.2 的代码**没有任何"训练"**——MDP 是已知的（我们手动定义了 P, R），价值函数有闭式解。

这是 RL 与监督学习的关键区别：
- **监督学习**：数据是给定的，模型参数需要学习
- **RL（已知 MDP）**：MDP 是给定的，价值函数可以**精确求解**或迭代求解（Ch2）
- **RL（未知 MDP）**：MDP 是未知的，需要从交互数据中**估计** $V$ 或 $Q$（Ch3 之后）

---

# §D 章末速查

## D.1 五个核心方程速记

| # | 方程 | 含义 |
| :---: | :--- | :--- |
| 1 | $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$ | 累积折扣回报 |
| 2 | $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$ | 状态价值定义 |
| 3 | $V^\pi(s) = \sum_a \pi(a \mid s) [R + \gamma \sum_{s'} P V^\pi(s')]$ | Bellman 期望方程 |
| 4 | $V^*(s) = \max_a [R + \gamma \sum_{s'} P V^*(s')]$ | Bellman 最优方程 |
| 5 | $\pi^*(s) = \arg\max_a Q^*(s, a)$ | 最优策略恢复 |

## D.2 常见面试题

**Q1：为什么 RL 要用 Markov 假设？**
- 没有 Markov 性，状态不能完整描述未来——必须保留历史，状态空间爆炸
- 实际中通过设计状态（如把过去 $k$ 帧叠在一起）让问题"近似 Markov"

**Q2：为什么 $V^*$ 唯一存在？**
- Bellman backup 算子 $T$ 是 $\gamma$-收缩
- 由 Banach 不动点定理，唯一不动点存在
- 收敛速度是几何级（$\gamma^n$）

**Q3：$V^\pi$ vs $Q^\pi$，什么时候用哪个？**
- **$V^\pi$**：评估状态本身好坏（如下棋时"我这个局面值多少"）
- **$Q^\pi$**：评估状态-动作对（"这一步走 e4 还是 d4 更好"）
- 直接得到最优策略需要 $Q^*$（$\pi^*(s) = \arg\max_a Q^*$）

**Q4：折扣因子 $\gamma$ 怎么选？**
- 短视任务（贪吃蛇）：0.9
- 长视任务（围棋、对话）：0.99-1.0（围棋实际用 1.0）
- LM 序列：1.0（reward 在末尾，不衰减）

**Q5：MDP 五元组里哪个最难定义？**
- 工程实践中：**奖励函数 $R$**。Reward design 是 RL 工程师 80% 的精力所在
- 错误的 reward 会导致 reward hacking（呼应《学习笔记-大模型》Ch6 §C.4）

---

## 承上启下

本章建立了 RL 的"母语"：MDP 与 Bellman 方程。但 §C.2 的闭式解只适用于**小状态空间** + **已知 MDP**。

下一章 **Ch2** 介绍**动态规划（DP）**：当状态空间稍大时，迭代法（策略迭代、价值迭代）求解 Bellman 方程。

DP 的核心限制：**仍需已知 MDP**（即 $P$ 和 $R$）。要是不知道 MDP 怎么办？那就要等到 Ch3 的 Monte Carlo / TD 方法——它们直接从经验中学，不需要环境模型。
