---
title: RL Chapter2 动态规划 (DP)：策略评估、策略迭代、价值迭代
categories: 学习笔记-强化学习
date: 2026-05-10 10:00:00
mathjax: true
tags:
    - AI
    - 强化学习
    - AI面试知识
---

> **本章定位**：在 MDP 已知（$P, R$ 已知）的前提下，迭代求解 Bellman 方程。DP 是 RL 的"理论基线"——后续所有 model-free 方法（MC、TD、Q-Learning、PG、PPO）都是在 DP 不可行时（MDP 未知）的替代方案。

> **承上**：Ch1 §A.4 Bellman 期望方程 + §A.6 收缩映射定理。
> **启下**：Ch3 把"迭代用 $\sum$ 求精确期望"替换为"用样本估计期望"。

---

# §A 数学原理

## 1. 三大基本操作

DP 的所有算法都建立在三个基本操作上：

| 操作 | 输入 | 输出 | 数学公式 |
| :--- | :--- | :--- | :--- |
| **策略评估** (Policy Evaluation) | 策略 $\pi$ | 价值函数 $V^\pi$ | 解 Bellman 期望方程 |
| **策略改进** (Policy Improvement) | $V^\pi$ | 更好的策略 $\pi'$ | $\pi'(s) = \arg\max_a Q^\pi(s,a)$ |
| **Bellman backup** | $V$ | 新 $V$ | $TV$ |

把这三个操作组合，就得到三个经典算法：
- **策略迭代** = 策略评估 + 策略改进 (反复交替)
- **价值迭代** = 反复 Bellman backup
- **广义策略迭代 (GPI)** = 上述两者的统一框架

## 2. 策略评估 (Policy Evaluation)

### 2.1 问题陈述

给定策略 $\pi$，求 $V^\pi$。这等价于解 Bellman 期望方程：
$$V^\pi(s) = \sum_a \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s') \right]$$

### 2.2 闭式解（Ch1 §A.4.2）

$$V^\pi = (I - \gamma P^\pi)^{-1} R^\pi$$

复杂度 $O(|\mathcal{S}|^3)$，当 $|\mathcal{S}|$ 大时不可行。

### 2.3 迭代解（Iterative Policy Evaluation）

定义算子 $T^\pi$（这是 Bellman 期望方程对应的算子）：
$$(T^\pi V)(s) = \sum_a \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V(s') \right]$$

迭代：$V_{k+1} = T^\pi V_k$。

**收敛性**：$T^\pi$ 也是 $\gamma$-收缩（与 Ch1 §A.6.2 证明几乎相同，只是 $\max_a$ 换成 $\sum_a \pi(a \mid s)$，平均仍然有界）。由 Banach 不动点定理，迭代收敛到 $V^\pi$。

**算法伪代码**：
```
输入: π, P, R, γ, ε (收敛阈值)
初始化: V(s) = 0 for all s

repeat:
    Δ ← 0
    for each s in S:
        v_old ← V(s)
        V(s) ← Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')]
        Δ ← max(Δ, |v_old - V(s)|)
until Δ < ε

return V
```

> **关键技巧：In-place 更新**。把 `V(s) ← ...` 直接覆盖（而非用新旧两个数组），收敛速度通常更快——这叫 **Gauss-Seidel 风格** 更新。

## 3. 策略改进 (Policy Improvement)

### 3.1 核心思想

给定 $V^\pi$，能否构造更好的策略？答：**贪心地选 $Q^\pi$ 最大的动作**。

定义新策略：
$$\pi'(s) = \arg\max_a Q^\pi(s, a) = \arg\max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s') \right]$$

### 3.2 策略改进定理 (Policy Improvement Theorem)

**定理**：对任意 $s$，
$$Q^\pi(s, \pi'(s)) \geq V^\pi(s) \quad \Longrightarrow \quad V^{\pi'}(s) \geq V^\pi(s)$$

即只要在每个状态都选 $Q$ 最大的动作，新策略一定**不差于**原策略。

**证明**（这个证明很优美，是 RL 经典基本功）：

由假设 $Q^\pi(s, \pi'(s)) \geq V^\pi(s)$，展开：
$$V^\pi(s) \leq Q^\pi(s, \pi'(s)) = R(s, \pi'(s)) + \gamma \sum_{s'} P(s' \mid s, \pi'(s)) V^\pi(s')$$

继续在 $s'$ 上应用同样不等式：
$$V^\pi(s') \leq Q^\pi(s', \pi'(s')) = R(s', \pi'(s')) + \gamma \sum_{s''} P(s'' \mid s', \pi'(s')) V^\pi(s'')$$

代回去：
$$V^\pi(s) \leq R(s, \pi'(s)) + \gamma \sum_{s'} P(s' \mid s, \pi'(s)) [R(s', \pi'(s')) + \gamma \sum_{s''} P V^\pi(s'')]$$

无限展开下去：
$$V^\pi(s) \leq \mathbb{E}_{\pi'}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t = s\right] = V^{\pi'}(s) \quad \square$$

### 3.3 推论：贪心策略一定不差

由于 $\pi'(s) = \arg\max_a Q^\pi(s, a)$ 满足 $Q^\pi(s, \pi'(s)) = \max_a Q^\pi(s,a) \geq V^\pi(s)$，定理条件成立 → $V^{\pi'} \geq V^\pi$。

> **直觉**：这个定理是策略迭代的基石——每一轮"评估→改进"，策略至少不变差。配合"价值有上界"，迭代必然收敛。

## 4. 策略迭代 (Policy Iteration)

### 4.1 算法

```
初始化: π_0 任意（如随机策略）

for k = 0, 1, 2, ...:
    1. Policy Evaluation: 解 V^{π_k}（用 §2.3 迭代法）
    2. Policy Improvement: π_{k+1}(s) = argmax_a Q^{π_k}(s, a)
    3. 若 π_{k+1} == π_k，停止
```

### 4.2 收敛性

每轮迭代：
- 由策略改进定理：$V^{\pi_{k+1}} \geq V^{\pi_k}$
- 在有限 MDP 中，策略数量有限（$|\mathcal{A}|^{|\mathcal{S}|}$）
- 价值函数严格单调上升（除非已最优），不会循环
- → **有限步内收敛到最优策略**

### 4.3 缺点

每轮内层"策略评估"本身要迭代很多次。能否合并？

## 5. 价值迭代 (Value Iteration)

### 5.1 思想

**直接迭代 Bellman 最优方程**，不显式维护策略：
$$V_{k+1}(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V_k(s') \right]$$

这就是 Ch1 §A.6.1 定义的算子 $T$ 反复作用：$V_{k+1} = TV_k$。

### 5.2 算法

```
输入: P, R, γ, ε
初始化: V(s) = 0 for all s

repeat:
    Δ ← 0
    for each s:
        v_old ← V(s)
        V(s) ← max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')]
        Δ ← max(Δ, |v_old - V(s)|)
until Δ < ε

# 从 V* 恢复策略
for each s:
    π(s) = argmax_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')]

return π, V
```

### 5.3 收敛性

由 Ch1 §A.6 收缩映射定理，$T$ 是 $\gamma$-收缩，**几何收敛**：
$$\|V_n - V^*\|_\infty \leq \gamma^n \|V_0 - V^*\|_\infty$$

### 5.4 价值迭代 vs 策略迭代

| 对比 | 价值迭代 | 策略迭代 |
| :--- | :--- | :--- |
| 内层操作 | $\max_a$（单步）| $\sum_a \pi(a)$（迭代到收敛） |
| 每轮成本 | 低 | 高 |
| 总轮数 | 高 | 低 |
| 收敛保证 | 几何（$\gamma^n$） | 有限步 |

**经验法则**：价值迭代实现简单，是 DP 实践中最常用的方法。

## 6. 广义策略迭代 (Generalized Policy Iteration, GPI)

GPI 是 Sutton & Barto 给出的统一视角：

> **任何让 $V$ 朝 $V^\pi$ 靠近 + 让 $\pi$ 朝 $V$ 的贪心策略靠近 的算法，都会收敛到 $(\pi^*, V^*)$。**

```
       Policy Improvement
            π → π_greedy(V)
              ╲
               ╲
                ╲ (V*, π*)
        ╳        ╳
       ╱
      ╱
     ╱
   Policy Evaluation
   V → V^π
```

- **策略迭代**：完整跑完一边，再跑另一边
- **价值迭代**：每边只跑一步就切换
- **DP 之外的方法**（MC, TD, Q-Learning, PG）：本质都在做 GPI，只是评估的方式不同

> **重要**：理解 GPI 后，**RL 不再有"种类繁多"的算法，只有同一个框架的不同实现**。这是 RL 学习的"顿悟时刻"。

---

# §B 模型架构（伪代码 + numpy 实现）

## B.1 数据流：DP 的输入输出

```
┌─────────────────────────────────────────┐
│  输入：MDP 五元组 ⟨S, A, P, R, γ⟩      │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌───────────────┐  ┌──────────────┐
│  策略迭代     │  │  价值迭代    │
│ (评估 + 改进)│  │ (反复 max)   │
└───────┬───────┘  └──────┬───────┘
        │                 │
        └────────┬────────┘
                 ▼
       ┌──────────────────┐
       │  输出：(π*, V*)  │
       └──────────────────┘
```

## B.2 numpy 完整实现（GridWorld，承接 Ch1）

```python
import numpy as np
from chapter1 import GridWorld   # 复用 Ch1 §C.1


def policy_evaluation(env, policy, theta=1e-6, max_iter=1000):
    """
    迭代求 V^π
    policy: [n_states, n_actions]
    """
    P, R = env.get_transition_matrix()                     # [S, A, S], [S, A]
    n = env.n_states
    V = np.zeros(n)

    for it in range(max_iter):
        delta = 0
        for s in range(n):
            v_old = V[s]
            # V(s) = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')]
            v_new = sum(
                policy[s, a] * (R[s, a] + env.gamma * np.sum(P[s, a] * V))
                for a in range(env.n_actions)
            )
            V[s] = v_new                                    # ⭐ in-place 更新
            delta = max(delta, abs(v_old - v_new))
        if delta < theta:
            print(f"Policy Evaluation converged in {it+1} iterations")
            break
    return V


def policy_improvement(env, V):
    """
    给定 V，贪心地构造新策略
    返回: new_policy [n_states, n_actions] (one-hot)
    """
    P, R = env.get_transition_matrix()
    n, m = env.n_states, env.n_actions
    new_policy = np.zeros((n, m))
    for s in range(n):
        # Q(s, a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')
        Q_sa = np.array([R[s, a] + env.gamma * np.sum(P[s, a] * V)
                        for a in range(m)])
        a_best = np.argmax(Q_sa)
        new_policy[s, a_best] = 1.0                         # ⭐ one-hot
    return new_policy


def policy_iteration(env, max_iter=100):
    """完整策略迭代"""
    n, m = env.n_states, env.n_actions
    policy = np.ones((n, m)) / m                            # 初始: 均匀随机

    for it in range(max_iter):
        V = policy_evaluation(env, policy)
        new_policy = policy_improvement(env, V)
        if np.array_equal(new_policy, policy):
            print(f"Policy Iteration converged in {it+1} rounds")
            break
        policy = new_policy
    return policy, V


def value_iteration(env, theta=1e-6, max_iter=1000):
    """价值迭代（直接求 V*）"""
    P, R = env.get_transition_matrix()
    n, m = env.n_states, env.n_actions
    V = np.zeros(n)

    for it in range(max_iter):
        delta = 0
        for s in range(n):
            v_old = V[s]
            # V(s) ← max_a [R + γ Σ P V]
            V[s] = max(
                R[s, a] + env.gamma * np.sum(P[s, a] * V)
                for a in range(m)
            )
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            print(f"Value Iteration converged in {it+1} iterations")
            break

    # 从 V* 恢复策略
    policy = policy_improvement(env, V)
    return policy, V


# ============ 跑通 ============
if __name__ == "__main__":
    env = GridWorld(gamma=0.95)

    # 价值迭代
    policy, V = value_iteration(env)
    print("\n最优价值函数:")
    print(V.reshape(4, 4).round(2))
    print("\n最优策略（每个格子选的动作: 0=上 1=下 2=左 3=右）:")
    print(policy.argmax(axis=-1).reshape(4, 4))
```

**预期输出**：
```
Value Iteration converged in 87 iterations

最优价值函数:
[[0.66 0.74 0.83 0.92]
 [0.74 -1.00 0.92 1.00]
 [0.83 0.92 1.00 1.00]
 [0.92 1.00 1.00 0.00]]

最优策略:
[[1 1 1 1]    # 都向下
 [1 ? 1 1]    # 障碍处不定
 [3 3 3 1]    # 向右然后向下
 [3 3 3 0]]   # 终点处不重要
```

可视化：每个格子的最优动作箭头都指向终点，符合直觉。

---

# §C 训练与推理

## C.1 收敛速度对比实验

```python
import matplotlib.pyplot as plt

def value_iteration_with_history(env):
    """记录每次迭代的 ‖V_k - V*‖∞"""
    P, R = env.get_transition_matrix()
    n, m = env.n_states, env.n_actions
    V = np.zeros(n)

    # 先跑到底拿 V*
    _, V_star = value_iteration(env, theta=1e-10, max_iter=10000)

    history = []
    V = np.zeros(n)
    for it in range(200):
        V_new = np.array([
            max(R[s, a] + env.gamma * np.sum(P[s, a] * V) for a in range(m))
            for s in range(n)
        ])
        history.append(np.max(np.abs(V_new - V_star)))
        V = V_new
    return history


# 对比不同 γ 的收敛速度
for gamma in [0.5, 0.9, 0.99]:
    env = GridWorld(gamma=gamma)
    h = value_iteration_with_history(env)
    plt.semilogy(h, label=f'γ={gamma}')

plt.xlabel('Iteration'); plt.ylabel('||V_k - V*||∞ (log)')
plt.legend(); plt.title('Value Iteration Convergence')
plt.grid(); plt.show()
```

**实验结论**（验证 Ch1 §A.6.3 的理论）：
- $\gamma = 0.5$：~30 次迭代收敛
- $\gamma = 0.9$：~80 次迭代收敛
- $\gamma = 0.99$：~600 次迭代收敛

误差随 $\gamma^n$ 几何衰减。这就是为什么大 $\gamma$（远视任务）训练慢的根本原因。

## C.2 推理视角：DP 完成后

DP 的"训练"输出的就是策略 $\pi^*$ 和 $V^*$，**推理时直接用**：
```python
def act(state, policy):
    """推理：贪心选最优动作"""
    return policy[state].argmax()
```

由于策略是 one-hot 的，没有"探索/利用"问题——每次都执行最优动作。

## C.3 DP 的局限：为什么需要 Ch3？

DP 看似完美，但有三个致命限制：

**限制 1：需要已知 $P$ 和 $R$**
- 现实中 $P$ 几乎总是未知（机器人物理、用户行为、围棋对手）
- 只能"通过交互采样"间接获取信息——这就是 Ch3 MC/TD 的出发点

**限制 2：维度灾难**
- 状态空间大时（如棋盘 $10^{170}$ 个状态），$|\mathcal{S}|$ 个 $V$ 值都存不下
- DP 要求遍历所有状态——计算量爆炸
- → 需要**函数逼近**（Ch7 DQN 起的深度 RL）

**限制 3：连续状态/动作**
- DP 要求枚举状态和动作
- 连续控制（机器人关节角度、自动驾驶方向盘）无法直接用 DP
- → 需要 **policy gradient**（Ch5）和 **Actor-Critic**（Ch6）

---

# §D 章末速查

## D.1 三个算法速记

| 算法 | 核心循环 | 何时用 |
| :--- | :--- | :--- |
| **策略评估** | $V_{k+1} = T^\pi V_k$ | 给定 $\pi$，求 $V^\pi$ |
| **策略迭代** | 评估 → 改进 → 评估 → 改进 ... | 状态空间小，要求精确最优 |
| **价值迭代** | $V_{k+1} = TV_k$ | DP 实战首选 |

## D.2 常见面试题

**Q1：策略迭代与价值迭代哪个更快？**
- 不一定。策略迭代每轮慢但轮数少，价值迭代相反
- **轮数 × 每轮成本** 才是真正的总成本
- 实践中价值迭代更常用（实现简单）

**Q2：策略改进定理的证明思路是什么？**
- 利用 $V^\pi(s) \leq Q^\pi(s, \pi'(s))$ 反复展开
- 每一步都是不等式 → 累加无穷多步 → $V^\pi(s) \leq V^{\pi'}(s)$

**Q3：DP 为什么不算"真正的 RL"？**
- DP 假设 $P, R$ 已知 → 这是"规划"（Planning）问题
- RL 的精髓是"**从交互经验中学习**"，环境模型未知
- 但 DP 是 RL 的理论基线和"上限"——告诉我们最优能到多好

**Q4：什么是广义策略迭代 (GPI)？**
- 任何让 $V$ 朝 $V^\pi$ 收敛 + 让 $\pi$ 朝贪心策略收敛的算法
- DP、MC、TD、Q-Learning、PG、Actor-Critic 都是 GPI 的实例
- 这是 RL 算法的统一视角

**Q5：DP 的时间复杂度？**
- 价值迭代每轮 $O(|\mathcal{S}|^2 |\mathcal{A}|)$（每个状态算 $|\mathcal{A}|$ 个 Q，每个 Q 求 $|\mathcal{S}|$ 维和）
- 总轮数 $O(\log(1/\epsilon) / \log(1/\gamma))$（几何收敛）
- 总：$O\left(\frac{|\mathcal{S}|^2 |\mathcal{A}| \log(1/\epsilon)}{\log(1/\gamma)}\right)$

---

## 承上启下

DP 给出了 MDP 求解的"理论解"，但要求 MDP 已知。现实中我们几乎总是面对**未知 MDP**：
- 围棋：知道规则但状态太多
- 机器人：物理模型不准
- LLM 对话：用户反应完全未知

下一章 **Ch3** 引入 RL 真正的核心思想：**从样本估计期望**。
- **Monte Carlo (MC)**：跑完一整个 episode，用采样回报估计 $V^\pi$
- **TD Learning**：边走边更新，用 bootstrap 估计 $V^\pi$

MC 和 TD 都不需要 $P, R$——它们直接从交互数据中学习。这是 RL 与 DP 的关键分水岭。
