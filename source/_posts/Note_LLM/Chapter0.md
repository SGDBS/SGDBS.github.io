---
title: RL Chapter0 全景与起源：从 Thorndike 的猫到 DeepSeek-R1
categories: 学习笔记-强化学习
date: 2026-05-08 10:00:00
mathjax: true
tags:
    - AI
    - 强化学习
    - AI面试知识
---

> **本系列定位**：从强化学习的思想起源讲起，建立完整的知识树。每章按"**数学原理 → 模型架构 → 训练与推理**"三视角组织。最终落到 LLM 时代的 RLHF、DPO、GRPO（与《学习笔记-大模型》系列形成双向链接）。

---

# §A 起源叙事：百年思想史

## A.1 时间轴

```
1898   Thorndike 提出"效应律"（Law of Effect）
       动物试错学习的心理学基础。猫在迷箱中通过反复尝试发现开门方法。
       ↓
1948   Wiener《控制论》
       反馈系统、自动调节，奠定"agent 与环境交互"的概念基础。
       ↓
1957   Bellman 提出"动态规划"
       最优控制问题的递归分解。Bellman 方程出现。
       ↓
1959   Samuel 西洋跳棋程序
       第一个真正意义上的 RL 系统：用 TD-like 更新自我对弈。
       ↓
1988   Sutton 提出"时序差分（TD）学习"
       缝合了试错学习 + Bellman 方程 + 蒙特卡洛方法。
       ↓
1989   Watkins 提出 Q-Learning
       第一个有收敛性保证的 model-free RL 算法。
       ↓
1992   Williams 提出 REINFORCE
       Policy Gradient 第一个实用算法，奠定 policy-based 方法基础。
       ↓
1995   Tesauro 的 TD-Gammon
       backgammon 达到人类世界冠军水平，证明 RL 实用价值。
       ↓
2013   Mnih 等提出 DQN（NIPS workshop）
       深度学习 + Q-Learning，玩 Atari 达到人类水平。RL 进入"深度时代"。
       ↓
2015   Schulman 等提出 TRPO
       第一次用 trust region 思想稳定 policy gradient。
       ↓
2016   AlphaGo 战胜李世石
       MCTS + 深度网络 + RL，里程碑式胜利。
       ↓
2017   Schulman 等提出 PPO
       工程化的 trust region，成为深度 RL 的"瑞士军刀"。
       ↓
2018   SAC（Soft Actor-Critic）
       最大熵 RL 框架，连续控制黄金标准。
       ↓
2022   InstructGPT / ChatGPT
       PPO 用于 LLM 对齐（RLHF），开启"语言 RL"时代。
       ↓
2024   DeepSeek-R1 / OpenAI o1
       GRPO + PRM，纯 RL 涌现 long CoT 推理能力。
```

## A.2 核心母题

整部 RL 史的母题只有一个：

> **如何让一个 agent 在不知道环境规则的前提下，仅凭 reward 信号，学会做出长期最优决策？**

这个问题的难点：
1. **环境未知**：不知道 $P(s' \mid s, a)$（不能像 DP 那样直接算）
2. **延迟奖励**：好坏要走到尽头才知道（不能像监督学习那样有即时标签）
3. **探索-利用窘境**：不试新动作可能错过更好选择，但试了又浪费时间
4. **信用分配**：得到 reward 后，怎么知道是哪一步功劳？

后续每一章都是对这四个难点的某种回应。

---

# §B 与监督/无监督学习的本质差异

| 维度 | 监督学习 | 无监督学习 | **强化学习** |
| :--- | :--- | :--- | :--- |
| **数据形式** | $(x, y)$ 配对 | 仅 $x$ | **$(s, a, r, s')$ 四元组流** |
| **反馈** | 即时 + 准确 | 无 | **延迟 + 评价性（仅好坏，不告诉正解）** |
| **数据分布** | 固定 | 固定 | **依赖于 agent 的策略**（策略变 → 数据分布变） |
| **目标** | 最小化预测误差 | 发现结构 | **最大化累积奖励** $\mathbb{E}[\sum \gamma^t r_t]$ |
| **核心难点** | 过拟合 | 评价指标缺失 | **探索 + 信用分配 + 非平稳性** |

> **关键差异**："监督学习教你认识猫"，"RL 教你成为捕鼠的猫"。前者要的是分类准确性，后者要的是序列决策的长期最优性。

---

# §C 知识树

```
强化学习
│
├── Part 1: 数学基础（Ch1-Ch3）
│   ├── Ch1: MDP 与 Bellman 方程
│   │   └── 五元组 / 价值函数 / Bellman 期望与最优方程 / 收缩映射
│   ├── Ch2: 动态规划（DP）
│   │   └── 策略评估 / 策略改进 / 策略迭代 / 价值迭代
│   └── Ch3: Monte Carlo 与 TD
│       └── MC 估计 / TD(0) / n-step / TD(λ) / 资格迹
│
├── Part 2: 经典算法（Ch4-Ch6）
│   ├── Ch4: Q-Learning 与 SARSA
│   │   └── Off-policy vs On-policy / ε-greedy / 收敛性
│   ├── Ch5: Policy Gradient
│   │   └── PG 定理 / REINFORCE / Baseline / Score function
│   └── Ch6: Actor-Critic
│       └── A2C/A3C / Advantage / GAE
│
├── Part 3: 深度强化学习（Ch7-Ch9）
│   ├── Ch7: DQN 家族
│   │   └── Replay Buffer / Target Network / Double/Dueling/Rainbow
│   ├── Ch8: TRPO → PPO
│   │   └── Trust Region / 重要性采样 / clip / 与 LLM RLHF 直接相连
│   └── Ch9: 连续控制（DDPG / SAC）
│       └── 确定性策略 / 最大熵 RL / 自动温度调节
│
├── Part 4: 进阶主题（Ch10-Ch12）
│   ├── Ch10: 探索（Exploration）
│   │   └── ε-greedy / UCB / Thompson / Curiosity / RND
│   ├── Ch11: 离线强化学习（Offline RL）
│   │   └── BCQ / CQL / IQL / 与 DPO 的关系
│   └── Ch12: 模仿学习与逆 RL
│       └── BC / DAgger / GAIL / IRL / 与 SFT 的关系
│
└── Part 5: RL × LLM（Ch13）
    └── 把 PPO/DPO/GRPO 在 LLM 上的应用串起来
        （与《学习笔记-大模型》Ch6-Ch8 互相引用）
```

---

# §D 三视角统一框架

继承自《学习笔记-大模型》，每章按这三个固定 section 组织：

| 视角 | 核心问题 | 内容形式 |
| :--- | :--- | :--- |
| **§A 数学原理** | "为什么这样设计？" | 完整推导，不跳步 |
| **§B 模型架构 / 算法伪代码** | "数据流长什么样？" | 输入输出图 + PyTorch 关键代码 |
| **§C 训练与推理** | "怎么用？" | Gymnasium/MuJoCo 实例 + 调参经验 |

---

# §E 章节路线与学习建议

## E.1 推荐阅读顺序

### 路线一：完整学习（推荐，约 30-40 小时）
**Ch0 → Ch1 → Ch2 → ... → Ch13**

按依赖顺序，不留盲区。

### 路线二：直奔 RLHF
**Ch0 → Ch1（精读）→ Ch5（PG）→ Ch6（AC）→ Ch8（TRPO/PPO）→ Ch13**

跳过表格法（Ch2-4）和深度 RL 视觉部分（Ch7、Ch9），最快路径到 LLM RLHF。但 **Ch1 MDP 必读**——后面所有概念都建立在它上面。

### 路线三：纯理论
**Ch1 → Ch2 → Ch3 → Ch5 → Ch11**

不写代码，纯数学推导。适合面试公式题。

## E.2 各章数学难度与工程深度

| 章节 | 数学难度 | 工程深度 | 前置依赖 |
| :--- | :---: | :---: | :--- |
| Ch1 MDP | ★★★ | ★ | 概率论 + 线性代数 |
| Ch2 DP | ★★ | ★★ | Ch1 |
| Ch3 MC/TD | ★★★ | ★★ | Ch1, Ch2 |
| Ch4 Q-Learning | ★★ | ★★★ | Ch3 |
| Ch5 Policy Gradient | ★★★★ | ★★ | Ch1（重新审视价值函数） |
| Ch6 Actor-Critic | ★★★ | ★★★ | Ch5 |
| Ch7 DQN | ★★ | ★★★★ | Ch4 + 深度学习基础 |
| Ch8 TRPO/PPO | ★★★★★ | ★★★★ | Ch5, Ch6 |
| Ch9 SAC | ★★★★ | ★★★★ | Ch8 + 信息论 |
| Ch10 Exploration | ★★★ | ★★ | Ch4 |
| Ch11 Offline RL | ★★★ | ★★★ | Ch4, Ch8 |
| Ch12 Imitation | ★★ | ★★★ | Ch5 |
| Ch13 RL × LLM | ★★★ | ★★★★ | 全部 |

---

# §F 必备数学预备

后续章节会假设你熟悉以下数学工具。如果不熟，建议先复习：

## F.1 概率论
- 期望、条件期望、全期望公式：$\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]]$
- 重要性采样（importance sampling）：$\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]$
- KL 散度（见《学习笔记-大模型》Ch1 §6）

## F.2 线性代数
- 矩阵特征值与谱半径
- 不动点与压缩映射

## F.3 优化
- 梯度下降、Adam
- 拉格朗日乘数法
- 凸优化基础（用于 TRPO 分析）

## F.4 信息论（仅 SAC、PPO entropy bonus 用到）
- 熵 $H(\pi) = -\sum_a \pi(a) \log \pi(a)$
- 互信息

---

# §G 编程环境推荐

```bash
# 基础环境
pip install torch torchvision
pip install gymnasium                  # OpenAI Gym 后继者
pip install stable-baselines3          # 工业级 RL 库（参考实现）
pip install tianshou                   # 清华开源的 PyTorch RL 库
pip install mujoco                     # 连续控制环境
pip install pettingzoo                 # 多智能体环境
```

**环境列表（贯穿全书）**：

| 环境 | 用途 |
| :--- | :--- |
| **GridWorld**（自实现） | Ch1-3 表格法演示 |
| **FrozenLake** | Ch4 Q-Learning |
| **CartPole** | Ch5-6 Policy Gradient |
| **Atari Pong/Breakout** | Ch7 DQN |
| **MuJoCo HalfCheetah/Humanoid** | Ch8-9 连续控制 |
| **Montezuma's Revenge** | Ch10 稀疏奖励探索 |
| **D4RL** | Ch11 Offline RL |

---

# §H 推荐参考资料

## 书籍
- **Sutton & Barto《Reinforcement Learning: An Introduction》(2nd ed.)** — 圣经，免费 PDF
  - http://incompleteideas.net/book/RLbook2020.pdf
- 周博磊《强化学习纲要》— 中文，覆盖现代深度 RL

## 视频课程
- **David Silver UCL RL 课程**（YouTube，B 站有翻译）— 经典中的经典
- 周博磊 RL 课程（B 站）

## 代码资源
- **Spinning Up in Deep RL**（OpenAI）— 教学版实现
  - https://spinningup.openai.com/
- **CleanRL**（GitHub）— **单文件 PyTorch 实现**，强烈推荐对照阅读
  - https://github.com/vwxyzjn/cleanrl
- **Stable-Baselines3** — 工业级，文档完善

## 论文
每章末尾会给出对应论文清单。开始写代码前先读对应论文 abstract + 算法部分。

---

# §I 学完后你应该能回答

- ✅ 什么是 MDP？为什么 RL 假设环境是 Markov 的？（Ch1）
- ✅ Bellman 方程的不动点为什么唯一？为什么价值迭代收敛？（Ch1, Ch2）
- ✅ Q-Learning 为什么是 off-policy 而 SARSA 是 on-policy？（Ch4）
- ✅ Policy Gradient 定理怎么推？为什么需要 baseline？（Ch5）
- ✅ DQN 的 Target Network 和 Replay Buffer 各解决什么问题？（Ch7）
- ✅ TRPO 的"单调改进定理"怎么证？PPO 的 clip 怎么近似 trust region？（Ch8）
- ✅ SAC 为什么要最大化熵？温度系数 α 怎么自动调？（Ch9）
- ✅ DPO 在数学上为什么是一种 offline RL？（Ch11, Ch13）
- ✅ DeepSeek-R1 的 "Aha moment" 是什么？为什么纯 RL 也能涌现 long CoT？（Ch13）

---

## 写在前面

RL 是一座"前期陡峭、后期开阔"的山。Ch1-Ch3 的数学密度可能会劝退一部分人，但翻过这座坡之后——从 Ch4 开始你会发现所有现代算法（DQN、PPO、SAC、DPO、GRPO）都是同一套数学语言的不同方言。

理解这套**统一语言**，比记住任何单一算法重要得多。

下一章 Ch1 我们正式进入 MDP——RL 的"母语"。
