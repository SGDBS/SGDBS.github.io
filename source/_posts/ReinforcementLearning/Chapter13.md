---
title: RL Chapter13 RL × LLM 综述：PPO、DPO、GRPO 的统一视角
categories: 学习笔记-强化学习
date: 2026-05-21 10:00:00
mathjax: true
tags:
    - AI
    - 强化学习
    - AI面试知识
---

> **本章定位**：整套笔记的终章。把前 12 章的 RL 知识应用到 LLM 时代——PPO/DPO/GRPO/RLAIF/PRM 在数学和工程上分别属于哪个 RL 范式，为什么不同公司选不同算法，以及 R1/o1 等推理模型如何"用 RL 涌现思考能力"。

> **承上**：本系列全部章节。
> **关联**：与《学习笔记-大模型》Ch5-Ch8 完全对应（双向引用）。

---

# §A 数学原理

## 1. LLM 作为 MDP

把 LLM 放进 RL 框架（呼应 Ch1 §A）：

| MDP 元素 | LLM 中的对应 |
| :--- | :--- |
| **状态 $s$** | (prompt, 已生成的 token 序列) |
| **动作 $a$** | 词表中的下一个 token |
| **状态转移 $P$** | 确定性（拼接 token） |
| **奖励 $r$** | 通常仅在 EOS 时由 RM 给一个标量；或每 token 加 KL penalty |
| **折扣 $\gamma$** | 通常 1.0（reward 只在末尾） |
| **策略 $\pi$** | LLM 自身 |

**特殊性**：
- 动作空间巨大（词表 ~5万-30万）
- 序列很长（几十到几千 token）
- Reward 极度稀疏（一条几百 token 的 response 才一个数）
- 状态确定性 → 不需要 model-free 的方差控制（用蒙特卡洛即可）

## 2. RLHF 三阶段在 RL 视角下的定位

### 2.1 阶段 1：SFT = Behavior Cloning（Ch12 §A.2）

$$\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x, y) \sim \mathcal{D}^E}\left[\sum_t \log \pi_\theta(y_t \mid x, y_{<t})\right]$$

- **专家** = 人工标注员
- **演示数据** = (prompt, response) 对
- **限制** = compounding errors（Ch12 §A.2.2）→ 这是为什么 SFT 不够

### 2.2 阶段 2：RM 训练 = MaxEnt IRL（Ch12 §A.5）

从偏好对学 reward 函数：
$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

这是 Bradley-Terry 模型的 MLE，对应 IRL 框架中"从偏好反推 reward"的特化。

### 2.3 阶段 3：策略优化 = RL on learned reward

PPO（Ch8）、DPO（本章 §3）、GRPO（本章 §4）的不同选择。

## 3. PPO RLHF：经典 RL 直接应用

### 3.1 完整算法（呼应《学习笔记-大模型》Ch6）

```
四模型架构：
  - Actor (Policy) π_θ：被更新
  - Critic V_φ：被更新（估计每 token 的累积 reward）
  - Reference π_ref：冻结，提供 KL 锚点
  - Reward Model r_φ：冻结，给末尾打分

每步流程：
  1. Rollout: 用 π_θ generate 多条 response
  2. Per-token reward shaping:
       r_t = -β · log(π_θ(y_t)/π_ref(y_t))     # per-token KL
       r_T += rm_score                          # 末尾加 RM 分
  3. GAE: 计算 advantage
  4. Clipped PPO update（同 Ch8 §A.3.2）
```

### 3.2 与 Ch8 经典 PPO 的两点差异

1. **Reward 来源不同**：经典 PPO 来自环境，LLM PPO 来自 RM
2. **per-token KL**：把 KL 注入 reward 而非 loss，本质是把 KL 约束局部化到每一步

### 3.3 为什么 PPO 在 LLM 上很难训？

- **4 个模型 × 大尺寸**：显存灾难（70B Policy + 70B Critic + 70B Ref + 70B RM = 280B 参数同框）
- **Rollout 慢**：generate 长 response 需要数千次 forward
- **超参敏感**：β、clip-ε、lr 互相耦合
- **Reward Hacking**：Policy 学会骗 RM（呼应《学习笔记-大模型》Ch6 §C.4）

→ 这些痛点催生了 DPO。

## 4. DPO：把 RL 变成监督学习

### 4.1 核心推导（呼应《学习笔记-大模型》Ch7 §A）

**Step 1**：KL-约束 RL 的闭式解（Ch11 §A.5.1）
$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp(r(x, y) / \beta)$$

**Step 2**：反解 reward
$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

**Step 3**：代入 Bradley-Terry
$$P(y_w \succ y_l) = \sigma\left(\beta \log \frac{\pi^*(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi^*(y_l)}{\pi_{\text{ref}}(y_l)}\right)$$

$\log Z(x)$ 自动消去！

**Step 4**：MLE → DPO loss
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right)\right]$$

### 4.2 DPO 在 RL 视角下是什么？

**DPO = Offline RL 在 LLM 偏好场景的特化**：
- KL-约束 RL 闭式解（Ch11 §A.5）
- 偏好对替代 advantage（无需 V 网络）
- 在固定偏好数据集上训练（offline）

> **DPO 与 IQL 的关系**：IQL 是 advantage-weighted BC，DPO 是 "log-ratio-pair-weighted MLE"，本质是同一思想的不同实现。

### 4.3 DPO 的局限

- **数据集限制**：只能在偏好数据覆盖的策略空间内优化
- **无法主动探索**：reward 高峰可能在数据外
- **Length bias**：长 response 自然 logp 小（呼应 SimPO 的 length normalize）

## 5. GRPO：去掉 Critic 的 PPO

### 5.1 算法（呼应《学习笔记-大模型》Ch8）

DeepSeek 的 GRPO 简化 PPO：
- **去掉 Critic**（不再估计 V）
- **用 Group 归一化优势替代**：每个 prompt 采样 G 条 response，组内归一化作 advantage

$$\hat{A}_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

其余（PPO clip、KL 约束）不变。

### 5.2 RL 视角

GRPO 本质是 **REINFORCE with batch baseline**：
- Vanilla REINFORCE: $\nabla J = \mathbb{E}[\nabla \log \pi · G]$
- GRPO: $\nabla J = \mathbb{E}[\nabla \log \pi · (G - \bar{G})]$，$\bar{G}$ 是同 prompt 内的平均 reward

是 Ch5 §A.5 baseline 思想的极简实现。

### 5.3 为什么 GRPO 在推理任务上爆火？

- **数学/代码任务有客观对错** → reward 由规则给（unit test、答案匹配）
- **同 prompt 多次采样**自然适合 best-of-N
- **去 Critic** → 显存够训更大 Policy
- **DeepSeek-R1-Zero 证明**：完全跳过 SFT，纯 GRPO + 规则 reward，模型自己学会 long CoT 推理

## 6. PRM：从 ORM 到 Process Reward

### 6.1 ORM 的局限

经典 RM 只对最终 response 打分（Outcome Reward Model）。问题：
- 多步推理任务（数学、代码）中间步骤错误难定位
- 模型可能"蒙对答案但中间步骤错"或"中间几步对最后一步错"

### 6.2 PRM 数学

PRM = Process Reward Model，对推理过程**每一步**打分：
$$r_{\text{PRM}}(x, s_{1:k}) = P(\text{step } s_k \text{ correct} \mid s_{<k})$$

训练（二分类）：
$$\mathcal{L}_{\text{PRM}} = -\sum_k [y_k \log \sigma(r_\theta(s_{1:k})) + (1-y_k) \log(1-\sigma(r_\theta(s_{1:k})))]$$

### 6.3 PRM 在 RL 中的使用

**方式 1（训练时）**：把 PRM 分数作为每 step 的 reward，用 PPO/GRPO 优化。
**方式 2（推理时）**：best-of-N + PRM scoring，选 PRM 最高的推理路径。

OpenAI o1、DeepSeek-R1 都用 PRM。

## 7. RLAIF：用 AI 替代人类标注

把 RM 训练数据来源从人类换成 LLM judge：
1. 用 candidate model 生成多对 response
2. 用强 LLM（如 GPT-4 / Claude）按"宪法原则"选优
3. 训练 RM
4. 后续与 RLHF 一致

数学上没有变化，**只是数据来源从人换成 AI**。但工程上完全改变 scale：从十万级偏好对扩展到百万级。

---

# §B 模型架构（统一对照）

## B.1 PPO RLHF 架构

```
                    ┌───────────────────┐
                    │   Reward Model    │  (frozen, 给末尾打分)
                    └─────────▲─────────┘
                              │ rm_score
                              │
   prompt ──→ ┌──────────────────────────┐ ──→ response
              │   Actor (Policy) π_θ      │
              └──────────┬───────────────┘
                         │ logp_θ
                         │
              ┌──────────▼───────────────┐
              │   Critic V_φ              │  (训练，估 V)
              └──────────────────────────┘
                         │ value
                         │
              ┌──────────▼───────────────┐
              │   GAE → advantage        │
              └──────────┬───────────────┘
                         │
              ┌──────────▼───────────────┐
              │   PPO Clipped Loss       │
              └──────────────────────────┘
                         ▲
                         │ logp_ref
              ┌──────────────────────────┐
              │   Reference π_ref         │  (frozen, 提供 KL 锚)
              └──────────────────────────┘
```

## B.2 DPO 架构（极简）

```
                ┌───────────────────┐
                │  Policy π_θ        │ ─── logp_θ(y_w), logp_θ(y_l)
                └───────────────────┘
                                          ↓
                                          ┌─── DPO Loss ───
                                          ↑
                ┌───────────────────┐
                │  Reference π_ref   │ ─── logp_ref(y_w), logp_ref(y_l)  (frozen)
                └───────────────────┘
```

**关键差异**：DPO 只需 2 个模型（policy + ref），没有 Critic、没有 RM。

## B.3 GRPO 架构

```
                    ┌───────────────────┐
                    │   Reward Model    │  (frozen)
                    └─────────▲─────────┘
                              │
   prompt ──→ ┌──────────────────────────┐ ──→ G 条 response
              │   Actor (Policy) π_θ      │
              └──────────┬───────────────┘
                         │
              ┌──────────▼───────────────┐
              │   Group 归一化:           │
              │   A_i = (r_i - mean)/std  │
              └──────────┬───────────────┘
                         │
              ┌──────────▼───────────────┐
              │   PPO Clipped Loss       │  ← 同 PPO，但用 group A
              │   + KL to π_ref          │
              └──────────────────────────┘
```

GRPO 比 PPO 少了 Critic，但比 DPO 多了 RM 和 online rollout。

## B.4 三种算法的核心 PyTorch 对照

```python
# ============ PPO RLHF ============
def ppo_loss(policy_logp, old_logp, advantages, eps=0.2):
    ratio = torch.exp(policy_logp - old_logp)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages
    return -torch.min(surr1, surr2).mean()

# ============ DPO ============
def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta=0.1):
    pi_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()

# ============ GRPO ============
def grpo_loss(policy_logp, old_logp, group_rewards,
              ref_logp, beta=0.04, eps=0.2):
    # ⭐ 组内归一化优势
    advantages = (group_rewards - group_rewards.mean(dim=1, keepdim=True)) \
               / (group_rewards.std(dim=1, keepdim=True) + 1e-8)

    # PPO Clip
    ratio = torch.exp(policy_logp - old_logp)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    # KL 直接加 loss（不像 PPO 加到 reward）
    kl = (torch.exp(policy_logp - ref_logp) - 1) - (policy_logp - ref_logp)
    return actor_loss + beta * kl.mean()
```

---

# §C 训练与推理

## C.1 各算法的工程对比

| 维度 | PPO | DPO | GRPO | RLAIF |
| :--- | :--- | :--- | :--- | :--- |
| **模型数量** | 4 | 2 | 3 | 4 |
| **是否需要 RM** | ✓ | ✗（隐式） | ✓ | ✓（AI 生成） |
| **是否需要 Critic** | ✓ | ✗ | ✗ | ✓ |
| **是否需要 online 采样** | ✓ | ✗ | ✓ | ✓ |
| **训练稳定性** | 中 | 高 | 中 | 中 |
| **数据效率** | 中 | 高 | 中 | 中 |
| **代表模型** | InstructGPT, GPT-4 | Llama 3, Mistral, Qwen 2.5 | DeepSeek-R1 | Claude |

## C.2 大公司的算法选择（2026 实战）

| 公司 / 模型 | 主要对齐算法 | 特点 |
| :--- | :--- | :--- |
| **OpenAI** | PPO + PRM | 资源足，追求上限 |
| **Anthropic** | RLAIF (CAI) + PPO | AI 反馈规模化 |
| **Meta (Llama 3)** | DPO + Iterative DPO | 工程友好，开源生态 |
| **Mistral** | DPO | 简单稳定 |
| **DeepSeek (R1)** | GRPO + 规则 reward | 推理任务效果惊人 |
| **Qwen 2.5** | DPO + PPO 混合 | 双管齐下 |

## C.3 R1-Zero 的 "Aha Moment"：纯 RL 涌现推理能力

DeepSeek-R1-Zero 的训练流程：
1. 直接从 base model 开始（**跳过 SFT**！）
2. 设计简单 reward：答案对 +1，格式对 +0.1
3. 用 GRPO 训练
4. ...等到训练 8000 步后，模型开始**自发地输出"等等，让我重新检查"** 之类的反思 token

**这是 RL 史上的一个里程碑**：证明纯 RL（无监督模仿）也能涌现复杂认知行为，反驳了"RLHF 必须以 SFT 为起点"的传统观点。

## C.4 LLM 推理时的"探索"

| 推理策略 | RL 类比 |
| :--- | :--- |
| `temperature=0` | DDPG 推理时取均值 |
| `temperature=0.7, top_p=0.9` | SAC 推理时采样 |
| **best-of-N** | 多臂老虎机 + 后选 |
| **best-of-N + PRM scoring** | UCB（Ch10 §A.2.2） |
| **MCTS (o1/R1 风格)** | model-based RL planning |

> **范式转移**：从"训练时 scaling"扩展到"**推理时 scaling**"。这是 2024-2026 的核心趋势。

## C.5 长 CoT 模型的特殊推理设置

R1 / o1 风格的模型：
- temperature 0.0 反而更差（陷入单一思路）
- 推荐 `temperature=0.6, top_p=0.95, max_tokens=4096`
- 必须给足 max_tokens（思考过程长达数千 token）

---

# §D 章末速查 + 整套笔记终结

## D.1 RL 概念到 LLM 的映射（终极版）

| RL 概念 | LLM 时代的对应 | 详见章节 |
| :--- | :--- | :--- |
| MDP 五元组 | (prompt, token, deterministic, RM score, γ=1) | Ch1 |
| Bellman 期望方程 | Per-token KL penalty 的累积 | Ch1, Ch6 |
| Policy Gradient | RLHF actor 更新 | Ch5, Ch8 |
| GAE | LLM PPO 的 advantage 计算 | Ch6, Ch8 |
| PPO clip | RLHF 的稳定性保证 | Ch8 |
| EMA Target | Reference Policy | Ch7, Ch9 |
| Off-policy + Replay | DPO 的固定偏好数据集 | Ch7, Ch11 |
| Behavior Cloning | SFT | Ch12 |
| MaxEnt IRL | RM 训练 | Ch12 |
| AWR / Offline RL 闭式解 | DPO 损失 | Ch11 |
| GAIL（对抗式偏好） | DPO 思想（间接） | Ch12 |
| Group baseline | GRPO 的优势归一化 | Ch5, 本章 |
| MCTS | o1/R1 推理时搜索 | （延伸） |

## D.2 整套笔记的核心串联（数学层面）

```
[Ch1 MDP + Bellman] → [Ch4 Q-Learning] → [Ch7 DQN]                  Value-based
                ↓
            [Ch5 PG] → [Ch6 AC + GAE] → [Ch8 PPO] ─────┐            Policy-based
                                                         │
                                            [Ch9 SAC]    │            Hybrid
                                                         │
                                  ┌──────────────────────┘
                                  ▼
                              【LLM 时代】
                                  │
        ┌──────────────────────┼──────────────────────────┐
        ▼                      ▼                          ▼
 [Ch12 BC = SFT]      [Ch6 + RM] = RLHF/PPO       [Ch8 PPO + clip]
                                                          │
        [Ch11 闭式解 + BT]                                │
                ↓                                          ▼
            [DPO]                              [GRPO]（去 Critic）
                                                          │
                                                          ▼
                                              [PRM + GRPO = R1]
```

## D.3 学完整套笔记后，你应该具备的能力

- ✅ 看到任何新的 RLHF 变种（KTO、ORPO、SimPO、IPO...），能立即定位它在哪个数学家族
- ✅ 看到 R1 / o1 / Claude 的训练流程，能拆解出每一步对应的 RL 算法
- ✅ 自己设计 LLM 对齐方案时，能根据数据形式（偏好对 / 单点评价 / 完整轨迹）选对应算法
- ✅ 看到一个 reward function 设计，能预判 reward hacking 风险
- ✅ 调 PPO/DPO/GRPO 超参时，能根据 KL 监控诊断训练状态

## D.4 整套笔记终极速查

**一句话总结每章**：

| 章 | 一句话 |
| :--- | :--- |
| Ch0 | RL 是"从 Thorndike 试错学习一路演化到 R1 涌现推理"的百年学科 |
| Ch1 | MDP + Bellman 方程是 RL 的"母语"，所有算法都建立在它上面 |
| Ch2 | DP 给出已知 MDP 的精确解，但需要枚举状态 |
| Ch3 | MC/TD 用样本估计期望，引入 bias-variance 权衡 |
| Ch4 | Q-Learning 是第一个 model-free 控制算法，off-policy 让数据复用成为可能 |
| Ch5 | Policy Gradient 用 log-derivative trick 直接对策略求梯度 |
| Ch6 | Actor-Critic + GAE 把 PG 工业化，bias-variance 在 advantage 估计上权衡 |
| Ch7 | DQN 用 Replay Buffer + Target Network 让 Q-Learning 适配神经网络 |
| Ch8 | PPO 用 clip 把 trust region 工程化为一阶优化，是 RLHF 的核心 |
| Ch9 | SAC 用最大熵 RL 框架，连续控制的 SOTA |
| Ch10 | 探索-利用窘境是 RL 永恒主题，RND 是稀疏 reward 的现代方案 |
| Ch11 | Offline RL 的核心是控制 OOD 高估，DPO 是它在 LLM 上的特化 |
| Ch12 | 模仿学习 = SFT 的 RL 视角，分布偏移问题催生 RLHF |
| Ch13 | LLM 时代的 PPO/DPO/GRPO/RLAIF 都是前 12 章 RL 概念的特定组合 |

---

## 终章寄语

整套笔记从 Thorndike 1898 年的迷箱实验起步，走到 2026 年 DeepSeek-R1 的"Aha moment"。**128 年间，RL 的核心问题始终是同一个**：

> 如何让一个 agent 在不知道环境规则的前提下，仅凭 reward 信号，学会做出长期最优决策？

不同年代的解法在变（DP → TD → DQN → PPO → DPO → GRPO），但底层数学语言惊人地连贯：
- **Bellman 方程** 是结构骨架
- **重要性采样 + 信赖域** 是稳定性保证
- **Stop-gradient + EMA** 是反塌缩工具
- **KL 散度** 是策略约束

理解这套**统一语言**，比记住任何一个具体算法都重要——因为新算法每年都会出现（明年说不定就有 GRPO++ 和 DPO 3.0），但底层的数学原语不会变。

如果你能在面试或工作中，从一个新算法的损失函数中**一眼看出它属于哪个 RL 家族、解决什么旧问题、可能引入什么新坑**，那这套笔记就完成了它的使命。

**祝学有所成，see you on the policy gradient.**
