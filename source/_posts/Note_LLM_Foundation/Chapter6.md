---
title: Chapter6 经典 RLHF：奖励模型 RM + PPO
categories: 学习笔记-大模型
date: 2026-04-01 10:00:00
mathjax: true
tags:
    - AI
    - AI面试知识
---

> **本章定位**：经典 RLHF 的"完整故事"。RM 把人类排序压缩成标量奖励，PPO 用它在 KL 约束下优化 Policy。RM + PPO 是不可分割的组合——PPO 的 reward 来自 RM。

> **承上**：Ch5 SFT 提供 Policy / Reference / RM 的初始化；Ch1 §6 的 KL 散度提供 PPO 的"防漂移"约束；Ch4 的 Stop-grad + EMA 思想直接对应 Reference Policy 的设计。
> **启下**：Ch7 用闭式解砍掉 RM 和 Critic（DPO）。

---

# §A 数学原理

## 1. 奖励模型 (RM) 的数学：Bradley-Terry 模型

### 1.1 为什么用排序而非打分？

人类不擅长打分（80 分还是 82 分？），但**非常擅长两两比较**。RM 的目标是把"人类偏好排序"压缩成一个标量打分函数 $r_\phi(x, y)$。

### 1.2 Bradley-Terry 模型

假设每个 response $y$ 有一个潜在分数 $r(x, y)$，则人类选 $y_w$ 优于 $y_l$ 的概率为：

$$P(y_w \succ y_l \mid x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} = \sigma(r(x, y_w) - r(x, y_l))$$

这是 logistic 模型的经典形式。

### 1.3 Pairwise Ranking Loss

对 BT 模型做极大似然估计，得到 RM 的损失：

$$\boxed{\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]}$$

**直觉**：拉大 $y_w$ 与 $y_l$ 的分差，分差越大 → $\sigma(\cdot)$ 越接近 1 → loss 越小。

> **关键观察**：BT 模型也是 Ch7 DPO 推导的起点，那里我们会看到这个 loss 怎么变成"无需 RM 的"DPO loss。

## 2. ORM vs PRM：推理模型时代的关键分化

| 类型 | 打分粒度 | 用途 | 代表 |
| :--- | :--- | :--- | :--- |
| **ORM (Outcome Reward Model)** | 整条 response 一个分 | 对话、写作、传统 RLHF | InstructGPT RM |
| **PRM (Process Reward Model)** | 推理过程**每一步**打分 | 数学/代码推理 (CoT) | OpenAI Let's Verify Step by Step |

PRM 是 o1 / R1 这类推理模型的核心组件——只奖励"对的最终答案"远不够，要奖励"对的中间步骤"。详细将在 Ch8 展开。

## 3. PPO 的奖励设计：Per-Token KL Penalty

经典 RLHF 的实际 reward **不是 RM 一个分**，而是逐 token 累加：

$$r_t = \begin{cases}
-\beta \cdot \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})} & t < T \\
-\beta \cdot \log \frac{\pi_\theta(y_T \mid \cdot)}{\pi_{\text{ref}}(y_T \mid \cdot)} + r_\phi(x, y) & t = T
\end{cases}$$

- **每个 token 上的 log-ratio**：防止 Policy 与 Reference 偏离（Ch1 §B.3 的 KL estimator）
- **末尾加 RM 分数**：只有最后一个 token 拿到 RM 给的最终回报
- **$\beta$**：KL 强度系数，通常 0.01–0.1

## 4. GAE：Generalized Advantage Estimation

PPO 不直接用原始 reward，需要计算**优势函数 (Advantage)** $A_t$：在状态 $s_t$ 选 action $a_t$ 比"平均水平"好多少。

GAE 用 TD-error 加权累加：
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}$$
其中 $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ 是 TD-error。

| 参数选择 | 偏差 | 方差 |
| :--- | :---: | :---: |
| $\lambda = 0$（纯 TD） | 高 | 低 |
| $\lambda = 1$（蒙特卡洛） | 0 | 高 |
| **$\lambda \approx 0.95$**（实践常用） | 中 | 中 |

LM 序列短，通常 $\gamma = 1.0$（不打折）。

## 5. PPO 目标函数：Clipped Surrogate Objective

定义重要性采样比：
$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

PPO 目标（Actor 部分）：
$$\boxed{\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]}$$

- $\epsilon$ 通常取 0.1 或 0.2
- **clip 的作用**：当 $r_t(\theta)$ 超出 $[1-\epsilon, 1+\epsilon]$ 时，梯度被截断为 0，**强制 Policy 每一步只能小幅更新**

完整目标（含 Critic value loss + entropy bonus）：
$$\mathcal{L}^{\text{PPO}} = \mathcal{L}^{\text{CLIP}} - c_1 \mathcal{L}^{\text{VF}} + c_2 \mathcal{L}^{\text{entropy}}$$

其中 $\mathcal{L}^{\text{VF}} = (V_\phi(s_t) - V_t^{\text{target}})^2$ 是 Critic 的 MSE 回归损失。

## 6. 为什么叫 "Proximal"：Trust Region 的工程化

PPO 的精神继承自 TRPO（Trust Region Policy Optimization）：
- **TRPO**：硬约束 $\text{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \le \delta$，需要二阶优化（Fisher 矩阵），计算昂贵
- **PPO**：用 clip 隐式实现"信赖域"，**只用一阶优化器（Adam）即可**——这就是 "Proximal"（接近原 Policy）的工程化

---

# §B 模型结构（PyTorch 实现）

## B.1 RM Scalar Head

由 SFT 模型改造：去掉 LM Head（输出 $V$ 维概率），换成 Scalar Head（输出 1 维分数）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTRewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.config = base_model.config
        self.backbone = base_model
        # ⭐ Scalar Head：随机初始化的线性层
        self.v_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]                # [B, L, D]

        # ⭐ 取最后一个非 padding token 的隐状态
        last_idx = attention_mask.sum(dim=1) - 1
        batch = input_ids.size(0)
        last_hidden = hidden_states[torch.arange(batch), last_idx]  # [B, D]

        return self.v_head(last_hidden)                           # [B, 1]
```

## B.2 RM 的 Pairwise Ranking Loss

```python
def compute_rm_loss(chosen_rewards, rejected_rewards):
    """
    chosen_rewards:   [B, 1] 胜出回答的分数
    rejected_rewards: [B, 1] 落败回答的分数
    """
    # ⭐ Bradley-Terry MLE：-log σ(r_w - r_l)
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
```

> **数值稳定性**：用 `F.logsigmoid` 而非 `torch.log(torch.sigmoid(x))`，前者底层用 `log(1 + exp(-x))` 更稳。

## B.3 PPO 完整训练循环

```python
def ppo_train_step(actor, critic, ref_model, reward_model,
                   prompts, optimizer_actor, optimizer_critic,
                   beta=0.05, gamma=1.0, lam=0.95, eps_clip=0.2,
                   ppo_epochs=4):

    # ============ Phase 1: Rollout（无梯度采样）============
    with torch.no_grad():
        responses = actor.generate(prompts, max_new_tokens=256)
        old_logprobs = compute_logprobs(actor, prompts, responses)    # π_old
        ref_logprobs = compute_logprobs(ref_model, prompts, responses)# π_ref
        values = critic(prompts, responses)                            # V(s_t)
        rewards_rm = reward_model(prompts, responses).squeeze(-1)      # 末尾 RM 分

    # ============ Phase 2: 计算 per-token reward + GAE ============
    # KL penalty: r_KL_t = -β * (log π_θ - log π_ref)
    kl = old_logprobs - ref_logprobs                                   # [B, L]
    rewards = -beta * kl                                                # [B, L]
    rewards[:, -1] += rewards_rm                                        # 末尾加 RM

    # GAE：从后往前累加
    advantages = compute_gae(rewards, values, gamma, lam)              # [B, L]
    returns = advantages + values                                       # [B, L]

    # ============ Phase 3: 多轮 minibatch 更新 ============
    for _ in range(ppo_epochs):                                         # 通常 4 轮
        new_logprobs = compute_logprobs(actor, prompts, responses)
        new_values = critic(prompts, responses)

        # ⭐ Importance Sampling Ratio
        ratio = torch.exp(new_logprobs - old_logprobs)

        # ⭐ Clipped Surrogate Objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic loss: MSE
        critic_loss = ((new_values - returns) ** 2).mean()

        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()

        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

    return actor_loss.item(), critic_loss.item()


def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """从后往前累加 GAE"""
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    T = rewards.size(1)
    for t in reversed(range(T)):
        next_v = values[:, t + 1] if t + 1 < T else 0
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae
    return advantages
```

## B.4 KL Estimator（呼应 Ch1 §B.3）

PPO 实现里 KL 实际是用 **k3 estimator**（无偏 + 非负）：

```python
def kl_div_k3(logp_theta, logp_ref):
    """John Schulman's k3 estimator: 无偏且总是非负"""
    log_ratio = logp_theta - logp_ref
    return (torch.exp(log_ratio) - 1) - log_ratio                       # ⭐
```

实际工程中，TRL 库对 reward 的计算就是：
```python
kl_per_token = kl_div_k3(logp_theta, logp_ref)                          # [B, L]
rewards = -beta * kl_per_token
rewards[:, -1] += rewards_rm
```

---

# §C 训练与推理

## C.1 训练流程：四模型架构总览

经典 PPO RLHF 的"四模并行"：

| 模型 | 角色 | 是否更新 | 显存占用 | 初始化 |
| :--- | :--- | :--- | :---: | :--- |
| **Actor (Policy) $\pi_\theta$** | 主角，生成 response | ✓ | 1× | SFT 模型 |
| **Critic (Value) $V_\phi$** | 教练，预估 state value | ✓ | 1× | RM 或独立初始化 |
| **Reference $\pi_{\text{ref}}$** | 标杆，防 Policy 跑偏 | ✗ 冻结 | 1× | SFT 模型副本 |
| **Reward Model $r_\phi$** | 判官，给 response 打分 | ✗ 冻结 | 1× | 第二阶段产物 |

> **回忆 Ch4 §D**：Reference Model 就是 BYOL 中的 Target Network——一个被 stop-gradient 的、提供稳定锚点的旧自己。

## C.2 训练流程：PPO 完整阶段图

```
┌──────────────────────────────────────────────────────────────────┐
│ Phase 1: Rollout（无梯度，纯采样）                                │
│   prompts → Actor.generate → responses                            │
│   responses → Actor / Ref / Critic / RM 各自前向，得到：           │
│      old_logprobs, ref_logprobs, values, RM 分数                  │
└──────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────┐
│ Phase 2: 计算 per-token reward + GAE                              │
│   rewards = -β · KL(π_θ || π_ref)                                 │
│   rewards[末尾] += RM 分数                                        │
│   advantages = GAE(rewards, values)                               │
└──────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────┐
│ Phase 3: 多轮 minibatch 更新（通常 4 轮）                          │
│   ratio = π_θ / π_θ_old                                           │
│   actor_loss = -min(ratio · A, clip(ratio, 1±ε) · A)              │
│   critic_loss = (V - returns)^2                                    │
└──────────────────────────────────────────────────────────────────┘
                                    ↓
                         回到 Phase 1，下一轮 rollout
```

## C.3 训练资源消耗（Llama-3 8B 估算）

| 项 | 数值 |
| :--- | :--- |
| **显存** | 4 × 16GB = 64GB（4 个 BF16 模型）+ 梯度 + Adam state ≈ 120GB+ |
| **典型硬件** | 8×A100 80GB |
| **训练时间** | 数十万 prompt rollout，需 1–2 周 |
| **数据需求** | RM：10万–100万对偏好；PPO：5万–20万 prompt 即可 |

> 这就是为什么 PPO 工程门槛极高，催生了 Ch7 的 DPO（少 2 个模型，offline，单机可训）。

## C.4 Reward Hacking：典型案例

PPO 训练中最常见的失败模式——Policy 学会"骗 RM 高分"而非"真正变好"：

| 类型 | 表现 | 缓解方法 |
| :--- | :--- | :--- |
| **长度偏差** | RM 偏爱长回答 → Policy 学会冗长 | RM 训练时加 length penalty |
| **Sycophancy（谄媚）** | Policy 迎合用户已有观点 | 偏好数据中加入"反 sycophancy"案例 |
| **Format gaming** | 滥用 markdown / emoji / 列表 | RM 训练数据多样化 |
| **特定 token 利用** | 重复 RM "见过的好回答里的标志短语" | online RM refresh |
| **拒答漂移** | 过度安全化，"我无法回答..." | RM 训练时平衡 helpful/harmless |

**核心防线**：
1. **per-token KL penalty** 是第一道防线（不让 Policy 跑离 SFT 太远）
2. **RM 训练数据多样化** 是治本之道
3. **online RM refresh**（每轮 PPO 后用新 Policy 生成的 response 重训 RM）

## C.5 推理视角：PPO 后的模型与 SFT 模型有何不同？

PPO 训完后的 Actor **结构上与 SFT 完全相同**——都是自回归 LM。但**输出分布发生显著变化**：

| 维度 | SFT 模型 | PPO 后模型 |
| :--- | :--- | :--- |
| **输出分布尖锐度** | 中等 | **更尖锐**（向 RM 偏好聚集） |
| **创造性** | 高 | 略下降（mode collapse 风险） |
| **遵循指令** | 中 | **更强** |
| **拒答倾向** | 中 | **更强**（向 harmless 偏好对齐） |
| **温度 0 输出质量** | 一般 | **显著更好** |

> **常见现象**：PPO 后的模型在 `temperature=0`（greedy）下表现最好，因为分布已经"足够尖锐"；继续加温度反而引入垃圾。这与 SFT 模型常用 temperature=0.7 形成鲜明对比。

---

# §D 章末速查：常见问题

**Q1：RM 模型和 Policy 一定要一样大吗？**
- 早期（InstructGPT 时代）RM 较小（如 175B Policy + 6B RM）
- **现代趋势是 RM ≥ Policy**（如 Llama 3.3 70B + 70B），因为 RM 质量直接决定 RL 上限，小 RM 容易被 Policy 钻空子

**Q2：为什么必须加 KL 惩罚？**
- 防止 Reward Hacking
- 没有 KL 时，Policy 会学到"骗 RM 高分"的捷径
- KL 把 Policy 锚定在 SFT 模型附近——这就是 Ch4 §D 的"Stop-grad + EMA Target"思想

**Q3：PPO 为什么训 4 轮 minibatch？**
- 每次 rollout 成本高（要 generate 整条序列）
- 一份数据训多轮 = 提高数据利用率
- 但训太多轮会让 $\pi_\theta$ 偏离 $\pi_{\theta_{\text{old}}}$ 太远，clip 失效

**Q4：Critic 和 RM 是同一个模型吗？**
- 形状相同（都输出标量），但**任务不同**：
  - RM：判断 (x, y) 这一对的好坏
  - Critic：给定状态 $s_t$，预估"未来累计 reward"
- 实践中 Critic 通常**用 RM 初始化**，但训练目标不同

**Q5：PPO 训完后能再做 DPO 吗？**
- 可以，常见做法是 PPO 后接 DPO 做"安全性微调"
- 但很少反过来（DPO → PPO），因为 DPO 已经把模型推到"边界"，PPO 容易让它崩

---

## 承上启下

PPO 是经典 RLHF 的完整故事，但工程上太重（4 模型、online 采样、不稳定）。**Ch7** 将看到一个革命性的简化——**DPO 用闭式解砍掉 RM 和 Critic，把 RL 变成纯监督学习**。这是 2024–2026 年的主流方向。

关键问题：**PPO 优化的 KL 约束 RL 目标，是否有闭式解？**
答案是：**有！** 而且这个闭式解一旦写出来，整个 RM + PPO 流程都可以坍缩成一个简单的 loss。下一章揭晓。
