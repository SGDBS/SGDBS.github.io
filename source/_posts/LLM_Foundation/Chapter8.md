---
title: Chapter8 推理时代与 AI Feedback：GRPO、PRM、RLAIF、Constitutional AI
categories: 学习笔记-大模型
date: 2026-04-05 10:00:00
mathjax: true
tags:
    - AI
    - AI面试知识
---

> **本章定位**：2024–2026 年最热门的对齐前沿。三条主线：
> 1. **GRPO**：去掉 Critic 的 PPO 简化版，DeepSeek-R1 引爆
> 2. **PRM**：对推理过程每一步打分，o1 / R1 时代的关键
> 3. **RLAIF / Constitutional AI**：用 AI 反馈替代人类标注，Claude 路线

> **承上**：Ch6 PPO + Ch7 DPO 的局限——前者太重，后者无法探索；本章给出新路径。
> **本章是整套笔记的终点**。

---

# §A 数学原理

## 1. GRPO：去掉 Critic 的 PPO 变体

### 1.1 动机：Critic 是 PPO 最大的痛点

回忆 Ch6，PPO 的"四模型架构"中 Critic 占用的资源：
- 显存：与 Policy 同尺寸
- 训练难度：response-level value 是高度抽象的概念，难学准
- 不稳定性：Critic 误差会传导到 advantage 估计

**有没有办法不要 Critic？**

### 1.2 GRPO 的核心思想：用 Group 平均替代 Value Baseline

GRPO（Group Relative Policy Optimization, DeepSeek 2024）：对每个 prompt $x$，**采样 $G$ 个不同 response** $\{y_1, \dots, y_G\}$，每个由 RM 打分得到 $\{r_1, \dots, r_G\}$。

定义**组内归一化优势**：
$$\boxed{\hat{A}_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}}$$

这就完全替代了 Critic：
- 不需要 $V_\phi$
- 不需要 GAE
- 优势是 **response-level**（而非 token-level）—— 一条 response 的所有 token 共享同一个 $\hat{A}_i$

### 1.3 GRPO 损失函数

形式上和 PPO 几乎一样：
$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[ \frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \min\left(r_{i,t}(\theta) \hat{A}_i,\ \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) \right] + \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

差异：
- 没有 $V_\phi$，没有 GAE
- KL **直接显式加到 loss**（不像 PPO 加到 reward 里）
- Advantage 是 response-level

### 1.4 为什么 GRPO 在推理任务上爆火？

1. **数学/代码任务有客观正确性** → RM 可以是规则验证器（unit test、答案匹配），完全无 hallucination
2. **Group 采样天然适合 best-of-N** → 强模型挖出强样本
3. **去 Critic** → 显存够训更大 Policy
4. **DeepSeek-R1-Zero 的惊人发现**：完全跳过 SFT，纯 GRPO + 规则 reward，模型自己学会了 long CoT 推理（"Aha moment"）

## 2. PRM：从 Outcome 到 Process

### 2.1 ORM 的局限

回忆 Ch6 §A.2，传统 RM 是 **ORM**（Outcome Reward Model）：只看最终答案对错。

但对于多步推理任务（数学、代码），ORM 有致命缺陷：
- 模型可能"蒙对答案但中间步骤错误"
- 模型可能"中间几步对，最后一步错"
- ORM 都给 0 分 / 1 分，无法定位问题

### 2.2 PRM 的数学

PRM（Process Reward Model）对推理过程**每一步**打分。设 response 由 $K$ 个 step 组成 $y = (s_1, s_2, \dots, s_K)$：

$$r_{\text{PRM}}(x, s_{1:k}) = \text{score that step } s_k \text{ is correct given } s_{<k}$$

**训练数据**：人工或 LLM 标注每一步是否正确。
- OpenAI PRM800K 数据集：80 万步级标注
- Math-Shepherd：用 MCTS 自动生成 PRM 数据

### 2.3 PRM 训练损失

把每步是否正确看作二分类：
$$\mathcal{L}_{\text{PRM}} = -\sum_{k=1}^K \left[ y_k \log \sigma(r_\theta(s_{1:k})) + (1 - y_k) \log(1 - \sigma(r_\theta(s_{1:k}))) \right]$$

其中 $y_k \in \{0, 1\}$ 是第 $k$ 步的人工标签。

### 2.4 PRM 在 RL 中的使用

**方式一：Step-level RL**
- 每一步 reward 由 PRM 给出
- 用 PPO/GRPO 优化整条推理路径
- DeepSeek-R1、OpenAI o1 都用这种方式

**方式二：Inference-time（best-of-N + PRM scoring）**
- 推理时采样 $N$ 个 reasoning chain
- 用 PRM 给每条打分
- 选 PRM 分数最高的（或加权平均）

## 3. RLAIF / Constitutional AI

### 3.1 动机：人类标注的瓶颈

- 人类标注**慢、贵、不一致**
- 复杂任务（代码、长文档）人类很难判断
- Scale 上不去：模型规模一旦增大，标注就成瓶颈

### 3.2 Constitutional AI 两阶段

Anthropic 的 CAI 方法：

**阶段一：SL-CAI（Self-Critique 监督学习）**
1. 让模型生成可能有害的 response
2. 用 "constitution"（一组成文原则，如"无害、有用、诚实"）让模型 **自我批判 (self-critique)**
3. 让模型 **根据批判改写 (self-revise)** response
4. 用改写后的 (prompt, revised response) 对做 SFT

**阶段二：RLAIF**
1. 让模型对 response 对做选择（"哪个更符合宪法？"），生成偏好数据
2. 用这些 AI-generated 偏好训练 RM
3. 后续与 PPO/DPO 一致

### 3.3 数学上 RLAIF 与 RLHF 完全等价

形式上没有任何变化：
- 都是 BT 模型 + Pairwise Ranking Loss 训练 RM
- 都是 PPO/DPO 优化 Policy

唯一区别在于**偏好数据来源**：
- RLHF：人类标注员
- RLAIF：用一个强 LLM（通常是另一个 Claude / GPT-4）作为 "judge"

---

# §B 模型结构（PyTorch 实现）

## B.1 GRPO 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def grpo_train_step(policy, ref_model, reward_fn,
                    prompts, optimizer,
                    G=8, beta=0.04, eps_clip=0.2, ppo_epochs=4):
    """
    GRPO：对每个 prompt 采样 G 个 response，组内归一化作 advantage
    """
    # ============ Phase 1: Group Rollout ============
    all_responses, all_logprobs_old, all_rewards = [], [], []
    with torch.no_grad():
        for _ in range(G):
            responses = policy.generate(prompts, do_sample=True, temperature=1.0)
            logprobs = compute_logprobs(policy, prompts, responses)        # [B, L]
            rewards = reward_fn(prompts, responses)                        # [B] response-level
            all_responses.append(responses)
            all_logprobs_old.append(logprobs)
            all_rewards.append(rewards)

    # 形状: [B, G, ...]
    rewards_group = torch.stack(all_rewards, dim=1)                        # [B, G]

    # ============ Phase 2: ⭐ 组内归一化优势 ============
    mean = rewards_group.mean(dim=1, keepdim=True)                          # [B, 1]
    std = rewards_group.std(dim=1, keepdim=True) + 1e-8                     # [B, 1]
    advantages = (rewards_group - mean) / std                               # [B, G] response-level

    # ============ Phase 3: 多轮更新 ============
    for _ in range(ppo_epochs):
        for g in range(G):
            responses = all_responses[g]
            logprobs_old = all_logprobs_old[g]
            adv = advantages[:, g].unsqueeze(-1)                            # [B, 1] 同一条 response 共享

            logprobs_new = compute_logprobs(policy, prompts, responses)
            ratio = torch.exp(logprobs_new - logprobs_old)                  # [B, L]

            # ⭐ Clipped Surrogate（与 PPO 完全一致）
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            # ⭐ KL 显式加到 loss（不像 PPO 加到 reward）
            with torch.no_grad():
                ref_logprobs = compute_logprobs(ref_model, prompts, responses)
            kl = (torch.exp(logprobs_new - ref_logprobs) - 1) - (logprobs_new - ref_logprobs)
            kl_loss = beta * kl.mean()

            loss = actor_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss.item()
```

**与 PPO 代码对比**：
- ❌ 没有 Critic、没有 `compute_gae`、没有 `V_target`
- ✅ 多了 group 维度的 sampling 和归一化
- ✅ KL 直接加 loss（而非通过 reward shaping）

## B.2 PRM 训练代码

```python
class PRM(nn.Module):
    """每个 step 输出一个分数"""
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.value_head = nn.Linear(base_model.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, step_end_indices, attention_mask):
        """
        input_ids:        [B, L]   完整推理序列
        step_end_indices: [B, K]   每个 step 结尾 token 的索引
        """
        outputs = self.backbone(input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
        hidden = outputs.hidden_states[-1]                                  # [B, L, D]

        # ⭐ 取每个 step 末尾 token 的隐状态，分别打分
        B, K = step_end_indices.shape
        step_hidden = torch.gather(
            hidden, 1, step_end_indices.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        )                                                                    # [B, K, D]

        return self.value_head(step_hidden).squeeze(-1)                     # [B, K]


def prm_loss(step_logits, step_labels, step_mask):
    """
    step_logits: [B, K]  PRM 对每步的打分
    step_labels: [B, K]  人工标注（0/1）
    step_mask:   [B, K]  哪些 step 有标注
    """
    loss = F.binary_cross_entropy_with_logits(
        step_logits, step_labels.float(), reduction='none'
    )                                                                        # [B, K]
    loss = (loss * step_mask).sum() / step_mask.sum()                       # 只在有标签的 step 上算
    return loss
```

## B.3 推理时的 PRM 应用：Best-of-N + Step Scoring

```python
def generate_with_prm(policy, prm, prompt, N=16):
    """采样 N 条推理路径，PRM 选最优"""
    candidates = []
    with torch.no_grad():
        for _ in range(N):
            response = policy.generate(prompt, do_sample=True, temperature=1.0)
            steps = split_into_steps(response)                              # 按 \n\n 等分割
            step_indices = compute_step_end_indices(response)
            step_scores = prm(response, step_indices)                       # [K]

            # 综合分数：最小值（最弱步骤决定整体）或平均
            score = step_scores.min()                                        # ⭐ "瓶颈步骤"策略
            candidates.append((response, score))

    return max(candidates, key=lambda x: x[1])[0]
```

> **OpenAI Let's Verify Step by Step 论文核心结论**：用 PRM 做 best-of-N 比用 ORM 好得多，尤其在数学题上 +20% 准确率。

## B.4 RLAIF：AI-as-a-judge 生成偏好数据

```python
def generate_ai_preferences(prompts, candidate_model, judge_model, constitution):
    """
    用 judge_model（通常更强的 LLM）对 candidate_model 的输出做偏好标注
    """
    preferences = []
    for prompt in prompts:
        # 1. Candidate 生成两个不同的 response
        y1 = candidate_model.generate(prompt, temperature=0.9)
        y2 = candidate_model.generate(prompt, temperature=0.9)

        # 2. ⭐ 让 judge 模型按宪法选择更好的
        judge_prompt = f"""根据以下宪法原则评判哪个回答更好：
{constitution}

问题：{prompt}
回答 A：{y1}
回答 B：{y2}

请输出 'A' 或 'B'。"""
        choice = judge_model.generate(judge_prompt)

        if choice == 'A':
            preferences.append((prompt, y1, y2))                            # (chosen, rejected)
        else:
            preferences.append((prompt, y2, y1))

    return preferences

# 后续：用这些 preferences 走标准 DPO 或 RM+PPO 流程
```

---

# §C 训练与推理

## C.1 DeepSeek-R1 训练流程：完整剖析

R1 是 GRPO + PRM 组合最经典的应用，训练流程：

```
┌──────────────────────────────────────────────────────────┐
│ 阶段 1: R1-Zero（无 SFT）                                 │
│   Base Model → 直接 GRPO + 规则 reward                    │
│   规则 reward: 答案正确 +1, 格式正确 +0.1                  │
│   → 模型自己学会 long CoT、自我反思、"Aha moment"          │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│ 阶段 2: 用 R1-Zero 生成"高质量推理数据" → 重新 SFT         │
│   解决可读性、混合语言等问题                               │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│ 阶段 3: GRPO（这次有 SFT 起点）                            │
│   规则 reward + 偏好 reward                                │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│ 阶段 4: 蒸馏到小模型                                       │
│   用 R1 生成数据 → SFT 7B/14B/32B 小模型                  │
└──────────────────────────────────────────────────────────┘
```

> **核心洞察**：R1-Zero 证明**纯 RL 也能涌现 long CoT 推理能力**——这是 LLM 训练史上的重要里程碑。

## C.2 GRPO vs PPO vs DPO：完整对比

| 维度 | PPO（Ch6） | DPO（Ch7） | GRPO（本章） |
| :--- | :--- | :--- | :--- |
| **是否需要 RM** | ✓ | ✗（隐式） | ✓ |
| **是否需要 Critic** | ✓ | ✗ | **✗** |
| **是否需要采样** | ✓ online | ✗ offline | ✓ online (group) |
| **模型数** | 4 | 2 | 3（policy + ref + RM） |
| **优势计算** | GAE (token-level) | — | Group 归一化 (response-level) |
| **KL 处理** | 加到 reward | 闭式解隐含 | 显式加到 loss |
| **代表模型** | InstructGPT, GPT-4 | Llama 3, Mistral | DeepSeek-R1, Qwen QwQ |

## C.3 RLAIF vs RLHF 工程对比

| 维度 | RLHF | RLAIF |
| :--- | :--- | :--- |
| **偏好来源** | 人类标注员 | LLM 自评 / 互评 |
| **规模** | 数十万对 | 数百万对（爬山自由） |
| **一致性** | 标注员之间分歧 | 同模型内部一致 |
| **复杂任务** | 难标注 | LLM 可处理长文档、代码 |
| **风险** | 人类偏见 | 模型偏见放大 |
| **成本** | 每对 $0.5–$5 | 每对 $0.001–$0.01 |

> **2024 后趋势**：Llama 3 用了 70%+ 的 AI 生成偏好数据；OpenAI、Anthropic、Meta 都在大规模 RLAIF。**人类反馈已经从主菜变成佐料**——只用于最关键的安全场景。

## C.4 推理视角：测试时计算 (Test-Time Compute) 的兴起

o1 / R1 时代的另一关键变化：**推理时也要花算力**。

| 推理策略 | 计算开销 | 准确率提升 |
| :--- | :---: | :---: |
| **Greedy** | 1× | baseline |
| **Best-of-N (ORM)** | N× | +5% |
| **Best-of-N (PRM)** | N× | **+15%** |
| **MCTS + PRM** | 10–100× | **+25%** |
| **Long CoT (R1 风格)** | 模型自决定 | **+30%** |

> **Scaling 范式转移**：从"训练时 scaling（更大模型 + 更多数据）"扩展到"**推理时 scaling**（生成更长 CoT、采样更多候选）"。

## C.5 推理视角：long CoT 的 sampling 策略

R1/o1 风格的 long CoT 模型（推理时主动生成几千 token 的思考链）需要不同的解码策略：

```python
# 错误做法：低温度 + 短 max_tokens
response = model.generate(prompt, temperature=0.0, max_tokens=200)
# → 模型还没"想完"就被截断

# 正确做法：保留多样性 + 充足空间
response = model.generate(
    prompt,
    temperature=0.6,                                                         # 不能 0，需要探索
    top_p=0.95,
    max_tokens=4096,                                                         # 给足思考空间
    stop=["</think>"],                                                       # 思考结束标志
)
```

**关键观察**：
- Long CoT 模型 `temperature=0` 反而更差（模型陷入单一思路）
- 温度 0.5–0.7 + 高 top_p 是 R1 / o1 的官方推荐
- max_tokens 必须 ≥ 4096，理想是 8192–16384

---

# §D 整章速查与终章总结

## D.1 现代对齐方法谱系（最终版）

| 方法 | 是否需要 RM | 是否需要 Critic | 是否需要 online 采样 | 代表模型 | 章节 |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **PPO** | ✓ | ✓ | ✓ | InstructGPT, GPT-4 | Ch6 |
| **DPO** | ✗（隐式） | ✗ | ✗ | Llama 3, Mistral, Qwen 2.5 | Ch7 |
| **IPO/KTO/ORPO/SimPO** | ✗ | ✗ | ✗ | （DPO 变种） | Ch7 |
| **GRPO** | ✓ | ✗ | ✓ | DeepSeek-R1, Qwen QwQ | **Ch8** |
| **PRM-RL** | ✓ (PRM) | 可选 | ✓ | OpenAI o1, R1 | **Ch8** |
| **RLAIF / CAI** | ✓ (AI 生成偏好) | ✓ | ✓ | Claude | **Ch8** |

> **演进核心逻辑**：
> 1. **从 PPO 到 DPO**：去掉 RM 和 Critic，把 RL 变成监督学习
> 2. **从 ORM 到 PRM**：从"看结果"到"看过程"
> 3. **从 RLHF 到 RLAIF**：从人类反馈到 AI 反馈
> 4. **从 Critic-based 到 Group-based**：从值函数估计到组内对比

## D.2 整套笔记（Ch1–Ch8）的核心串联

```
[Ch1] 数学工具箱
   │ 点积 → Attention                CE → LM Loss             KL → 对齐约束
   │   ↓                                ↓                        ↓
[Ch2] InfoNCE 视觉对比          [Ch5] SFT (CE)             [Ch6] PPO (KL penalty)
[Ch3] CLIP / SimCSE / BGE                                  [Ch7] DPO (KL 闭式解)
[Ch4] BYOL / SimSiam / DINO   ←──── stop-grad + EMA ────→  Reference Policy
                                                            [Ch8] GRPO / PRM / RLAIF
```

每一章都建立在前面章节的概念之上：
- Attention 的点积 ⊆ Ch1
- 对比学习的 InfoNCE ⊆ Ch2
- 多模态/文本对比的 CLIP/SimCSE ⊆ Ch3
- BYOL 的 stop-grad ⊆ Ch4 → RLHF 的 reference policy ⊆ Ch6
- Bradley-Terry ⊆ Ch6 → DPO 闭式解推导 ⊆ Ch7
- PPO clip ⊆ Ch6 → GRPO 简化 ⊆ Ch8

## D.3 终章答题：你应该已经能回答这些

- ✅ 为什么 Transformer 注意力要除 $\sqrt{d_k}$？（**Ch1**）
- ✅ SimCLR / MoCo / CLIP 的 InfoNCE 公式分别长什么样？（**Ch2, Ch3**）
- ✅ BYOL 没有负样本为什么不会塌缩？（**Ch4**）
- ✅ LoRA 的低秩分解为什么有效？为什么 SFT 要把 prompt 部分 label 设为 -100？（**Ch5**）
- ✅ PPO 的 clip 为什么叫 "Proximal"？per-token KL penalty 怎么计算？（**Ch6**）
- ✅ DPO 怎么从 KL 约束 RL 闭式解推出来的？$\log Z(x)$ 为什么会消去？（**Ch7**）
- ✅ GRPO 为什么能去掉 Critic？DeepSeek-R1 怎么靠纯 RL 学会 long CoT？（**Ch8**）

如果有任何一题答不上，回到对应章节。

## D.4 常见面试题（终章版）

**Q1：为什么 GRPO 不用 Critic 也能稳？**
- Group-level 归一化提供了 baseline（mean/std）
- 当 group size $G \geq 8$ 时，归一化后的方差天然受控
- 代价是采样成本 × G，但比训练 Critic 便宜

**Q2：PRM 和 ORM 的训练数据怎么区别？**
- ORM：(prompt, response, 0/1) 三元组，10 万级
- PRM：(prompt, partial_response, 0/1) 大量条目（每条推理 K 个 step），需 80 万级
- PRM 数据更贵，但只有它能"诊断哪一步出错"

**Q3：RLAIF 的偏见放大问题怎么解？**
- 用多个不同 LLM 作为 judge（ensemble）
- 关键场景仍保留人类标注（hybrid pipeline）
- 用 constitutional principles 显式约束 judge 的判断标准

**Q4：DeepSeek-R1-Zero 的 "Aha moment" 是什么？**
- 训练后期模型自发地在 CoT 中产生"等等，让我重新检查"这类反思 token
- 这表明纯 RL（无监督模仿）也能涌现高级认知行为
- 反驳了"RLHF 必须以 SFT 为起点"的传统观点

**Q5：当前最 SOTA 的对齐 pipeline 长什么样？**
- 基本配方（2026 年）：
  1. **预训练** → Base Model
  2. **SFT**（少量高质量数据，LIMA 风格）
  3. **DPO**（大规模 AI 生成偏好对，RLAIF 路线）
  4. **GRPO** + **规则 reward**（针对推理/代码/数学专项强化）
  5. **PRM** + **best-of-N**（推理时增强，Test-Time Compute）

## D.5 推荐继续学习方向

- **更深的数学**：策略梯度定理、Natural Policy Gradient、TRPO 完整推导
- **训练加速**：FlashAttention、ZeRO、TP/PP/SP 并行策略
- **推理加速**：vLLM 的 PagedAttention、Speculative Decoding、Quantization (AWQ, GPTQ)
- **多模态**：VLM (LLaVA, Qwen-VL), 多模态 RL (RLAIF for VLM)
- **Agent 范式**：Tool Use, ReAct, AutoGen 类框架

---

## 终章寄语

从 **Ch1 的点积** 一路走到 **Ch8 的 GRPO**，看似跨越了相似度度量、表示学习、对齐算法三个看似不相关的领域，但其实它们都建立在**少数几个数学原语**上：

- **点积** 串联 Attention、InfoNCE、对比学习
- **KL 散度** 串联训练损失、PPO 约束、DPO 闭式解
- **Stop-gradient + EMA** 串联 BYOL、Reference Policy、Iterative DPO

理解这些原语之间的转换规律，比记忆任何单个算法都重要。**这套笔记的真正目标，不是让你记住 SimCLR 和 GRPO 的公式，而是让你看到它们背后的同一套数学语言**。

如果你能在面试或工作中，从一个新算法的损失函数中**一眼看出它属于哪一族、解决什么旧问题、可能引入什么新坑**——那这套笔记就完成了它的使命。
