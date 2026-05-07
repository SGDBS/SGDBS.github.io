---
title: Chapter4 强化学习 RLHF 与 LLM 对齐
categories: 学习笔记-大模型
date: 2026-03-30 16:03:34
mathjax: true
tags:
    - AI
    - AI面试知识
---

我们可以把 RLHF 看作是将人类的"价值观"和"偏好"量化为奖励信号，并指导模型进化的过程。本章覆盖经典 RLHF 三阶段（SFT → RM → PPO），并扩展到 2024–2026 年主流的对齐方法（DPO、GRPO、RLAIF）。

## 0. 概览：现代对齐方法谱系（TL;DR）

| 方法 | 是否需要显式 RM | 是否需要 Critic | 是否需要 online 采样 | 代表模型 |
| :--- | :---: | :---: | :---: | :--- |
| **PPO** (经典 RLHF) | ✓ | ✓ | ✓ | InstructGPT, GPT-4 |
| **DPO** | ✗（隐式） | ✗ | ✗ (offline) | Llama 3, Mistral, Qwen 2.5 |
| **GRPO** | ✓ | ✗（移除） | ✓ | DeepSeek-R1, Qwen QwQ |
| **RLAIF / CAI** | ✓（AI 生成偏好） | ✓ | ✓ | Claude |
| **KTO** | ✗ | ✗ | ✗ | （单点偏好场景） |

> **演进核心逻辑**：从"四个模型并行训"（PPO）→ "去掉 RM"（DPO）→ "去掉 Critic"（GRPO）→ "去掉人类标注"（RLAIF）。每一步都在剥离系统复杂度。

---

## 1. 第一阶段：SFT (Supervised Fine-Tuning)

将"只会预测下一个词"的**预训练语言模型（Base Model）转变为"能听懂指令"的对话模型（Assistant Model）**的关键步骤。

### 1.1 数据准备
- 格式：(Prompt, Response) 对。
- 来源：专业标注人员编写或高质量种子任务衍生。
- 规模：通常几万条到几十万条，**质量 >> 数量**（LIMA 论文：1000 条高质量数据足以解锁对话能力）。

### 1.2 数学本质：最大似然估计 (MLE)

$$\mathcal{L}_{SFT}(\theta) = - \mathbb{E}_{(x, y) \sim D_{SFT}} \left[ \sum_{t=1}^{|y|} \log P_\theta(y_t \mid x, y_{<t}) \right]$$

直观：让模型生成的分布尽可能贴近人类标注的分布。

### 1.3 关键工程细节：为什么只对 Response 计算 Loss？

虽然整条序列 `[Prompt][Response]` 都喂给模型，但 Loss 只在 Response 部分计算。

**实现方式**：构造 `labels` 张量时，把 Prompt 对应位置设为 `-100`（PyTorch 交叉熵默认 `ignore_index=-100`）：

```python
labels = input_ids.clone()
labels[:, :prompt_length] = -100   # Prompt 部分不计算 loss
# 后续 model(input_ids, labels=labels) 自动忽略 -100 位置
```

**原因**：我们不希望模型学习"用户怎么提问"（Prompt 是外部输入），只要求模型学习"给定 Prompt 如何生成 Response"。

### 1.4 全参数微调 vs LoRA

| 方式 | 训练参数 | 显存需求 | 效果 |
| :--- | :--- | :--- | :--- |
| **Full FT** | 所有权重（$W_Q, W_K, W_V, W_O$, FFN, Embedding, LM Head） | 模型参数量 4–8 倍 | 上限最高 |
| **LoRA** | 只训练旁路低秩矩阵 $A (d \times r), B (r \times d)$ | 参数量降至 0.1–1% | 略逊全量但工程友好 |

LoRA 公式：$h = W_0 x + BA x$，其中 $W_0$ 冻结。

- 早期：LoRA 通常只加在 $W_Q, W_V$。
- 现代趋势：**全层加 LoRA**（包括 MLP），效果显著提升（QLoRA、DoRA 等变种）。

### 1.5 关键代码（修正版）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from trl import DataCollatorForCompletionOnlyLM   # 正确的 SFT collator

model_id = "your-base-model-path"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],   # 现代趋势：全层加 LoRA
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./output_sft",
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
)

# 关键：DataCollatorForCompletionOnlyLM 自动把 prompt 部分的 labels 设为 -100
collator = DataCollatorForCompletionOnlyLM(
    response_template="### Response:",
    tokenizer=tokenizer,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)
trainer.train()
```

> **修正说明**：原代码用了 `DataCollatorForSeq2Seq` 是错的——那是给 T5 这类 encoder-decoder 模型用的，**不会自动 mask prompt**。Causal LM SFT 应该用 `DataCollatorForCompletionOnlyLM`（TRL 库）或自定义 collator。

### 1.6 SFT 的局限性
- **标注成本高**：写高质量回答非常累，难覆盖边界情况。
- **多解性问题**：写代码、创作类任务有很多正确答案，SFT 强迫模型只学某一种写法。
- **错误积累**：本质是模仿学习，标注里的错误会被模型学得很扎实。
- **无法学到"什么是不好的"**：SFT 只展示正例，没有负例对比。

→ 这就是为什么需要 RM + RL 阶段。

---

## 2. 第二阶段：RM (Reward Modeling)

人类不擅长打分（80 分还是 82 分？），但擅长**排序**。RM 就是把"人类排序偏好"压缩成一个标量打分函数。

### 2.1 数据构造：Pairwise 偏好数据

- **采样**：给 Prompt $x$，让 SFT 模型生成 $k$ 个不同回答 $\{y_1, \dots, y_k\}$（不同 random seed）。
- **标注**：人类两两比较，得到 $(x, y_w, y_l)$，$y_w$ 是 winner，$y_l$ 是 loser。
- **规模**：数十万到数百万对。

### 2.2 模型结构改造

由 SFT 模型改造而来：
- **保留 backbone**：Transformer Decoder 不变。
- **移除 LM Head**：原本输出维度 $V$（词表大小，如 50,257）的线性层。
- **添加 Scalar Head**：随机初始化的 `Linear(d_model, 1)`。
- **取序列末尾 token 的隐状态**经 Scalar Head → 标量分数 $r_\phi(x, y)$。

### 2.3 损失函数：Pairwise Ranking Loss (Bradley-Terry 模型)

$$L(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma\left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right]$$

**直觉**：通过拉大正负样本分差，迫使模型学习人类偏好的底层特征（安全性、事实性、语气）。

**数学背景（Bradley-Terry 模型）**：假设人类选 $y_w$ 优于 $y_l$ 的概率为
$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$$
对该模型做极大似然估计，就得到上面的 loss。这个推导在后面 §4 DPO 中是核心起点。

### 2.4 RM 规模：早期做法 vs 现代趋势

| 时代 | 典型配置 | 代表 |
| :--- | :--- | :--- |
| **早期 (2022-2023)** | RM 比 Policy 小（如 Policy 175B + RM 6B） | InstructGPT |
| **现代 (2024-2026)** | RM ≥ Policy（同尺寸或更大） | Llama 3.3 (70B + 70B), Nemotron (340B RM) |

**为什么变大**：RM 质量直接决定 RL 阶段上限。小 RM 在分布外（OOD）场景容易被 Policy 钻空子（reward hacking）。

### 2.5 ORM vs PRM：推理模型时代的关键分化

| 类型 | 打分粒度 | 用途 | 代表 |
| :--- | :--- | :--- | :--- |
| **ORM (Outcome Reward Model)** | 整条 response 一个分 | 对话、写作、传统 RLHF | InstructGPT RM |
| **PRM (Process Reward Model)** | 推理过程**每一步**打分 | 数学/代码推理（CoT） | OpenAI Let's Verify Step by Step、DeepSeek-R1 |

PRM 是 o1 / R1 这类推理模型的核心组件——只奖励"对的最终答案"远远不够，要奖励"对的中间步骤"。

### 2.6 RM 关键代码

```python
import torch
import torch.nn as nn

class GPTRewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.config = base_model.config
        self.backbone = base_model
        self.v_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state           # [B, L, D]

        # 取每个序列最后一个非 padding token
        last_idx = attention_mask.sum(dim=1) - 1
        batch = input_ids.size(0)
        last_hidden = hidden_states[torch.arange(batch), last_idx]   # [B, D]
        return self.v_head(last_hidden)                     # [B, 1]


def compute_ranking_loss(chosen_rewards, rejected_rewards):
    return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()


def train_step(model, batch, optimizer):
    model.train()
    rewards_chosen = model(batch['input_ids_chosen'], batch['attention_mask_chosen'])
    rewards_rejected = model(batch['input_ids_rejected'], batch['attention_mask_rejected'])
    loss = compute_ranking_loss(rewards_chosen, rewards_rejected)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()
```

---

## 3. 第三阶段：PPO (Proximal Policy Optimization)

经典 RLHF 的"压轴戏"。利用 RM 作为裁判，用强化学习更新 Policy。

### 3.1 四模型架构

| 模型 | 角色 | 是否更新 | 初始化 |
| :--- | :--- | :--- | :--- |
| **Actor (Policy) $\pi_\theta$** | 主角，生成 response | ✓ 更新 | SFT 模型 |
| **Critic (Value) $V_\phi$** | 教练，预估 state value | ✓ 更新 | RM 或独立初始化 |
| **Reference $\pi_{\text{ref}}$** | 标杆，防 Policy 跑偏 | ✗ 冻结 | SFT 模型副本 |
| **Reward Model $r_\phi$** | 判官，给最终 response 打分 | ✗ 冻结 | 第二阶段产物 |

> **回忆 Chapter3**：Reference Model 就是 BYOL 中的 Target Network——一个被 stop-gradient 的、提供稳定锚点的旧自己。

### 3.2 Reward 设计：per-token KL penalty

经典 RLHF 的实际 reward **不是 RM 一个分**，而是逐 token 累加：

$$r_t = \begin{cases}
-\beta \cdot \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})} & t < T \\
-\beta \cdot \log \frac{\pi_\theta(y_T \mid \cdot)}{\pi_{\text{ref}}(y_T \mid \cdot)} + r_\phi(x, y) & t = T \text{ (序列末尾)}
\end{cases}$$

- **per-token KL 惩罚**：每个 token 上的 log-ratio，防止 Policy 与 Reference 偏离。
- **末尾加 RM 分数**：只有最后一个 token 拿到 RM 给的最终回报。

> **细节**：KL 实际是用 sample-based 估计的（k1, k2, k3 estimator），TRL 库默认用 k3：$\text{KL} \approx (e^{x} - 1) - x$，比直接 log-ratio 更稳。

### 3.3 GAE：Generalized Advantage Estimation

PPO 不能用原始 reward，需要计算**优势函数 (Advantage)** $A_t$ —— "在状态 $s_t$ 选 action $a_t$ 比平均水平好多少"。

GAE 用 TD-error 加权累加：
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}$$
其中 $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ 是 TD-error。

- $\lambda = 0$：纯 TD（高偏差低方差）
- $\lambda = 1$：蒙特卡洛（无偏高方差）
- 实践常用 $\lambda = 0.95, \gamma = 1.0$（LM 序列短，无折扣）。

### 3.4 PPO 目标函数：Clipped Surrogate Objective

定义重要性采样比：
$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

PPO 目标（Actor 部分）：
$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

- $\epsilon$ 通常取 0.1 或 0.2。
- **clip 的作用**：当 $r_t(\theta)$ 超出 $[1-\epsilon, 1+\epsilon]$ 时，梯度被截断为 0，**强制 Policy 每一步只能小幅更新** —— 这就是 "**Proximal**" 的含义（接近原 Policy）。

**完整目标**（含 Critic value loss + entropy bonus）：
$$\mathcal{L}^{\text{PPO}} = \mathcal{L}^{\text{CLIP}} - c_1 \mathcal{L}^{\text{VF}} + c_2 \mathcal{L}^{\text{entropy}}$$
其中 $\mathcal{L}^{\text{VF}} = (V_\phi(s_t) - V_t^{\text{target}})^2$ 是 Critic 的回归损失。

### 3.5 为什么叫 "Proximal"：Trust Region 的工程化

PPO 的精神继承自 TRPO（Trust Region Policy Optimization）：
- TRPO：硬约束 $\text{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \le \delta$，需要二阶优化（Fisher 矩阵），计算昂贵。
- PPO：用 clip 隐式实现"信赖域"，**只用一阶优化器 (Adam) 即可**，工程上极其友好。

### 3.6 完整 PPO 训练循环

```python
# 简化版 PPO RLHF 循环
for iteration in range(num_iterations):
    # ============ Rollout 阶段 ============
    prompts = sample_prompts(batch_size)
    with torch.no_grad():
        responses = actor.generate(prompts)               # Policy 生成
        old_logprobs = actor(prompts, responses)          # π_old
        ref_logprobs = ref_model(prompts, responses)      # π_ref
        values = critic(prompts, responses)               # V(s)
        rewards_rm = reward_model(prompts, responses)     # 末尾 RM 分

    # ============ 计算 per-token reward + GAE ============
    kl = old_logprobs - ref_logprobs                      # per-token KL
    rewards = -beta * kl
    rewards[:, -1] += rewards_rm                          # 末尾加 RM
    advantages = compute_gae(rewards, values, gamma=1.0, lam=0.95)
    returns = advantages + values

    # ============ 多轮 minibatch 更新 ============
    for ppo_epoch in range(ppo_epochs):                   # 通常 4 轮
        new_logprobs = actor(prompts, responses)
        new_values = critic(prompts, responses)

        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = ((new_values - returns) ** 2).mean()
        loss = actor_loss + 0.5 * critic_loss
        loss.backward()
        optimizer.step()
```

### 3.7 Reward Hacking：典型案例

| 类型 | 表现 |
| :--- | :--- |
| **长度偏差 (Length bias)** | RM 偏爱长回答，Policy 学会冗长（最常见） |
| **Sycophancy（谄媚）** | Policy 学会迎合用户已有观点，即使错误 |
| **Format gaming** | 滥用 markdown / emoji / 列表骗 RM 高分 |
| **特定 token 利用** | 重复某些 RM "见过的好回答里的标志短语" |
| **拒答漂移** | 过度安全化，"我无法回答..."刷无害分 |

KL 惩罚是缓解这些的第一道防线，但治本要靠**RM 训练数据多样化** + **online RM refresh**。

### 3.8 PPO 的局限
- **训练不稳定**：超参数敏感（$\epsilon$、$\beta$、学习率）
- **显存压力大**：四个模型同时驻留显存
- **采样昂贵**：每次更新都要 generate
- **RM 误差累积**：Policy 越强，越容易暴露 RM 的弱点

→ 这些痛点催生了 DPO。

---

## 4. DPO (Direct Preference Optimization) —— 现代主流方案

> **DPO 是 2024-2026 年最流行的对齐方法**：Llama 3、Mistral、Qwen 2.5 都以 DPO 为主。它直接用偏好数据训练 Policy，**无需显式 RM、无需采样、无需 Critic**。

### 4.1 核心洞察：KL 约束 RL 的闭式解

经典 RLHF 在第三阶段优化的目标可以写为：
$$\max_\pi \mathbb{E}_{x, y \sim \pi}[r(x,y)] - \beta \cdot \text{KL}(\pi(\cdot \mid x) \| \pi_{\text{ref}}(\cdot \mid x))$$

这个优化问题有**闭式解**：
$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

反解出 reward：
$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

### 4.2 DPO 损失推导

把上面的 reward 表达代入 Bradley-Terry 偏好概率：
$$P(y_w \succ y_l \mid x) = \sigma\left( \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)$$

注意 $\log Z(x)$ 在两项相减时**自动消去**（这是 DPO 神奇的关键点）。

最终 DPO 损失：
$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)\right]$$

### 4.3 DPO 直觉理解

把 DPO 损失当成"分类任务"：
- 输入：$(x, y_w, y_l)$
- 输出：模型应该让 $y_w$ 在自己分布下的相对概率（vs ref）**高于** $y_l$
- 训练：标准的 BCE 损失

**关键洞察**：Policy 自己就是它自己的 reward model——根本不需要单独训练 RM。

### 4.4 DPO vs PPO 对比

| 维度 | PPO | DPO |
| :--- | :--- | :--- |
| 显式 RM | 需要 | 不需要 |
| Critic | 需要 | 不需要 |
| 采样 | online（每次重新生成） | offline（直接用偏好数据） |
| 模型数 | 4 个 | 2 个（Policy + Reference） |
| 显存 | 极高 | 中等 |
| 训练稳定性 | 难 | 容易 |
| 上限 | 高（online 探索） | 受限于偏好数据集 |
| 工程复杂度 | 高 | 低 |

### 4.5 DPO 的局限与变种

- **Length bias 仍存在**：DPO 也会学到偏好长回答 → **Length-Normalized DPO**、**SimPO** 用平均 log-prob 缓解。
- **离线限制**：数据集没覆盖到的策略空间无法探索 → **Online DPO / Iterative DPO**（用当前 policy 重新生成偏好对）。
- **过拟合 chosen / 抑制 rejected**：可能导致 NLL 下降但分布"变窄" → **IPO** 用平方损失替代 sigmoid。
- **变种家族**：KTO（单点偏好）、ORPO（无 reference 模型）、SLiC、RRHF、DPO-P 等。

---

## 5. GRPO (Group Relative Policy Optimization) —— 推理模型的标配

> DeepSeek-R1 引爆的算法。**移除 Critic**，用 group-level 归一化优势替代 value baseline。Qwen QwQ、各种 R1 复刻方案都在用。

### 5.1 动机：Critic 的代价

PPO 的 Critic 占用大量显存（与 Policy 同尺寸），且训练不稳定。在 LLM 场景下，**Critic 很难学准**——response-level value 是高度抽象的概念。

### 5.2 GRPO 核心思想：用 group 平均替代 Critic

对每个 prompt $x$，**采样 $G$ 个不同 response** $\{y_1, \dots, y_G\}$，每个由 RM 打分得到 $\{r_1, \dots, r_G\}$。

定义**组内归一化优势**：
$$\hat{A}_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}$$

这个优势在每个 token 上都使用同一个值（response-level）。

### 5.3 GRPO 目标函数

形式上和 PPO 几乎一样：
$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[ \frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \min\left(r_{i,t}(\theta) \hat{A}_i, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) \right] + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

差异：
- 没有 $V_\phi$，没有 GAE
- KL 直接显式加到 loss 里（不像 PPO 加到 reward 里）
- Advantage 是 response-level 而非 token-level

### 5.4 为什么 GRPO 在推理任务上爆火？

1. **数学/代码任务**有客观正确性 → RM 可以是规则验证器（unit test、答案匹配）
2. **Group 采样**天然适合 best-of-N → 强模型挖出强样本
3. **去 Critic** → 显存够训更大 Policy
4. **DeepSeek-R1-Zero** 证明：完全跳过 SFT，纯 GRPO + 规则 reward，模型自己学会 long CoT 推理

---

## 6. RLAIF / Constitutional AI (CAI) —— Anthropic 路线

> 用 **AI feedback** 替代 human feedback，大规模生成偏好数据。Claude 的对齐核心。

### 6.1 动机：人类标注的瓶颈

- 人类标注慢、贵、不一致
- 复杂任务（代码、长文档）人类很难判断
- Scale 上不去，模型规模一旦增大，标注就成瓶颈

### 6.2 Constitutional AI 两阶段

**阶段一：SL-CAI（Self-Critique 监督学习）**
1. 让模型生成可能有害的 response
2. 用 "constitution"（一组成文原则，如"无害、有用、诚实"）让模型**自我批判 (self-critique)**
3. 让模型**根据批判改写 (self-revise)** response
4. 用改写后的 (prompt, revised response) 对做 SFT

**阶段二：RLAIF**
1. 让模型对 response 对做选择（"哪个更符合宪法？"），生成偏好数据
2. 用这些 AI-generated 偏好训练 RM
3. 后续与 PPO/DPO 一致

### 6.3 RLAIF vs RLHF

| 维度 | RLHF | RLAIF |
| :--- | :--- | :--- |
| 偏好来源 | 人类标注员 | LLM 自评 / 互评 |
| 规模 | 数十万对 | 数百万对（爬山自由） |
| 一致性 | 标注员之间分歧 | 同模型内部一致 |
| 复杂任务 | 难标注 | LLM 可处理长文档、代码 |
| 风险 | 人类偏见 | 模型偏见放大 |

> **2024 后趋势**：Llama 3 用了 70%+ 的 AI 生成偏好数据；OpenAI、Anthropic、Meta 都在大规模 RLAIF。**人类反馈已经从主菜变成佐料**。

---

## 7. 章末速记：每个方法解决了什么痛点？

| 方法 | 解决的核心痛点 |
| :--- | :--- |
| **SFT** | 让 Base Model 学会"指令格式" |
| **RM + PPO** | 用人类排序而非绝对打分定义"好坏" |
| **per-token KL** | 防止 reward hacking |
| **PPO clip** | 把 trust region 工程化为一阶优化 |
| **DPO** | 砍掉 RM 和 Critic，把 RL 变成监督学习 |
| **GRPO** | 砍掉 Critic，用 group baseline 替代 |
| **RLAIF** | 砍掉人类标注瓶颈 |
| **PRM** | 奖励推理过程而非只看答案 |

---

## 8. 常见问题汇总

**Q1：RM 模型和 Policy 一定要一样大吗？**
早期（InstructGPT 时代）RM 较小（如 175B Policy + 6B RM）。现代趋势是 RM ≥ Policy，因为 RM 质量直接决定 RL 上限，小 RM 容易被 Policy 钻空子。

**Q2：为什么要加 KL 惩罚？**
防止 Reward Hacking。没有约束的话，Policy 会学到"骗 RM 高分"的捷径（重复 token、特定格式、谄媚）。KL 把 Policy 锚定在 SFT 模型附近。

**Q3：DPO 没有显式 RM，怎么"学到"reward？**
DPO 的关键：Policy 自身的 log-ratio $\log(\pi_\theta / \pi_{\text{ref}})$ **就是** reward 的隐式表达。这是"KL 约束 RL 闭式解"反解的结果——见 §4.1。

**Q4：GRPO 去掉 Critic 不会方差爆炸吗？**
靠 group-level normalization 控制方差。当 group size $G$ 足够大（通常 8-16），$\hat A_i$ 的方差被均值/标准差除掉了。代价是采样成本 × G。

**Q5：RLHF 总共需要多少数据？**
- SFT：1 万 – 100 万对
- RM：10 万 – 100 万 pairwise
- PPO/DPO：可以用 RM 阶段同一批偏好数据，也可以再补充

**Q6：LoRA 在 RLHF 里能用吗？**
完全可以。SFT 阶段标配；RM 阶段次之；PPO/DPO 阶段也有方案（如 PEFT-PPO）。Critic 一般也跟着 LoRA。
