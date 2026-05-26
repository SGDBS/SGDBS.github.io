---
title: Chapter7 离线对齐：DPO 家族（IPO/KTO/ORPO/SimPO）
categories: 学习笔记-大模型
date: 2026-04-03 10:00:00
mathjax: true
tags:
    - AI
    - AI面试知识
---

> **本章定位**：2024–2026 年 LLM 对齐的主流方向。DPO 用一个漂亮的数学推导**砍掉了 RM 和 Critic**，把 RL 问题变成纯监督学习。Llama 3、Mistral、Qwen 2.5 都以 DPO 为主。

> **承上**：Ch6 §A.1 的 Bradley-Terry 模型 + Ch6 §A.5 的 KL-约束 RL 目标。
> **启下**：Ch8 推理时代会回到 RL（GRPO），但 DPO 仍是绝大多数对齐场景的首选。

---

# §A 数学原理

## 1. 起点：KL-约束 RL 的优化目标

回忆 Ch6，RLHF 优化目标是：
$$\max_\pi \mathbb{E}_{x, y \sim \pi}[r(x,y)] - \beta \cdot \text{KL}(\pi(\cdot \mid x) \| \pi_{\text{ref}}(\cdot \mid x))$$

PPO 用迭代采样的方式优化它，但**这个问题其实有闭式解**。

## 2. 闭式解推导（DPO 核心一）

把 KL 展开：
$$\max_\pi \mathbb{E}_{y \sim \pi}\left[r(x,y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\right]$$

对 $\pi$ 求变分极值。设 $\pi(y \mid x)$ 的拉格朗日函数（含归一化约束 $\sum_y \pi = 1$），对 $\pi(y)$ 求导并令其为零：

$$r(x,y) - \beta(\log \pi(y \mid x) - \log \pi_{\text{ref}}(y \mid x)) - \beta - \lambda(x) = 0$$

整理得：
$$\boxed{\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)}$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp(r(x,y)/\beta)$ 是**归一化常数**（仅依赖 $x$）。

**几何直觉**：最优 Policy = 在 reference 分布上"按 reward 重加权"的分布。Reward 越高的 response 概率被乘以一个指数放大因子。

## 3. 反解 reward（DPO 核心二）

对 $\pi^*$ 取对数：
$$\log \pi^*(y \mid x) = \log \pi_{\text{ref}}(y \mid x) + \frac{1}{\beta} r(x, y) - \log Z(x)$$

反解 reward：
$$\boxed{r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)}$$

这是 DPO 最关键的等式：**reward 可以用 policy 的 log-ratio 表达**。

## 4. 代入 Bradley-Terry：log Z(x) 神奇消去

回忆 Ch6 §A.1.2，BT 模型给出偏好概率：
$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$$

把第 3 步的 reward 表达代入：
$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi^*(y_l)}{\pi_{\text{ref}}(y_l)} + \underbrace{\beta \log Z(x) - \beta \log Z(x)}_{= 0}$$

**$\log Z(x)$ 自动消去！** 这是 DPO 推导最神奇的一步——$Z(x)$ 是个棘手的、不可计算的归一化常数（要对所有 $y$ 求和），但它只依赖 $x$，在两项相减时正好抵消。

## 5. 最终：DPO 损失函数

把上面代回 BT 模型，再做 MLE，得到：

$$\boxed{\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)\right]}$$

**核心洞察**：
- 不需要训练 RM——Policy 自身的 log-ratio **就是** reward 的隐式表达
- 不需要 Critic——直接用偏好对训练
- 不需要 online 采样——纯 offline 监督学习

## 6. DPO 的几何解读

设 $\Delta_w = \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}$，$\Delta_l = \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}$。

DPO 损失变为：
$$\mathcal{L}_{\text{DPO}} = -\log \sigma(\beta(\Delta_w - \Delta_l))$$

**梯度方向**：
$$\nabla_\theta \mathcal{L}_{\text{DPO}} \propto \underbrace{(1 - \sigma(\beta(\Delta_w - \Delta_l)))}_{\text{越确信差距大，梯度越小}} \cdot (\nabla \Delta_l - \nabla \Delta_w)$$

直觉：让 $\pi_\theta$ 在 $y_w$ 上的"相对 ref 提升"超过在 $y_l$ 上的提升。差距越大（已经学得很好），梯度越小。

---

# §B 模型结构（PyTorch 实现）

## B.1 DPO 损失函数（核心）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta=0.1):
    """
    所有 logp 都是 [B] —— 整条 response 的 sum log-prob
    """
    # 计算 log-ratio
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # ⭐ 核心公式：β · (log π_θ - log π_ref) 的差
    logits = beta * (pi_logratios - ref_logratios)

    # ⭐ 等价于 -log σ(...)，但 logsigmoid 数值更稳
    losses = -F.logsigmoid(logits)
    return losses.mean()
```

## B.2 计算整条 response 的 sum log-prob

```python
def get_batch_logps(logits, labels, average=False):
    """
    logits: [B, L, V]   labels: [B, L]
    返回每条样本的 sum log-prob：Σ_t log π(y_t | y_<t)
    labels 中 -100 的位置不计入（相当于 prompt 部分）
    """
    # 错位（呼应 Ch1 §B.2）
    labels = labels[:, 1:].clone()                                       # [B, L-1]
    logits = logits[:, :-1, :]                                            # [B, L-1, V]

    loss_mask = (labels != -100)                                          # [B, L-1]
    labels = labels.masked_fill(~loss_mask, 0)                            # 填 0 防 gather 报错

    # gather 出每个位置目标 token 的 log-prob
    per_token_logps = torch.gather(
        F.log_softmax(logits, dim=-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)                                                          # [B, L-1]

    # 用 mask 屏蔽 prompt 部分，sum 或 mean
    per_token_logps = per_token_logps * loss_mask
    if average:
        return per_token_logps.sum(-1) / loss_mask.sum(-1)               # SimPO 用
    else:
        return per_token_logps.sum(-1)                                    # 标准 DPO 用
```

## B.3 完整 DPO 训练循环

```python
def dpo_train_step(policy, reference, batch, optimizer, beta=0.1):
    """
    batch 包含：
      chosen_input_ids,   chosen_labels,    chosen_attention_mask
      rejected_input_ids, rejected_labels,  rejected_attention_mask
    """
    # 1. Policy 前向：计算 sum log-prob
    chosen_logits = policy(batch['chosen_input_ids'],
                          attention_mask=batch['chosen_attention_mask']).logits
    rejected_logits = policy(batch['rejected_input_ids'],
                            attention_mask=batch['rejected_attention_mask']).logits
    policy_chosen_logps = get_batch_logps(chosen_logits, batch['chosen_labels'])
    policy_rejected_logps = get_batch_logps(rejected_logits, batch['rejected_labels'])

    # 2. ⭐ Reference 前向：no_grad，节省显存
    with torch.no_grad():
        ref_chosen_logits = reference(batch['chosen_input_ids'],
                                     attention_mask=batch['chosen_attention_mask']).logits
        ref_rejected_logits = reference(batch['rejected_input_ids'],
                                       attention_mask=batch['rejected_attention_mask']).logits
        ref_chosen_logps = get_batch_logps(ref_chosen_logits, batch['chosen_labels'])
        ref_rejected_logps = get_batch_logps(ref_rejected_logits, batch['rejected_labels'])

    # 3. DPO 损失
    loss = dpo_loss(policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps, beta=beta)

    # 4. 反传
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 监控指标：reward margin
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return loss.item(), (chosen_reward - rejected_reward).mean().item()
```

## B.4 IPO：DPO 的"防过拟合"变种

DPO 有个问题：当 chosen-rejected 差距已经很大时，loss 仍可能继续推高 $\Delta_w - \Delta_l$，导致 $\pi_\theta$ 过度偏离 $\pi_{\text{ref}}$。

IPO（Identity-PO, 2023）用平方损失替代 sigmoid：
$$\mathcal{L}_{\text{IPO}} = \mathbb{E}\left[\left(\Delta_w - \Delta_l - \frac{1}{2\beta}\right)^2\right]$$

```python
def ipo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps, beta=0.1):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios
    # ⭐ 让 logits 收敛到目标值 1/(2β) 而非无穷
    return ((logits - 1 / (2 * beta)) ** 2).mean()
```

## B.5 KTO：单点偏好（不要 pairwise）

KTO（Kahneman-Tversky Optimization）的洞察：很多场景没有成对偏好，只有"这个回答好/坏"的单点标注。

$$\mathcal{L}_{\text{KTO}} = \mathbb{E}_{(x,y)}\left[\lambda(y) - v(\Delta(x,y))\right]$$

其中 $v$ 是 prospect theory 的 value function（不对称：损失比收益感受更强）。

```python
def kto_loss(policy_logps, reference_logps, labels,
             beta=0.1, desirable_weight=1.0, undesirable_weight=1.0):
    """
    labels: [B] 标签（1=好, 0=坏）
    """
    logratios = policy_logps - reference_logps                            # [B]
    # 估算 KL 作为 prospect value 的"参考点"
    KL = (policy_logps - reference_logps).mean().clamp(min=0).detach()

    chosen_logits = beta * (logratios - KL)
    # ⭐ 好样本：用 sigmoid，坏样本：用 1 - sigmoid
    chosen_loss = desirable_weight * (1 - F.sigmoid(chosen_logits[labels == 1]))
    rejected_loss = undesirable_weight * (1 - F.sigmoid(-chosen_logits[labels == 0]))

    return torch.cat([chosen_loss, rejected_loss]).mean()
```

## B.6 ORPO：不需要 reference 模型

ORPO（Odds Ratio Preference Optimization, 2024）把 SFT 损失和偏好损失合二为一，**完全不需要 reference 模型**：

$$\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w) + \lambda \cdot \mathcal{L}_{\text{OR}}$$

其中 odds ratio loss：
$$\mathcal{L}_{\text{OR}} = -\log \sigma\left(\log \frac{\text{odds}_\theta(y_w)}{\text{odds}_\theta(y_l)}\right)$$

$\text{odds}(y) = \frac{\pi(y)}{1 - \pi(y)}$。直觉：让"选 $y_w$ 的几率"显著高于"选 $y_l$ 的几率"。

**显存节省**：DPO 需要 2 个模型（policy + ref），ORPO 只需要 1 个。

## B.7 SimPO：去掉 reference 的另一种思路

SimPO（Simple Preference Optimization, 2024）观察到 DPO 的 log-ratio 形式与"长度"耦合（response 越长，logp 越小）。

它直接用 **平均 token log-prob** 替代 sum：
$$\mathcal{L}_{\text{SimPO}} = -\mathbb{E}\left[\log \sigma\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l) - \gamma\right)\right]$$

* **平均化**：消除长度偏差
* **margin $\gamma$**：避免 logits 无限增大
* **无 reference**：直接用 policy 的 logp，不需要 $\pi_{\text{ref}}$

```python
def simpo_loss(policy_chosen_logps_avg, policy_rejected_logps_avg,
               beta=2.0, gamma=1.4):
    # ⭐ 注意是 average logp（除以长度），不是 sum
    logits = beta * (policy_chosen_logps_avg - policy_rejected_logps_avg) - gamma
    return -F.logsigmoid(logits).mean()
```

---

# §C 训练与推理

## C.1 DPO vs PPO：完整对比

| 维度 | PPO（Ch6） | DPO（本章） |
| :--- | :--- | :--- |
| **显式 RM** | 需要 | 不需要 |
| **Critic** | 需要 | 不需要 |
| **采样方式** | online（每次重新生成） | offline（直接用偏好数据） |
| **模型数** | 4 个 | 2 个（Policy + Reference） |
| **显存** | 极高 | 中等 |
| **训练稳定性** | 难 | 容易 |
| **超参敏感度** | 高 | 低 |
| **上限** | 高（online 探索） | 受限于偏好数据集 |
| **代表模型** | InstructGPT, GPT-4 | Llama 3, Mistral, Qwen 2.5 |

## C.2 DPO 训练流程

```
1. 准备偏好数据集 D = {(x, y_w, y_l)}（与 RM 训练同源）
2. 加载 SFT 模型作为 Policy 初始化
3. 复制一份冻结作为 Reference
4. 训练循环：
   for batch in D:
       计算 policy_chosen/rejected_logps
       计算 reference_chosen/rejected_logps  (no_grad)
       loss = -log σ(β · (Δ_w - Δ_l))
       backward + step
5. 训完即对齐
```

**关键超参**：
- $\beta = 0.1$（最常用），范围 0.01–0.5
  - $\beta$ 大 → 更强 KL 约束，更接近 ref
  - $\beta$ 小 → 更激进对齐，但易过拟合 chosen
- 学习率：5e-7 到 5e-6（比 SFT 小 10 倍）
- Epoch：1–3（DPO 容易过拟合）

## C.3 DPO 的局限与常见 pitfalls

### Pitfall 1：Length Bias（最常见）

DPO 的 $\sum_t \log \pi(y_t)$ 与长度相关，导致模型倾向**写长**。

**缓解**：
- SimPO（用 average logp）
- 训练数据中加入长度多样的 chosen/rejected 对

### Pitfall 2：Chosen reward 下降

监控指标：`policy_chosen_logps - reference_chosen_logps`
- 健康训练：这个数值应该上升或稳定
- 不健康：chosen 和 rejected **都在下降**，但 chosen 下降得慢——这是 DPO 的"边际效应"，模型为了拉开差距，可能同时降低两者

**缓解**：
- 加 NLL 辅助损失（在 chosen 上做 SFT 损失，正则）
- 用 IPO 替代 DPO

### Pitfall 3：Distribution shift

如果偏好数据 $(x, y_w, y_l)$ 不是来自当前 SFT 模型的输出（来自其他模型），DPO 效果会变差。

**缓解**：
- 用 SFT 模型自己生成 $y_w, y_l$（self-generated preference）
- Iterative DPO / Online DPO（详见 C.5）

## C.4 推理视角：DPO 后的模型与 PPO 后有何不同？

| 维度 | PPO 后 | DPO 后 |
| :--- | :--- | :--- |
| **输出分布** | 向 RM 偏好聚集，可能 mode collapse | 向 chosen 分布对齐，相对温和 |
| **创造性** | 显著下降 | 略下降 |
| **风格一致性** | 强 | 中等 |
| **适合的 temperature** | 0.0（greedy 最优） | 0.5–0.7（保留多样性） |
| **拒答倾向** | 强（向 harmless 偏好对齐） | 中等 |

> **经验法则**：DPO 后的模型在采样时更"灵活"，PPO 后的模型更"刻板"。这是因为 DPO 是 offline 学习，没有 online 探索 RM 的"顶峰"。

## C.5 Iterative DPO / Online DPO：闭环演进

DPO 的 offline 限制可以通过迭代缓解：

```
Round 1:
  用 SFT 模型生成多个 response → 人类/AI 标注偏好 → DPO
Round 2:
  用 Round 1 的 DPO 模型生成新 response → 重新标注 → 再次 DPO
...
```

这就是 **SPIN / Self-Rewarding LM / Online DPO** 的核心思想——本质是 BYOL 中"周期性更新 Target"思想的离散化（呼应 Ch4 §D）。

```python
# 伪代码
policy_t = SFT_model
for round in range(N):
    # 1. 用当前 policy 生成 response
    responses = [policy_t.generate(p) for p in prompts]

    # 2. 用 RM 或 LLM-as-a-judge 标注
    preferences = label(responses)

    # 3. ⭐ 用上一轮 policy 当作 reference
    policy_t = DPO_train(policy_t, reference=policy_t.copy(), preferences)
```

---

# §D 章末速查

## D.1 DPO 家族对比

| 方法 | 核心思路 | 是否需要 ref | 长度归一化 | 适用 |
| :--- | :--- | :---: | :---: | :--- |
| **DPO** | KL-RL 闭式解 + BT | ✓ | ✗ | 主流场景 |
| **IPO** | DPO 用平方损失防过拟合 | ✓ | ✗ | 数据量小、想稳 |
| **KTO** | 单点偏好 | ✓ | ✗ | 没有 pairwise 标注 |
| **ORPO** | SFT + odds ratio | **✗** | ✗ | 显存极限 |
| **SimPO** | 平均 logp 去长度偏差 | **✗** | ✓ | 长度偏差严重 |

## D.2 选型指南

- **首选 DPO**：90% 场景的默认选择
- **数据量小 (< 1万对)**：选 IPO，更稳
- **没有 pairwise 数据**：选 KTO
- **极限显存**：选 ORPO 或 SimPO（不需要 ref model）
- **想要 online 探索**：用 Iterative DPO

## D.3 常见面试题

**Q1：DPO 怎么从 KL 约束 RL 推导出来的？**
- 第一步：写出 KL 约束 RL 的拉格朗日 → 闭式解 $\pi^* = \frac{1}{Z}\pi_{\text{ref}}\exp(r/\beta)$
- 第二步：反解 $r = \beta \log(\pi^*/\pi_{\text{ref}}) + \beta \log Z$
- 第三步：代入 BT 模型，$\log Z(x)$ 在差中消去
- 第四步：MLE → DPO 损失

**Q2：DPO 的 $\beta$ 起什么作用？**
- 数学上：原 RL 目标里的 KL 强度系数
- 直觉上：控制 policy 偏离 reference 的程度
  - $\beta$ 大 → 接近 ref（保守）
  - $\beta$ 小 → 激进对齐（容易过拟合）

**Q3：为什么 DPO 不需要采样？**
- 数据集 $(x, y_w, y_l)$ 已经预先准备好
- DPO loss 只需要在 $y_w$ 和 $y_l$ 上计算 logp，不需要 model.generate()
- 这是 DPO 比 PPO 工程门槛低的根本原因

**Q4：DPO 的 reference 一定要是 SFT 模型吗？**
- 不一定，但**必须是和偏好数据"分布匹配"的模型**
- 如果偏好数据来自 SFT 模型的输出，reference 就用 SFT
- 如果偏好数据来自其他源，需要谨慎处理 distribution shift

---

## 承上启下

DPO 把"经典 RLHF"工程化成了纯监督学习，是 2024–2026 年 LLM 对齐的**主流路径**。但它仍有局限：

- **离线限制**：无法探索 RM 边界（PPO 能）
- **依赖偏好数据质量**：好数据贵
- **不适合纯客观奖励的场景**（如数学题对错）

下一章 **Ch8** 进入"推理时代"，看 GRPO / PRM / RLAIF 如何分别解决这三个问题：
- **GRPO**：去掉 Critic 的 PPO 简化版，DeepSeek-R1 同款
- **PRM**：对推理过程每一步打分（不只看最终答案）
- **RLAIF / Constitutional AI**：用 AI 反馈替代人类标注，规模化生成偏好数据
