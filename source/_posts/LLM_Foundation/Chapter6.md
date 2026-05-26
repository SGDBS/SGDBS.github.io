---
title: Chapter6 经典 RLHF：奖励模型 RM + PPO
categories: 学习笔记-大模型
date: 2026-05-22 10:00:00
mathjax: true
tags:
    - AI
    - AI面试知识
---

> **本章定位**：经典 RLHF 的"完整故事"。RM 把人类排序压缩成标量奖励，PPO 用它在 KL 约束下优化 Policy。RM + PPO 是不可分割的组合——PPO 的 reward 来自 RM。

> **承上**：Ch5 SFT 提供 Policy / Reference / RM 的初始化；Ch1 §6 的 KL 散度提供 PPO 的"防漂移"约束；Ch3 的 FlashAttention / KV Cache 等 inference 知识在 §C Infra 部分会复用；Ch4 的 Stop-grad + EMA 思想直接对应 Reference Policy 的设计。
>
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

PRM 给推理链条上的每个中间步骤都打分——这对长链推理（math、code）非常关键，因为最终答案对的不一定每一步都对，反之亦然。

---

## 3. PPO 的完整推导链（核心）

> 这一节是本章最关键的部分。PPO 的每一个设计——ratio、clip、min、trust region——都不是凭空的，而是被"前一步的问题"逼出来的。理解这条链条，PPO 就不再是一堆 trick 的堆砌。

完整逻辑链：

$$\underbrace{\text{PG 是 on-policy}}_{\text{样本浪费}} \xrightarrow{\text{IS}} \underbrace{\text{surrogate } L^{\text{IS}}}_{\text{但 ratio 会爆炸}} \xrightarrow{\text{加约束}} \underbrace{\text{TRPO 硬 KL}}_{\text{但要二阶优化}} \xrightarrow{\text{工程化}} \underbrace{\text{PPO 用 clip 隐式约束}}_{\text{只要一阶}}$$

### 3.1 Policy Gradient 的出生：log-derivative trick

强化学习目标：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int \pi_\theta(\tau) R(\tau)\, d\tau$$

求梯度：

$$\nabla_\theta J(\theta) = \int \nabla_\theta \pi_\theta(\tau) \cdot R(\tau)\, d\tau \quad (\star)$$

🛑 **问题**：$(\star)$ 式在数学上成立，**但不能用采样估计**。

**为什么不能采样估计**：Monte Carlo 估计要求被积函数能写成 $\mathbb{E}_{x \sim p(x)}[f(x)]$ 的形式，其中 $p(x)$ 是真正的概率分布（非负、积分为 1）。$\nabla \pi_\theta$ 不是概率分布（可以为负、加起来不等于 1），所以 $(\star)$ 写不成期望。

**log-derivative trick** 解决这个问题：

$$\nabla_\theta \log \pi_\theta(\tau) = \frac{\nabla_\theta \pi_\theta(\tau)}{\pi_\theta(\tau)} \Rightarrow \nabla_\theta \pi_\theta(\tau) = \pi_\theta(\tau) \cdot \nabla_\theta \log \pi_\theta(\tau)$$

本质上就是 $\frac{d \log x}{dx} = \frac{1}{x}$，但它把 $\pi_\theta(\tau)$ "挪"到了乘子位置——让积分变回期望。代入：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \nabla_\theta \log \pi_\theta(\tau) \cdot R(\tau) \right]$$

加上 reward-to-go + baseline（减方差），得到带 advantage 的形式：

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{(s,a) \sim \pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A^{\pi_\theta}(s,a) \right]}$$

> **PG 本质上是"加权的最大似然"**：$A > 0$ 时推高 $\pi_\theta(a|s)$（像 SFT），$A < 0$ 时压低它。这是 PG 和 SFT 在工程上能复用同一套代码的根本原因。

### 3.2 VPG 的死结：严格 on-policy

注意期望下标 $(s,a) \sim \pi_\theta$——**采样分布必须是当前策略**。每次 $\theta$ 更新，旧数据立刻作废。

对 LLM 来说这是灾难——一次 rollout 是 7B 模型 generate 256 token，**比 train 慢 5-10 倍**。如果一批 rollout 只能用一次梯度更新就扔，整个训练经济性崩溃。

**核心矛盾**：我们想用 $\pi_{\theta_{\text{old}}}$ 采的数据估 $\nabla J(\theta)$。这在数学上叫 **off-policy correction**。

### 3.3 重要性采样 (IS)：ratio 的真正出生

IS 恒等式：

$$\mathbb{E}_{x \sim p}[f(x)] = \int p(x) f(x)\, dx = \int q(x) \cdot \frac{p(x)}{q(x)} f(x)\, dx = \mathbb{E}_{x \sim q}\left[ \frac{p(x)}{q(x)} f(x) \right]$$

**前提**：$q(x) = 0 \Rightarrow p(x) = 0$（不然密度比无定义）。

应用到 PG（采样从 $\pi_\theta$ 换成 $\pi_{\theta_{\text{old}}}$）：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_{\theta_{\text{old}}}}\left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \cdot \nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a) \right]$$

利用
$$\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \cdot \nabla_\theta \log \pi_\theta(a|s) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \cdot \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$$

而由于 $\pi_{\theta_{\text{old}}}$ 是常数（不依赖 $\theta$），所以：

$$\frac{\nabla_\theta \pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} = \nabla_\theta \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \right]$$
可以写成"某个目标函数的梯度"：

$$\boxed{L^{\text{IS}}(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\theta_{\text{old}}}}\left[ r(\theta) \cdot A(s,a) \right]}$$

其中：

$$r(\theta) := \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$$

> **ratio 是 IS 密度比 $p/q$ 的具体化，它必须长这样，没有别的选择**。看似随意的定义其实是数学上的必然。

### 3.4 IS 的致命问题：ratio 爆炸

如果 $\pi_{\theta_{\text{old}}}(a|s) = 0.01$ 而 $\pi_\theta(a|s) = 0.5$，那 $r = 50$。一个样本主导整个 batch 的梯度——**方差爆炸**。

而 $\theta$ 离 $\theta_{\text{old}}$ 越远，ratio 越离谱。$L^{\text{IS}}$ 只在 **trust region**（信赖域）内是 $J$ 的有效代理。

🛑 **关键判断**：PPO 想约束的是**分布空间**距离（$\pi_\theta$ 和 $\pi_{\theta_{\text{old}}}$ 输出概率接近），**不是**参数空间距离（$\|\theta - \theta_{\text{old}}\|$ 小）。原因：

- **场景 A（危险）**：模型在临界区，参数动一点就让分布剧变
- **场景 B（安全）**：参数走很远但所有关心的 $(s,a)$ 上 $\pi$ 几乎不变

IS 的失效条件是**密度比偏离 1**，**和参数欧氏距离没关系**。深度网络的参数空间和分布空间严重不等距——这是 RL 里几乎所有 trust region 方法用 KL 散度（分布距离）而不是参数距离的根本原因。

### 3.5 TRPO：硬 KL 约束

$$\max_\theta L^{\text{IS}}(\theta) \quad \text{s.t.} \quad \mathbb{E}_{s \sim \pi_{\theta_{\text{old}}}}\left[ \text{KL}(\pi_{\theta_{\text{old}}}(\cdot|s) \,\|\, \pi_\theta(\cdot|s)) \right] \le \delta$$

数学优雅但**工程上不可行**——要算 Fisher 矩阵（二阶量）、共轭梯度求 natural gradient、line search 投影。对深度网络是天方夜谭。

### 3.6 PPO Clipped Surrogate：用一阶代替二阶

PPO 的 insight：**不要"硬约束 KL"，直接在目标里加"刹车"**——ratio 超出 $[1-\epsilon, 1+\epsilon]$ 就截断梯度。

$$\boxed{L^{\text{CLIP}}(\theta) = \mathbb{E}\left[ \min\left( r(\theta) A, \ \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A \right) \right]}$$

典型 $\epsilon = 0.2$，意思是：每个 $(s,a)$ 上新策略给 action 的概率，相对旧策略只能涨/跌 20% 之内。

### 3.7 clip + min 的几何（精髓）

PPO clip 的行为是**非对称**的，A>0 和 A<0 的"刹车方向"不一样：

| 区域 | 行为 | 直觉 |
| :--- | :--- | :--- |
| **A>0, r<1−ε** | 有梯度 | 好动作概率太低，继续推 |
| **A>0, r∈[1±ε]** | 有梯度 | 正常推 |
| **A>0, r>1+ε** | **梯度=0** | 好动作够高了，刹住 |
| **A<0, r<1−ε** | **梯度=0** | 坏动作够低了，刹住 |
| **A<0, r∈[1±ε]** | 有梯度 | 正常压 |
| **A<0, r>1+ε** | 有梯度 | 策略走反了，**必须**保留梯度救回来 |

> **核心原则**：clip 只在"按 A 的方向走且走过头"时刹车；"走反方向"时绝不刹车——min 保证修正梯度不被截断。

**为什么是 min（不是单独的 clip）**：考虑 A<0, r=5（策略严重走反）。

- 有 min：$L = \min(5A, 1.2A) = 5A$（A 负，5A 更负），梯度强烈拉回
- 无 min（错误设计）：$L = 1.2A$ 是常数，梯度=0。模型已经犯了大错，loss 却毫不在意

> min 是 PPO 设计里**唯一让"修正错误"和"防止过更新"并存**的机制。

### 3.8 一句话总结 PPO 设计

> **PPO 用 clip 实现"信赖域"，但只在"做对方向"时启用；min 保证"做错方向"时梯度不被截断，让模型有机会修正。**

---

## 4. GAE：把 advantage 估计也站稳

### 4.1 Advantage 的定义和两难

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

$V, Q$ 都是期望值，**我们手上拿不到真值，只能估计**。估 $Q$ 有两个极端：

**Monte Carlo (MC)**：

$$\hat{Q}^{\text{MC}}_t = \sum_{l=0}^{T-t-1} \gamma^l r_{t+l}, \quad \hat{A}^{\text{MC}}_t = \hat{Q}^{\text{MC}}_t - V_\phi(s_t)$$

- 无偏（轨迹跑完了，真实 reward）
- 高方差（未来 reward 随机性堆叠）

**1-step TD**：

$$\hat{A}^{\text{TD}}_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) =: \delta_t$$

- 高偏差（依赖学出来的 critic $V_\phi$）
- 低方差（只 1 步随机）

> **类比**：MC 是"亲自走一遍测时间"（无偏但抖），TD 是"信 Google Maps 的预测"（稳但可能错）。
>
> **PPO 里 critic 永远追着移动目标**——actor 一变，$V^\pi$ 就变。所以 critic 的"准确性"是个动态概念，从不真正收敛。这是 RL 和 SFT 的根本区别——监督学习的目标是固定分布，RL 的目标是被自己改变的分布。

### 4.2 GAE：用 λ 在两极间插值

$$\hat{A}^{\text{GAE}(\gamma, \lambda)}_t = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \hat{A}^{(n)}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

两种形式数学等价（用望远镜消除可证），但**计算复杂度差一个数量级**：

- 加权和形式：$O(T^2)$
- TD-error 累加形式：$O(T)$（从后往前递推：$\hat{A}_t^{\text{GAE}} = \delta_t + \gamma\lambda \cdot \hat{A}_{t+1}^{\text{GAE}}$）

工程实现一律用后者。

### 4.3 λ 的物理意义

| λ | 等价于 | 偏差/方差 |
| :--- | :--- | :--- |
| 0 | 1-step TD | 高偏差，低方差 |
| 1 | MC | 无偏，高方差 |
| **0.95（实践最优）** | 中间档 | 平衡 |

> **为什么 0.95 实践最优**：critic 在 PPO 初期肯定不准，所以不能完全信它（不能 λ=0）；但完全 MC 方差又太大（不能 λ=1）。0.95 让前几步真实 reward 主导，后面 critic 接力——这正好和"critic 离当前位置越远预测越不准"匹配。
>
> **为什么实践中不对 λ 退火**：PPO 训练本身不稳，多调度一个超参数风险高于收益。critic 在 PPO 里永远不会"训好"，所以"等 critic 收敛后降 λ"的前提不成立。

### 4.4 LM RLHF 特殊性：γ = 1

经典 RL 用 $\gamma < 1$ 是为了：(a) 序列可能无限长，不打折发散；(b) 表达"看重眼前"的时间偏好。

LM RLHF 都不成立：
- 序列长度有 `max_new_tokens` 上限，不会发散
- RM 只在 EOS 给一次 reward，不存在"早期 reward 比后期 reward 更值钱"
- 每个 token 对最终答案的贡献等权

**$\gamma=1$ 在 LM 上的语义就是"序列里每个位置等权"**。这是 LM RLHF 和经典 RL 最大的语义差别之一。

### 4.5 LM 上的工程细节：序列末尾的 V

```python
next_v = values[:, t + 1] if t + 1 < T else 0
```

这隐含假设：**$V(s_T) = 0$**（序列结束后没有更多 reward）。

- 自然 EOS：合理，RM 已经在 EOS 打过分了
- max_tokens 截断：**有偏**——序列其实没结束，未来本来可能有 reward。TRL 等实现会区分这两种情况，对截断的用 $V_\phi(s_T)$ 而不是 0 来 bootstrap

---

## 5. PPO 完整目标函数

PPO 的总 loss（actor + critic 联合优化）：

$$\boxed{L^{\text{PPO}}(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_1 \cdot L^{VF}(\phi) - c_2 \cdot H(\pi_\theta)}$$

其中 $c_1 \in [0.5, 1.0]$，$c_2$ 在 LM RLHF 里通常为 0（不需要熵奖励）。

### 5.1 Actor loss

$$L^{\text{CLIP}}_t = \min\left( r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t \right)$$

### 5.2 Critic loss（带 value clipping）

$$L^{VF}_t = \max\left( (V_\phi(s_t) - R_t)^2,\ (\text{clip}(V_\phi(s_t), V_{\phi_{\text{old}}} - \epsilon_v, V_{\phi_{\text{old}}} + \epsilon_v) - R_t)^2 \right)$$

其中 $R_t = \hat{A}_t + V_{\phi_{\text{old}}}(s_t)$。

🛑 **注意 actor 用 min，critic 用 max**——目的相反：
- Actor 最大化 $L^{\text{CLIP}}$ → min 选保守值（不要太乐观）
- Critic 最小化 MSE → max 选悲观值（双向都罚）

### 5.3 Per-Token KL Penalty

在 reward 信号里加 KL 惩罚（**注意**：这是融到 reward 里的 per-token KL，和 PPO 总 loss 里没有显式 KL 项不一样）：

$$\tilde{r}_t = \begin{cases} -\beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} & t < T \\ r_\phi(x, y) -\beta \log \frac{\pi_\theta(a_T|s_T)}{\pi_{\text{ref}}(a_T|s_T)} & t = T \end{cases}$$

这个 per-token KL 实际上是把 GAE 中每个 $\delta_t$ 的 $r_t$ 项替换成 $\tilde{r}_t$ —— 让 KL 惩罚通过 advantage 反向传播到策略梯度上。

> **β 是固定的吗？不是**。OpenAI 原版 PPO 论文用 **adaptive KL controller**：每 step 后检查实际 KL，若 > target × 1.5 则 β×=2，若 < target / 1.5 则 β /= 2。LLaMA-2 论文显式提到用 target_kl=0.01。这是面试加分细节。

---

## 6. Trust Region 视角：PPO 是 TRPO 的工程简化

| | TRPO | PPO |
| :--- | :--- | :--- |
| 约束方式 | 全局 KL 期望 ≤ δ（硬约束） | per-sample ratio clip（软约束） |
| 求解 | 二阶（Fisher + 共轭梯度） | 一阶（普通 Adam） |
| 实现复杂度 | 极高 | 低 |
| 数学严格性 | 强 | 弱（clip 不是真 trust region） |

⚠️ **微妙之处**：PPO 的 clip 是**逐样本、采样位置上的**约束，不是全局 KL。所以实践中**还要额外监控** $\text{KL}(\pi_\theta \| \pi_{\theta_{\text{old}}})$——光靠 clip 不够，KL 还是可能慢慢漂。

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

## B.3 PPO 完整训练循环（精修版）

### 数据流总览

```
Rollout 阶段（无梯度）:
  prompts → Actor.generate → responses
  responses → Actor.forward → old_logprobs (logπ_θ_old)
  responses → Ref.forward   → ref_logprobs (logπ_ref)
  responses → RM.forward    → rewards (scalar at EOS)
  responses → Critic.forward → old_values (per-token V)
  
  → 合成 per-token reward (rewards + KL penalty)
  → 算 GAE → advantages, returns
  → 全部 detach（整个 PPO epoch 内冻结）

Train 阶段（有梯度，跑 ppo_epochs × n_minibatch 次）:
  Actor.forward  → new_logprobs → ratio = exp(new - old) → actor_loss (clipped surrogate)
  Critic.forward → new_values  → critic_loss (clipped MSE)
  loss = actor_loss + c1 * critic_loss
  backward + step
  
  每个 epoch 后检查 KL，若超阈值则 early stop
```

### 完整代码

```python
def ppo_step(actor, critic, ref, rm, prompts, optimizer, cfg):
    # ============ Phase 1: Rollout (no grad) ============
    with torch.no_grad():
        responses = actor.generate(prompts, max_new_tokens=cfg.max_new_tokens)
        
        # 注意：old_logprobs 必须用 train 端精度重算
        # 不要用 vLLM 返回的 logprobs（精度不一致导致 ratio 漂移）
        old_logprobs = compute_logprobs(actor, prompts, responses)  # [B, L]
        ref_logprobs = compute_logprobs(ref,   prompts, responses)  # [B, L]
        rewards      = rm(prompts, responses)                        # [B], scalar
        old_values   = critic(prompts, responses)                    # [B, L]
        
        # ===== Per-token reward = KL penalty + RM at EOS =====
        per_token_rewards = -cfg.beta * (old_logprobs - ref_logprobs)
        per_token_rewards[:, -1] += rewards  # RM 只加在最后位置
        
        # ===== GAE =====
        advantages = compute_gae(per_token_rewards, old_values, cfg.gamma, cfg.lam)
        returns = advantages + old_values
        
        # ===== Advantage normalization（必加，否则 ratio 失控）=====
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # detach 一切（PPO 设计哲学：rollout 时刻冻结）
        old_logprobs = old_logprobs.detach()
        old_values   = old_values.detach()
        advantages   = advantages.detach()
        returns      = returns.detach()
    
    # ============ Phase 2 & 3: Train ============
    for epoch in range(cfg.ppo_epochs):
        for batch in dataloader(data, cfg.mini_batch_size):
            # ===== Actor =====
            logits = actor(batch.input_ids)
            new_logprobs = compute_logprobs_from_logits(logits, batch.labels)
            
            ratio = torch.exp(new_logprobs - batch.old_logprobs)
            
            # Clipped surrogate（注意 min）
            surr1 = ratio * batch.advantages
            surr2 = torch.clamp(ratio, 1 - cfg.eps, 1 + cfg.eps) * batch.advantages
            actor_loss = -torch.min(surr1, surr2)
            actor_loss = (actor_loss * batch.mask).sum() / batch.mask.sum()
            
            # ===== Critic with value clipping（注意 max）=====
            new_values = critic(batch.input_ids)
            values_clipped = batch.old_values + torch.clamp(
                new_values - batch.old_values, -cfg.eps_v, cfg.eps_v
            )
            vf_loss1 = (new_values - batch.returns) ** 2
            vf_loss2 = (values_clipped - batch.returns) ** 2
            critic_loss = 0.5 * torch.max(vf_loss1, vf_loss2)
            critic_loss = (critic_loss * batch.mask).sum() / batch.mask.sum()
            
            # ===== 联合更新 =====
            loss = actor_loss + cfg.c1 * critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(),  1.0)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
        
        # ===== Early stop by KL =====
        kl = compute_kl(actor, ref, data)
        if kl > cfg.target_kl * 1.5:
            print(f"Early stop at PPO epoch {epoch}, KL={kl:.4f}")
            break
    
    # ===== Adaptive KL controller =====
    if kl < cfg.target_kl / 1.5:
        cfg.beta /= 2
    elif kl > cfg.target_kl * 1.5:
        cfg.beta *= 2
```

## B.4 Actor 和 Critic 的更新机制对比

这是 PPO 工程上最容易混淆的地方。

| 维度 | Actor | Critic |
| :--- | :--- | :--- |
| 任务类型 | 分类（输出 token 分布） | 回归（输出 V 标量） |
| 网络结构 | LM 主体 | LM 主体 + value head |
| Loss 形式 | clipped surrogate（要**最大化**） | MSE（要**最小化**） |
| Clip min/max | **min**（最大化时取保守） | **max**（最小化时取悲观） |
| Target 来源 | advantage（GAE 算的，常量） | return = adv + old_value（常量） |
| 梯度通过谁 | $\log \pi_\theta$（softmax 输出） | $V_\phi$（value head 输出） |
| 典型学习率 | 1e-6 ~ 5e-6 | 5e-6 ~ 2e-5（**比 actor 高 5-10×**） |

> **为什么 critic LR 比 actor 高**：critic 在做"追逐移动目标"——actor 一变 $V^\pi$ 就变，critic 必须更新得快才跟得上。actor 在做"对策略本身的微调"，慢一点更稳。

### PPO 内层循环的核心哲学

PPO epoch 内**所有 rollout 时刻的快照量都冻结**：

| 量 | epoch 内是否变 |
| :--- | :--- |
| `old_logprobs`, `old_values`, `rewards` | ❌ 不变 |
| `advantages`, `returns` | ❌ **不变**（即使 critic 已经更新了多次） |
| `new_logprobs`, `new_values`, `ratio` | ✅ 每次都重算 |

> **为什么 advantage 不重新算**（即使 critic 已变好）？
>
> 1. **理论上**：advantage 必须配 $\pi_{\theta_{\text{old}}}$，重算就破坏了 IS 的数学结构
> 2. **工程上**：advantage 必须是常量（detach），否则 actor 的梯度被 critic 污染
> 3. **稳定性**：所有"约束量"建立在 rollout 时刻快照上，重算就丢掉了 trust region 参考点
> 4. **代价**：advantage 过时是已经被 ppo_epochs 上限控制住的，不应该用"重算 advantage"来解决

**一句话总结 PPO 工程哲学**：

> 冻结 rollout 时刻的一切，让 actor / critic 在这个固定参考系上更新有限次（ppo_epochs × n_minibatch），更新完丢掉这批数据，重新 rollout 建立新参考系。

## B.5 KL Estimator

PPO 实现 KL 估计有两种常用方法：

```python
# 方法 1：直接定义（无偏但方差大，偶尔负值）
kl_1 = (old_logprobs - new_logprobs).mean()

# 方法 2：k3 estimator（John Schulman 推荐，偏一点但稳定且非负）
log_ratio = new_logprobs - old_logprobs
kl_2 = (torch.exp(log_ratio) - 1 - log_ratio).mean()
```

OpenRLHF 默认用方法 2。

---

# §C 训练与推理（Infra 视角）

## C.1 四模型的真实显存账

> **重点纠正**：很多笔记说"PPO 要 4 倍显存"——**严重低估**。我们仔细算账。

四个模型地位完全不同：

| 模型 | 训练？ | 需要梯度？ | 需要 Adam 状态？ | 用在哪个阶段 |
| :--- | :--- | :--- | :--- | :--- |
| **Actor**（Policy） | ✅ | ✅ | ✅ | Rollout (generate) + Train |
| **Critic**（Value） | ✅ | ✅ | ✅ | Rollout (forward) + Train |
| **Reference**（SFT 冻结） | ❌ | ❌ | ❌ | Rollout (forward) |
| **Reward Model** | ❌ | ❌ | ❌ | Rollout (forward) |

🛑 **关键观察**：只有 Actor 需要 generate——Critic/Ref/RM 都只 forward。这个不对称是所有 Infra 优化的起点。

### 单 7B 模型显存（bf16 训练）

| 组件 | Actor (训练) | Critic (训练) | Ref (冻结) | RM (冻结) |
| :--- | :--- | :--- | :--- | :--- |
| 参数 (bf16) | 14 GB | 14 GB | 14 GB | 14 GB |
| 梯度 (bf16) | 14 GB | 14 GB | — | — |
| Adam 状态 (fp32 m, v) | 56 GB | 56 GB | — | — |
| Master weights (fp32) | 28 GB | 28 GB | — | — |
| **静态小计** | **112 GB** | **112 GB** | **14 GB** | **14 GB** |

四个加起来静态就 **252 GB**——单张 H100 (80 GB) 根本装不下。

### 还要加上激活和 KV Cache

**激活值** (训练 forward 留下的中间结果)：bs=8, seq=1024
- 不开 gradient checkpointing：~40-60 GB
- 开了：~10-15 GB

**KV Cache** (Actor generate 必需)：

$$\text{KV Cache} = 2 \times L \times B \times S \times n_h \times d_h \times \text{bytes}$$

LLaMA-7B (L=32, n_h=32, d_h=128), bs=8, seq=1024：约 **4.3 GB**
若 bs=32, seq=2048：**34 GB**——经常被忽视！

### 总账（7B, bs=8, seq=1024, bf16）

| 阶段 | Actor | Critic | Ref | RM | KV Cache | 激活 | 总计 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rollout** | 14 GB (推理) | 14 GB | 14 GB | 14 GB | 4 GB | ~0 | **60 GB** |
| **Train** | 112 GB | 112 GB | — | — | — | 50 GB | **274 GB** |

> 🛑 **反直觉**：Rollout 显存压力小，Train 显存压力大；但**时间消耗正好相反**——rollout 是时间瓶颈。

## C.2 Rollout 为什么是时间瓶颈

| 阶段 | 单 step 时间 | GPU 利用率 | 为什么 |
| :--- | :--- | :--- | :--- |
| Train (bs=8, seq=1024) | 0.5-1s | 40-50% | compute-bound，参数读 1 次服务全 batch |
| Rollout (gen=256, 朴素 HF) | 5-7s | **<5%** | **memory-bound**，256 步串行 decode |

**朴素 HF generate 一个 PPO step 时间分布**：

| 阶段 | 时间 | 占比 |
| :--- | :--- | :--- |
| Actor generate | 5.5s | **73%** |
| Ref/RM/Critic forward | 0.4s | 5% |
| Train (4 PPO epochs) | 1.6s | 22% |

**rollout 占 70%+——把它优化掉，整体训练能翻 3 倍**。

## C.3 Decode 阶段的根本瓶颈：HBM 带宽

🛑 **极易踩坑的概念**：很多人以为 decode 慢是因为 attention 计算——错。

回忆 prefill vs decode：

| | Prefill | Decode |
| :--- | :--- | :--- |
| 处理 | 一次性多 token (256) | 一次 1 token |
| 计算密度 (FLOPs/byte) | 高 | **极低** |
| 瓶颈 | compute (TFLOPs) | **memory bandwidth (HBM I/O)** |
| FlashAttention 帮助 | **巨大** | 有限 |

**decode 时 HBM 上必须读取**：

1. **全部参数** 14 GB（每层 weights 都要参与计算）
2. **当前序列的 KV Cache** ~0.5 GB

**14 GB vs 0.5 GB——参数读取是 KV Cache 的 28 倍**。decode 的带宽**主要被参数吃掉**，不是 KV Cache。

> decode 阶段的 attention matrix 是 1×t 的向量（不是 t×t 矩阵），FlashAttention 优化的"$N^2$ 中间矩阵"问题在 decode 阶段**不存在**。
>
> 真正的 decode 优化要直接攻击参数读取：
> - **量化**：直接缩小要读的数据量（INT8/INT4）
> - **MoE**：每次只激活部分参数
> - **Continuous batching**：参数读 1 次服务 N 条序列（横向复用）
> - **Speculative decoding**：参数读 1 次产出多个 token（纵向复用）

## C.4 vLLM 救场的三件套

| 优化 | 攻击瓶颈 | 加速贡献 |
| :--- | :--- | :--- |
| **PagedAttention** | KV Cache 碎片化 | 间接——让 batch 能开更大 |
| **Continuous Batching** | 长尾 + GPU 闲置 | **巨大**——参数读 1 次服务 N 条 |
| **Fused Kernels** | kernel launch overhead | 中等 |

**vLLM 在 PPO rollout 上 5-10× 加速的真正来源是 continuous batching**：参数读取的成本被摊销到更多序列上。

具体算：
- 朴素 bs=8：每 token 参数读取摊销 = 14 GB / 8 = 1.75 GB
- vLLM bs=64：14 GB / 64 = 0.22 GB → **有效带宽 8 倍**

> **vLLM 的角色**：只用于 rollout 的 generate 步骤——它不能训练，也算不了 logprobs（只输出 token）。但这一步就值回票价。

### PPO 里**不**用 speculative decoding 的原因

虽然 spec decoding 是 LLM serving 的标配，但 PPO rollout 上一般不用：

1. **PPO bs 已经够大**（64-128），continuous batching 已经把 GPU 喂饱，spec 边际收益小
2. **需要严格的 old_logprobs**：spec decoding 的接受/拒绝机制虽然分布等价，但工程实现稍有 bug 就违反 on-policy 假设
3. **需要可控的采样分布**：actor 输出必须是当前策略 $\pi_\theta$ 的样本，spec 引入额外复杂性

## C.5 现代 RLHF 框架的两条架构路线

### 路线 A：Hybrid Engine（DeepSpeed-Chat）

**思路**：训练和推理共用 GPU、共用权重，在两个 mode 间切换。

```
Phase 1 (rollout): GPU 加载推理 mode，权重 TP 排列，开 KV Cache
Phase 2/3 (train): GPU 切换训练 mode，权重 ZeRO 重组，开梯度/Adam
```

- ✅ 显存利用最高效
- ❌ mode 切换有 reshard 开销
- ❌ vLLM 不支持 hybrid engine，rollout 引擎只能自研（比 vLLM 慢）

### 路线 B：Disaggregated（OpenRLHF / veRL，业界主流）

**思路**：训练 GPU 和 rollout GPU 物理分开，各用最优框架。

```
Rollout GPUs: vLLM (Actor inference) + Ref/RM/Critic forward
Train GPUs:   DeepSpeed (Actor + Critic 训练)
每 step 后:   train → rollout 做 NCCL broadcast 同步权重
```

- ✅ 每边都用最优引擎
- ✅ 实现清晰，扩展性好
- ❌ 双份 Actor 副本，显存翻倍
- ❌ 需要解决参数同步

### 对比

| 维度 | Hybrid (DeepSpeed-Chat) | Disaggregated (OpenRLHF) |
| :--- | :--- | :--- |
| 资源效率 | 高 | 低 |
| Rollout 引擎 | 自研，慢 | vLLM，最快 |
| 实现复杂度 | 极高 | 中等 |
| 参数同步 | 不需要 | NCCL broadcast |
| 业界主流 | 小规模 | **大规模（DeepSeek, 字节 veRL, 千问）** |

## C.6 参数同步：disaggregated 的核心难题

每 PPO step 训练完，必须把更新后的 actor 权重同步到 vLLM。三个工程难题：

1. **Sharding 格式不一致**：训练用 ZeRO-3/FSDP，vLLM 用 TP，切法不一样
2. **量级巨大**：7B = 14 GB，磁盘往返要 30s+，naive 实现完全不可接受
3. **共享 GPU 时显存冲突**：train mode 和 infer mode 不能同时占用

### NCCL Broadcast 方案（OpenRLHF）

```
Train side (8 GPUs, ZeRO-3):                  Rollout side (4 GPUs, vLLM TP=4):
  GPU 0: param shard 0                          GPU 0: full params (TP shard 0)
  GPU 1: param shard 1     ────NCCL group────►  GPU 1: full params (TP shard 1)
  ...                       逐层 broadcast        GPU 2: full params (TP shard 2)
  GPU 7: param shard 7                          GPU 3: full params (TP shard 3)
```

**做法**：
1. Train ranks 和 Rollout ranks 建立公共 NCCL group
2. 按 layer 顺序：train 端 all-gather 完整 layer → rank 0 broadcast → rollout 端按 TP 切分
3. 流水线化：layer N+1 的 all-gather 和 layer N 的 broadcast 并行

**为什么用 NCCL 而不是磁盘**：NCCL 走 NVLink/InfiniBand 带宽几十倍于 PCIe→磁盘。14 GB 在 NVLink (600 GB/s) 下理论 23ms，磁盘要 30s。

**实际同步时间**：1-3s。

## C.7 一个易踩的坑：精度不一致

PPO rollout 用 vLLM (bf16)，train 用 DeepSpeed (可能 fp32)，**即使权重相同，算出的 logprobs 不一致**。

后果：即使 $\theta = \theta_{\text{old}}$（PPO 第一个 minibatch），ratio 也不精确等于 1，而是 0.98-1.02。这个"假 ratio"驱动训练，导致策略漂移。

**解决方案**：
- **重新计算 old_logprobs**：rollout 完用 train 端精度重算（OpenRLHF 默认做法）
- 或者**统一精度**：rollout 和 train 都 bf16

> 这是面试杀手锏问题——"PPO 训练里精度有什么坑"，能讲清楚这个就明显是做过的。

## C.8 系统设计参考：16×H100 训 7B PPO

**配置**：单机 8×H100 × 2，通过 InfiniBand 互连，目标 bs=64, seq=1024。

**分配**：
- 机器 A（8×H100）：Train cluster（DeepSpeed ZeRO-3 跑 Actor + Critic 训练）
- 机器 B（8×H100）：Rollout cluster（vLLM TP=2 跑 Actor inference + Ref/RM/Critic forward）

**预期时间分布**：

| 阶段 | 时间 | 占比 |
| :--- | :--- | :--- |
| Actor generate (vLLM) | 1.5s | 35% |
| Ref/RM/Critic forward | 0.5s | 12% |
| NCCL param sync | 0.3s | 7% |
| Actor+Critic train (4 epochs) | 2s | 46% |
| **Total** | **4.3s/step** | 100% |

🛑 **判断 infra 是否做对了的信号**：用了 vLLM 之后，**train 应该占大头**，rollout 占比降到 35-40%。如果 rollout 还占 70%，说明推理引擎没优化好。

**优化方向**（短中长期）：
- 短期：FA3 + 大 batch 压 train；layer-wise async sync 压参数同步
- 中期：resource borrowing（train 阶段借 rollout 卡做 forward）
- 长期：上 70B 后切 Megatron + veRL

## C.9 推理视角：PPO 后的模型与 SFT 模型有何不同？

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

# §D 失败模式与调参

> 这一节是 PPO 工程的"暗面"——只有真正调过 PPO 的人才答得上来。

## D.1 四大失败模式速查

| 失败模式 | 主要症状 | 关键指标 | 处方 |
| :--- | :--- | :--- | :--- |
| **KL 爆炸** | KL 从 0.05 涨到 5+，输出乱码/重复 | `kl_divergence`, `kl_max` | 加大 β / 降 LR / 减 ppo_epochs / adaptive KL |
| **Reward Hacking** | RM reward 涨，人工 eval 反而退步 | `mean_response_length`, held-out win-rate | length penalty / RM ensemble / iterative DPO |
| **Value Loss Collapse** | value_loss 突增 10×，ratio 方差爆炸 | `value_loss`, `explained_variance` | value clipping / 降 critic LR / critic pretrain |
| **Ratio 失控** | ratio.std > 0.3, clip_fraction > 50% | `ratio.std`, `clip_fraction` | 减 ppo_epochs / early stop by KL |

## D.2 KL 爆炸：最常见的发散模式

### 恶性循环

```
actor 偏离 ref
  → KL penalty 在 per-token reward 里越来越大（负值）
  → reward 噪声变大
  → advantage 估计抖动
  → policy update 不稳
  → 输出乱码或重复
```

### 健康/警报阈值

| 指标 | 健康 | 警报 |
| :--- | :--- | :--- |
| `kl_divergence` (mean) | 0.01 - 0.1 | > 0.5 |
| `kl_divergence` (max per token) | < 1 | > 5 |
| `ratio.std` | < 0.1 | > 0.3 |
| `clip_fraction` | 5% - 20% | > 50% |
| 输出熵 | 稳定 | 骤降 = mode collapse |

### 处方（按优先级）

1. **加大 β**（0.05 → 0.1-0.2）
2. **降 actor LR**（1e-6 → 5e-7）
3. **减 ppo_epochs**（4 → 1-2）
4. **缩小 clip ε**（0.2 → 0.1）
5. **加 adaptive KL controller**
6. **回滚 + 严调超参重跑**（PPO 一旦发散就回不来）

## D.3 Reward Hacking：最阴险的失败

### 经典案例

| 类型 | 表现 | 缓解方法 |
| :--- | :--- | :--- |
| **长度偏差** | RM 偏爱长回答 → Policy 学会冗长 | RM 训练时加 length penalty |
| **Sycophancy（谄媚）** | Policy 迎合用户已有观点 | 偏好数据中加入"反 sycophancy"案例 |
| **Format gaming** | 滥用 markdown / emoji / 列表 | RM 训练数据多样化 |
| **特定 token 利用** | 重复 RM "见过的好回答里的标志短语" | online RM refresh |
| **拒答漂移** | 过度安全化，"我无法回答..." | RM 训练时平衡 helpful/harmless |
| **过度礼貌** | "Great question!" 开头泛滥 | length / format penalty |

### 根本原因（Goodhart's Law）

> **"When a measure becomes a target, it ceases to be a good measure."**

RM 不是真实 reward，是真实 reward 的代理。actor 探索到 RM 训练分布之外时，RM 不可靠，actor 会专门**攻击 RM 的弱点**。

### 必备：独立 eval

光看 RM reward 上涨**毫无意义**。必须有：
- 独立人工 eval（每 N step）
- 或更强 LLM 当 judge

### 处方

1. Length penalty: $r' = r - \alpha \cdot \text{len}$
2. RM ensemble (3-5 个 RM 取均值或最小值)
3. 加 KL penalty（把 actor 拉回 RM 训练分布）
4. 重训 RM（用 actor 当前输出作为新偏好数据 — online RM refresh）
5. Iterative DPO / online DPO

**核心防线总结**：
1. **per-token KL penalty** 是第一道防线（不让 Policy 跑离 SFT 太远）
2. **RM 训练数据多样化** 是治本之道
3. **online RM refresh**（每轮 PPO 后用新 Policy 生成的 response 重训 RM）

## D.4 Value Loss Collapse

### 触发条件

- reward outlier（RM 给某条样本异常高分）
- critic LR 太大
- 没有 value clipping

### 诊断指标

| 指标 | 健康 | 警报 |
| :--- | :--- | :--- |
| `value_loss` | 平稳下降 | 突增 5× 以上 |
| `value.std` | 稳定 | 飙升 |
| `explained_variance` | > 0.5 | < 0 → critic 没学到东西 |

### 处方

1. **加 value clipping**（前面 §A.5.2 的公式）
2. **降 critic LR**
3. **Reward clipping/normalization**：限制 $r \in [-5, 5]$ 或 z-score
4. **Critic pretraining**：PPO 前用 SFT 数据预训 critic 几步（OpenRLHF `--critic-pretrain-steps`）

## D.5 Ratio 失控

### 症状与原因

`ratio.mean` 远离 1，`ratio.max` 飙到 100+，`clip_fraction` 超过 80%。

**根本原因**：PPO 一次 rollout 后做 N 个 epoch 训练。N 越大，最后几个 epoch 的 $\theta$ 离 $\theta_{\text{old}}$ 越远，ratio 越偏离 1。**这就是 ppo_epochs 不能设太大的根本原因——不是过拟合，是 IS 失效**。

### Clip Fraction 诊断

- < 5%：模型几乎没在更新（LR 太低？没 advantage 信号？）
- 5-20%：健康
- 20-50%：偏高但能跑
- > 50%：**严重 off-policy**

### 处方

1. **减少 ppo_epochs**（4 → 2 → 1）
2. **加大 mini-batch size**：减少每个 epoch 的 grad step 数
3. **Early stopping by KL**：每个 epoch 算 KL，超阈值就提前停

## D.6 失败模式的因果链

🛑 **关键认知**：这些失败往往**连环触发**：

```
LR 太大 / reward outlier
  → ratio 爆炸（最早出现）
  → KL 爆炸
  → value collapse
  → mode collapse + reward hacking
```

**调试第一原则**：**最早出问题的指标是真正的病因，后面都是症状**。

按"指标第一次异常的时间顺序"诊断：
1. step 0-10: `ratio.std` 异常 → 怀疑 LR
2. step 20-30: KL 涨快 → 怀疑 β
3. step 30+: `value_loss` 抖 → 怀疑 critic LR
4. step 50+: reward 涨但 eval 跌 → reward hacking

## D.7 PPO 上线前 checklist

| 项 | 建议 | 重要性 |
| :--- | :--- | :--- |
| KL adaptive controller | target_kl=0.01, β 自动调 | ★★★ |
| Advantage normalization | per-batch z-score | ★★★ |
| Value clipping | $\epsilon_v = 0.2$ | ★★★ |
| Early stopping by KL | per-epoch 检查 | ★★★ |
| 监控 held-out eval | 每 N step 跑 | ★★★ |
| **重算 old_logprobs**（精度） | train 端重算 | ★★★ |
| Critic pretraining | PPO 前预热几步 | ★★ |
| Reward clipping/normalization | $r \in [-5, 5]$ | ★★ |
| Gradient clipping | grad_norm ≤ 1.0 | ★★ |
| Length penalty | $\alpha \cdot \text{len}$ | ★★ |
| Save checkpoint frequently | 每 50 step | ★★ |

---

# §E 章末速查：常见问题

### 数学推导类

**Q1: PPO 的 ratio 为什么长那样？凭空定义的吗？**
不是。ratio 是重要性采样密度比 $p/q$ 在 PG 语境下的具体化——为了让旧策略采的数据能估当前策略的梯度，必须乘上 $\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$ 修正。它的形式是数学上的必然，不是设计选择。

**Q2: PPO 的 clip 为什么要配 min？只用 clip 不行吗？**
不行。考虑 A<0, r=5（策略严重走反）：
- 只 clip：$L = 1.2A$ 是常数，梯度=0，模型犯了大错却没法修正
- min(r·A, clip·A) = 5A，保留强烈修正梯度

min 让"做对方向但走过头"启动刹车，"做错方向"绝不刹车。

**Q3: log-derivative trick 解决什么核心问题？**
让 $\nabla J$ 能写成期望的形式，从而能用采样估计。$\nabla \pi_\theta$ 不是概率分布，所以 $\int \nabla \pi \cdot R\, d\tau$ 不是期望，不能采样。通过 $\nabla \pi = \pi \cdot \nabla \log \pi$，重新引入 $\pi$ 作乘子，让积分变回期望。

**Q4: GAE 的 λ 调到多少？怎么解释？**
实践中 0.95。本质是在 1-step TD（λ=0，高偏低方差）和 MC（λ=1，无偏高方差）之间几何加权平均。critic 在 PPO 里永远训不准（追移动靶），所以不能完全信它；但完全 MC 又方差太大。0.95 让前几步真实 reward 主导，后面 critic 接力。

**Q5: γ 在 LM RLHF 里为什么是 1？**
经典 RL 用 γ<1 因为：序列可能无限长 / 表达时间偏好。LM RLHF 都不成立：序列有 max_tokens 上限、RM 只在 EOS 给一次 reward、每个 token 对最终答案等权。γ=1 的语义就是"每个位置贡献等权"。

### Infra 类

**Q6: PPO 训练的真实显存是 4 倍 SFT 吗？**
不是——严重低估。4 个 7B 模型静态需求 ~252 GB（Actor/Critic 各 112 GB，Ref/RM 各 14 GB），还要加激活（50+ GB）和 KV Cache（4-34 GB）。Train 阶段需求 ~270 GB，rollout 阶段 ~60 GB——两者不对称是 hybrid engine 设计的动机。

**Q7: 为什么 rollout 是 PPO 的时间瓶颈？**
decode 是 memory-bound（HBM 带宽瓶颈），每生成 1 token 要读 14 GB 参数。串行 256 步，朴素实现 5-7 秒/step，占总时间 70%+。GPU 利用率 <5%。

**Q8: FlashAttention 对 PPO 的 decode 有帮助吗？**
有限。FlashAttention 优化的是 $N^2$ 中间矩阵的 HBM I/O，但 decode 时 attention matrix 只是 1×t 向量（不是 t×t 矩阵），这个问题不存在。decode 真正的瓶颈是**参数读取**，要靠 continuous batching 或量化解决。

**Q9: vLLM 为什么能加速 5-10×？真正的加速来源是哪个？**
Continuous batching——把参数读取摊销到更多序列上。bs 从 8 升到 64 → 每 token 摊销的参数读取从 1.75 GB 降到 0.22 GB，有效带宽 8×。PagedAttention 是间接帮助（让 bs 能开更大），Fused Kernels 是中等贡献。

**Q10: PPO 里要不要用 speculative decoding？**
通常不用。PPO bs 已经够大（64-128），continuous batching 已喂饱 GPU；spec decoding 还可能引入 logprobs 计算复杂性和分布偏移风险，违反 on-policy 假设。

**Q11: Hybrid Engine vs Disaggregated 怎么选？**

| 场景 | 选择 |
| :--- | :--- |
| 资源紧张 / 单机 | Hybrid Engine（DeepSpeed-Chat） |
| **大规模 / 多机 / 生产** | **Disaggregated（OpenRLHF / veRL）** |

业界趋势：disaggregated 在 2024-2025 成为主流，因为 vLLM 优势太显著。

**Q12: Disaggregated 架构里参数怎么同步？**
NCCL broadcast：
1. Train ranks 和 Rollout ranks 建立公共 NCCL group
2. 按 layer 顺序：train 端 all-gather → rank 0 broadcast → rollout 端按 TP 切分
3. layer 间流水线化

14 GB 参数在 NVLink 下 23ms 理论值，实际 1-3s。

**Q13: PPO 训练里有什么精度坑？**
rollout 和 train 精度不一致 → old_logprobs 不准 → ratio 漂移。即使 $\theta = \theta_{\text{old}}$，ratio 也不精确等于 1。

**解决**：rollout 完用 train 端精度**重算** old_logprobs（OpenRLHF 默认）。这是面试加分项。

### 工程与调参类

**Q14: ppo_epochs 为什么不能设太大？**
两个原因：
1. **IS 失效**：$\theta$ 离 $\theta_{\text{old}}$ 越远，ratio 越离谱
2. **Advantage 过时**：critic 改进了但 actor 用的还是旧 advantage

经验上 4 是平衡点（OpenAI 原版），实践中常用 early stop by KL 让平均 epoch 更小。

**Q15: 调试 PPO 的第一原则是什么？**
看最早出问题的指标——它是真正的病因，后面都是症状。失败模式连环触发：ratio 异常 → KL 爆炸 → value collapse → mode collapse。

**Q16: Reward Hacking 怎么发现？**
光看 RM reward 涨毫无意义——必须有独立 eval（人工或更强 LLM judge）。监控 `mean_response_length` 飙升、n-gram 重复率上升、held-out RM-vs-人工评分相关性下降。

### 综合类

**Q17: RM 模型和 Policy 一定要一样大吗？**
- 早期（InstructGPT 时代）RM 较小（如 175B Policy + 6B RM）
- 现代趋势是 **RM ≥ Policy**（如 Llama 3.3 70B + 70B），因为 RM 质量直接决定 RL 上限，小 RM 容易被 Policy 钻空子

**Q18: 为什么必须加 KL 惩罚？**
- 防止 Reward Hacking
- 没有 KL 时，Policy 会学到"骗 RM 高分"的捷径
- KL 把 Policy 锚定在 SFT 模型附近——这就是 Ch4 §D 的"Stop-grad + EMA Target"思想

**Q19: Critic 和 RM 是同一个模型吗？**
形状相同（都输出标量），但**任务不同**：
- RM：判断 (x, y) 这一对的好坏（固定打分）
- Critic：给定状态 $s_t$，预估"未来累计 reward"（追逐移动目标）

实践中 Critic 通常**用 RM 初始化**，但训练目标不同。

**Q20: PPO 训完后能再做 DPO 吗？**
- 可以，常见做法是 PPO 后接 DPO 做"安全性微调"
- 但很少反过来（DPO → PPO），因为 DPO 已经把模型推到"边界"，PPO 容易让它崩

---

## 承上启下

PPO 的复杂度——**4 模型、3 阶段、2 套 loss、1 堆 trick**——是 RLHF 工程落地的核心痛点。DPO 在 Ch7 会展示如何用一个**闭式解**直接砍掉 RM 和 Critic：

> **PPO 用 IS + clip 维护 trust region；DPO 用 BT 模型 + ref 闭式解直接绕过整个 RL 框架。**

关键问题：**PPO 优化的 KL 约束 RL 目标，是否有闭式解？**

答案是：**有！** 而且这个闭式解一旦写出来，整个 RM + PPO 流程都可以坍缩成一个简单的 loss。

但 DPO 也有自己的失败模式（preference 数据集上的过拟合、模式集中等），不是"PPO 的完美替代品"。两者在 2024-2025 是**并存关系**：高复杂场景（长链推理 RLHF）仍用 PPO，简单偏好对齐用 DPO 或其变种（SimPO, IPO, KTO）。

下一章揭晓。

---

> **本章核心收获**：PPO 的每一个设计都不是凭空的。理解 PG → IS → TRPO → PPO 的逻辑链 + clip+min 的几何 + GAE 的 bias-variance 权衡 + 失败模式因果链 + Infra 视角的 rollout 瓶颈和 vLLM 救场，PPO 就从"一堆 trick"变成"被工程问题逼出来的必然方案"。