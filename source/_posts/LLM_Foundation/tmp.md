好，你跳过了 A/B 的确认，那我假设你心里已经清楚了——但**为了保险**，我在进入 IS 之前用一句话把关键结论钉死，你边读边自检：

> **A 的答案**：期望要求采样自 $\pi_\theta$（当前策略），但手上数据来自 $\pi_{\theta_{\text{old}}}$（采样时的旧策略）。$\theta$ 一更新两者就错位。
>
> **B 的答案**：$A > 0$ → 推**高** $\pi_\theta(a|s)$。链条是：梯度上升沿 $+\nabla \log \pi$ 方向走 → $\log \pi$ 增 → $\pi$ 增。

如果这两句你都觉得"对，我懂"，我们继续。

---

# §1.2 重要性采样：把旧数据"翻译"到新策略

## 重要性采样恒等式

这是统计里一个非常基础的等式：

$$\mathbb{E}_{x \sim p}[f(x)] = \int p(x) f(x)\, dx$$

现在我"乘 1"——分子分母同乘 $q(x)$：

$$= \int q(x) \cdot \frac{p(x)}{q(x)} f(x)\, dx = \mathbb{E}_{x \sim q}\left[ \frac{p(x)}{q(x)} f(x) \right]$$

**前提**：在 $q(x) = 0$ 的地方必须 $p(x) = 0$（不然 $p/q$ 没定义）。直觉：q 没采过的地方，p 也不能有概率质量。

**这个恒等式的本质**：你可以用任何分布 $q$ 去采样，估计 $p$ 下的期望——**只要乘上密度比修正**。

## 应用到 PG

我们想算（采样分布 = 当前 $\pi_\theta$）：

$$\nabla_\theta J(\theta) = \mathbb{E}_{(s,a) \sim \pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a) \right]$$

但手上数据来自 $\pi_{\theta_{\text{old}}}$。套 IS：

$$= \mathbb{E}_{(s,a) \sim \pi_{\theta_{\text{old}}}}\left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \cdot \nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a) \right]$$

🛑 **停一下，这里有个数学小动作非常关键**。

注意 $\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} \cdot \nabla \log \pi_\theta$ 这个组合。我们做一步代数：

$$\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \cdot \nabla_\theta \log \pi_\theta(a|s) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \cdot \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$$

而由于 $\pi_{\theta_{\text{old}}}$ 是常数（不依赖 $\theta$），所以：

$$\frac{\nabla_\theta \pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} = \nabla_\theta \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \right]$$

**这意味着什么**？意味着 IS 后的梯度可以写成"某个 ratio 表达式的梯度"：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_{\theta_{\text{old}}}}\left[ \nabla_\theta \left( \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \right) \cdot A(s,a) \right]$$

如果把梯度往外挪（同样靠正则性，$A$ 与 $\theta$ 无关——它来自旧策略下计算的优势）：

$$= \nabla_\theta \underbrace{\mathbb{E}_{\pi_{\theta_{\text{old}}}}\left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \cdot A(s,a) \right]}_{\text{这就是 PPO/TRPO 的 surrogate objective } L(\theta)}$$

**于是我们定义 surrogate objective**：

$$\boxed{L^{\text{IS}}(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\theta_{\text{old}}}}\left[ r(\theta) \cdot A(s,a) \right], \quad r(\theta) := \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$$

它满足一个性质：在 $\theta = \theta_{\text{old}}$ 这一点，$L^{\text{IS}}$ 的梯度等于真实 $J$ 的梯度。

$$\nabla_\theta L^{\text{IS}}(\theta) \Big|_{\theta = \theta_{\text{old}}} = \nabla_\theta J(\theta) \Big|_{\theta = \theta_{\text{old}}}$$

**意义**：在 $\theta_{\text{old}}$ 的"附近"，最大化 $L^{\text{IS}}$ 就近似等于最大化 $J$——而 $L^{\text{IS}}$ **只需要 $\pi_{\theta_{\text{old}}}$ 的样本，不用重新 rollout**。

## 你笔记里的 ratio 现在"出生"了

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

这个看似随意的定义，实际上是 IS 密度比 $p/q$ 在 PG 语境下的具体化。它**必须长这样**，没有别的选择。

---

## 但是 IS 有个致命问题

我们说了 IS 的前提：在 $\pi_{\theta_{\text{old}}}$ 没覆盖的地方 $\pi_\theta$ 也不能有概率质量。如果违反呢？**ratio 会爆炸**。

考虑一个极端例子：
- $\pi_{\theta_{\text{old}}}(a|s) = 0.01$（旧策略很少选这个 action）
- $\pi_\theta(a|s) = 0.5$（新策略变了，现在很喜欢选）

那么 $r(\theta) = 50$。如果这一项的 $A > 0$，单个样本的贡献就是 $50 \cdot A$——**一个样本主导整个 batch 的梯度**。这就是高方差。

更糟糕的是：**$\theta$ 离 $\theta_{\text{old}}$ 越远，ratio 越离谱**。所以 surrogate objective 的"近似"性质只在**信赖域**（trust region）内有效——出了这个区域，$L^{\text{IS}}$ 不再是 $J$ 的合理代理。

## 这就是 TRPO 的动机

TRPO 提出：**最大化 $L^{\text{IS}}(\theta)$，但加一个硬约束**让 $\theta$ 不能跑太远：

$$\max_\theta \mathbb{E}\left[ \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} A \right] \quad \text{s.t.} \quad \mathbb{E}_{s \sim \pi_{\theta_{\text{old}}}}\left[ \text{KL}(\pi_{\theta_{\text{old}}}(\cdot|s) \,\|\, \pi_\theta(\cdot|s)) \right] \le \delta$$

KL 是新旧策略的"距离"，$\delta$ 是信赖域半径（典型值 0.01）。**只要 KL 小，IS 的近似就有效**。

但这个约束优化在数学上要解一个 quadratic program，需要：
- 计算 Fisher 信息矩阵（二阶量）
- 共轭梯度求 natural gradient
- line search 投影回信赖域

对深度网络（参数百万级以上）来说**计算上不可行**——Fisher 矩阵的存储和求逆都吃不消。

## PPO 的核心 insight

> **能不能用一个更简单的方式实现"不让 $\theta$ 跑太远"？**

PPO 的答案是：**直接在目标函数里加一个"刹车"**——当 ratio 超出 $[1-\epsilon, 1+\epsilon]$ 时，**把这一项的梯度截断为 0**。

这就是 clipped surrogate：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\left[ \min\left( r(\theta) A, \ \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A \right) \right]$$

**为什么是 min？为什么 clip 之后还要取 min？这里有非常精妙的几何**——我们下一节专门讲。

---

## 🛑 阶段总结 + 一个验证题

到这里你应该看到一条完整的逻辑链：

$$\underbrace{\text{PG 是 on-policy}}_{\text{样本浪费}} \xrightarrow{\text{IS}} \underbrace{\text{surrogate } L^{\text{IS}} \text{ 能复用旧样本}}_{\text{但 ratio 会爆炸}} \xrightarrow{\text{加约束}} \underbrace{\text{TRPO 用硬 KL}}_{\text{但要二阶优化}} \xrightarrow{\text{工程化}} \underbrace{\text{PPO 用 clip 隐式约束}}_{\text{只要一阶}}$$

**每一步都是被前一步的问题"逼"出来的**，没有任何凭空的设计。

**验证题**（这次只问一个，因为答案需要你想清楚才能进入下一节）：

> PPO 论文里说"clip 实现了 trust region 的效果"。具体来说，trust region 是要求 $\theta$ 不能离 $\theta_{\text{old}}$ 太远。**clip 这个操作是怎么和 "$\theta$ 不远离 $\theta_{\text{old}}$" 联系起来的？**
>
> 提示：$\theta = \theta_{\text{old}}$ 时 ratio = ? $\theta$ 偏离时 ratio 会怎样？clip 把 ratio 限制在 $[1-\epsilon, 1+\epsilon]$，意味着什么？

把你的理解写出来，对错都行——这个问题想清楚了，下一节"clip 的几何意义"会变得非常自然。