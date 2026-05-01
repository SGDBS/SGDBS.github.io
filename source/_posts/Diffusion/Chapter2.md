---
title: Chapter 2 Score-based model 模型
categories: Diffusion Models
date: 2026-05-01 13:00:00
mathjax: true
tags:
    - AI
    - Diffusion Models
---

好,我们正式进入**第二讲:Score Matching**。我会按你熟悉的节奏走——动机、数学、关键洞察、和后面的衔接。

这条线的核心问题始终是同一个:**如何建模数据分布 $p(x)$**。但和第一讲的变分推断完全不同的思路——不学 $p(x)$,而学 $\nabla_x \log p(x)$。

---

## 一、动机:为什么不直接建模 $p(x)$?

先看直接建模 $p(x)$ 会遇到什么问题。

任何概率密度都可以写成:

$$p_\theta(x) = \frac{\tilde p_\theta(x)}{Z(\theta)}$$

其中 $\tilde p_\theta(x) \geq 0$ 是非归一化的(unnormalized),$Z(\theta) = \int \tilde p_\theta(x)\, dx$ 是**归一化常数(配分函数)**,确保积分为 1。

比如最经典的能量模型(EBM):

$$p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z(\theta)},\quad Z(\theta) = \int \exp(-E_\theta(x))\, dx$$

最大似然要最大化 $\log p_\theta(x) = -E_\theta(x) - \log Z(\theta)$,梯度:

$$\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) - \nabla_\theta \log Z(\theta)$$

**第二项是地狱**。$Z(\theta)$ 是高维积分,通常没有闭式,而且每个 $\theta$ 都不一样。

第二项展开:

$$\nabla_\theta \log Z(\theta) = \frac{\nabla_\theta Z(\theta)}{Z(\theta)} = \mathbb{E}_{x \sim p_\theta(x)}[-\nabla_\theta E_\theta(x)]$$

注意期望是对 **$p_\theta$ 自己**取的——要从当前模型采样!这通常需要 MCMC,慢且方差大。这就是经典 EBM 训练困难的根源。

**核心问题**:$Z(\theta)$ 让最大似然几乎不可行。能不能绕开 $Z$?

---

## 二、Hyvärinen 的关键洞察(2005)

**Score** 定义:

$$s(x) := \nabla_x \log p(x)$$

注意是对 **$x$** 求梯度,不是对 $\theta$。

关键观察:

$$\nabla_x \log p_\theta(x) = \nabla_x \log \frac{\tilde p_\theta(x)}{Z(\theta)} = \nabla_x \log \tilde p_\theta(x) - \underbrace{\nabla_x \log Z(\theta)}_{= 0}$$

**$Z(\theta)$ 不依赖 $x$,所以它的 $x$ 梯度恒为 0!**

也就是说,**score 完全不需要知道 $Z$**。我们只要建模 $\nabla_x \log p(x)$,就绕开了归一化的灾难。

这是 Hyvärinen 2005 的根本动机:**学 score 而不是学密度**。

---

## 三、Score 的几何意义

直观一下,$\nabla_x \log p(x)$ 是什么?

它是一个向量场,每点的方向**指向 $\log p$ 增长最快的方向**——也就是**指向高密度区域**。

```
低密度
 ↑↑↑↑↑
 ↑↑↑↑↑      ← score 向量在四周都"指向中心"
[高密度]
 ↓↓↓↓↓
 ↓↓↓↓↓
低密度
```

**所以 score 是一个"指向数据流形"的向导场**。如果你站在低密度区域,顺着 score 走,会走向高密度区域(数据集中的地方)。

这给了我们采样的思路:**有了 score,就能用类似"梯度上升"的过程从噪声走向数据**。这就是 Langevin 动力学的雏形,我们后面会展开。

---

## 四、Score Matching 损失

我们想学一个网络 $s_\theta(x)$ 逼近真实的 $\nabla_x \log p_{\text{data}}(x)$。最自然的损失:

$$J_{\text{ESM}}(\theta) = \frac{1}{2}\mathbb{E}_{x \sim p_{\text{data}}}\left[\| s_\theta(x) - \nabla_x \log p_{\text{data}}(x) \|^2\right]$$

(下标 ESM = Explicit Score Matching)

**问题**:这没法直接算——我们**不知道** $\nabla_x \log p_{\text{data}}(x)$,这正是要学的东西!陷入鸡生蛋蛋生鸡。

Hyvärinen 的关键贡献是证明了一个**奇迹般的恒等式**:

$$J_{\text{ESM}}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}}\left[\frac{1}{2}\| s_\theta(x) \|^2 + \text{tr}(\nabla_x s_\theta(x))\right] + \text{const}$$

**意义**:右边**完全不需要 $\nabla_x \log p_{\text{data}}$**!只用 $s_\theta$ 自身和它的雅可比迹。我们可以直接最小化它。

---

## 五、Hyvärinen 恒等式的推导

这个推导值得细看,只用一次分部积分(integration by parts)。

展开 $J_{\text{ESM}}$:

$$J_{\text{ESM}} = \frac{1}{2}\mathbb{E}_{p_{\text{data}}}\left[\| s_\theta(x) \|^2\right] - \mathbb{E}_{p_{\text{data}}}\left[s_\theta(x)^\top \nabla_x \log p_{\text{data}}(x)\right] + \text{const}$$

(最后一项 $\frac{1}{2}\mathbb{E}\|\nabla_x \log p_{\text{data}}\|^2$ 不依赖 $\theta$,是常数)

第一项已经只含 $s_\theta$,好。**麻烦在第二项**,展开为积分:

$$\mathbb{E}_{p_{\text{data}}}\left[s_\theta(x)^\top \nabla_x \log p_{\text{data}}(x)\right] = \int p_{\text{data}}(x)\, s_\theta(x)^\top \nabla_x \log p_{\text{data}}(x)\, dx$$

用对数导数公式 $\nabla_x \log p = \nabla_x p / p$:

$$= \int p_{\text{data}}(x)\, s_\theta(x)^\top \frac{\nabla_x p_{\text{data}}(x)}{p_{\text{data}}(x)}\, dx = \int s_\theta(x)^\top \nabla_x p_{\text{data}}(x)\, dx$$

**$p_{\text{data}}$ 被消掉了!** 这一步是关键——它让我们不再需要知道 $p_{\text{data}}$ 的密度形式,只需要它的存在。

现在用**分部积分**(假设 $p_{\text{data}}(x) \to 0$ 在边界上,这在标准条件下成立):

$$\int s_\theta(x)^\top \nabla_x p_{\text{data}}(x)\, dx = -\int p_{\text{data}}(x)\, \nabla_x \cdot s_\theta(x)\, dx = -\mathbb{E}_{p_{\text{data}}}[\nabla_x \cdot s_\theta(x)]$$

其中 $\nabla_x \cdot s_\theta = \sum_i \partial s_{\theta,i} / \partial x_i = \text{tr}(\nabla_x s_\theta)$ 是散度,等于雅可比矩阵的迹。

**代回**:

$$J_{\text{ESM}} = \mathbb{E}_{p_{\text{data}}}\left[\frac{1}{2}\| s_\theta(x) \|^2 + \text{tr}(\nabla_x s_\theta(x))\right] + \text{const}$$

这就是 Hyvärinen 恒等式。**完全可以用 $p_{\text{data}}$ 的样本(数据集)蒙特卡洛估计**——不需要知道密度本身。

---

## 六、致命问题:trace 项算不动

理论很美,工程很疼。

$\text{tr}(\nabla_x s_\theta(x))$ 是 $s_\theta$ **雅可比矩阵的迹**。如果 $x \in \mathbb{R}^d$:

$$\text{tr}(\nabla_x s_\theta(x)) = \sum_{i=1}^d \frac{\partial s_{\theta,i}(x)}{\partial x_i}$$

要算这个,你需要 **$d$ 次反向传播**(每次拿一个偏导)。对于 $d = 32 \times 32 \times 3 \approx 3000$(小图像)就已经吃不消,对于 $d = 256 \times 256 \times 3 \approx 200{,}000$ 完全没法算。

**结果**:Hyvärinen 2005 在低维问题上能 work,但在图像生成这种高维场景下,原版 score matching 完全不实用。沉寂了 14 年。

---

## 七、关键转机:Denoising Score Matching(Vincent 2011)

这是后来一切的基础。Vincent 想:**能不能避开 trace 项?**

他的洞察:**给数据加点噪声,再学加噪后分布的 score**。

设噪声 $\sigma > 0$,定义加噪分布:

$$q_\sigma(\tilde x \mid x) = \mathcal{N}(\tilde x; x, \sigma^2 I)$$

$$q_\sigma(\tilde x) = \int q_\sigma(\tilde x \mid x) p_{\text{data}}(x)\, dx$$

$q_\sigma(\tilde x)$ 是**数据分布加高斯噪声**后的边际分布。我们要学 $q_\sigma$ 的 score,即 $\nabla_{\tilde x} \log q_\sigma(\tilde x)$。

定义 **Denoising Score Matching (DSM)** 损失:

$$\boxed{J_{\text{DSM}}(\theta) = \frac{1}{2}\mathbb{E}_{x \sim p_{\text{data}}, \tilde x \sim q_\sigma(\tilde x \mid x)}\left[\| s_\theta(\tilde x) - \nabla_{\tilde x} \log q_\sigma(\tilde x \mid x) \|^2\right]}$$

注意里面是 $\log q_\sigma(\tilde x \mid x)$(条件概率,有闭式),不是 $\log q_\sigma(\tilde x)$(边际,无闭式)。

Vincent 证明了:

$$\arg\min_\theta J_{\text{DSM}}(\theta) = \arg\min_\theta J_{\text{ESM}}^{(\sigma)}(\theta)$$

也就是 **DSM 和 ESM(对 $q_\sigma$)的最优解相同**。学 $q_\sigma$ 的 score 这件事,可以通过 DSM 来做——而 DSM **没有 trace 项**!

---

## 八、DSM 损失化简

把 $q_\sigma(\tilde x \mid x) = \mathcal{N}(\tilde x; x, \sigma^2 I)$ 代入:

$$\log q_\sigma(\tilde x \mid x) = -\frac{\| \tilde x - x \|^2}{2\sigma^2} + \text{const}$$

$$\nabla_{\tilde x} \log q_\sigma(\tilde x \mid x) = -\frac{\tilde x - x}{\sigma^2}$$

代回 DSM 损失:

$$J_{\text{DSM}} = \frac{1}{2}\mathbb{E}_{x, \tilde x}\left[\left\| s_\theta(\tilde x) + \frac{\tilde x - x}{\sigma^2} \right\|^2\right]$$

用重参数化 $\tilde x = x + \sigma \epsilon,\, \epsilon \sim \mathcal{N}(0, I)$:

$$\frac{\tilde x - x}{\sigma^2} = \frac{\epsilon}{\sigma}$$

所以:

$$J_{\text{DSM}} = \frac{1}{2}\mathbb{E}_{x, \epsilon}\left[\left\| s_\theta(\tilde x) + \frac{\epsilon}{\sigma} \right\|^2\right]$$

**这是一个干净的 MSE!** 没有 trace,没有积分,可以直接训练。

---

## 九、第一个奇迹:DSM ≡ "学去噪"

让我们用另一种参数化。定义"去噪函数":

$$D_\theta(\tilde x) := \tilde x + \sigma^2 s_\theta(\tilde x)$$

代入 DSM 损失:

$$J_{\text{DSM}} = \frac{1}{2\sigma^2}\mathbb{E}_{x, \tilde x}\left[\| D_\theta(\tilde x) - x \|^2\right]$$

**意义**:DSM 等价于训练一个去噪器——给加噪图像 $\tilde x$,输出原始图像 $x$。

**这就是 score 视角和 diffusion 视角第一次"对上"了**:

- 第一讲(变分视角):DDPM 让网络预测 $\epsilon$ 或 $\mu$,本质上是去噪
- 这一讲(score 视角):DSM 学 score 也等价于去噪
- **它们本质上是同一件事**!score 就是去噪方向(乘上一个标量)

更精确地:

$$s_\theta(\tilde x) = \frac{D_\theta(\tilde x) - \tilde x}{\sigma^2} = -\frac{\epsilon_\theta(\tilde x)}{\sigma}$$

(其中 $\epsilon_\theta$ 是预测的噪声)

**这是连接两条线的第一个公式**。后面 Score SDE 会把这个对应推到完整框架。

---

## 十、有了 score 之后怎么采样?

答案:**Langevin 动力学(Langevin dynamics)**。

物理背景:Langevin 方程描述布朗运动中的粒子。在我们这里,它给出了一个从 $p$ 采样的方法,**只需要 $\nabla_x \log p(x)$**:

$$x_{k+1} = x_k + \frac{\eta}{2}\nabla_x \log p(x_k) + \sqrt{\eta}\, z_k,\quad z_k \sim \mathcal{N}(0, I)$$

其中 $\eta > 0$ 是步长。可以证明:当 $\eta \to 0$、$k \to \infty$ 时,$x_k$ 的分布收敛到 $p(x)$。

**直观**:

- 漂移项 $\frac{\eta}{2}\nabla_x \log p$:把粒子拉向高密度区域
- 噪声项 $\sqrt{\eta}\, z$:防止粒子卡在局部模式上,保证遍历全分布

如果我们学到了 $s_\theta \approx \nabla_x \log p$,直接代进去:

$$x_{k+1} = x_k + \frac{\eta}{2}s_\theta(x_k) + \sqrt{\eta}\, z_k$$

——就能从 $p$ 采样了。**无需归一化常数,无需采样链复杂的 EBM,只要 score**。

---

## 十一、致命问题二:低密度区域 score 不准

理论很美,实践又疼一次。

考虑实际数据:图像在 $\mathbb{R}^{d}$($d \approx 10^5$)中其实集中在一个**低维流形**上,绝大部分空间是低密度区域。

问题是:

- **训练数据稀疏**:低密度区域几乎没有训练样本
- **score 学不准**:网络在低密度区域输出基本是随机的
- **Langevin 卡住**:从随机噪声 $\mathcal{N}(0, I)$ 出发,初始点几乎肯定在低密度区域,$s_\theta$ 给出的方向是错的,Langevin 完全不收敛

这正是 score-based 方法 2011 年之后又沉寂多年的原因——理论可行,但**实际从噪声开始采样不 work**。

---

## 十二、第二个奇迹:NCSN(Song & Ermon 2019)

直到 2019 年,Yang Song 想到:**用多种尺度的噪声**。

核心思路:

- **大噪声 $\sigma_1$**:加大量噪声后,$q_{\sigma_1}$ 在整个空间都有显著密度,score 处处都能学准
- **小噪声 $\sigma_L$**:噪声很小时,$q_{\sigma_L} \approx p_{\text{data}}$,采样得到的接近真实数据
- **中间过渡**:从 $\sigma_1$ 退火到 $\sigma_L$

具体地,选 $L$ 个噪声尺度 $\sigma_1 > \sigma_2 > \cdots > \sigma_L$(几何递减,如 $\sigma_l = \sigma_1 \cdot (\sigma_L/\sigma_1)^{(l-1)/(L-1)}$)。

训练一个**噪声条件 score 网络** $s_\theta(x, \sigma)$,加权 DSM 损失:

$$J_{\text{NCSN}} = \frac{1}{L}\sum_{l=1}^L \lambda(\sigma_l) \cdot J_{\text{DSM}}(\theta; \sigma_l)$$

其中 $\lambda(\sigma_l) = \sigma_l^2$(平衡不同尺度的损失大小)。

代入展开,加权后形式漂亮:

$$J_{\text{NCSN}} = \frac{1}{2L}\sum_l \mathbb{E}_{x, \epsilon}\left[\| \sigma_l \, s_\theta(\tilde x_l, \sigma_l) + \epsilon \|^2\right]$$

(其中 $\tilde x_l = x + \sigma_l \epsilon$)

这是个统一的 MSE,可以训练。

---

## 十三、Annealed Langevin Dynamics

采样时,从大噪声到小噪声**退火**:

```
初始化: x ~ N(0, σ_1^2 I)         # 大噪声分布

for l = 1, 2, ..., L:              # 逐步退火
    η_l = ε · σ_l^2 / σ_L^2        # 步长(随尺度变小而变小)
    for k = 1, ..., K:             # 每个尺度跑 K 步 Langevin
        z ~ N(0, I)
        x ← x + (η_l / 2) · s_θ(x, σ_l) + sqrt(η_l) · z

返回 x
```

**直观**:大尺度先把 $x$ 拉到正确的"大方向"上,小尺度再做精修。这绕开了"从纯噪声出发 score 不准"的问题——因为初始点 $\mathcal{N}(0, \sigma_1^2 I)$ 正好是 $q_{\sigma_1}$ 的高密度区域。

NCSN 是 2019 年第一个用 score-based 方法在 CIFAR-10 上做出能看的图像的工作——比 GAN 还差一点,但已经证明了路线可行。

---

## 十四、和第一讲的对照

让我把两条线放一起,你会看到惊人的对称性:

| | 变分视角(第一讲) | Score 视角(这一讲) |
|---|---|---|
| 起点 | Sohl-Dickstein 2015 | Hyvärinen 2005 |
| 核心问题 | 边际化算不出来 | 归一化常数 $Z$ 算不出来 |
| 解决思路 | ELBO 下界 | 学 score 绕开 $Z$ |
| 第一版工程问题 | 方差大 | trace 项算不动 |
| 关键补丁 | DDPM(简化目标) | DSM(去噪等价) |
| 工程问题二 | 仍要 1000 步采样 | 低密度区域 score 不准 |
| 突破 | 使用多步小噪声(自带) | NCSN 多尺度噪声 |
| 训练目标 | $\| \epsilon - \epsilon_\theta(x_t, t) \|^2$ | $\sigma \| s_\theta(\tilde x, \sigma) + \epsilon/\sigma \|^2$ |
| 采样 | 逆向链 | Annealed Langevin |
| 提出年份 | 2020(DDPM) | 2019(NCSN) |

**两条独立发展的线,2019-2020 几乎同时成熟,得出几乎相同形式的目标函数**。这不是巧合——它们在数学上**本质等价**。下一讲 Score SDE 会把这件事完全说清。

---

## 十五、要点回顾

- **Score** 是 $\nabla_x \log p(x)$,绕开了 $Z$ 的问题
- **原版 Score Matching**(Hyvärinen 2005):用分部积分把损失变成不需要真实 score 的形式,但 trace 项算不动
- **DSM**(Vincent 2011):用加噪后的 score,损失变成漂亮的 MSE,等价于"学去噪"
- 关键洞察:**score ≈ 去噪方向**——这是连接两条线的桥梁
- **Langevin 采样**:用 score 做"梯度上升 + 噪声"
- **NCSN**(Song & Ermon 2019):多尺度噪声 + annealed Langevin,解决低密度区域问题
- 形式上和 DDPM 几乎相同——下一讲统一两条线

---

到这里,**两条独立的源头**都讲完了。下一步可以选:

1. **第三讲 DDPM**:推贝叶斯改写 + $\epsilon$-预测,把第一讲的"原始版本"升级成"现代版本"。这是工程上的关键突破。
2. **第四讲 Score SDE**:直接进入统一框架,看两条线如何在 SDE 视角下完全合二为一。
3. **再展开 NCSN**:比如详细讨论 noise schedule 的选择、Langevin 收敛理论。

你想怎么走?

我个人建议按**1 → 4** 的顺序——先把 DDPM 完整推完(它是 score SDE 的"离散版基石"),再看 SDE 框架如何把所有东西统一,这样会比直接跳到 SDE 更有"恍然大悟"的感觉。