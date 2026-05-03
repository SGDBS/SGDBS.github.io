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

现在用**分部积分**(假设 $p_{\text{data}}(x) \to 0$ 在边界上,这在标准条件下成立，完整证明见附录1):

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

Vincent 证明了(详细证明在附录2):

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

## 附录1： $\int s_\theta(x)^\top \nabla_x p_{\text{data}}(x)\, dx = -\int p_{\text{data}}(x)\, \nabla_x \cdot s_\theta(x)\, dx$

这一步是 score matching 推导里的关键一步——分部积分。我们一行一行推清楚,顺便把这个工具讲透,因为它在后面 SDE 推导里还会频繁出现。

---

### 一、先看一维情形(最容易理解)

一维分部积分公式(高中知识):

$$\int u(x)\, v'(x)\, dx = u(x) v(x) - \int u'(x)\, v(x)\, dx$$

或者写成定积分:

$$\int_{-\infty}^{\infty} u(x)\, v'(x)\, dx = \left[u(x) v(x)\right]_{-\infty}^{\infty} - \int_{-\infty}^{\infty} u'(x)\, v(x)\, dx$$

我们要推:

$$\int s_\theta(x)\, \frac{dp_{\text{data}}(x)}{dx}\, dx = -\int p_{\text{data}}(x)\, \frac{ds_\theta(x)}{dx}\, dx$$

(暂且把 $s_\theta, p_{\text{data}}$ 都当作一维标量函数)

设 $u = s_\theta$,$v = p_{\text{data}}$。那么 $v' = \frac{dp_{\text{data}}}{dx}$,$u' = \frac{ds_\theta}{dx}$。代入分部积分公式:

$$\int_{-\infty}^{\infty} s_\theta(x)\, \frac{dp_{\text{data}}}{dx}\, dx = \underbrace{\left[s_\theta(x)\, p_{\text{data}}(x)\right]_{-\infty}^{\infty}}_{\text{边界项}} - \int_{-\infty}^{\infty} \frac{ds_\theta}{dx}\, p_{\text{data}}(x)\, dx$$

**关键步骤:边界项为 0**。

这要求一个**正则性假设**:$\lim_{|x| \to \infty} s_\theta(x)\, p_{\text{data}}(x) = 0$。

什么时候成立?

- $p_{\text{data}}(x) \to 0$ 在 $|x| \to \infty$ 处 ✓(任何合理的概率密度都满足,因为 $\int p\, dx = 1$ 强迫它衰减)
- $s_\theta(x)$ 不能"爆炸得比 $p_{\text{data}}$ 衰减得还快"——通常 $p_{\text{data}}$ 是指数衰减(高斯尾巴),$s_\theta$ 是神经网络多项式增长,这个条件几乎总是成立

所以边界项消失,得到:

$$\int s_\theta(x)\, \frac{dp_{\text{data}}}{dx}\, dx = -\int p_{\text{data}}(x)\, \frac{ds_\theta}{dx}\, dx$$

这就是一维版本的恒等式。✓

---

### 二、为什么这个变换能成立?直觉

为什么"导数能从一边换到另一边"?用一个最直观的图景:

$$\int s_\theta\, \frac{dp_{\text{data}}}{dx}\, dx = \int s_\theta\, dp_{\text{data}}$$

(微分形式的写法)用乘法的微分法则:

$$d(s_\theta \cdot p_{\text{data}}) = (ds_\theta) \cdot p_{\text{data}} + s_\theta \cdot (dp_{\text{data}})$$

整理:

$$s_\theta \cdot dp_{\text{data}} = d(s_\theta \cdot p_{\text{data}}) - p_{\text{data}} \cdot ds_\theta$$

两边积分:

$$\int s_\theta\, dp_{\text{data}} = \int d(s_\theta \cdot p_{\text{data}}) - \int p_{\text{data}}\, ds_\theta = [s_\theta \cdot p_{\text{data}}]_{-\infty}^{\infty} - \int p_{\text{data}}\, ds_\theta$$

边界项消失:

$$\int s_\theta\, dp_{\text{data}} = -\int p_{\text{data}}\, ds_\theta$$

**本质上就是乘法求导法则反向用了一次**。

---

### 三、推广到多维:散度定理

多维情形,我们要推的是:

$$\int s_\theta(x)^\top \nabla_x p_{\text{data}}(x)\, dx = -\int p_{\text{data}}(x)\, \nabla_x \cdot s_\theta(x)\, dx$$

注意类型:

- $s_\theta(x) \in \mathbb{R}^d$:向量场
- $\nabla_x p_{\text{data}}(x) \in \mathbb{R}^d$:梯度向量
- $s_\theta(x)^\top \nabla_x p_{\text{data}}(x) \in \mathbb{R}$:两个向量的点积,是标量
- $\nabla_x \cdot s_\theta(x) = \sum_i \partial s_{\theta,i}/\partial x_i \in \mathbb{R}$:散度,是标量

所以两边都是标量积分,类型对得上。

#### 用散度恒等式推

有一个标准的向量分析恒等式(乘积法则的多维版):

$$\nabla \cdot (f \mathbf{v}) = \nabla f \cdot \mathbf{v} + f\, \nabla \cdot \mathbf{v}$$

对任意标量函数 $f$ 和向量场 $\mathbf{v}$ 成立(就是把一维的 $(fg)' = f'g + fg'$ 推广到多维,这里"导数"对应"梯度",左边的"和的一阶导"对应"散度")。

我们这里取 $f = p_{\text{data}}$,$\mathbf{v} = s_\theta$:

$$\nabla \cdot (p_{\text{data}} s_\theta) = \nabla p_{\text{data}} \cdot s_\theta + p_{\text{data}}\, \nabla \cdot s_\theta$$

整理:

$$\nabla p_{\text{data}} \cdot s_\theta = \nabla \cdot (p_{\text{data}} s_\theta) - p_{\text{data}}\, \nabla \cdot s_\theta$$

注意 $\nabla p_{\text{data}} \cdot s_\theta$ 就是 $s_\theta^\top \nabla p_{\text{data}}$。两边积分:

$$\int s_\theta^\top \nabla p_{\text{data}}\, dx = \int \nabla \cdot (p_{\text{data}} s_\theta)\, dx - \int p_{\text{data}}\, \nabla \cdot s_\theta\, dx$$

#### 第一项消失:散度定理

第一项 $\int \nabla \cdot (p_{\text{data}} s_\theta)\, dx$ 是关键。**散度定理(高斯-奥斯特罗格拉茨基定理)**:

$$\int_V \nabla \cdot \mathbf{F}\, dV = \oint_{\partial V} \mathbf{F} \cdot \mathbf{n}\, dS$$

意思是:**一个向量场在体积内的散度积分,等于它在边界上的法向通量**。

我们的积分域 $V = \mathbb{R}^d$ 是整个空间,边界 $\partial V$ 是"无穷远的球面"。要让这个边界积分为 0,需要:

$$\lim_{\|x\| \to \infty} p_{\text{data}}(x)\, s_\theta(x) = 0$$

(且衰减得足够快,具体是要快于 $1/\|x\|^{d-1}$)

这个条件在标准设定下成立——$p_{\text{data}}$ 是合理的概率密度(指数衰减),$s_\theta$ 是神经网络(最多多项式增长),乘积一定衰减到 0。

**所以散度积分 = 边界通量 = 0**。

#### 得到结果

$$\int s_\theta^\top \nabla p_{\text{data}}\, dx = 0 - \int p_{\text{data}}\, \nabla \cdot s_\theta\, dx = -\int p_{\text{data}}\, \nabla \cdot s_\theta\, dx$$

写成期望形式:

$$\int s_\theta^\top \nabla p_{\text{data}}\, dx = -\mathbb{E}_{p_{\text{data}}}[\nabla \cdot s_\theta]$$

---

### 四、再说一句关于 trace 和散度的关系

我们之前写成 $\text{tr}(\nabla_x s_\theta)$,这里又写成 $\nabla_x \cdot s_\theta$,这两个其实是**同一个东西**:

$$\nabla_x \cdot s_\theta(x) = \sum_{i=1}^d \frac{\partial s_{\theta,i}}{\partial x_i}$$

而 $\nabla_x s_\theta$ 是 $s_\theta$ 的雅可比矩阵 $J \in \mathbb{R}^{d \times d}$,$J_{ij} = \partial s_{\theta,i}/\partial x_j$。雅可比矩阵的迹:

$$\text{tr}(\nabla_x s_\theta) = \sum_{i=1}^d J_{ii} = \sum_{i=1}^d \frac{\partial s_{\theta,i}}{\partial x_i}$$

所以**散度 = 雅可比的迹**,只是不同记号。

---

### 五、把推导整合在一起

为了把整个 score matching 的关键化简看清楚,我把完整链条写一遍:

我们要算:

$$\mathbb{E}_{p_{\text{data}}}\left[s_\theta(x)^\top \nabla_x \log p_{\text{data}}(x)\right]$$

**第一步**:展开期望,用对数导数公式 $\nabla_x \log p = \nabla_x p / p$:

$$= \int p_{\text{data}}(x)\, s_\theta(x)^\top \frac{\nabla_x p_{\text{data}}(x)}{p_{\text{data}}(x)}\, dx = \int s_\theta(x)^\top \nabla_x p_{\text{data}}(x)\, dx$$

**$p_{\text{data}}$ 神奇地约掉了**——这是为什么这个推导能 work。

**第二步**:用乘积法则 + 散度定理(刚才推的):

$$= -\int p_{\text{data}}(x)\, \nabla_x \cdot s_\theta(x)\, dx$$

**第三步**:写回期望:

$$= -\mathbb{E}_{p_{\text{data}}}[\nabla_x \cdot s_\theta(x)]$$

完成。✓

---

### 六、这个技巧的"关键魔法"在哪?

值得停下来欣赏一下这个推导为什么这么神奇。

**起点**:$\mathbb{E}_{p_{\text{data}}}[s_\theta^\top \nabla_x \log p_{\text{data}}]$——里面有个 $\nabla_x \log p_{\text{data}}$,**这正是我们不知道、想学的东西**。

**终点**:$-\mathbb{E}_{p_{\text{data}}}[\nabla_x \cdot s_\theta]$——里面只有 $s_\theta$ 和它的导数,**完全不需要知道 $p_{\text{data}}$ 的密度形式**,只要能从 $p_{\text{data}}$ 采样就行(数据集就是采样!)。

**两步操作**做到了这个变换:

1. **对数导数公式**:把 $\log p$ 的梯度变成 $\nabla p / p$,然后 $p$ 约掉
2. **分部积分(散度定理)**:把"$\nabla$ 在 $p$ 上"换成"$\nabla$ 在 $s_\theta$ 上",代价是边界项(为 0)

第二步是核心:**它把"未知函数 $p_{\text{data}}$ 的导数"换成了"已知函数 $s_\theta$ 的导数"**,梯度从积不动的对象转移到了能算的对象。

---

### 七、这个技巧后面还会出现

为什么花这么多时间讲这个推导?因为**分部积分(散度定理)在 diffusion / score matching 框架里是反复出现的核心工具**。后面你会看到:

1. **Score Matching 损失推导**(就是我们刚做的)
2. **Stein's identity**:$\mathbb{E}_p[\nabla \cdot f + f^\top \nabla \log p] = 0$,用同样手法推
3. **Fokker-Planck 方程**:推 SDE 下分布的演化,反复用分部积分
4. **Reverse SDE 推导(Anderson 1982)**:把前向 SDE 翻转成反向,核心一步就是分部积分
5. **Probability Flow ODE**:从 Fokker-Planck 推 ODE,又是分部积分

**学透这一步,后面的 SDE 推导你会觉得"怎么又是这一招"——确实是同一招。**

---

### 八、要点回顾

- 一维分部积分:$\int u v' = [uv] - \int u' v$,边界项消失则得到对称形式
- 多维推广:用乘积法则 $\nabla \cdot (fv) = \nabla f \cdot v + f \nabla \cdot v$ + 散度定理
- 散度定理把"体积内的散度积分"变成"边界上的通量",在合理衰减条件下边界项为 0
- 这个技巧的本质:**把未知函数的导数换成已知函数的导数**
- 散度 = 雅可比矩阵的迹,两个名字一回事
- 这个工具在后续 SDE 推导里反复出现,值得彻底理解



## 附录2：$\arg\min_\theta J_{\text{DSM}}(\theta) = \arg\min_\theta J_{\text{ESM}}^{(\sigma)}(\theta)$
这是 Vincent 2011 论文的核心定理,推导很优雅。我们彻底证一遍——这个证明的技巧和你刚学的分部积分非常类似,都是"把不知道的东西通过代数变换约掉"。

---

### 一、先把两个目标写清楚

设噪声分布:

$$q_\sigma(\tilde x \mid x) = \mathcal{N}(\tilde x; x, \sigma^2 I)$$

加噪后的边际分布:

$$q_\sigma(\tilde x) = \int q_\sigma(\tilde x \mid x)\, p_{\text{data}}(x)\, dx$$

**ESM(对 $q_\sigma$)**:学 $q_\sigma$ 的 score:

$$J_{\text{ESM}}^{(\sigma)}(\theta) = \frac{1}{2}\mathbb{E}_{\tilde x \sim q_\sigma}\left[\| s_\theta(\tilde x) - \nabla_{\tilde x} \log q_\sigma(\tilde x) \|^2\right]$$

**DSM**:学条件 score:

$$J_{\text{DSM}}(\theta) = \frac{1}{2}\mathbb{E}_{x \sim p_{\text{data}},\, \tilde x \sim q_\sigma(\tilde x \mid x)}\left[\| s_\theta(\tilde x) - \nabla_{\tilde x} \log q_\sigma(\tilde x \mid x) \|^2\right]$$

**注意区别**:

- ESM 里是 $\nabla \log q_\sigma(\tilde x)$ —— 边际的 score,**没有闭式**(里面有积分)
- DSM 里是 $\nabla \log q_\sigma(\tilde x \mid x)$ —— 条件的 score,**有闭式**(高斯)

我们要证:**两者作为 $\theta$ 的函数,只差一个常数**——所以最优 $\theta$ 相同。

---

### 二、整体策略

我们**不直接证两者相等**,而是证明:

$$J_{\text{DSM}}(\theta) = J_{\text{ESM}}^{(\sigma)}(\theta) + C$$

其中 $C$ **不依赖于 $\theta$**。这样:

$$\arg\min_\theta J_{\text{DSM}} = \arg\min_\theta J_{\text{ESM}}^{(\sigma)}$$

策略:把两个目标都展开,看交叉项是否相等。

---

### 三、把两个目标展开

平方展开 $\| s_\theta - t \|^2 = \| s_\theta \|^2 - 2 s_\theta^\top t + \| t \|^2$,所以:

**ESM 展开**:

$$J_{\text{ESM}}^{(\sigma)} = \underbrace{\frac{1}{2}\mathbb{E}_{q_\sigma(\tilde x)}\left[\| s_\theta(\tilde x) \|^2\right]}_{(A)} - \underbrace{\mathbb{E}_{q_\sigma(\tilde x)}\left[s_\theta(\tilde x)^\top \nabla_{\tilde x} \log q_\sigma(\tilde x)\right]}_{(B)} + \underbrace{\frac{1}{2}\mathbb{E}_{q_\sigma(\tilde x)}\left[\| \nabla_{\tilde x} \log q_\sigma(\tilde x) \|^2\right]}_{C_1}$$

**DSM 展开**:

$$J_{\text{DSM}} = \underbrace{\frac{1}{2}\mathbb{E}_{p_{\text{data}}(x)\, q_\sigma(\tilde x \mid x)}\left[\| s_\theta(\tilde x) \|^2\right]}_{(A')} - \underbrace{\mathbb{E}_{p_{\text{data}}(x)\, q_\sigma(\tilde x \mid x)}\left[s_\theta(\tilde x)^\top \nabla_{\tilde x} \log q_\sigma(\tilde x \mid x)\right]}_{(B')} + \underbrace{\frac{1}{2}\mathbb{E}_{p_{\text{data}}(x)\, q_\sigma(\tilde x \mid x)}\left[\| \nabla_{\tilde x} \log q_\sigma(\tilde x \mid x) \|^2\right]}_{C_2}$$

**目标**:证明 $(A) = (A')$、$(B) = (B')$,这样剩下 $C_1, C_2$ 都不依赖 $\theta$,差就是常数。

---

### 四、证 $(A) = (A')$:边际化等价

这一步几乎是定义。

$(A')$ 里的双重期望可以重组:

$$(A') = \frac{1}{2}\int p_{\text{data}}(x) \int q_\sigma(\tilde x \mid x)\, \| s_\theta(\tilde x) \|^2\, d\tilde x\, dx$$

交换积分顺序:

$$= \frac{1}{2}\int \| s_\theta(\tilde x) \|^2 \underbrace{\int q_\sigma(\tilde x \mid x)\, p_{\text{data}}(x)\, dx}_{= q_\sigma(\tilde x)}\, d\tilde x$$

$$= \frac{1}{2}\int q_\sigma(\tilde x)\, \| s_\theta(\tilde x) \|^2\, d\tilde x = \frac{1}{2}\mathbb{E}_{q_\sigma(\tilde x)}[\| s_\theta(\tilde x) \|^2] = (A)$$

所以 **$(A) = (A')$** ✓

直觉:被积函数 $\| s_\theta(\tilde x) \|^2$ 只依赖 $\tilde x$,和 $x$ 无关,所以对 $x$ 边际化把联合分布塌缩成边际分布,期望不变。

---

### 五、证 $(B) = (B')$:核心一步

这是关键的一步。我们要证:

$$\mathbb{E}_{q_\sigma(\tilde x)}\left[s_\theta(\tilde x)^\top \nabla_{\tilde x} \log q_\sigma(\tilde x)\right] \stackrel{?}{=} \mathbb{E}_{p_{\text{data}}(x)\, q_\sigma(\tilde x \mid x)}\left[s_\theta(\tilde x)^\top \nabla_{\tilde x} \log q_\sigma(\tilde x \mid x)\right]$$

左边的"边际 score"和右边的"条件 score"**在期望意义下相等**。这是个非平凡的事实——表面上看两者完全不同。

#### 推导左边 $(B)$

$$(B) = \int q_\sigma(\tilde x)\, s_\theta(\tilde x)^\top \nabla_{\tilde x} \log q_\sigma(\tilde x)\, d\tilde x$$

用对数导数公式 $\nabla_{\tilde x} \log q_\sigma(\tilde x) = \nabla_{\tilde x} q_\sigma(\tilde x) / q_\sigma(\tilde x)$:

$$= \int q_\sigma(\tilde x)\, s_\theta(\tilde x)^\top \frac{\nabla_{\tilde x} q_\sigma(\tilde x)}{q_\sigma(\tilde x)}\, d\tilde x = \int s_\theta(\tilde x)^\top \nabla_{\tilde x} q_\sigma(\tilde x)\, d\tilde x$$

(还是那个"$q_\sigma$ 神奇消掉"的把戏)

#### 推 $\nabla_{\tilde x} q_\sigma(\tilde x)$

$q_\sigma(\tilde x) = \int q_\sigma(\tilde x \mid x)\, p_{\text{data}}(x)\, dx$,对 $\tilde x$ 求梯度($p_{\text{data}}$ 不依赖 $\tilde x$,梯度可以进入积分):

$$\nabla_{\tilde x} q_\sigma(\tilde x) = \int \nabla_{\tilde x} q_\sigma(\tilde x \mid x)\, p_{\text{data}}(x)\, dx$$

再用一次对数导数公式($\nabla_{\tilde x} q_\sigma(\tilde x \mid x) = q_\sigma(\tilde x \mid x)\, \nabla_{\tilde x} \log q_\sigma(\tilde x \mid x)$):

$$= \int q_\sigma(\tilde x \mid x)\, \nabla_{\tilde x} \log q_\sigma(\tilde x \mid x)\, p_{\text{data}}(x)\, dx$$

#### 代回 $(B)$

$$(B) = \int s_\theta(\tilde x)^\top \left[\int q_\sigma(\tilde x \mid x)\, \nabla_{\tilde x} \log q_\sigma(\tilde x \mid x)\, p_{\text{data}}(x)\, dx\right] d\tilde x$$

把 $s_\theta(\tilde x)$ 移进内层积分(它不依赖 $x$):

$$= \int \int p_{\text{data}}(x)\, q_\sigma(\tilde x \mid x)\, s_\theta(\tilde x)^\top \nabla_{\tilde x} \log q_\sigma(\tilde x \mid x)\, dx\, d\tilde x$$

这就是双重期望:

$$= \mathbb{E}_{p_{\text{data}}(x)\, q_\sigma(\tilde x \mid x)}\left[s_\theta(\tilde x)^\top \nabla_{\tilde x} \log q_\sigma(\tilde x \mid x)\right] = (B')$$

✓

---

### 六、合起来

我们证明了:

- $(A) = (A')$
- $(B) = (B')$
- $C_1, C_2$ 都不依赖 $\theta$

所以:

$$J_{\text{ESM}}^{(\sigma)}(\theta) = (A) - (B) + C_1$$

$$J_{\text{DSM}}(\theta) = (A') - (B') + C_2 = (A) - (B) + C_2$$

两式相减:

$$\boxed{J_{\text{DSM}}(\theta) - J_{\text{ESM}}^{(\sigma)}(\theta) = C_2 - C_1 = \text{常数(不依赖 } \theta \text{)}}$$

最小化谁结果都一样:

$$\arg\min_\theta J_{\text{DSM}} = \arg\min_\theta J_{\text{ESM}}^{(\sigma)}$$

证毕。

---

### 七、关键技巧总结

整个证明就靠两个把戏,反复用:

#### 技巧 1:对数导数公式

$$\nabla \log p = \frac{\nabla p}{p}$$

它的威力:把 $\log p$ 拆开后,$p$ 出现在分母,容易和外面的 $p$ 约掉。我们用了**两次**:

- 一次把 $\nabla \log q_\sigma(\tilde x)$ 变成 $\nabla q_\sigma(\tilde x) / q_\sigma(\tilde x)$,然后约掉 $q_\sigma(\tilde x)$
- 一次把 $\nabla q_\sigma(\tilde x \mid x)$ 写成 $q_\sigma(\tilde x \mid x)\, \nabla \log q_\sigma(\tilde x \mid x)$,造出条件 score

#### 技巧 2:交换积分顺序

边际分布是积分的结果 $q_\sigma(\tilde x) = \int q_\sigma(\tilde x \mid x) p_{\text{data}}(x)\, dx$。当我们对 $\tilde x$ 求梯度后,这个积分依然在,只是被积函数变了。

**梯度算子可以"穿过"对 $x$ 的积分**(因为求导和积分变量不同),这让我们能把"边际 score 的期望"和"条件 score 的期望"联系起来。

---

### 八、为什么这个等价"看起来不可思议"?

直觉上,边际 score $\nabla \log q_\sigma(\tilde x)$ 和条件 score $\nabla \log q_\sigma(\tilde x \mid x)$ **完全不同**:

- 边际 score:在所有可能的 $x$ 上平均后的"指向高密度方向"
- 条件 score:已知 $x$ 时,$\tilde x$ 偏离 $x$ 的方向(有闭式 $-(\tilde x - x)/\sigma^2$)

它们怎么会"等价"?

关键是**等价的是期望**,不是逐点相等。具体地:

$$\nabla_{\tilde x} \log q_\sigma(\tilde x) = \mathbb{E}_{p(x \mid \tilde x)}[\nabla_{\tilde x} \log q_\sigma(\tilde x \mid x)]$$

(这其实是 Tweedie 公式的雏形——边际 score 是条件 score 在后验下的期望)

我们证明的不是逐点相等,而是**在更外层的期望下相等**。这就是为什么 $J_{\text{ESM}}$ 和 $J_{\text{DSM}}$ 只差一个不依赖 $\theta$ 的常数——交叉项 $(B), (B')$ 被这个隐藏的恒等式连接起来。

---

### 九、为什么这件事如此重要?

回顾我们绕了一大圈做了什么:

| | 我们想要的 | 实际可算的 |
|---|---|---|
| ESM | $\nabla \log p_{\text{data}}$(根本不知道) | 没法直接算 |
| ESM($q_\sigma$) | $\nabla \log q_\sigma(\tilde x)$(边际,无闭式) | 没法直接算 |
| DSM | $\nabla \log q_\sigma(\tilde x \mid x)$(条件,**有闭式**!) | **可以直接算!** |

DSM 把"想学边际 score"这件事,通过"加点噪声 + 用条件 score 当目标"的迂回,变成了一个**完全可计算**的 MSE 损失。

代入条件高斯的 score $\nabla \log q_\sigma(\tilde x \mid x) = -(\tilde x - x)/\sigma^2$,得到:

$$J_{\text{DSM}} = \frac{1}{2}\mathbb{E}_{x, \tilde x}\left[\left\| s_\theta(\tilde x) + \frac{\tilde x - x}{\sigma^2} \right\|^2\right]$$

——干净的回归损失,可以直接训练。这就是 DSM 在工程上的伟大之处。

---

### 十、和上一讲分部积分对比

注意到这个证明**没有用分部积分**——它纯粹靠"对数导数 + 积分顺序交换"。这是因为:

- Hyvärinen ESM 的化简(上一讲)需要把 $\nabla p_{\text{data}}$ 转移到 $s_\theta$ 上,所以要分部积分
- Vincent DSM 的等价性证明则是把"边际 score"通过定义本身展开成"条件 score 的积分",不需要"导数转移"

两个证明思路不同,但**共同的精神**是:利用**对数导数公式**让"未知函数"自动从分子分母约掉,最终得到一个只用已知量的式子。

这是 score 视角下反复出现的核心技术——记住它,后面在 Tweedie 公式、reverse SDE 推导里你都会再看到。

---

### 十一、要点回顾

- DSM 和 ESM(对 $q_\sigma$)只差一个不依赖 $\theta$ 的常数,所以最优解相同
- 证明靠两步:展开平方损失,然后证明 $(A) = (A')$ 和 $(B) = (B')$
- $(A) = (A')$ 用积分顺序交换得到边际化
- $(B) = (B')$ 用对数导数公式让 $q_\sigma$ 在分母分子约掉,然后造出 $p_{\text{data}}\, q_\sigma(\tilde x \mid x)$ 的形式
- 隐藏的核心事实:**边际 score = 条件 score 在后验下的期望**(Tweedie 公式)
- 这让我们绕开"边际 score 没有闭式"的问题,转而学有闭式的"条件 score"——目标完全等价但完全可计算

