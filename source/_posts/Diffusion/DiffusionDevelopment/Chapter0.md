---
title: Chapter 0. 数学知识
categories: Diffusion Models
date: 2026-05-01 12:00:00
mathjax: true
tags:
    - AI
    - Diffusion Models
---


## 1. 核心动机:为什么是"扩散"?

2015 年之前,生成模型主要有几条路:

- **VAE**(2013):用变分下界,但生成质量有限
- **GAN**(2014):效果好但训练不稳定,没有显式概率
- **自回归模型**(PixelRNN 等):生成慢,不适合连续数据
- **Normalizing Flow**:要求可逆架构,表达力受限

Sohl-Dickstein 想要的是:**既有显式概率(像 VAE),训练又稳定,表达力还强**。

灵感来自非平衡热力学。考虑一滴墨水滴入水中:

- **正向过程**:墨水从有结构(集中一点)逐渐扩散成无结构(均匀分布)。这个过程**简单、可解析**。
- **逆向过程**:让均匀分布的水"凝聚"回一滴墨水。这看起来违反热力学第二定律(熵减),物理上不可能自发发生。**但如果我们知道每一步该怎么做(学出来的逆过程),数学上是可以的。**

这就是 diffusion model 的核心思想:**用一个简单的、可解析的破坏过程,然后学习它的逆过程**。


## 2. 高斯分布的一些性质

一维高斯 $\mathcal{N}(x; \mu, \sigma^2)$ 的密度:

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

多维高斯 $\mathcal{N}(x; \mu, \Sigma)$ 的密度:

$$p(x) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)\right)$$

**关键性质 —— 高斯的可加性**:如果 $x \sim \mathcal{N}(\mu_1, \sigma_1^2)$ 且 $y \mid x \sim \mathcal{N}(ax, \sigma_2^2)$,那么边际 $y \sim \mathcal{N}(a\mu_1, a^2\sigma_1^2 + \sigma_2^2)$。这个性质后面 DDPM 会大量用。

## 3. KL 散度

衡量两个分布的"差异":

$$D_{\text{KL}}(q \| p) = \mathbb{E}_{x \sim q}\left[\log \frac{q(x)}{p(x)}\right]$$

性质:非负、不对称、$D_{\text{KL}}(q\|p) = 0 \iff q = p$。

**两个高斯之间的 KL 有闭式解**(这个非常重要,后面会反复用):

$$D_{\text{KL}}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

## 4. 马尔可夫链

序列 $x_0, x_1, \ldots, x_T$ 满足马尔可夫性:

$$p(x_t \mid x_{t-1}, x_{t-2}, \ldots, x_0) = p(x_t \mid x_{t-1})$$

即"未来只依赖于现在,与过去无关"。联合分布可以分解:

$$p(x_{0:T}) = p(x_0) \prod_{t=1}^T p(x_t \mid x_{t-1})$$

## 5. 变分下界(ELBO)

这是从 VAE 继承来的核心工具。我们想最大化数据似然 $\log p_\theta(x_0)$,但它通常算不出来(因为要对所有隐变量积分):

$$p_\theta(x_0) = \int p_\theta(x_0, x_{1:T})\, dx_{1:T}$$

引入一个辅助分布 $q(x_{1:T} \mid x_0)$,用 Jensen 不等式:

$$\log p_\theta(x_0) = \log \int p_\theta(x_{0:T})\, dx_{1:T} = \log \int q(x_{1:T} \mid x_0) \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\, dx_{1:T}$$

$$\geq \mathbb{E}_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right] =: \mathcal{L}(\theta)$$

这个下界 $\mathcal{L}$ 就是 **ELBO**(Evidence Lower BOund)。最大化 ELBO 就是在(近似)最大化对数似然。



### 一、为什么需要 ELBO?

我们的目标是**最大似然**:给定数据 $\{x^{(1)}, \ldots, x^{(N)}\}$,找参数 $\theta$ 使

$$\theta^* = \arg\max_\theta \sum_{i=1}^N \log p_\theta(x^{(i)})$$

但问题来了:模型里有**隐变量** $z$(在 diffusion 里 $z = x_{1:T}$,在 VAE 里 $z$ 就是潜变量)。我们写出的是**联合分布** $p_\theta(x, z)$,而要的是**边际**:

$$p_\theta(x) = \int p_\theta(x, z)\, dz$$

这个积分通常**算不出来**。比如在 diffusion 里:

$$p_\theta(x_0) = \int p_\theta(x_0, x_{1:T})\, dx_{1:T} = \int p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)\, dx_{1:T}$$

这是一个 $T$ 重高维积分($T$ 可能是 1000),没有解析解,蒙特卡洛估计方差也极大。

**ELBO 的作用**:既然 $\log p_\theta(x)$ 算不出来,那就找一个能算的**下界**,最大化下界。

---

### 二、推导一:Jensen 不等式视角(最直接)

**Jensen 不等式**:对凹函数 $f$ 和随机变量 $X$,

$$f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$$

$\log$ 是凹函数,所以 $\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$。

引入任意一个分布 $q(z \mid x)$(称为**变分分布**),做"乘 1 除 1"的技巧:

$$\log p_\theta(x) = \log \int p_\theta(x, z)\, dz = \log \int q(z \mid x) \cdot \frac{p_\theta(x, z)}{q(z \mid x)}\, dz$$

注意中间那部分就是期望:

$$= \log \mathbb{E}_{z \sim q(z \mid x)}\left[\frac{p_\theta(x, z)}{q(z \mid x)}\right]$$

用 Jensen 把 $\log$ 移进期望:

$$\geq \mathbb{E}_{z \sim q(z \mid x)}\left[\log \frac{p_\theta(x, z)}{q(z \mid x)}\right] =: \mathcal{L}(\theta, q; x)$$

这个 $\mathcal{L}$ 就是 **ELBO**。它的精妙之处:

- 我们有了一个**期望形式**——可以用蒙特卡洛采样估计
- $q(z \mid x)$ 是我们自己选的——可以选一个采样和计算都方便的形式
- 它是 $\log p_\theta(x)$ 的下界——最大化它**至少不会让似然变差**

---

### 三、推导二:KL 散度视角(最深刻)

这个推导更优美,能直接告诉你**下界有多紧**。

从 KL 散度的定义出发:

$$D_{\text{KL}}(q(z \mid x) \,\|\, p_\theta(z \mid x)) = \mathbb{E}_q\left[\log \frac{q(z \mid x)}{p_\theta(z \mid x)}\right]$$

用贝叶斯公式 $p_\theta(z \mid x) = \frac{p_\theta(x, z)}{p_\theta(x)}$:

$$= \mathbb{E}_q\left[\log q(z \mid x) - \log p_\theta(x, z) + \log p_\theta(x)\right]$$

注意 $\log p_\theta(x)$ 不依赖 $z$,可以从期望里提出来:

$$= \mathbb{E}_q[\log q(z \mid x)] - \mathbb{E}_q[\log p_\theta(x, z)] + \log p_\theta(x)$$

整理一下,把 $\log p_\theta(x)$ 单独放一边:

$$\log p_\theta(x) = \underbrace{\mathbb{E}_q\left[\log \frac{p_\theta(x, z)}{q(z \mid x)}\right]}_{\mathcal{L}(\theta, q; x) \text{ = ELBO}} + \underbrace{D_{\text{KL}}(q(z \mid x) \,\|\, p_\theta(z \mid x))}_{\geq 0}$$

**这是一个等式!** 不是不等式!

它告诉我们三件极其重要的事:

1. $\log p_\theta(x) = \text{ELBO} + \text{KL}$,因为 KL ≥ 0,所以 $\log p_\theta(x) \geq \text{ELBO}$ —— 重新得到了下界
2. **下界的"间隙"恰好是** $D_{\text{KL}}(q(z \mid x) \,\|\, p_\theta(z \mid x))$。也就是说,$q$ 越接近真实后验 $p_\theta(z \mid x)$,下界越紧
3. 当 $q(z \mid x) = p_\theta(z \mid x)$ 时,KL = 0,**ELBO 等于真实对数似然**

这告诉我们 ELBO 优化的**本质是什么**:同时在做两件事——拉高 $\log p_\theta(x)$,以及让 $q$ 逼近真实后验。

---

### 四、推导三:能量分解视角(最实用)

把 ELBO 重新整理一下:

$$\mathcal{L} = \mathbb{E}_q[\log p_\theta(x, z)] - \mathbb{E}_q[\log q(z \mid x)]$$

用 $p_\theta(x, z) = p_\theta(x \mid z) p(z)$:

$$\mathcal{L} = \mathbb{E}_q[\log p_\theta(x \mid z)] + \mathbb{E}_q[\log p(z)] - \mathbb{E}_q[\log q(z \mid x)]$$

后两项合并成 KL:

$$\boxed{\mathcal{L} = \underbrace{\mathbb{E}_{q(z \mid x)}[\log p_\theta(x \mid z)]}_{\text{重建项}} - \underbrace{D_{\text{KL}}(q(z \mid x) \,\|\, p(z))}_{\text{先验匹配项}}}$$

这就是 **VAE 论文里的经典 ELBO 形式**。两项的含义:

- **重建项**:从 $q$ 采样 $z$,用 $z$ 重建 $x$ 的对数似然要高 —— 即"信息要保留"
- **先验匹配项**:$q(z \mid x)$ 不能离先验 $p(z)$ 太远 —— 即"潜空间要规整"

最大化 ELBO = 重建好 + 潜空间规整。

---

### 五、应用到 diffusion 上

在 diffusion 里:

- $x = x_0$(数据)
- $z = x_{1:T}$(所有加噪后的隐变量)
- $q(z \mid x) = q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})$ —— **固定的**前向过程,不需要学
- $p_\theta(x, z) = p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)$ —— **要学的**逆向过程

直接套推导一的结果:

$$\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T} \mid x_0)}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right]$$

把分子分母展开:

$$= \mathbb{E}_q\left[\log \frac{p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)}{\prod_{t=1}^T q(x_t \mid x_{t-1})}\right]$$

$$= \mathbb{E}_q\left[\log p(x_T) + \sum_{t=1}^T \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1})}\right]$$

这就是 Sohl-Dickstein 2015 用的 ELBO 形式。

但这个形式**训练效果不好**——分子是逆向 $p_\theta(x_{t-1} \mid x_t)$,分母是前向 $q(x_t \mid x_{t-1})$,方向不一致,梯度方差大。

DDPM 的关键贡献之一就是用**贝叶斯公式重写这个 ELBO**,让每一项都变成两个高斯之间的 KL,从而:

- 有闭式解,不需要采样估计
- 方差大幅降低
- 最终化简为漂亮的 $\| \epsilon - \epsilon_\theta \|^2$ 形式

这部分我们留到 DDPM 那一讲(第三讲)详细推。

---

### 六、几个常见困惑点

**Q1:为什么 $q$ 可以"任意选"?选错了会怎样?**

ELBO 对**任何** $q$ 都成立(Jensen 不等式不要求 $q$ 是什么具体形式)。但选得不好,下界很松,优化下界相当于在优化一个跟 $\log p_\theta(x)$ 关系很弱的东西。

在 VAE 里,$q_\phi(z \mid x)$ 是用神经网络参数化、和 $\theta$ 一起学的。在 diffusion 里更聪明:$q$ 是**预先固定的扩散过程**——简单到不需要学,但又因为 $T$ 大、每步小,使得 $q$ 自动接近真实后验。这是 diffusion 比 VAE 强的一个本质原因。

**Q2:为什么不直接最大化 $\log p_\theta(x)$?**

算不出来。即使能用蒙特卡洛 $p_\theta(x) \approx \frac{1}{N}\sum_i \frac{p_\theta(x, z_i)}{q(z_i \mid x)}$,这是有偏估计(因为 $\log$ 套外面),而且方差极大,实际不可行。

**Q3:Jensen 不等式损失了多少?**

正好是 $D_{\text{KL}}(q \| p_\theta(z \mid x))$。这就是为什么推导二比推导一更深刻——它精确量化了"损失"。



要点回顾:

- ELBO 来源于"$\log$ 移进期望"(Jensen)或等价地"似然 = ELBO + KL"
- 最大化 ELBO 同时做两件事:逼近真实似然 + 让 $q$ 逼近真实后验
- 在 diffusion 里 $q$ 是固定的前向过程,这是和 VAE 的关键区别
- 直接套出来的 ELBO 形式可训练但方差大,DDPM 的关键改进是用贝叶斯重写成 KL 之和


## 6. 蒙特卡洛估计



### 一、问题的起点:期望算不出来

我们经常遇到这种形式:

$$\mathbb{E}_{x \sim p(x)}[f(x)] = \int f(x) p(x)\, dx$$

比如:

- ELBO 里的 $\mathbb{E}_{q(z \mid x)}[\log p_\theta(x, z) / q(z \mid x)]$
- 训练损失 $\mathbb{E}_{x \sim p_{\text{data}}}[\mathcal{L}(x; \theta)]$
- 物理里的配分函数、金融里的期权定价……

这些积分大多数情况下**没有解析解**。维度一高(比如 $x$ 是 1000 维),数值积分(网格法)也完全不可行——计算量随维度指数爆炸,这叫"维度诅咒"。

---

### 二、蒙特卡洛的核心思想

**用样本平均近似期望**。

如果我们能从 $p(x)$ 里采样出 $x_1, x_2, \ldots, x_N$(独立同分布),那么:

$$\hat{f}_N = \frac{1}{N} \sum_{i=1}^N f(x_i) \approx \mathbb{E}_{x \sim p}[f(x)]$$

就这么简单。把"积分"变成"采样后求平均"。

**为什么这能行?** 两个数学保证:

**(1) 大数定律**:当 $N \to \infty$,

$$\frac{1}{N} \sum_{i=1}^N f(x_i) \xrightarrow{\text{a.s.}} \mathbb{E}[f(x)]$$

样本越多,估计越准,最终一定收敛到真值。

**(2) 中心极限定理**:估计的误差服从

$$\hat{f}_N - \mathbb{E}[f(x)] \approx \mathcal{N}\left(0, \frac{\text{Var}(f(x))}{N}\right)$$

也就是误差大约是 $O(1/\sqrt{N})$。这意味着想把误差减半,要 4 倍样本;减到 1/10,要 100 倍样本。慢,但**和维度无关** —— 这是蒙特卡洛最神奇的地方,完全打破了维度诅咒。

---

### 三、一个直观例子:估计 $\pi$

经典例子。在 $[0,1]^2$ 单位正方形里随机撒点,计算落在四分之一圆 $x^2 + y^2 \leq 1$ 里的比例,乘以 4 就近似 $\pi$。

为什么?因为

$$\frac{\pi}{4} = \int_0^1 \int_0^1 \mathbb{1}[x^2 + y^2 \leq 1]\, dx\, dy = \mathbb{E}_{(x,y) \sim \text{Uniform}([0,1]^2)}[\mathbb{1}[x^2 + y^2 \leq 1]]$$

撒 $N$ 个点,落在圆里的有 $k$ 个,$\pi \approx 4k/N$。

```
N=100:    π ≈ 3.16   (误差 ~0.02)
N=10000:  π ≈ 3.142  (误差 ~0.001)
```

完美体现 $O(1/\sqrt{N})$ 的收敛速度。

---

### 四、应用到 ELBO

回到我们的场景。ELBO:

$$\mathcal{L} = \mathbb{E}_{z \sim q(z \mid x)}\left[\log \frac{p_\theta(x, z)}{q(z \mid x)}\right]$$

这个期望算不出来(高维积分)。**蒙特卡洛估计**:

$$\hat{\mathcal{L}} = \frac{1}{N} \sum_{i=1}^N \log \frac{p_\theta(x, z_i)}{q(z_i \mid x)}, \quad z_i \sim q(z \mid x)$$

实践中,**通常 $N = 1$ 就够了**——因为我们做随机梯度下降,每个 batch 里有很多 $x$,跨 batch 的随机性已经平均掉了估计噪声。

**在 diffusion 里更具体**:训练时,对每个数据点 $x_0$:

1. 从 $\{1, \ldots, T\}$ 均匀采一个 $t$
2. 从 $q(x_t \mid x_0)$ 采一个 $x_t$
3. 计算这一项的损失

这就是用单样本蒙特卡洛估计完整的 ELBO。每次只看一个 $t$,看似很粗,但跨 batch、跨 epoch 平均下来,梯度的期望是对的——这就是**随机梯度下降**能工作的根本原因。

---

### 五、估计的"质量":偏差 vs 方差

蒙特卡洛估计有两个关键性质:

**(1) 无偏性(unbiased)**:

$$\mathbb{E}\left[\frac{1}{N}\sum_i f(x_i)\right] = \mathbb{E}[f(x)]$$

估计的期望等于真值。这是好性质——只要采足够多次,平均会对。

**(2) 方差**:

$$\text{Var}(\hat{f}_N) = \frac{\text{Var}(f(x))}{N}$$

$N$ 越大方差越小。但如果 $\text{Var}(f(x))$ 本身很大,即使 $N$ 大,单次估计也很不稳。

**这就是 Sohl-Dickstein 原始 ELBO 训练困难的根源**:

$$\mathcal{L} = \mathbb{E}_q\left[\log p(x_T) + \sum_{t=1}^T \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1})}\right]$$

被估计量 $\log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1})}$ **方差极大**——前向、逆向方向不一致,采样得到的值跳得很厉害。$N=1$ 的蒙特卡洛估计噪声太大,梯度信号被淹没。

DDPM 的关键改进就是**用贝叶斯把 ELBO 重写成 KL 之和,而高斯之间的 KL 有闭式解,根本不需要蒙特卡洛**——直接消掉了一大部分方差源。剩下需要采样的部分(选 $t$、采 $x_t$)方差也小得多。

---

### 六、什么时候蒙特卡洛会失效?

理论上 $O(1/\sqrt{N})$ 不依赖维度,看起来无敌。但实际有几个陷阱:

**(1) 方差爆炸**:如果 $f(x)$ 在 $p(x)$ 下方差极大(比如 $f$ 在 $p$ 的尾部取极大值),需要的 $N$ 可能大到不实用。这激发了**重要性采样**(importance sampling):换个分布采。

**(2) 采样本身困难**:能做蒙特卡洛的前提是能从 $p(x)$ 采样。但很多时候我们只知道 $p(x)$ 的非归一化形式 $\tilde{p}(x) \propto p(x)$,没法直接采。这激发了 **MCMC**(马尔可夫链蒙特卡洛)、Langevin dynamics 等方法——后面讲 score-based model 时会用到这些。

**(3) 有偏估计的陷阱**:有些量看起来能蒙特卡洛,但其实**有偏**。最典型的是 $\log \mathbb{E}[X]$:

$$\log \mathbb{E}[X] \neq \mathbb{E}[\log X]$$

(这正是 Jensen 不等式!)所以你不能用 $\log \frac{1}{N}\sum_i x_i$ 估计 $\log \mathbb{E}[X]$ 然后说它无偏——它**不是无偏的**(虽然渐近正确)。

回想 ELBO 的推导:

$$\log p_\theta(x) = \log \mathbb{E}_q\left[\frac{p_\theta(x, z)}{q(z \mid x)}\right]$$

理论上可以蒙特卡洛估计 $\frac{1}{N}\sum_i \frac{p_\theta(x, z_i)}{q(z_i \mid x)}$,然后取 $\log$。但这是有偏的(且方差极大),所以我们绕开它,改用 ELBO——ELBO 是无偏估计的(对 $\mathbb{E}_q[\log \cdots]$ 而言),代价是它只是下界不是真值。

---

### 七、要点回顾

- 蒙特卡洛 = 用样本平均近似期望
- 大数定律保证收敛,中心极限定理给出 $O(1/\sqrt{N})$ 的误差,**不依赖维度**
- 实际训练里 $N = 1$ 就够,因为 SGD 的批量平均已经在做"更多采样"
- 估计的**方差**直接决定训练稳定性—— DDPM 比 Sohl-Dickstein 强很大程度上是因为减小了这个方差
- 当不能直接采样时,需要 MCMC、Langevin 等更高级的方法(下一讲会用到)

---

清楚了吗?这个概念之所以重要,是因为后面讲 score matching 的时候,我们会反复在"理论积分形式"和"实际可计算的蒙特卡洛形式"之间转换。比如 score matching 原始形式有个 trace 项算不动,denoising score matching 通过等价变换变成了可以蒙特卡洛估计的形式——这是它能 work 的关键。

要不要我们继续进入**第二讲:Score Matching**?


## 7. 闭式解
好问题——这是我们一直在用但没正式定义过的术语。讲清楚后,你会更深刻地理解为什么 diffusion 选高斯、为什么改写 ELBO 这么重要。

---

### 一、什么是闭式解?

**闭式解(closed-form solution)**:用有限多个**标准运算和函数**(加减乘除、根号、指数、对数、三角函数等)**直接写出来的精确表达式**。

对比的是:

- **数值解**:得通过迭代算法逼近(比如牛顿法)
- **采样估计**:得通过随机采样平均(比如蒙特卡洛)

**举例对比**:

| 问题 | 闭式解? | 形式 |
|------|--------|------|
| $\int_{-\infty}^\infty x e^{-x^2/2}\, dx$ | ✓ 有 | $= 0$(直接算) |
| $\int_{-\infty}^\infty e^{-x^2/2}\, dx$ | ✓ 有 | $= \sqrt{2\pi}$(高斯归一化) |
| $\int_0^1 e^{-x^2}\, dx$ | ✗ 无 | 只能数值积分(误差函数 erf) |
| 一元二次方程 $ax^2+bx+c=0$ | ✓ 有 | $x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}$ |
| 五次方程通解 | ✗ 无 | 阿贝尔-鲁菲尼定理:不存在根式解 |

**直观理解**:闭式解 = "可以写在纸上的公式"。

---

### 二、为什么"闭式"这件事很重要?

机器学习里,"闭式" vs "需要近似"决定了**两件实际的事**:

#### 1. 计算效率

闭式:一行代码,一次浮点运算,$O(1)$ 算完。
非闭式:要么数值迭代(可能不收敛),要么蒙特卡洛(要采样很多次)。

#### 2. 梯度精确

闭式公式可以直接对参数求导(自动微分一路畅通)。
蒙特卡洛估计的梯度本身就有噪声,$N=1$ 时方差大,训练不稳。

**所以一个量是不是闭式,直接决定它能不能高效、稳定地参与训练。**

---

### 三、KL 散度和闭式的关系

回忆 KL 的定义:

$$D_{\text{KL}}(p \| q) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right] = \int p(x) \log \frac{p(x)}{q(x)}\, dx$$

这是个积分。**对一般的 $p, q$,这个积分没有闭式**——只能蒙特卡洛估计。

但**在某些特殊的分布族之间,这个积分有闭式**。最重要的几个:

#### 3.1 高斯之间的 KL(diffusion 的核心)

两个一维高斯 $\mathcal{N}(\mu_1, \sigma_1^2)$ 和 $\mathcal{N}(\mu_2, \sigma_2^2)$:

$$\boxed{D_{\text{KL}}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}}$$

多维各向同性 $\mathcal{N}(\mu_1, \sigma^2 I)$ 和 $\mathcal{N}(\mu_2, \sigma^2 I)$($d$ 维):

$$D_{\text{KL}} = \frac{\| \mu_1 - \mu_2 \|^2}{2\sigma^2}$$

**注意!** 第二个公式简洁得吓人——当两个分布方差相同时,KL 就是均值之差的平方除以方差。

#### 3.2 其他有闭式 KL 的分布族

- 两个伯努利分布
- 两个分类分布(categorical)
- 两个指数分布
- 两个 Dirichlet 分布
- 一般地,任何**两个属于同一指数族的分布**

但**两个不同分布族之间的 KL 通常没有闭式**(比如高斯和 Cauchy,或者高斯和混合高斯)。

---

### 四、回到 diffusion:这怎么决定了一切?

我们之前讲到 DDPM 改写 ELBO,把每一项变成

$$D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))$$

让我们看为什么这一步如此关键:

#### 步骤 1:确认两边都是高斯

- $q(x_{t-1} \mid x_t, x_0)$:可以证明这是高斯(用贝叶斯公式 + 高斯运算)。具体均值方差有闭式,我们下一讲推
- $p_\theta(x_{t-1} \mid x_t)$:我们设计模型时就把它**参数化成高斯** $\mathcal{N}(\mu_\theta, \Sigma_\theta)$

#### 步骤 2:套用高斯 KL 闭式公式

直接代公式:

$$D_{\text{KL}}(q \| p_\theta) = \frac{\| \tilde\mu(x_t, x_0) - \mu_\theta(x_t, t) \|^2}{2\sigma_t^2} + \text{const}$$

(假设方差固定为 $\sigma_t^2$,DDPM 的简化设定)

**这是闭式!** 没有积分、没有蒙特卡洛、可以直接对 $\theta$ 求导。

#### 步骤 3:得到训练目标

最小化这个 KL 等价于最小化

$$\| \tilde\mu(x_t, x_0) - \mu_\theta(x_t, t) \|^2$$

——一个干净的均方误差!**训练 diffusion 本质上变成了一个回归问题**。

后来 DDPM 进一步用重参数化把 $\tilde\mu, \mu_\theta$ 都换成 $\epsilon, \epsilon_\theta$,得到最终的:

$$L_{\text{simple}} = \| \epsilon - \epsilon_\theta(x_t, t) \|^2$$

——所有这些干净美丽的形式,**根源都是高斯 KL 有闭式**。

---

### 五、和 Sohl-Dickstein 的对比:为什么原始形式不闭式?

回顾 Sohl-Dickstein 的损失里每一项:

$$\log \frac{q(x_t \mid x_{t-1})}{p_\theta(x_{t-1} \mid x_t)}$$

它**不是**一个 KL——只是一个对数比值。所以无法套用 KL 闭式公式。要算它,必须:

1. 采样具体的 $x_{t-1}, x_t$
2. 把这俩值代入两个高斯密度公式
3. 得到一个具体的对数比值数

每次得到的数是**随机的**(依赖采样),方差大。

**对比一下两种形式**:

| 形式 | 是不是 KL | 闭式? | 方差 |
|------|----------|------|------|
| Sohl-Dickstein:$\log \frac{q(x_t\|x_{t-1})}{p_\theta(x_{t-1}\|x_t)}$ | 不是 | 不闭式(只能采样) | 大 |
| DDPM:$D_{\text{KL}}(q(x_{t-1}\|x_t,x_0) \| p_\theta(x_{t-1}\|x_t))$ | 是 | 闭式(高斯 KL 公式) | 小 |

**这就是改写带来的本质差异**——不是数学上更"正确"(两者都是 ELBO 的一部分),而是**计算上更"干净"**(从需要随机采样估计变成了直接闭式计算)。

---

### 六、闭式 vs 蒙特卡洛 vs 数值

最后给你一张全景图,看三种"算积分"方式的对比:

| 方式 | 何时可用 | 误差 | 速度 |
|------|---------|------|------|
| **闭式** | 极少数特殊情况(高斯 KL 等) | 0(精确) | 极快 |
| **蒙特卡洛** | 任何能采样的情况 | $O(1/\sqrt{N})$ | 中等(需多次采样) |
| **数值积分** | 低维($\leq 3$ 维)且光滑 | 取决于网格 | 维度高时不可用 |

机器学习里我们的优先级:**能闭式就闭式,不能闭式才退而求其次用蒙特卡洛**。这就是 diffusion 一直在追求的事。

---

### 七、要点回顾

- **闭式解**:用标准运算和函数直接写出的精确表达式
- 闭式 → 计算快、梯度精确、训练稳定
- 一般 KL 没有闭式,**但同分布族之间的 KL 有闭式**(高斯之间是最重要的)
- DDPM 改写 ELBO 让每一项变成两个高斯的 KL → 套闭式公式 → 训练目标变成漂亮的 MSE
- 这是为什么 diffusion 选高斯、为什么改写 ELBO 如此关键的根本数学原因
