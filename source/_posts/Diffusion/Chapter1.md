---
title: Chapter 0: 前置知识
categories: Diffusion Models
date: 2026-05-01 12:00:00
mathjax: true
tags:
    - AI
    - Diffusion Models
---

好,那我们从头系统讲起。这一讲先把**最早的源头**讲透,把数学基础打牢,后面才能层层递进。

我会按这个路径展开:

1. 这一讲:**Sohl-Dickstein 2015** —— diffusion 的真正起点,以及它依赖的物理直觉和数学工具
2. 下一讲:Score Matching(Hyvärinen 2005)和 NCSN(2019)
3. 再下一讲:DDPM(2020),把 ELBO 推导彻底讲完
4. 之后:Score SDE 统一框架
5. 最后:DDIM、CFG、Flow Matching 等现代发展

每讲我会假设你跟得上前面的推导,如果哪里卡住随时打断我。

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

## 2. 必要的数学预备

在讲具体模型前,需要这几个工具:

**(1) 高斯分布的一些性质**

一维高斯 $\mathcal{N}(x; \mu, \sigma^2)$ 的密度:

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

多维高斯 $\mathcal{N}(x; \mu, \Sigma)$ 的密度:

$$p(x) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)\right)$$

**关键性质 —— 高斯的可加性**:如果 $x \sim \mathcal{N}(\mu_1, \sigma_1^2)$ 且 $y \mid x \sim \mathcal{N}(ax, \sigma_2^2)$,那么边际 $y \sim \mathcal{N}(a\mu_1, a^2\sigma_1^2 + \sigma_2^2)$。这个性质后面 DDPM 会大量用。

**(2) KL 散度**

衡量两个分布的"差异":

$$D_{\text{KL}}(q \| p) = \mathbb{E}_{x \sim q}\left[\log \frac{q(x)}{p(x)}\right]$$

性质:非负、不对称、$D_{\text{KL}}(q\|p) = 0 \iff q = p$。

**两个高斯之间的 KL 有闭式解**(这个非常重要,后面会反复用):

$$D_{\text{KL}}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

**(3) 马尔可夫链**

序列 $x_0, x_1, \ldots, x_T$ 满足马尔可夫性:

$$p(x_t \mid x_{t-1}, x_{t-2}, \ldots, x_0) = p(x_t \mid x_{t-1})$$

即"未来只依赖于现在,与过去无关"。联合分布可以分解:

$$p(x_{0:T}) = p(x_0) \prod_{t=1}^T p(x_t \mid x_{t-1})$$

**(4) 变分下界(ELBO)**

这是从 VAE 继承来的核心工具。我们想最大化数据似然 $\log p_\theta(x_0)$,但它通常算不出来(因为要对所有隐变量积分):

$$p_\theta(x_0) = \int p_\theta(x_0, x_{1:T})\, dx_{1:T}$$

引入一个辅助分布 $q(x_{1:T} \mid x_0)$,用 Jensen 不等式:

$$\log p_\theta(x_0) = \log \int p_\theta(x_{0:T})\, dx_{1:T} = \log \int q(x_{1:T} \mid x_0) \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\, dx_{1:T}$$

$$\geq \mathbb{E}_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right] =: \mathcal{L}(\theta)$$

这个下界 $\mathcal{L}$ 就是 **ELBO**(Evidence Lower BOund)。最大化 ELBO 就是在(近似)最大化对数似然。

---

## 附录1. ELBO的证明

好,ELBO 是整个 diffusion 数学骨架的基石,我们彻底讲透。我会从最根本的动机讲起,推导给你三种角度——它们结果一样,但视角不同,能帮你深刻理解。


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



