---
title: Chapter 1. 贝叶斯 & 变分推断
categories: Diffusion Models
date: 2026-05-01 12:30:00
mathjax: true
tags:
    - AI
    - Diffusion Models
---

数学知识打底之后,我们要把"为什么 diffusion 长这个样子"的根本数学骨架讲清楚:**贝叶斯推断 → 隐变量框架 → 变分推断 → ELBO**。后续 DDPM、Score SDE、VAE 全部都站在这条逻辑链上。

## 1. 一切的源头:贝叶斯推断算不动

### 1.1 贝叶斯推断的基本设定

我们有:

- **先验** $p(z)$:我们对隐变量 $z$ 的初始信念
- **似然** $p(x \mid z)$:给定 $z$,数据 $x$ 怎么生成
- **证据**(数据) $x$:观测到的东西

我们想要的是 **后验** $p(z \mid x)$:看到数据后,$z$ 的更新信念。

由贝叶斯公式:

$$p(z \mid x) = \frac{p(x \mid z)\, p(z)}{p(x)}$$

**这看起来很简单**,但分母

$$p(x) = \int p(x \mid z)\, p(z)\, dz$$

——**这个积分通常算不出来**。

### 1.2 为什么算不出来?几个例子

具体看几个例子,你会感受到这个问题的普遍性:

**例 1:贝叶斯线性回归(简单情形)** $z = w$(权重),先验 $p(w) = \mathcal{N}(0, \sigma^2 I)$,似然 $p(x \mid w) = \mathcal{N}(Xw, \tau^2 I)$。这里**有闭式**——后验也是高斯。但这是少数运气好的情形。

**例 2:贝叶斯神经网络** $z$ 是网络的所有权重(几百万维)。$\int p(x \mid z) p(z)\, dz$ 是几百万维的积分,**绝无闭式**。

**例 3:主题模型(LDA)** 后验依赖于复杂的离散+连续混合,**没有闭式**。

**例 4:生成模型(VAE, diffusion)** $z$ 是隐变量(VAE 的潜码,diffusion 的 $x_{1:T}$)。给定 $z$ 我们能生成 $x$,但反过来"给定 $x$,$z$ 是什么"——后验复杂,**算不出来**。

### 1.3 问题的本质

**贝叶斯推断的核心障碍**:**后验 $p(z \mid x)$ 几乎总是算不出来**,因为:

- 分子 $p(x \mid z) p(z)$ **可以算**(模型定义里给出)
- 分母 $p(x) = \int p(x \mid z) p(z)\, dz$ **算不出来**

**比值"算不出来"——但分子可以算**。整个变分推断的故事,就是怎么利用这个"分子可算"的事实,绕开分母。

{% details 传统方案为什么失败:解析解 / MCMC / 拉普拉斯 %}

在变分推断之前,人们怎么处理这个问题?

**解析解**:如果先验和似然是**共轭对**,后验有闭式。比如高斯-高斯、Beta-二项、Dirichlet-多项。模型一复杂(神经网络、深度模型),共轭性消失,这条路堵死。

**蒙特卡洛(MCMC)**:不算后验的解析形式,而是**从后验采样**。Metropolis-Hastings、Gibbs sampling、HMC 等。

- 优点:理论上渐近精确——样本足够多就接近真实后验
- 缺点:**慢**(高维下要采几百万步)、**不可微**(没法和深度学习配合)、**难诊断**(不知道链有没有真的混合好)

MCMC 在统计学里是主流,但**在大规模深度学习里几乎不可用**——太慢,而且不能放进 SGD。

**拉普拉斯近似**:用一个高斯近似后验,中心是后验的众数,方差是 Hessian 的逆。只在后验"接近高斯"时准——多模态、复杂分布全废。

{% enddetails %}

---

## 2. 为什么生成模型必然走向"隐变量 + 贝叶斯"?

回到生成模型最根本的目标:**学一个分布 $p(x)$,使得能从中采样,生成新数据**。听起来简单,但仔细想:这个 $p(x)$ 怎么"参数化"?

### 2.1 路线一:直接参数化 $p_\theta(x)$

最朴素的想法:让 $p_\theta(x)$ 是个直接由 $\theta$ 决定的密度函数。

- **混合高斯**:$p_\theta(x) = \sum_k \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)$
- **指数族**:$p_\theta(x) = h(x) \exp(\theta^\top T(x) - A(\theta))$
- **能量模型**:$p_\theta(x) = \exp(-E_\theta(x)) / Z(\theta)$

**问题**:简单分布(混合高斯)**表达力弱**——拟合不了复杂的图像分布;复杂分布(神经网络作为能量函数)**归一化常数 $Z$ 算不出来**。直接参数化的路在简单时太弱、在复杂时太难。

### 2.2 路线二:把复杂分布"拆"成简单部件

复杂分布很难直接表达,但一个普遍的数学策略是:**把复杂的东西分解成简单部件的组合**。

**链式分解(自回归)**:$p(x_1, \ldots, x_d) = p(x_1)\, p(x_2 \mid x_1) \cdots$。每个条件分布简单,但连乘表达力强。这是 PixelRNN、PixelCNN、GPT 的路。优点:精确似然;缺点:必须串行。

**可逆变换(Normalizing Flow)**:$x = f_\theta(z)$,$z \sim p(z)$。要求 $f$ 可逆,精确算 $\log p(x)$ 通过雅可比公式。优点:精确似然;缺点:可逆要求严重限制网络架构。

**隐变量分解** ★:

$$p(x) = \int p(x, z)\, dz = \int p(x \mid z)\, p(z)\, dz$$

引入一个**未观测的隐变量 $z$**,通过它把 $p(x)$ 变成"对 $z$ 的边际化"。**这就是隐变量模型——VAE、diffusion、GMM 都是这条路**。

### 2.3 隐变量框架为什么这么自然?

**表达力理由**:任何分布都可以写成"简单分布 + 简单变换"的隐变量形式。如果 $p(z)$ 是标准高斯,$p(x \mid z)$ 是高斯(均值是 $z$ 的函数),那么 $p(x) = \int p(x \mid z) p(z) dz$ 可以是**任意复杂的分布**。

**物理/认知理由**:**真实数据的复杂性**通常**确实**来自隐变量。图像不是任意像素组合,而是来自"真实 3D 物体的投影"——物体有形状、纹理、姿态、光照,这些是**隐藏的因子**。

**推断理由**:学到隐变量 $z$ 的好处不只是生成数据,还在于 $z$ 本身**就是数据的有用表示**——可以做插值、操控属性。**生成 + 表示一举两得**。

**历史理由**:隐变量框架在统计学里**有 100 多年历史**——Spearman 1904 因子分析、HMM 1950s、EM 1977、图模型 1980s-90s、LDA 2000s、VAE 2013。

### 2.4 引入隐变量后,贝叶斯推断必然出现

定义了 $p(x, z)$ 之后,有两件事要做:

**学习**:估计 $\theta$,使 $p_\theta(x) = \int p_\theta(x, z) dz$ 最大化数据似然。
**推断**:给定观测 $x$,**$z$ 是什么?**——也就是 $p(z \mid x)$。

这两件事**不能分开**——这是关键。EM 算法就是教科书式的例子:E 步做推断,M 步用推断结果更新 $\theta$,交替进行。

为什么必须耦合?因为最大似然里 $\log p_\theta(x) = \log \int p_\theta(x, z) dz$,这个积分 = "对隐变量边际化"——而**边际化和推断是同一件事的两面**:

- 边际化 = 平均所有可能的 $z$ 来算 $p(x)$
- 推断 = 给定 $x$,算每个 $z$ 的概率

所以**只要用了隐变量,就避不开贝叶斯推断**。

---

## 3. 变分推断:把积分问题变成优化问题

### 3.1 关键转变

变分推断的核心想法,用一句话:

> **不要试图精确算后验 $p(z \mid x)$,而是从一族"容易处理的分布"里找一个,让它最接近后验。**

数学上:

- 选一个分布族 $\mathcal{Q}$(比如所有对角高斯)
- 在 $\mathcal{Q}$ 中找 $q^*(z)$ 使得 $q^{*}(z)$ 最接近 $p(z \mid x)$

"最接近"用 KL 散度衡量:

$$q^*(z) = \arg\min_{q \in \mathcal{Q}}\, D_{\text{KL}}(q(z) \| p(z \mid x))$$

**这是个优化问题,不是积分问题**——这就是变分推断的精髓。

> "变分(variational)"来自变分法(calculus of variations)——研究"在函数空间里优化"的数学分支。我们在函数 $q$ 上优化,而不是在数(参数)上优化,所以叫"变分"。

### 3.2 但 $D_{\text{KL}}(q \| p(z \mid x))$ 也算不出来啊!

这是关键的"卡点"。展开 KL:

$$D_{\text{KL}}(q(z) \| p(z \mid x)) = \mathbb{E}_q[\log q(z)] - \mathbb{E}_q[\log p(z \mid x)]$$

第二项需要 $\log p(z \mid x)$,而 $p(z \mid x) = p(x, z)/p(x)$,**还是要算 $p(x)$**——绕了一圈又回到原点!

### 3.3 关键代数变换

但这里有个救命的代数变换。继续展开第二项:

$$\log p(z \mid x) = \log p(x, z) - \log p(x)$$

代入:

$$D_{\text{KL}}(q \| p(z \mid x)) = \mathbb{E}_q[\log q(z)] - \mathbb{E}_q[\log p(x, z)] + \log p(x)$$

(注意 $\log p(x)$ 不依赖 $z$,可以从期望里提出来)

整理:

$$\boxed{\log p(x) = \underbrace{\mathbb{E}_q[\log p(x, z)] - \mathbb{E}_q[\log q(z)]}_{=: \text{ELBO}(q)} + D_{\text{KL}}(q \| p(z \mid x))}$$

这就是著名的 **ELBO 等式**。

### 3.4 关键魔法

注意:

- $\log p(x)$ 是**固定的**(不依赖 $q$)
- $D_{\text{KL}} \geq 0$,且依赖 $q$
- ELBO 也依赖 $q$

由等式:**最小化 KL 等价于最大化 ELBO**(因为它们之和是常数)!

而 ELBO 里**只用 $\log p(x, z)$ 和 $\log q(z)$**——**两者都能算**!

- $\log p(x, z) = \log p(x \mid z) + \log p(z)$ 是模型定义,直接给出
- $\log q(z)$ 是我们自选的近似,知道它的形式

**变分推断的"魔法时刻"**:

> **我们想优化"$q$ 接近 $p(z \mid x)$"——但 $p(z \mid x)$ 不知道。**
>
> **代数变换发现:这等价于优化 ELBO——而 ELBO 只用已知量。**

---

## 4. ELBO 三种推导视角

ELBO 至少有三种推导,看似不同,实际等价。每种视角揭示一个本质。

### 4.1 推导一:Jensen 不等式视角(最直接)

**Jensen 不等式**:对凹函数 $f$ 和随机变量 $X$,$f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$。$\log$ 是凹函数,所以 $\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$。

引入任意一个分布 $q(z \mid x)$(称为**变分分布**),做"乘 1 除 1"的技巧:

$$\log p_\theta(x) = \log \int p_\theta(x, z)\, dz = \log \int q(z \mid x) \cdot \frac{p_\theta(x, z)}{q(z \mid x)}\, dz$$

注意中间那部分就是期望:

$$= \log \mathbb{E}_{z \sim q(z \mid x)}\left[\frac{p_\theta(x, z)}{q(z \mid x)}\right]$$

用 Jensen 把 $\log$ 移进期望:

$$\geq \mathbb{E}_{z \sim q(z \mid x)}\left[\log \frac{p_\theta(x, z)}{q(z \mid x)}\right] =: \mathcal{L}(\theta, q; x)$$

它的精妙之处:

- 我们有了一个**期望形式**——可以用蒙特卡洛采样估计
- $q(z \mid x)$ 是我们自己选的——可以选一个采样和计算都方便的形式
- 它是 $\log p_\theta(x)$ 的下界——最大化它**至少不会让似然变差**

### 4.2 推导二:KL 散度视角(最深刻)

这个推导更优美,能直接告诉你**下界有多紧**。

从 KL 散度的定义出发:

$$D_{\text{KL}}(q(z \mid x) \,\|\, p_\theta(z \mid x)) = \mathbb{E}_q\left[\log \frac{q(z \mid x)}{p_\theta(z \mid x)}\right]$$

用贝叶斯公式 $p_\theta(z \mid x) = \frac{p_\theta(x, z)}{p_\theta(x)}$:

$$= \mathbb{E}_q\left[\log q(z \mid x) - \log p_\theta(x, z) + \log p_\theta(x)\right]$$

注意 $\log p_\theta(x)$ 不依赖 $z$,可以从期望里提出来:

$$= \mathbb{E}_q[\log q(z \mid x)] - \mathbb{E}_q[\log p_\theta(x, z)] + \log p_\theta(x)$$

整理一下,把 $\log p_\theta(x)$ 单独放一边:

$$\log p_\theta(x) = \underbrace{\mathbb{E}_q\left[\log \frac{p_\theta(x, z)}{q(z \mid x)}\right]}_{\mathcal{L}(\theta, q; x) \text{ = ELBO}} + \underbrace{D_{\text{KL}}(q(z \mid x) \,\|\, p_\theta(z \mid x))}_{\geq 0}$$

**这是一个等式!** 不是不等式!它告诉我们三件极其重要的事:

1. $\log p_\theta(x) = \text{ELBO} + \text{KL}$,因为 KL ≥ 0,所以 $\log p_\theta(x) \geq \text{ELBO}$ —— 重新得到了下界
2. **下界的"间隙"恰好是** $D_{\text{KL}}(q(z \mid x) \,\|\, p_\theta(z \mid x))$。也就是说,$q$ 越接近真实后验 $p_\theta(z \mid x)$,下界越紧
3. 当 $q(z \mid x) = p_\theta(z \mid x)$ 时,KL = 0,**ELBO 等于真实对数似然**

最大化 ELBO 同时在做两件事——拉高 $\log p_\theta(x)$,以及让 $q$ 逼近真实后验。

### 4.3 推导三:能量分解视角(最实用)

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

## 5. 应用到 diffusion

在 diffusion 里:

- $x = x_0$(数据)
- $z = x_{1:T}$(所有加噪后的隐变量)
- $q(z \mid x) = q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})$ —— **固定的**前向过程,不需要学
- $p_\theta(x, z) = p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)$ —— **要学的**逆向过程

直接套推导一的结果:

$$\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T} \mid x_0)}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right]$$

把分子分母展开:

$$= \mathbb{E}_q\left[\log p(x_T) + \sum_{t=1}^T \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1})}\right]$$

这就是 Sohl-Dickstein 2015 用的 ELBO 形式,我们下一讲会从这里出发。

但这个形式**训练效果不好**——分子是逆向 $p_\theta(x_{t-1} \mid x_t)$,分母是前向 $q(x_t \mid x_{t-1})$,方向不一致,梯度方差大。

DDPM 的关键贡献之一就是用**贝叶斯公式重写这个 ELBO**,让每一项都变成两个高斯之间的 KL,从而:

- 有闭式解,不需要采样估计
- 方差大幅降低
- 最终化简为漂亮的 $\| \epsilon - \epsilon_\theta \|^2$ 形式

这部分留到 DDPM 那一讲详细推。

{% details 几个常见困惑点 %}

**Q1:为什么 $q$ 可以"任意选"?选错了会怎样?**

ELBO 对**任何** $q$ 都成立(Jensen 不等式不要求 $q$ 是什么具体形式)。但选得不好,下界很松,优化下界相当于在优化一个跟 $\log p_\theta(x)$ 关系很弱的东西。

在 VAE 里,$q_\phi(z \mid x)$ 是用神经网络参数化、和 $\theta$ 一起学的。在 diffusion 里更聪明:$q$ 是**预先固定的扩散过程**——简单到不需要学,但又因为 $T$ 大、每步小,使得 $q$ 自动接近真实后验。这是 diffusion 比 VAE 强的一个本质原因。

**Q2:为什么不直接最大化 $\log p_\theta(x)$?**

算不出来。即使能用蒙特卡洛 $p_\theta(x) \approx \frac{1}{N}\sum_i \frac{p_\theta(x, z_i)}{q(z_i \mid x)}$,这是有偏估计(因为 $\log$ 套外面),而且方差极大,实际不可行。

**Q3:Jensen 不等式损失了多少?**

正好是 $D_{\text{KL}}(q \| p_\theta(z \mid x))$。这就是为什么推导二比推导一更深刻——它精确量化了"损失"。

{% enddetails %}

---

## 6. 为什么 KL 散度?为什么是 $D_{\text{KL}}(q \| p)$ 而不是反过来?

值得回答这个常被问的问题。

### 6.1 为什么用 KL 而不是其他距离

KL 散度是衡量分布距离的诸多方法之一,还有总变差距离、Wasserstein 距离、JS 散度、Hellinger 距离。**为什么 ML 偏爱 KL?** 几个原因:

1. **可拆分**:$\log$ 让乘积变求和,样本独立时似然加项变成 KL 项加项
2. **和最大似然天然对应**:$\arg\max_\theta \mathbb{E}_{p_{\text{data}}}[\log p_\theta] = \arg\min_\theta D_{\text{KL}}(p_{\text{data}} \| p_\theta)$
3. **和信息论挂钩**:KL = 多用 $\log p$ 编码 $q$ 样本时的额外比特数
4. **数学上易处理**:对很多分布族(指数族)有闭式

### 6.2 $D_{\text{KL}}(q \| p)$ vs $D_{\text{KL}}(p \| q)$:不对称!

KL 不对称——选哪个方向有实际差别。

**变分推断用 $D_{\text{KL}}(q \| p(z \mid x))$**(forward KL,从 $q$ 到真实后验)。

- $D_{\text{KL}}(q \| p) = \mathbb{E}_q[\log q - \log p]$:对 $q$ 求期望——**我们能从 $q$ 采样**(因为 $q$ 是我们自选的)
- $D_{\text{KL}}(p \| q) = \mathbb{E}_p[\log p - \log q]$:对 $p$ 求期望——**我们不能从 $p$ 采样**(因为 $p$ 是真实后验,本来就算不出来)

**所以方向选择不是设计选择,是必然的**——只有 $D_{\text{KL}}(q \| p)$ 能算,反方向算不了。

特性上:$D_{\text{KL}}(q \| p)$ 是"mode-seeking",$q$ 倾向于覆盖真实后验的某个模式;反方向是"mode-covering",会让 $q$ 覆盖整个真实后验(更平展)。

---

## 7. 一条完整逻辑链

让我把这一讲的内容串成一条完整逻辑链:

```
我想学 p(x)(生成模型目标)
        ↓
直接表达 p(x) 的方法都有问题
(简单的太弱,复杂的归一化算不动)
        ↓
策略:用"简单分布 + 隐变量边际化"表达
        ↓
得到 p(x) = ∫ p(x, z) dz
        ↓
最大似然要算这个积分——通常算不动
        ↓
等价问题:推断 p(z | x)——也算不动
(因为分母还是 ∫ p(x, z) dz)
        ↓
变分推断:用优化代替积分
找一个 q(z) ≈ p(z | x),最大化 ELBO
        ↓
ELBO 只用 p(x, z) 和 q(z),都能算
        ↓
完成训练
```

**每一步都是被前一步逼出来的**。没有"凭空想到"的环节,全是"上一步堵了,这一步怎么走"的自然推导。

---

## 8. 多视角汇聚

为什么这么多不同的研究路径(物理直觉、信息论、神经网络)最终都汇聚到隐变量 + 贝叶斯 + 变分推断?

**这反映了一个深刻的事实——任何想严肃处理"复杂分布"的方法,最终都要面对"边际化算不动"这个核心障碍**。绕开它的工具有限,变分推断是其中最通用的。

不同路径只是"从哪里进入"这个框架:

- **统计学家**(1990s):从贝叶斯推断进来
- **物理学家**(平均场近似):从统计力学进来,得到等价数学
- **信息论家**(MDL,信源编码):从最优编码进来,ELBO 等价于压缩理论
- **神经网络派**(VAE):从 autoencoder 加上概率视角进来
- **Diffusion 派**:从 SDE/score 进来,但本质还是变分

**这种"多视角汇聚"是数学结构深刻性的标志**——你触到了真正普适的东西。

---

## 9. 要点回顾

- **变分推断的根本动机**:贝叶斯后验 $p(z \mid x)$ 几乎总是算不出来(分母 $\int p(x, z) dz$ 算不动)
- **核心技巧**:把"算后验"变成"找一个最接近后验的简单分布"——积分问题变优化问题
- **关键代数**:$\log p(x) = \text{ELBO}(q) + D_{\text{KL}}(q \| p(z \mid x))$,优化 ELBO 等价于让 $q$ 逼近后验
- **ELBO 三视角**:Jensen 不等式 / KL 等式 / 能量分解
- **方向 $D_{\text{KL}}(q \| p)$ 的必然性**:只有这个方向才计算可行($q$ 是我们能采样的)
- **diffusion 的特殊性**:把 $q$ 做成固定但精心设计的扩散过程,让 ELBO 变得几乎紧
- **更深的视角**:变分推断是"用近似换可计算性"的通用方法论,在物理、数学、ML 里到处出现

下一讲我们用变分推断 + 隐变量这个框架,具体看 EM 算法和 VAE——它们是 diffusion 在精神上的两个直接前辈。
