---
title: Chapter 2. EM & VAE
categories: Diffusion Models
date: 2026-05-01 13:00:00
mathjax: true
tags:
    - AI
    - Diffusion Models
---

上一讲我们把 ELBO 和变分推断这个数学骨架立起来了。这一讲讲两个**站在这个骨架上**的经典模型:**EM 算法**(经典隐变量模型的标准训练工具)和 **VAE**(深度学习版的隐变量模型)。理解这两个,后面 diffusion 是一脉相承。

我会按这个顺序展开:

1. EM 算法:动机、推导、收敛性
2. GMM 上的 EM:具体跑一遍
3. VAE 的动机和数学:重新走一遍 ELBO
4. VAE 的工程结构:encoder / decoder / 重参数化
5. 几何直觉与 VAE 的局限
6. EM 和 VAE 的对应关系

---

## 1. EM 算法详解

EM(Expectation-Maximization,期望最大化)算法是 1977 年由 Dempster、Laird 和 Rubin 系统化提出的,用来处理**含有隐变量的极大似然估计问题**。它和 VAE 有着深刻的联系——实际上 VAE 可以看成 EM 的神经网络版本。理解 EM 之后,再回头看 VAE 会有"原来如此"的感觉。

### 1.1 要解决的问题

我们有一堆观测数据 $X = \{x^{(1)}, \dots, x^{(N)}\}$,模型有参数 $\theta$,想做极大似然估计:

$$\theta^* = \arg\max_\theta \log p(X; \theta)$$

如果模型很简单(比如直接是高斯),求导置零就解决了。但很多模型含有**隐变量** $z$,似然要写成:

$$p(x; \theta) = \sum_z p(x, z; \theta) \quad \text{(离散)} \qquad \text{或} \qquad p(x; \theta) = \int p(x, z; \theta) \, dz \quad \text{(连续)}$$

对数里有求和(或积分),求导后没有解析解,直接优化非常困难。

**典型例子:高斯混合模型(GMM)**。数据由 $K$ 个高斯混合而成,每个数据点 $x$ 来自哪个高斯是未知的——这就是隐变量 $z \in \{1, \dots, K\}$。似然是:

$$p(x; \theta) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x; \mu_k, \Sigma_k)$$

直接对 $\theta = \{\pi_k, \mu_k, \Sigma_k\}$ 求导,会发现各分量耦合在一起,解不出来。

### 1.2 核心思想

EM 的想法非常优雅:**如果我们知道每个数据点的隐变量 $z$,问题就简单了**。

回到 GMM:如果我事先知道每个 $x^{(i)}$ 来自哪个高斯,那 $\mu_k$ 就是属于第 $k$ 个高斯的所有点的样本均值,一步到位。

但我们不知道 $z$。EM 的策略是**迭代地"猜"和"算"**:

- **E 步**:用当前参数 $\theta^{(t)}$,猜每个点的 $z$ 的分布(即后验 $p(z|x; \theta^{(t)})$)
- **M 步**:在这个猜测下,把 $\theta$ 更新成"最对得起这个猜测"的值

然后用新的 $\theta$ 再猜、再更新,如此往复。直觉上像"先估计模糊的归属,再用归属反过来精修参数",两边互相搀扶,逐步收敛。

### 1.3 数学推导

这部分和 VAE 的 ELBO 推导**几乎一模一样**——这不是巧合。我们想最大化 $\log p(x; \theta)$。引入任意分布 $q(z)$,做相同的代数变形:

$$\log p(x; \theta) = \mathbb{E}_{q(z)}\left[\log \frac{p(x, z; \theta)}{q(z)}\right] + \mathrm{KL}(q(z) \| p(z|x; \theta))$$

$$= \underbrace{\mathcal{L}(q, \theta)}_{\text{ELBO}} + \mathrm{KL}(q(z) \| p(z|x; \theta))$$

由于 KL $\geq 0$,$\log p(x; \theta) \geq \mathcal{L}(q, \theta)$。

EM 算法就是**交替地优化 $q$ 和 $\theta$**,以推高这个下界。

**E 步**:固定 $\theta^{(t)}$,优化 $q$。要最大化 ELBO,等价于最小化 KL 项(因为 $\log p(x;\theta)$ 此时是常数)。KL 在 $q(z) = p(z|x; \theta^{(t)})$ 时为零。所以:

$$q^{(t+1)}(z) = p(z | x; \theta^{(t)})$$

**这一步的关键**:$q$ 直接取真后验,KL = 0,bound 被推到与 $\log p(x; \theta)$ **完全相切**。

**M 步**:固定 $q^{(t+1)}$,优化 $\theta$。ELBO 中只有一项依赖 $\theta$:

$$\theta^{(t+1)} = \arg\max_\theta \, \mathbb{E}_{q^{(t+1)}(z)}[\log p(x, z; \theta)]$$

这个期望称为 **Q 函数**,记作 $Q(\theta; \theta^{(t)})$。M 步就是最大化 Q 函数。

### 1.4 为什么 EM 一定收敛(到局部最优)

每次迭代,$\log p(x; \theta)$ **单调不减**。论证如下:

- **E 步后**:$\log p(x; \theta^{(t)}) = \mathcal{L}(q^{(t+1)}, \theta^{(t)})$ ——下界与真值相切。
- **M 步后**:$\mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)})$ ——M 步推高下界。
- 而 $\log p(x; \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)})$ ——下界永远 $\leq$ 真值。

合起来:

$$\log p(x; \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \log p(x; \theta^{(t)})$$

**所以 $\log p$ 单调不减**。由于 $\log p$ 通常有上界,EM 必然收敛——不过只能保证收敛到**局部最优或鞍点**,不是全局最优。

### 1.5 几何直觉

把 EM 想象成"造梯子"爬山:

- **E 步**:在当前 $\theta^{(t)}$ 处,造一个**与 $\log p(\theta)$ 相切**的下界曲面 $\mathcal{L}(q, \theta)$。这个下界处处 $\leq \log p$,但在 $\theta^{(t)}$ 处相等。
- **M 步**:爬上这个下界曲面的最高点,得到 $\theta^{(t+1)}$。
- 在新位置,$\log p$ 必然 $\geq$ 下界的最高值,所以 $\log p$ 也升高了。
- 重复:在新位置造新的相切下界,再爬。

每一轮都用一个**容易优化的下界**代替**难优化的真函数**,但这个下界总是与真函数在当前点相切——所以爬下界等于爬真函数。这是 EM 最深的洞察。

---

## 2. 具体例子:GMM 的 EM

**模型**:$p(x; \theta) = \sum_{k=1}^K \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)$

**隐变量**:$z^{(i)} \in \{1, \dots, K\}$,表示第 $i$ 个点属于哪个分量。

**E 步**:计算每个点属于每个分量的后验概率(称为 responsibility):

$$\gamma_{ik} = p(z^{(i)} = k \mid x^{(i)}; \theta^{(t)}) = \frac{\pi_k \mathcal{N}(x^{(i)}; \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j)}$$

直观:$\gamma_{ik}$ 是"第 $i$ 个点有多大概率来自第 $k$ 个高斯"。

**M 步**:用 responsibility 加权,更新参数(都是闭式解):

$$N_k = \sum_i \gamma_{ik}$$

$$\mu_k^{(t+1)} = \frac{1}{N_k} \sum_i \gamma_{ik} \, x^{(i)}$$

$$\Sigma_k^{(t+1)} = \frac{1}{N_k} \sum_i \gamma_{ik} (x^{(i)} - \mu_k^{(t+1)})(x^{(i)} - \mu_k^{(t+1)})^\top$$

$$\pi_k^{(t+1)} = \frac{N_k}{N}$$

直观:每个高斯的均值 = 属于它的点的加权平均;协方差类似;权重 = 属于它的点的比例。

**和 K-means 的关系**:K-means 是 GMM-EM 的"硬"版本——E 步把每个点硬分配给最近的中心($\gamma_{ik}$ 是 0 或 1),M 步用分配的点更新中心。所以 K-means 是 EM 在协方差固定为 $\sigma^2 I$、$\sigma \to 0$ 时的极限。

{% details GMM 一个数值例子(手算) %}

我用一个**具体的小例子**带你走一遍。

#### 一、问题设定

假设我们有 5 个一维数据点:

$$x^{(1)} = 1.0, \quad x^{(2)} = 1.5, \quad x^{(3)} = 5.0, \quad x^{(4)} = 5.5, \quad x^{(5)} = 6.0$$

我们假设这些点来自 $K=2$ 个高斯分布的混合。模型有以下参数要学:

- $\pi_1, \pi_2$:两个高斯的混合权重(加起来 = 1)
- $\mu_1, \mu_2$:两个高斯的均值
- $\sigma_1^2, \sigma_2^2$:两个高斯的方差

肉眼看,数据明显分成两团:$\{1.0, 1.5\}$ 和 $\{5.0, 5.5, 6.0\}$。但**算法不知道这个**,它要自己学出来。

#### 二、关键概念:隐变量 $z$

对每个点 $x^{(i)}$,我们引入一个隐变量 $z^{(i)} \in \{1, 2\}$,表示**这个点来自哪个高斯**。

**如果我们知道 $z$**,问题就变得超级简单。**但我们不知道 $z$**。EM 的策略是:不要硬猜 $z$ 是 1 还是 2,而是计算"$x^{(i)}$ 来自高斯 $k$ 的概率"——这是一个**软分配**。

#### 三、初始化

EM 需要初始值。随便给一组(实践中通常用 K-means 的结果初始化):

$$\pi_1 = 0.5, \quad \pi_2 = 0.5$$
$$\mu_1 = 2.0, \quad \mu_2 = 4.0$$
$$\sigma_1^2 = 1.0, \quad \sigma_2^2 = 1.0$$

注意初始的 $\mu_1, \mu_2$ 故意选得不太准,看 EM 怎么把它们调对。

#### 四、E 步:计算 responsibility(软分配)

E 步要算的是 $\gamma_{ik}$:**第 $i$ 个点属于第 $k$ 个高斯的后验概率**。公式来自贝叶斯定理:

$$\gamma_{ik} = \frac{\pi_k \, \mathcal{N}(x^{(i)}; \mu_k, \sigma_k^2)}{\sum_{j=1}^K \pi_j \, \mathcal{N}(x^{(i)}; \mu_j, \sigma_j^2)}$$

**具体计算 $x^{(1)} = 1.0$**:

一维高斯密度 $\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$

- 高斯 1($\mu_1 = 2.0, \sigma_1^2 = 1.0$):$\mathcal{N}(1.0; 2.0, 1.0) \approx 0.242$
- 高斯 2($\mu_2 = 4.0, \sigma_2^2 = 1.0$):$\mathcal{N}(1.0; 4.0, 1.0) \approx 0.0044$

$$\gamma_{11} = \frac{0.5 \times 0.242}{0.5 \times 0.242 + 0.5 \times 0.0044} \approx 0.982,\quad \gamma_{12} \approx 0.018$$

**解读**:$x^{(1)} = 1.0$ 有 98.2% 的概率来自高斯 1,1.8% 来自高斯 2。这非常合理——1.0 离 $\mu_1 = 2.0$ 近,离 $\mu_2 = 4.0$ 远。

**对所有 5 个点都算一遍**:

| $i$ | $x^{(i)}$ | $\gamma_{i1}$(来自高斯 1) | $\gamma_{i2}$(来自高斯 2) |
|---|---|---|---|
| 1 | 1.0 | 0.982 | 0.018 |
| 2 | 1.5 | 0.924 | 0.076 |
| 3 | 5.0 | 0.018 | 0.982 |
| 4 | 5.5 | 0.002 | 0.998 |
| 5 | 6.0 | 0.0001 | 0.9999 |

每行加起来都是 1。这就是 E 步的全部输出——一张"软分配表"。

#### 五、M 步:更新参数

**先算"有效样本数" $N_k$**(第 $k$ 个高斯"拥有"多少个点):

$$N_1 \approx 1.926,\quad N_2 \approx 3.074$$

注意 $N_1 + N_2 = 5$。

**更新均值**:

$$\mu_1^{\text{new}} = \frac{0.982 \times 1.0 + 0.924 \times 1.5 + 0.018 \times 5.0 + 0.002 \times 5.5 + 0.0001 \times 6.0}{1.926} \approx 1.282$$

$$\mu_2^{\text{new}} = \frac{0.018 \times 1.0 + 0.076 \times 1.5 + 0.982 \times 5.0 + 0.998 \times 5.5 + 0.9999 \times 6.0}{3.074} \approx 5.378$$

**对比初始值**:$\mu_1$ 从 2.0 → 1.282(更接近左边一团),$\mu_2$ 从 4.0 → 5.378(更接近右边一团)。EM 在自动找两团数据!

**更新方差和混合权重**:类似公式,算出 $\sigma_1^{2,\text{new}} \approx 0.078$、$\sigma_2^{2,\text{new}} \approx 0.227$、$\pi_1^{\text{new}} \approx 0.385$、$\pi_2^{\text{new}} \approx 0.615$。

#### 六、再迭代

把更新后的参数拿回 E 步,重新算 $\gamma_{ik}$。这一次,因为 $\mu_1, \mu_2$ 更准了,$\sigma$ 也更小,$\gamma_{ik}$ 变得更"极端"。然后再做 M 步,参数继续微调。通常几十轮后,$\mu_1 \to 1.25$,$\mu_2 \to 5.5$,完美找到两团数据的中心。

#### 七、为什么 M 步是这些公式?

M 步是最大化 Q 函数:

$$Q(\theta; \theta^{(t)}) = \sum_i \sum_k \gamma_{ik} \log\left[\pi_k \mathcal{N}(x^{(i)}; \mu_k, \sigma_k^2)\right]$$

对 $\mu_k$ 求导置零:

$$\frac{\partial Q}{\partial \mu_k} = \sum_i \gamma_{ik} \cdot \frac{x^{(i)} - \mu_k}{\sigma_k^2} = 0 \implies \mu_k = \frac{\sum_i \gamma_{ik} x^{(i)}}{N_k}$$

正是加权平均公式。

#### 八、整体直觉

软分配是关键——它让算法**不必硬选**,可以"骑墙"。一个处于两个高斯之间的点,可以以 0.5/0.5 的方式同时影响两边。这个柔软性让 EM 能平滑收敛,而不会像硬分配那样陷入剧烈震荡。

| 步骤 | 输入 | 输出 | 直觉 |
|---|---|---|---|
| **E 步** | 当前参数 $\theta^{(t)}$ | 软分配 $\gamma_{ik}$ | 在当前参数下,每个点属于每个高斯的概率 |
| **M 步** | 软分配 $\gamma_{ik}$ | 新参数 $\theta^{(t+1)}$ | 用加权数据,重新估计每个高斯的参数 |

{% enddetails %}

### 2.1 EM 的其他经典应用与变体

**经典应用**:HMM(Baum-Welch)、因子分析/概率 PCA、话题模型 pLSA、缺失数据填补、Mixture of Experts。

**变体**:广义 EM(M 步只要让 Q 增大)、变分 EM(E 步用近似分布)、Monte Carlo EM(E 步采样估计)、Stochastic EM(mini-batch)、Hard EM(取后验众数)。

**EM 的局限**:局部最优、收敛慢、模型受限(要求 E 步后验可计算、M 步有闭式解)、奇异性(GMM 中某高斯只覆盖一点会让似然爆炸)。

---

## 3. VAE:深度学习版的隐变量模型

VAE(Variational Autoencoder)是 2013 年由 Kingma 和 Welling 提出的生成模型,它把概率图模型的思想和神经网络结合在一起,既是一个生成模型,也是一个表示学习工具。

### 3.1 要解决的问题

我们手头有一堆数据 $\{x^{(1)}, x^{(2)}, \dots, x^{(N)}\}$(比如人脸图像),希望学到这堆数据背后的概率分布 $p(x)$。学到这个分布之后,我们就能:

- **生成**新样本(从 $p(x)$ 采样)
- **评估**一张图片"有多真实"(计算 $p(x)$)
- **学到表示**:每张图像背后有一些更简洁的"潜在因素"(姿态、光照、表情等)

直接学 $p(x)$ 很难。VAE 的策略是引入**潜变量** $z$,把这个分布拆成:

$$p(x) = \int p(x \mid z) \, p(z) \, dz$$

其中 $p(z)$ 是简单的先验(标准高斯),$p(x|z)$ 是用神经网络参数化的条件分布。

### 3.2 核心困难:后验不可计算

学这个模型,需要做两件事:

**1. 最大化数据似然** $\log p(x) = \log \int p(x|z) p(z) dz$:这个积分对 $z$ 是高维的,而且 $p(x|z)$ 是神经网络,**无法直接计算**。

**2. 推断后验** $p(z|x) = \frac{p(x|z) p(z)}{p(x)}$:分母 $p(x)$ 算不了,所以后验也算不了。

VAE 的破局之道是:**用一个神经网络去近似后验**,记作 $q_\phi(z|x)$,称为**变分分布**(或 encoder)。这就是"变分"的由来——把推断问题转成优化问题(上一讲的标准变分推断)。

### 3.3 VAE 的 ELBO

把上一讲的 ELBO 等式代入这个具体的 $(x, z)$ 设定,得到:

$$\log p(x) = \mathcal{L}(\theta, \phi; x) + \mathrm{KL}(q_\phi(z|x) \| p(z|x))$$

由 KL $\geq 0$:

$$\log p(x) \geq \mathcal{L}(\theta, \phi; x) =: \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x,z)}{q_\phi(z|x)}\right]$$

把 $p(x,z) = p(x|z)p(z)$ 代入并整理(等同于上一讲的"推导三:能量分解视角"):

$$\boxed{\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重建项}} - \underbrace{\mathrm{KL}(q_\phi(z|x) \| p(z))}_{\text{正则项}}}$$

这就是 VAE 训练用的目标函数,有非常清晰的解读:

**重建项**:从 $q_\phi(z|x)$ 采一个 $z$,decoder 用这个 $z$ 应该能重建出 $x$。最大化这一项 = 最小化重建误差。

**正则项**:把每个 $q_\phi(z|x)$ 拉向先验 $p(z) = \mathcal{N}(0, I)$。

整个目标:**让 decoder 能从 $z$ 重建 $x$,同时让 $z$ 的分布形状像高斯**。

---

## 4. VAE 的工程结构

### 4.1 三个网络模块

**Encoder $q_\phi(z|x)$**:输入图像 $x$,输出一个高斯分布的均值和方差:

$$q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi(x)^2 I)$$

通常神经网络最后一层输出一个 $\mu$ 向量和一个 $\log \sigma^2$ 向量(用 log 是为了保证方差为正)。

**Decoder $p_\theta(x|z)$**:输入 $z$,输出 $x$ 的分布参数。如果 $x$ 是连续的(如归一化图像),通常用高斯;如果是二值的(如 MNIST 黑白),通常用伯努利:

$$p_\theta(x|z) = \mathcal{N}(x; \mu_\theta(z), \sigma^2 I) \quad \text{或} \quad \text{Bernoulli}(\mu_\theta(z))$$

**先验 $p(z)$**:固定为 $\mathcal{N}(0, I)$,不学。

### 4.2 重参数化技巧

训练时我们要计算 $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$,做法是从 $q_\phi(z|x)$ 采一个 $z$,代入 decoder 算 $\log p_\theta(x|z)$。问题是:**采样操作不可导**,梯度传不到 encoder $\phi$。

**重参数化**的技巧是把随机性从计算图里"抽离"出来:

$$\epsilon \sim \mathcal{N}(0, I), \quad z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$$

这样 $z$ 是 $\mu, \sigma$ 的可导函数,$\epsilon$ 是外部输入的噪声。梯度可以正常通过 $z$ 反传到 $\phi$。

这是 VAE 工程上能 work 的核心技巧之一。

### 4.3 KL 项的解析解

当 $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$ 且 $p(z) = \mathcal{N}(0, I)$ 时,KL 散度有解析解:

$$\mathrm{KL}(q \| p) = \frac{1}{2} \sum_{i=1}^{d} \left( \mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1 \right)$$

这个公式直接进入 loss,不用采样近似,数值稳定。

### 4.4 完整训练 / 生成流程

**训练**:对每个 batch 的图像 $x$:

1. encoder 前向:$\mu, \log\sigma^2 = \text{Encoder}(x)$
2. 采样 $\epsilon \sim \mathcal{N}(0, I)$,计算 $z = \mu + \sigma \odot \epsilon$
3. decoder 前向:$\hat{x} = \text{Decoder}(z)$
4. 计算重建损失(如 MSE 或交叉熵)
5. 计算 KL 损失(用上面解析公式)
6. 总 loss = 重建损失 + KL 损失,反向传播

**生成**:训练完成后,丢掉 encoder,只用 decoder:

1. 采样 $z \sim \mathcal{N}(0, I)$
2. $\hat{x} = \text{Decoder}(z)$ 就是新样本

---

## 5. 几何直觉与一个常见困惑

### 5.1 VAE 在做什么(几何视角)

可以把 VAE 想象成两件事的合奏:

**Encoder** 把图像空间(高维、复杂)中的每个点 $x$ 映射到潜空间(低维、简单)中的一个**小区域**(高斯团)。相似图像的小区域在潜空间靠近,因为这样 decoder 才好重建。

**Decoder** 学了一个从潜空间到图像空间的连续光滑映射。在潜空间相邻的两个 $z$,decoder 输出的图像也相邻——这就是 VAE 的潜空间常常很"平滑"、可以做插值(在两张人脸的 $z$ 之间线性插值,生成中间过渡的人脸)的原因。

KL 项确保所有这些小区域**汇总起来**填满标准高斯,这样从 $\mathcal{N}(0,I)$ 采样才能落在 decoder 见过的区域里,生成有意义的图像。

### 5.2 一个常见困惑

> 标准高斯 $\mathcal{N}(0, I)$ 在原点附近概率密度高,远离原点的地方概率密度低。如果不同图像被映射到 $z$ 空间的不同位置,那靠近原点的图像"更可能",远离原点的图像"更不可能"——这听起来确实有点奇怪。

**关键澄清**:不是"不同图像 → 高斯的不同部位",而是更像:

> 不同图像被映射到 $z$ 空间的**不同小区域**(每个 $q(z|x)$ 是一个小高斯),但这些小区域**整体拼起来**恰好填满了标准高斯的形状——中心区域被很多图像的小高斯覆盖,外围区域被较少的图像覆盖。

打个比方:想象你要把很多小水滴铺在一张纸上,要求最终铺出来的样子整体看起来像一个高斯钟形。那么中心区域必然有更多水滴重叠,边缘区域水滴稀疏。**每个水滴(每张图像的 $q(z|x)$)本身没有"概率高低"之分**,是它们集体的密度构成了高斯。

VAE 训练时**同时优化两件事**(ELBO 的两项):

- **重建损失**:鼓励 encoder 把每张图像映射到一个能让 decoder 重建出来的 $z$。这一项希望不同图像的 $z$ 分散开、彼此可区分。
- **KL 散度**:把每个 $q(z|x)$ 拉向标准高斯 $p(z) = \mathcal{N}(0, I)$。这一项希望所有图像的 $z$ 都挤在原点附近。

这两项是**对抗的**。最终的平衡结果是:**所有图像合在一起的分布(聚合后验)$q(z) = \mathbb{E}_x[q(z|x)]$ 近似等于标准高斯 $\mathcal{N}(0, I)$**,而不是每张图像被映射到高斯的某个固定"部位"。

生成图像时,从 $\mathcal{N}(0, I)$ 采 $z$ 更可能采到中心区域 → 更可能生成那些被映射到中心的图像类型(通常是"典型"、"常见"的图像);偶尔采到外围 → 生成那些被映射到外围的图像(可能是"罕见"、"特殊"的图像)。这其实**反映了真实数据的频率结构**。

---

## 6. VAE 的局限和变体

### 6.1 VAE 的局限

**生成图像偏模糊**:这是 VAE 最常被诟病的问题。原因有几层:decoder 用高斯似然 + MSE 重建,本质上是在做平均;encoder 输出分布而非点,引入额外噪声;ELBO 是下界,优化它不等于优化真似然。

**KL 与重建的张力**:KL 太强,encoder 退化成只输出先验(posterior collapse),$z$ 不带信息;KL 太弱,潜空间不平滑,采样生成质量差。需要小心调节。

**Posterior collapse**:特别是当 decoder 表达能力很强时(如自回归 decoder),decoder 倾向于忽略 $z$,只用自身能力建模数据,导致 $z$ 完全没用。这是 VAE 实际部署中的典型坑。

### 6.2 VAE 和其他模型的对比

- **AE(自编码器)**:也是 encoder-decoder 结构,但潜空间没有概率约束,encoder 输出确定的 $z$ 而不是分布。AE 不能生成,因为潜空间没有"采样起点"。
- **GAN**:不显式学 $p(x)$,通过对抗训练直接学一个生成器。生成质量通常更高,但训练不稳定,也没有 encoder。
- **Normalizing Flow**:用可逆神经网络精确计算 $p(x)$,不需要变分近似。但要求网络可逆,架构受限。
- **Diffusion Model**:可以看成多步 VAE 的级联。训练稳定、生成质量极高,是当前主流。

### 6.3 常见变体

- **β-VAE**:把 KL 项乘上系数 $\beta$,$\beta > 1$ 鼓励解耦的(disentangled)表示
- **Conditional VAE**:condition 在标签上,$p(x|z, y)$,可控生成
- **VQ-VAE**:用离散 codebook 代替连续 $z$,生成质量大幅提升,是 DALL-E、Stable Diffusion 的祖先之一
- **Hierarchical VAE**:多层潜变量 $z_1, z_2, \dots$,表达能力更强
- **NVAE、VDVAE**:深度层级 VAE,生成质量已经能和 GAN 一较高下

---

## 7. EM 与 VAE 的对应关系

如果你刚读完上面的 VAE,这里的对应应该非常清晰:

| | EM | VAE |
|---|---|---|
| E 步 | $q(z) = p(z \mid x; \theta)$,精确后验 | $q_\phi(z \mid x)$,神经网络近似后验 |
| M 步 | 闭式解最大化 Q 函数 | 梯度下降最大化 ELBO |
| 模型 $p(x \mid z)$ | 简单(如高斯) | 神经网络 |
| 后验 | 通常可解析计算 | 一般算不出,必须近似 |
| KL 项 | E 步后为 0,bound 紧 | 一般 > 0,bound 不紧 |
| 优化方式 | 交替优化,每步闭式 | 端到端梯度,联合优化 |

**核心区别**:经典 EM 假设后验可算、M 步有闭式解,所以两步交替、各自求极值。VAE 处理的模型太复杂,后验和 M 步都没有闭式解,只能用神经网络做近似 + 梯度下降。

换个角度看,**VAE = 摊销变分 EM(amortized variational EM)**:

- "变分":E 步用 $q_\phi$ 近似后验,而不是精确后验
- "摊销":不为每个 $x$ 单独优化一个 $q$,而是训练一个共享 encoder,输入 $x$ 直接输出 $q(z|x)$ 的参数
- 用神经网络参数化所有分布,用 SGD 联合优化

---

## 8. 这一讲和 diffusion 的关系

VAE 到 diffusion 的转变,可以这样理解:

- VAE:**单步**隐变量,$x \leftrightarrow z$,encoder 学一个复杂的 $q_\phi(z\|x)$
- diffusion:**多步**隐变量,$x_0 \leftrightarrow x_1 \leftrightarrow \cdots \leftrightarrow x_T$,$q$ 是**预定义的固定扩散过程**(不学)

diffusion 的"巧思"在于:不学 $q$,但精心设计 $q$——多步小噪声让固定的 $q$ 自动接近真实后验,绕开了"$q$ 学不准"的老问题。

VAE 和 diffusion 共享整个变分推断骨架,只是在"$q$ 怎么选"这一点上分道扬镳。VAE 学一个 encoder $q_\phi$,diffusion 用固定但精心设计的扩散过程。

下一讲(Sohl-Dickstein)我们就具体看 diffusion 怎么把这个变分骨架展开成多步链式结构。

---

## 9. 要点回顾

- **EM** 是处理"含隐变量的极大似然"的经典工具——E 步用当前参数推断 $z$ 的分布,M 步用 $z$ 的分布更新参数,交替进行
- **EM 收敛性**:每轮 $\log p$ 单调不减,证明依靠"E 步让下界与真值相切"
- **GMM 上的 EM** 是最经典的具体应用,K-means 是它的"硬版本"
- **VAE = 神经网络化的变分推断**:用 encoder $q_\phi(z\|x)$ 近似后验,用重参数化让梯度可流
- **重参数化技巧**:$z = \mu + \sigma \odot \epsilon$,把随机性和待优化参数解耦
- **VAE 的几何直觉**:每张图像在潜空间是一个小高斯团,所有小团合起来填满 $\mathcal{N}(0, I)$
- **EM 和 VAE 是同一个数学骨架的两端**——经典与深度学习版本
