---
title: Chapter 6. DDPM
categories: Diffusion Models
date: 2026-05-01 12:00:00
mathjax: true
tags:
    - AI
    - Diffusion Models
---

这一讲是整个 diffusion 系列里最关键、最优美的一讲——它把 Sohl-Dickstein 的"理论上正确但实际不好用"的框架,改造成了"现代 diffusion 的标准形式"。所有后续工作(Stable Diffusion、Imagen、SDXL、FLUX 等)都建立在 DDPM 之上。

我会按这个顺序展开:
1. DDPM 的贡献概览
2. 关键引理:$q(x_{t-1} \mid x_t, x_0)$ 是高斯,有闭式
3. 用贝叶斯重写 ELBO,得到 KL 之和
4. 化简 KL,从 $\mu$-预测到 $\epsilon$-预测
5. $L_{\text{simple}}$ 的诞生
6. 训练和采样算法

---

## 一、DDPM 的核心贡献

Ho et al. 2020 的论文("Denoising Diffusion Probabilistic Models")做了三件事:

1. **数学改写**:用贝叶斯把 ELBO 改成"两个高斯之间 KL"的求和——每一项都有闭式
2. **重参数化**:把网络的预测目标从均值 $\mu_\theta$ 换成噪声 $\epsilon_\theta$,损失变成简单的 MSE
3. **工程突破**:UNet 架构 + 时间嵌入,在 CIFAR-10 上首次达到 GAN 级别的质量

**这一讲专注前两件事**——它们是数学上的核心,理解了它们,后面的所有 diffusion 工作你都能秒懂。

---

## 二、关键引理:$q(x_{t-1} \mid x_t, x_0)$ 是闭式高斯

这是整个 DDPM 推导的基石。让我们看它说什么、为什么重要、怎么算。

### 2.1 它是什么?

注意条件:**给定 $x_t$ 和 $x_0$**,求 $x_{t-1}$ 的分布。

为什么要加 $x_0$?因为没有 $x_0$ 时,$q(x_{t-1} \mid x_t)$(真实的逆向条件分布)**没有闭式**——它要对所有可能的 $x_0$ 边际化,涉及对真实数据分布的积分。

但**条件**在 $x_0$ 上之后,问题变得 tractable——这是因为给定 $x_0$,前向过程是一条已知的高斯链。

### 2.2 直观

想象一个三角形:

```
         x_0
        /    \
       /      \
      ↓        ↓
    x_{t-1} → x_t
```

我们已经知道:

- $q(x_{t-1} \mid x_0)$:从 $x_0$ 一步跳到 $x_{t-1}$ 的高斯(用边际公式)
- $q(x_t \mid x_{t-1})$:前向一步的高斯
- $q(x_t \mid x_0)$:从 $x_0$ 跳到 $x_t$ 的高斯

三者都是高斯,且都已知。**用贝叶斯把它们组合**,就能得到 $q(x_{t-1} \mid x_t, x_0)$。

### 2.3 推导

贝叶斯公式(条件版本):

$$q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0)\, q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$$

注意 $q(x_t \mid x_{t-1}, x_0) = q(x_t \mid x_{t-1})$(因为前向是马尔可夫,$x_t$ 只依赖 $x_{t-1}$)。所以:

$$q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1})\, q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$$

三个都是高斯:

- $q(x_t \mid x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{\alpha_t} x_{t-1},\, \beta_t I)$,其中 $\alpha_t = 1 - \beta_t$
- $q(x_{t-1} \mid x_0) = \mathcal{N}(x_{t-1};\, \sqrt{\bar\alpha_{t-1}} x_0,\, (1-\bar\alpha_{t-1}) I)$
- $q(x_t \mid x_0) = \mathcal{N}(x_t;\, \sqrt{\bar\alpha_t} x_0,\, (1-\bar\alpha_t) I)$

把高斯密度的表达式代入,展开 $\log$,合并 $x_{t-1}$ 的二次项和一次项——**结果一定还是个关于 $x_{t-1}$ 的高斯**(因为指数里是 $x_{t-1}$ 的二次型)。

### 2.4 闭式结果

整理后,$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde\mu_t(x_t, x_0), \tilde\beta_t I)$,其中:

$$\boxed{\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{1 - \bar\alpha_t}\, x_0 + \frac{\sqrt{\alpha_t} (1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\, x_t}$$

$$\boxed{\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\, \beta_t}$$

**关键观察**:

- 均值 $\tilde\mu_t$ 是 $x_0$ 和 $x_t$ 的**线性组合**——非常干净
- 方差 $\tilde\beta_t$ **不依赖于 $x_0$ 或 $x_t$**,只依赖于 $t$ 和预设的 $\beta_t$
- 所有项都用 schedule 中已知的 $\alpha, \bar\alpha, \beta$ 算出,没有任何待估计的量

这个公式是 DDPM 数学的**核心引擎**——后面的一切都建立在它上面。

(完整推导见附录1)

---

## 三、用贝叶斯重写 ELBO

回忆 Sohl-Dickstein 的 ELBO:

$$-\mathcal{L} = \mathbb{E}_q\left[-\log p(x_T) + \sum_{t=1}^T \log \frac{q(x_t \mid x_{t-1})}{p_\theta(x_{t-1} \mid x_t)}\right]$$

我们之前指出问题:每一项分子是前向 $q(x_t \mid x_{t-1})$,分母是逆向 $p_\theta(x_{t-1} \mid x_t)$,**条件方向不一致**,不是 KL 形式,无法用闭式。

**DDPM 的修复**:用贝叶斯把分子的方向翻过来。

### 3.1 贝叶斯翻转

由马尔可夫性,$q(x_t \mid x_{t-1}) = q(x_t \mid x_{t-1}, x_0)$(条件在 $x_0$ 上不影响)。再用贝叶斯:

$$q(x_t \mid x_{t-1}, x_0) = \frac{q(x_{t-1} \mid x_t, x_0)\, q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}$$

代回 ELBO 的求和项(对 $t \geq 2$ 做这件事,$t=1$ 单独处理):

$$\sum_{t=2}^T \log \frac{q(x_t \mid x_{t-1})}{p_\theta(x_{t-1} \mid x_t)} = \sum_{t=2}^T \log \frac{q(x_{t-1} \mid x_t, x_0)\, q(x_t \mid x_0)}{p_\theta(x_{t-1} \mid x_t)\, q(x_{t-1} \mid x_0)}$$

把 $\log$ 拆开:

$$= \sum_{t=2}^T \log \frac{q(x_{t-1} \mid x_t, x_0)}{p_\theta(x_{t-1} \mid x_t)} + \sum_{t=2}^T \log \frac{q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}$$

### 3.2 第二个求和:望远镜求和(telescoping)

第二个求和是个望远镜结构:

$$\sum_{t=2}^T \log \frac{q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)} = \log q(x_T \mid x_0) - \log q(x_1 \mid x_0)$$

(中间所有项依次抵消,这就是望远镜求和的威力)

### 3.3 处理 $t=1$ 项

把 $t=1$ 的项 $\log \frac{q(x_1 \mid x_0)}{p_\theta(x_0 \mid x_1)}$ 拿出来,和望远镜结果合并:

$$\log \frac{q(x_1 \mid x_0)}{p_\theta(x_0 \mid x_1)} + \log q(x_T \mid x_0) - \log q(x_1 \mid x_0) = -\log p_\theta(x_0 \mid x_1) + \log q(x_T \mid x_0)$$

### 3.4 整合

把所有项放回 $-\mathcal{L}$:

$$-\mathcal{L} = \mathbb{E}_q\left[-\log p(x_T) + \log q(x_T \mid x_0) + \sum_{t=2}^T \log \frac{q(x_{t-1} \mid x_t, x_0)}{p_\theta(x_{t-1} \mid x_t)} - \log p_\theta(x_0 \mid x_1)\right]$$

把前两项合并成一个 KL:

$$-\log p(x_T) + \log q(x_T \mid x_0) = \log \frac{q(x_T \mid x_0)}{p(x_T)}$$

它的期望是 $D_{\text{KL}}(q(x_T \mid x_0) \| p(x_T))$。

中间求和的每一项 $\log \frac{q(x_{t-1} \mid x_t, x_0)}{p_\theta(x_{t-1} \mid x_t)}$ 在期望下也是 KL。所以最终:

$$\boxed{-\mathcal{L} = \mathbb{E}_q\left[\underbrace{D_{\text{KL}}(q(x_T \mid x_0) \,\|\, p(x_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t))}_{L_{t-1}} - \underbrace{\log p_\theta(x_0 \mid x_1)}_{-L_0}\right]}$$

这是 DDPM 论文里 Eq. (5) 的著名形式。

### 3.5 三项的物理意义

- **$L_T$**:最终噪声分布要接近先验 $p(x_T) = \mathcal{N}(0, I)$。**不依赖 $\theta$**(因为 $q$ 不学),训练时是常数,**直接忽略**
- **$L_{t-1}$**($t = 2, \ldots, T$):每一步的逆向匹配——**这是训练的核心**
- **$L_0$**:最后一步从 $x_1$ 重建 $x_0$,作为离散数据时的对数似然项

所以**实际训练时,我们最关心的就是中间这一项 $L_{t-1}$**。

---

## 四、化简 $L_{t-1}$:核心一步

现在魔法发生。$L_{t-1}$ 是两个高斯之间的 KL:

- $q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde\mu_t(x_t, x_0), \tilde\beta_t I)$ ← 闭式,刚才推过
- $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$ ← 网络输出

DDPM 做了一个简化:**固定方差不学**。设 $\Sigma_\theta = \sigma_t^2 I$,其中 $\sigma_t^2$ 是预设的(取 $\beta_t$ 或 $\tilde\beta_t$ 都行,论文实验两者效果相近，详细解释见附录2)。

(为什么可以固定方差?因为 $\tilde\beta_t$ 已经是闭式且不依赖 $x_0, x_t$,网络再去学一个方差既冗余又不稳。Improved DDPM 后来证明学方差能略微提升似然,但 DDPM 原版坚持固定。)

固定方差后,套用各向同性高斯 KL 闭式公式:

$$D_{\text{KL}}(\mathcal{N}(\mu_1, \sigma^2 I) \| \mathcal{N}(\mu_2, \sigma^2 I)) = \frac{\| \mu_1 - \mu_2 \|^2}{2\sigma^2}$$

得到:

$$L_{t-1} = \frac{1}{2\sigma_t^2}\| \tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t) \|^2 + C$$

其中 $C$ 不依赖于 $\theta$。**训练目标变成了一个 MSE**:让网络输出的 $\mu_\theta$ 逼近真实后验均值 $\tilde\mu_t$。

这已经是巨大的进步——但 DDPM 还要再走一步。

---

## 五、从 $\mu$-预测到 $\epsilon$-预测

### 5.1 重写 $\tilde\mu_t$

$\tilde\mu_t$ 的原始公式包含 $x_0$,但训练时网络看到的是 $x_t$,看不到 $x_0$。我们用重参数化把 $x_0$ 表达成 $x_t$ 和 $\epsilon$ 的函数。

由 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$:

$$x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\, \epsilon}{\sqrt{\bar\alpha_t}}$$

代入 $\tilde\mu_t$ 的公式(代入后大量代数化简,这一步比较繁琐但纯机械):

$$\tilde\mu_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\, \epsilon\right)$$

**这个形式非常优美**:$\tilde\mu_t$ 是 $x_t$ 减去"按比例缩放后的噪声"。

### 5.2 让 $\mu_\theta$ 取相同的参数化形式

既然真实后验均值长这样,我们让模型输出也长这样:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\, \epsilon_\theta(x_t, t)\right)$$

这里的 $\epsilon_\theta(x_t, t)$ 是一个新网络——给定 $x_t$ 和 $t$,**预测一个噪声**。这是个**重参数化选择**——我们把"输出 $\mu_\theta$"换成等价地"输出 $\epsilon_\theta$"。

### 5.3 损失变成噪声 MSE

代入 $L_{t-1}$:

$$\| \tilde\mu_t - \mu_\theta \|^2 = \left\| \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}(\epsilon - \epsilon_\theta(x_t, t)) \right\|^2 = \frac{\beta_t^2}{\alpha_t (1-\bar\alpha_t)}\| \epsilon - \epsilon_\theta(x_t, t) \|^2$$

代回 $L_{t-1}$ 的完整表达:

$$L_{t-1} = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar\alpha_t)}\, \| \epsilon - \epsilon_\theta(x_t, t) \|^2$$

这是一个加权 MSE——**权重依赖于 $t$,但损失函数本身就是"预测噪声"的 MSE**。

---

## 六、$L_{\text{simple}}$ 的诞生

DDPM 论文做了最后一个决定性的简化:**把权重扔掉**,得到

$$\boxed{L_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[\| \epsilon - \epsilon_\theta(x_t, t) \|^2\right]}$$

其中 $t \sim \text{Uniform}(1, T)$,$x_0 \sim p_{\text{data}}$,$\epsilon \sim \mathcal{N}(0, I)$,$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$。

这是整个 diffusion 训练的"圣杯"——一个干净的、**无权重**的 MSE。

### 6.1 为什么扔权重?

理论上原始权重 $\beta_t^2 / (2\sigma_t^2 \alpha_t (1-\bar\alpha_t))$ 是 ELBO 的一部分,扔了就不再是严格的最大似然。但 DDPM 实验发现:**扔了之后效果反而更好**。

直觉:原始权重对小 $t$(噪声小)的项加很大权,而那些项的"信号"实际上很弱——网络在小噪声时容易学,学到的细节对生成质量影响不大。扔掉权重相当于让网络在所有 $t$ 上**均匀重视**,实际上这强化了大噪声时的学习,而大噪声时的去噪决定了生成图像的整体结构,更重要。

后来研究者发现,$L_{\text{simple}}$ 等价于一种"去权重"的最大似然——不是严格的 ELBO,但仍然是个合理的 surrogate。

### 6.2 与 score matching 的对应

把 $\epsilon$-预测和 score 联系起来。从 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$,我们有

$$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar\alpha_t} x_0}{1-\bar\alpha_t} = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

所以"预测 $\epsilon$"和"预测 score"只差一个标量 $-1/\sqrt{1-\bar\alpha_t}$:

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}}$$

**两条线在这里完全对上了**——DDPM 的 $\epsilon$-预测和 NCSN 的 score-预测在数学上是同一件事。这正是下一讲 Score SDE 要正式统一的内容。

---

## 七、训练算法

**完整训练流程**:

```
for each iteration:
    x_0 ~ p_data                              # 采一个数据点
    t   ~ Uniform({1, ..., T})                # 采一个时刻
    eps ~ N(0, I)                             # 采噪声
    x_t = sqrt(α_bar[t]) * x_0 + sqrt(1 - α_bar[t]) * eps   # 一步加噪
    loss = || eps - eps_θ(x_t, t) ||^2        # 噪声 MSE
    update θ via SGD
```

仅仅 5 行——这就是 DDPM 训练的全部。回头看 Sohl-Dickstein 那种"整条链采样、逐项算对数比值"的复杂流程,简洁度天差地别。

---

## 八、采样算法

训练完后,从纯噪声生成图像:

```
x_T ~ N(0, I)                                 # 起点:纯噪声
for t = T, T-1, ..., 1:
    z = N(0, I) if t > 1 else 0               # 最后一步不加噪声
    μ = (1/sqrt(α_t)) * (x_t - β_t/sqrt(1-α_bar_t) * eps_θ(x_t, t))
    x_{t-1} = μ + σ_t * z
return x_0
```

每一步的核心:**预测噪声 → 算后验均值 → 加一点新噪声**,$T$ 次后得到一张图。

注意采样要走 **$T = 1000$ 步**(DDPM 默认)——这慢,是 DDPM 最大的工程问题。后面 DDIM、DPM-Solver 等专门解决这个问题(下一讲后展开)。

{% details 采样算法怎么推的 %}

---

### 一、先把采样代码贴出来

```
x_T ~ N(0, I)                                 # 起点:纯噪声
for t = T, T-1, ..., 1:
    z = N(0, I) if t > 1 else 0               # 最后一步不加噪声
    μ = (1/sqrt(α_t)) * (x_t - β_t/sqrt(1-α_bar_t) * eps_θ(x_t, t))
    x_{t-1} = μ + σ_t * z
return x_0
```

四个值得问"为什么"的地方:

1. 为什么从 $\mathcal{N}(0, I)$ 出发?
2. 那个 $\mu$ 公式怎么来的?
3. 为什么再加一个 $\sigma_t z$ 噪声?
4. 为什么最后一步($t=1$)不加噪声?

我们逐个拆。

---

### 二、为什么从 $\mathcal{N}(0, I)$ 出发?

回忆训练目标 $L_T$:

$$L_T = D_{\text{KL}}\big(q(x_T \mid x_0) \,\big\|\, p(x_T)\big)$$

当 $T$ 大、$\bar\alpha_T \to 0$ 时,$q(x_T \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_T} x_0, (1-\bar\alpha_T) I) \to \mathcal{N}(0, I)$。

我们定义先验 $p(x_T) = \mathcal{N}(0, I)$。这两者几乎相同,所以 $L_T \approx 0$,我们不训练它。

**采样时**:既然真实的 $x_T$ 分布(无论 $x_0$ 是什么)都接近 $\mathcal{N}(0, I)$,那我们从 $\mathcal{N}(0, I)$ 采样作为起点,就是从训练分布采样——**不是任意选的"起点",而是数学上对应的那个分布**。

---

### 三、那个 $\mu$ 的公式怎么来的?

这是采样里最关键的一行。我们在训练时已经设计了:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(\mu_\theta(x_t, t),\, \sigma_t^2 I\big)$$

所以采样时,从这个高斯采 $x_{t-1}$。**问题是 $\mu_\theta$ 等于什么?**

回忆我们在训练阶段做了**重参数化**——让 $\mu_\theta$ 取以下形式:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\, \epsilon_\theta(x_t, t)\right)$$

**这个公式不是采样时凭空写出的——它是训练时网络的参数化定义本身**。我们就是这样让网络的输出(通过 $\epsilon_\theta$ 间接)给出 $\mu_\theta$ 的。

所以采样代码里那行

```
μ = (1/sqrt(α_t)) * (x_t - β_t/sqrt(1-α_bar_t) * eps_θ(x_t, t))
```

就是把训练时的参数化定义代入,**给定网络预测的 $\epsilon_\theta$,算出 $\mu_\theta$**。这一步纯粹是代数代入,没有新东西。

#### 这个公式的直观

让我们看清楚它在做什么。回忆 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$。如果网络完美预测 $\epsilon_\theta = \epsilon$,那么:

**第一步**:从 $x_t$ 反推 $x_0$:

$$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\, \epsilon_\theta}{\sqrt{\bar\alpha_t}}$$

这是把"加噪公式"反过来用。

**第二步**:算逆向后验均值 $\tilde\mu_t(x_t, \hat x_0)$。直接套真实后验的公式:

$$\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\hat x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} x_t$$

**两步合并**(代入 $\hat x_0$,化简——这就是 $\tilde\mu_t \to \epsilon$ 那步代数化简的逆向使用),得到那个 $\mu$ 公式。

**所以采样的 $\mu$ 步可以理解为**:

> "用网络预测噪声 → 反推一个 $\hat x_0$ 估计 → 用真实后验公式算 $x_{t-1}$ 的均值"

每次去噪一步,**间接地是在估计 $x_0$ 然后向 $x_0$ 走一步**。

---

### 四、为什么还要再加 $\sigma_t z$ 噪声?

这是最容易让初学者疑惑的一步——既然要去噪,为什么还要加噪声?

#### 4.1 数学原因

我们的逆向模型是

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 I)$$

——一个**有方差的高斯**,不是只输出均值。**从这个分布采样**就是"均值 + 标准差 × 标准高斯",即:

$$x_{t-1} = \mu_\theta + \sigma_t z,\quad z \sim \mathcal{N}(0, I)$$

代码里的 `μ + σ_t * z` 就是这个采样。所以这个噪声**本来就是逆向高斯分布的一部分**,没加就不是从 $p_\theta$ 采样了。

#### 4.2 直觉原因

为什么模型设计成有方差?因为**真实的逆向后验 $q(x_{t-1} \mid x_t, x_0)$ 也是有方差的高斯**($\tilde\beta_t I$)——不是确定性映射。

直观看:已知 $x_t$ 和 $x_0$,$x_{t-1}$ **不是唯一确定的**——它有一个分布的可能值。$\tilde\beta_t$ 衡量了这种不确定性。

更直观:前向加噪是随机过程,撤销它每一步也应该是随机的——给定 $x_t$,可能的"上一步" $x_{t-1}$ 有多种。如果逆向是确定的,我们就丢失了随机性,**最终生成的 $x_0$ 永远只有一个**(给定起点),失去多样性。

#### 4.3 物理类比

想象前向是"墨水扩散到水里"。如果你能逆转每个分子的运动,逆向也是确定的——但实际上,信息在扩散过程中**以热力学方式丢失**,逆向就必须有随机性来"补偿"丢失的信息。

每一步采样加的 $\sigma_t z$,正是"补偿"前向那一步丢的信息。

#### 4.4 一个常被问的细节:$\sigma_t$ 应该取什么?

DDPM 论文里实验了两种选择:

- $\sigma_t^2 = \beta_t$(前向方差)
- $\sigma_t^2 = \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$(真实后验方差)

两者效果相近。前者对应"假设 $x_0 \sim \mathcal{N}(0, I)$"的逆向后验,后者对应"假设 $x_0$ 是确定值"的逆向后验。实践中默认用前者(更简单)。

后来 DDIM(我们之后讲)发现:**$\sigma_t$ 可以是任意值,甚至 $\sigma_t = 0$**(完全确定性采样),仍然能生成正确分布的样本——这是个重要的发现,会让采样可以加速 10-50 倍。

---

### 五、为什么最后一步($t=1$)不加噪声?

代码里:

```
z = N(0, I) if t > 1 else 0
```

也就是当 $t = 1$ 时,$z = 0$,我们只取 $\mu$,不加噪声。

#### 5.1 数学原因

最后一步是从 $x_1$ 生成 $x_0$。$x_0$ 是**最终输出**,我们希望它是图像本身,而不是"图像 + 一点点噪声"。

更精确地,DDPM 里 $L_0 = -\log p_\theta(x_0 \mid x_1)$ 是单独处理的——它对应真实数据的对数似然。在最后一步,我们直接取**最大似然估计**,即 $\mu_\theta$ 本身,不再加噪声。

#### 5.2 直观原因

如果最后一步加噪声 $\sigma_1 z$,你的图像就会带一个最后的小噪声。$\sigma_1$ 虽然小,但对生成图像的清晰度还是有影响。

把这一步噪声去掉,相当于"在最后做个 MAP(maximum a posteriori)估计"——取分布的众数(高斯的众数就是均值),作为最佳点估计。

#### 5.3 这是个工程选择

严格说,**完全按 $p_\theta$ 采样应该最后一步也加噪声**。"$t=1$ 时不加"是个**实践上的优化**——理论上稍微偏离了概率模型,但生成质量更好。

后来 Improved DDPM、DDIM 等都采用了这个约定。

---

### 六、把整个采样过程串起来

让我把所有"为什么"放在一起,画出完整图景:

```
起点 x_T = N(0, I)             ← 因为前向终点近似就是 N(0, I)

for t = T, ..., 1:
    
    用网络预测 ε_θ(x_t, t)
    
    算 μ = ...                  ← 训练时的参数化定义,等价于
                                  "反推 x_0 估计 → 算逆向后验均值"
    
    if t > 1:
        x_{t-1} = μ + σ_t z     ← 从 p_θ(x_{t-1}|x_t) 采样
                                  (有方差是因为真实后验也有方差)
    else:
        x_0 = μ                 ← 最后一步取最大似然估计

return x_0
```

每一步都不是"魔法",而是训练阶段数学结构的直接对应:

- 起点 ↔ $L_T$ 的训练目标
- $\mu$ 公式 ↔ 训练时的参数化设计
- 加噪声 ↔ $p_\theta$ 是高斯分布
- 最后不加噪 ↔ $L_0$ 的处理约定

---

### 七、采样到底在干什么?(更高层理解)

最后一句话总结整个采样的"灵魂":

**采样是把训练时学到的"逆向条件分布 $p_\theta(x_{t-1} \mid x_t)$"按马尔可夫链**反向运行**,从 $\mathcal{N}(0, I)$ 一步步走到数据分布 $p_{\text{data}}$。**

每一步本质上是:

> "网络看着 $x_t$,猜里面的噪声 $\epsilon$,然后:
>  - 大致知道'清掉 $\epsilon$ 后应该长什么样'
>  - 但不完全清掉,只走一小步,留一点噪声给下一步处理
>  - 这就是 $x_{t-1}$"

走 $T$ 步,最终噪声彻底清完,得到 $x_0$。

---

### 八、几个值得知道的细节

#### 8.1 为什么必须 $T$ 步?能不能跳步?

直接套这个采样过程,**必须走完整的 $T$ 步**——因为模型训练时假设了离散的 $t = 1, 2, \ldots, T$,逆向也按这个 schedule 走。

跳步不行,因为:

- 网络只在 $t \in \{1, \ldots, T\}$ 这些时刻"见过"
- 跳步意味着用错误的方差/均值组合

DDIM 后来重新设计了一个不依赖马尔可夫性的采样过程,**可以跳步,10-50 步搞定**。但那是另一个故事。

#### 8.2 这个采样方法叫什么?

正式名字叫 **Ancestral Sampling**(祖先采样)——按马尔可夫链定义的方向逐步采样。它是按字面意义"从 $p_\theta$ 采样"——所以是数学上"原汁原味"的方法。

#### 8.3 和 Langevin 动力学的对比

回忆第二讲里 NCSN 的 Langevin 采样:

$$x_{k+1} = x_k + \frac{\eta}{2}\, s_\theta(x_k) + \sqrt{\eta}\, z_k$$

DDPM 采样的形式:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t z$$

把 $\epsilon_\theta = -\sqrt{1-\bar\alpha_t}\, s_\theta$ 代入,展开后……

**两者形式上几乎一样!** 都是"当前点 + 学到的方向 + 一点噪声"。这绝非巧合——下一讲 Score SDE 会证明它们其实是**同一个 SDE 的不同离散化**。

---

### 九、要点回顾

- 起点 $\mathcal{N}(0, I)$:对应训练时 $L_T$ 的目标分布
- $\mu$ 公式:训练时的参数化定义,等价于"反推 $\hat x_0$ 后算后验均值"
- 加噪声 $\sigma_t z$:$p_\theta$ 是有方差的高斯,采样必须加
- 最后一步不加噪:工程优化,取 MAP
- 整个采样 = 按马尔可夫链反向运行 $p_\theta$,从噪声走到数据
- 每步直觉:**预测噪声 → 部分清除 → 留下少量给下一步**
- 形式上和 Langevin 动力学几乎一样——这预示了下一讲的统一框架


{% enddetails %}

---

## 九、完整 PyTorch 实现:从数据到生成

数学讲完了,这一节落到代码上。我们以 **CIFAR-10 / MNIST 上的图像生成**为例,从数据预处理一路写到采样,**每一步都标清楚输入、输出、形状、含义**。完整可运行版本约 200 行。

### 9.1 数据预处理

| 步骤 | 输入 | 输出 | 含义 |
|---|---|---|---|
| `ToTensor` | PIL `Image` | `Tensor` 形状 `(C, H, W)`,值 `[0, 1]` | 像素归一到 `[0, 1]` 并加通道维 |
| `Normalize((0.5,...), (0.5,...))` | `[0, 1]` | `[-1, 1]` | 把数据中心化到 0,**这一步对 diffusion 至关重要** |

为什么是 `[-1, 1]` 而不是 `[0, 1]`?因为前向 $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\, x_0, (1-\bar\alpha_t) I)$ 假设噪声是 0 均值的高斯。如果 $x_0 \in [0, 1]$,加上 0 均值高斯后会"变黑"(均值往 0 偏);中心化到 `[-1, 1]` 让 $x_0$ 的均值约为 0,加噪后**均值守恒为 0**,和先验 $\mathcal{N}(0, I)$ 对得上。

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # → [-1, 1]
])

train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
```

每个 batch 出来的 `x_0` 形状 `(128, 3, 32, 32)`,值在 `[-1, 1]`。

### 9.2 Noise schedule:预先算好的常数表

| 量 | 形状 | 含义 |
|---|---|---|
| `beta` | `(T,)` | $\beta_1, \ldots, \beta_T$,每步的加噪强度,**预设的非学习参数** |
| `alpha = 1 - beta` | `(T,)` | $\alpha_t = 1 - \beta_t$,每步的"信号保留比例" |
| `alpha_bar` | `(T,)` | $\bar\alpha_t = \prod_{s=1}^t \alpha_s$,从 $x_0$ 一步跳到 $x_t$ 的累积衰减 |
| `sqrt_alpha_bar` | `(T,)` | $\sqrt{\bar\alpha_t}$,$x_t$ 公式里 $x_0$ 的系数 |
| `sqrt_one_minus_alpha_bar` | `(T,)` | $\sqrt{1-\bar\alpha_t}$,$x_t$ 公式里 $\epsilon$ 的系数 |

这些都**只依赖 $t$**,训练前一次性算出来存好,后面查表用。

```python
T = 1000
beta = torch.linspace(1e-4, 0.02, T)               # 线性 schedule
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# 训练/采样要用的预计算量
sqrt_alpha_bar = torch.sqrt(alpha_bar)
sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
sqrt_recip_alpha = torch.sqrt(1.0 / alpha)         # 采样里的 1/√α_t

# 后验方差 σ_t² (取 β_t 这种简化设定)
posterior_var = beta.clone()
```

后来的工作发现 **cosine schedule** 更好(Improved DDPM),但线性 schedule 是最朴素也最稳的版本。

### 9.3 前向加噪 `q_sample`:核心一步公式

| 项 | 内容 |
|---|---|
| **输入** | $x_0$ 形状 `(B, C, H, W)`、$t$ 形状 `(B,)`(每个样本独立采的 timestep)、$\epsilon$ 形状同 $x_0$(可选) |
| **输出** | $x_t$,形状 `(B, C, H, W)` |
| **意义** | 把"$T$ 步加噪"塌缩成**一步公式** $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$,免去逐步采样 |

```python
def q_sample(x_0, t, eps=None):
    if eps is None:
        eps = torch.randn_like(x_0)
    # 取出对应 t 的系数,reshape 成 (B, 1, 1, 1) 以广播到 (B, C, H, W)
    a = sqrt_alpha_bar[t].view(-1, 1, 1, 1)
    b = sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
    return a * x_0 + b * eps
```

一行公式,**没有任何网络**,没有循环——这是高斯可加性带来的礼物。

### 9.4 时间嵌入:让网络知道当前是第几步

DDPM 的网络要**接受 $t$ 作为输入**,因为不同时刻去噪强度不同。直接把整数 $t$ 喂进去太"low-level",标准做法是**正弦位置编码**(和 Transformer 一样):

| 项 | 内容 |
|---|---|
| **输入** | $t$,形状 `(B,)`,整数 |
| **输出** | 嵌入向量,形状 `(B, dim)` |
| **意义** | 把离散的整数时间映射到一个连续、平滑、不同频率组成的向量,网络容易学 |

```python
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,)  →  output: (B, dim)
        half = self.dim // 2
        # 生成 half 个递减频率
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
```

实际使用时,通常再过两层 MLP:

```python
time_mlp = nn.Sequential(
    SinusoidalTimeEmbedding(128),
    nn.Linear(128, 512),
    nn.SiLU(),
    nn.Linear(512, 512),
)
# 调用: t_emb = time_mlp(t)   →  (B, 512)
```

这个 `t_emb` 会被注入到 UNet 的**每一层**,告诉每层"现在我们在第几步"。

### 9.5 UNet 架构

DDPM 的 noise predictor 是个 **UNet**:输入 `x_t` (加了噪的图)+ `t` (当前 timestep),输出 `ε_θ` (预测的噪声)——形状和输入一样。

| 项 | 内容 |
|---|---|
| **输入** | $x_t$ 形状 `(B, C, H, W)` + $t$ 形状 `(B,)` |
| **输出** | $\hat\epsilon$,形状 `(B, C, H, W)`,和输入完全一样 |
| **架构** | encoder(下采样路径) → bottleneck → decoder(上采样路径),每层带 skip connection |
| **关键设计** | 每个 ResBlock 都注入时间嵌入,让网络在不同 $t$ 下表现不同 |

最小可工作版的 UNet 模块:

```python
class ResBlock(nn.Module):
    """带时间条件的残差块"""
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))           # (B, out_ch, H, W)
        h = h + self.t_proj(t_emb)[:, :, None, None]    # 把时间信息加进 feature
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)                          # 残差
```

时间条件**怎么注入**?把 `t_emb` 通过 `Linear` 投影到 `out_ch` 维,reshape 成 `(B, out_ch, 1, 1)`,加到 feature map 上(每个空间位置加同一个时间偏移)。这是最常见的做法。

{% details 完整 UNet 骨架代码 %}

```python
class UNet(nn.Module):
    """简化版 UNet for 32×32 图像"""
    def __init__(self, in_ch=3, base_ch=64, t_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_ch),
            nn.Linear(base_ch, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        # Encoder (下采样)
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down1 = ResBlock(base_ch,     base_ch,     t_dim)   # 32×32
        self.down2 = ResBlock(base_ch,     base_ch * 2, t_dim)   # 16×16(下采样后)
        self.down3 = ResBlock(base_ch * 2, base_ch * 4, t_dim)   # 8×8

        # Bottleneck
        self.mid = ResBlock(base_ch * 4, base_ch * 4, t_dim)

        # Decoder (上采样,带 skip)
        self.up3 = ResBlock(base_ch * 4 + base_ch * 4, base_ch * 2, t_dim)
        self.up2 = ResBlock(base_ch * 2 + base_ch * 2, base_ch,     t_dim)
        self.up1 = ResBlock(base_ch     + base_ch,     base_ch,     t_dim)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, 3, padding=1),  # 输出和输入同形状
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)                       # (B, t_dim)

        x0 = self.in_conv(x)                           # (B, C, 32, 32)
        x1 = self.down1(x0, t_emb)
        x2 = self.down2(F.avg_pool2d(x1, 2), t_emb)    # (B, 2C, 16, 16)
        x3 = self.down3(F.avg_pool2d(x2, 2), t_emb)    # (B, 4C, 8, 8)

        m = self.mid(x3, t_emb)

        u3 = self.up3(torch.cat([m, x3], dim=1), t_emb)
        u3 = F.interpolate(u3, scale_factor=2)         # (B, 2C, 16, 16)
        u2 = self.up2(torch.cat([u3, x2], dim=1), t_emb)
        u2 = F.interpolate(u2, scale_factor=2)         # (B, C, 32, 32)
        u1 = self.up1(torch.cat([u2, x1], dim=1), t_emb)

        return self.out_conv(u1)                        # (B, in_ch, 32, 32)
```

实际生产级 UNet(Stable Diffusion、Imagen)还会加:

- **Self-Attention 块**:在低分辨率层(8×8、4×4)插自注意力,捕捉全局结构
- **多个 ResBlock 串联**:每个分辨率层放 2 个 ResBlock 而不是 1 个
- **EMA 模型权重**:保存一份指数平滑的权重用于推理,生成质量更好
- **Dropout**:在 ResBlock 里加一点(0.1)

但骨架就是上面这个:**编码下采样 → bottleneck → 解码上采样 + skip**,带时间条件。

{% enddetails %}

### 9.6 训练循环:5 行核心逻辑

整套 DDPM 训练就是把 $L_{\text{simple}}$ 实现出来:

```python
device = 'cuda'
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# 把所有 schedule tensor 搬到 GPU
beta = beta.to(device); alpha = alpha.to(device)
alpha_bar = alpha_bar.to(device)
sqrt_alpha_bar = sqrt_alpha_bar.to(device)
sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.to(device)

for epoch in range(epochs):
    for x_0, _ in train_loader:
        x_0 = x_0.to(device)                                       # (B, 3, 32, 32) ∈ [-1, 1]
        B = x_0.size(0)

        # 1. 每个样本随机一个 timestep
        t = torch.randint(0, T, (B,), device=device)                # (B,)

        # 2. 采噪声
        eps = torch.randn_like(x_0)                                 # (B, 3, 32, 32)

        # 3. 一步加噪到 x_t
        x_t = q_sample(x_0, t, eps)                                 # (B, 3, 32, 32)

        # 4. 网络预测噪声
        eps_pred = model(x_t, t)                                    # (B, 3, 32, 32)

        # 5. MSE 损失,一行
        loss = F.mse_loss(eps_pred, eps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

回头看每一步在做什么:

1. **采 $t$**:不同样本独立采 timestep,等价于对 ELBO 的 $\sum_t$ 做蒙特卡洛
2. **采 $\epsilon$**:这是真实"标签"——网络要预测的就是它
3. **`q_sample`**:用闭式公式造出 $x_t$,避免一步步加噪
4. **`model(x_t, t)`**:UNet 看着 $x_t$ 和当前时刻,猜里面藏的噪声
5. **MSE**:$L_{\text{simple}} = \|\epsilon - \epsilon_\theta\|^2$,直接 `F.mse_loss`

### 9.7 采样循环:逆向走 T 步

训练完后,从 $\mathcal{N}(0, I)$ 出发,逐步去噪到图:

| 项 | 内容 |
|---|---|
| **输入** | 形状 `shape = (n, 3, 32, 32)`,生成 n 张图 |
| **输出** | $x_0$,形状 `(n, 3, 32, 32)`,值大致 `[-1, 1]` |
| **意义** | 把训练学到的 $p_\theta(x_{t-1}\mid x_t)$ 按马尔可夫链反向运行 $T$ 次 |

```python
@torch.no_grad()
def sample(model, shape, device='cuda'):
    model.eval()
    x = torch.randn(shape, device=device)                           # x_T ~ N(0, I)

    for t in reversed(range(T)):                                    # t = T-1, ..., 0
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # 1. 网络预测此刻的噪声
        eps_pred = model(x, t_batch)                                # (n, 3, 32, 32)

        # 2. 算后验均值(训练时参数化的 μ_θ 公式)
        coef1 = 1.0 / torch.sqrt(alpha[t])
        coef2 = beta[t] / torch.sqrt(1.0 - alpha_bar[t])
        mean = coef1 * (x - coef2 * eps_pred)

        # 3. 加一点新噪声(最后一步不加)
        if t > 0:
            sigma = torch.sqrt(beta[t])                              # σ_t = √β_t
            z = torch.randn_like(x)
            x = mean + sigma * z
        else:
            x = mean

    return x  # x_0
```

每一步在做的事(对应"采样算法"那节的公式):

1. **`eps_pred = model(x, t_batch)`**:UNet 估一下当前 $x_t$ 里的噪声成分
2. **`mean = ...`**:用重参数化的 $\mu_\theta = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta)$ 公式算后验均值
3. **`x = mean + sigma * z`**:从高斯 $p_\theta(x_{t-1}\mid x_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 I)$ 采样
4. **最后一步 `t = 0`**:取 MAP(均值),不加噪——直接得到清晰图

### 9.8 把生成图变回去:反归一化

采样得到的 `x` 在 `[-1, 1]`,显示前要还原到 `[0, 1]`:

```python
def to_image(x):
    # x: (n, 3, 32, 32) ∈ [-1, 1]
    x = (x + 1) / 2          # → [0, 1]
    x = x.clamp(0, 1)        # 防止越界
    return x
```

### 9.9 整体流程图

```
训练:
                      时间 t ──→ time_mlp ──→ t_emb
                                                 │
                                                 ↓
  x_0 ──→ q_sample(t) ──→ x_t  ──UNet(x_t, t)──→ ε_θ
                          │                       │
                          └──→ ε  ───────MSE──────┘
                                                 │
                                                 ↓
                                              更新 θ

生成:
  x_T ~ N(0, I)
        │
        ├──→ for t = T-1, ..., 0:
        │        ε_θ = UNet(x_t, t)
        │        μ   = (x_t - β_t/√(1-ᾱ_t) · ε_θ) / √α_t
        │        x_{t-1} = μ + σ_t · z   (t=0 时不加噪)
        │
        ↓
       x_0
```

整个 DDPM 工程量不大——核心代码大约:

- 数据预处理 5 行
- noise schedule 10 行
- `q_sample` 5 行
- 时间嵌入 10 行
- UNet 80 行
- 训练循环 15 行
- 采样循环 20 行

合计 **不到 150 行**,就能在 CIFAR-10 上训出能看的生成图。生产级 Stable Diffusion 的复杂度主要堆在 UNet 的精细架构、attention、EMA、混合精度训练等工程上,**核心数学就是上面这些**。

{% details 训练超参数参考(DDPM 原版) %}

| 项 | CIFAR-10 默认值 | 备注 |
|---|---|---|
| 图像分辨率 | 32×32 | 也跑过 256×256 LSUN |
| Batch size | 128 | 跨 8 GPU 数据并行 |
| 优化器 | Adam | 没换 AdamW,β1=0.9, β2=0.999 |
| 学习率 | 2e-4 | 不用 warmup |
| Dropout | 0.1 | 在 ResBlock 内部 |
| EMA decay | 0.9999 | 推理用 EMA 权重而非原始权重 |
| 训练步数 | 800K | 32×32 大概 24 小时 8×V100 |
| $T$ | 1000 | 推理也走完整 1000 步 |
| schedule | linear $\beta_t$ from 1e-4 to 0.02 | DDPM 原版 |
| UNet base channels | 128 | 三层下采样,channel 倍率 [1, 2, 2, 2] |
| Attention 分辨率 | 16×16 | 在这一层加 self-attention |

后来 Improved DDPM 改进了几处:

- **cosine schedule** 替代 linear:训练更稳,FID 更好
- **学方差** $\Sigma_\theta$ 而非固定:似然估计更准
- **Importance sampling 选 $t$**:聚焦在高 loss 的 timestep

{% enddetails %}

---

## 十、复盘:为什么这一切如此优雅?

让我们回头看,DDPM 究竟做了什么:

1. **加了一个条件 $x_0$**:$q(x_{t-1} \mid x_t)$ 不闭式 → $q(x_{t-1} \mid x_t, x_0)$ 闭式
2. **用贝叶斯翻转方向**:让 ELBO 的每一项变成两个高斯的 KL
3. **套高斯 KL 闭式**:KL → MSE
4. **重参数化预测目标**:$\mu_\theta \to \epsilon_\theta$
5. **去掉权重**:得到 $L_{\text{simple}}$

每一步都是把"复杂、不闭式、方向不对"的形式替换成"简单、闭式、可训练"的形式。最终结果:

- 训练 loss 一行 MSE
- 训练 5 行代码
- 完全消除蒙特卡洛的内层方差
- 数值稳定、效果惊艳

**这就是 DDPM 比 Sohl-Dickstein 好的本质**——不是改了模型,而是**改了表达方式**。同一个模型,换个写法,从难训变成易训,从凑合到 SOTA。

---

## 十一、要点回顾

- DDPM 没有改 Sohl-Dickstein 的模型,只改了**表达方式**
- 关键引理:$q(x_{t-1} \mid x_t, x_0)$ 是闭式高斯,均值 $\tilde\mu_t$ 是 $x_0, x_t$ 的线性组合
- 贝叶斯重写:把 ELBO 改成"$L_T + \sum L_{t-1} + L_0$"的形式
- $L_T$ 是常数,$L_0$ 单独处理,**核心训练**是 $L_{t-1}$
- $L_{t-1}$ 是两个高斯 KL → 闭式 MSE on $\mu_\theta$
- 重参数化 $\mu_\theta \to \epsilon_\theta$,变成噪声预测
- $L_{\text{simple}}$ 去掉权重,纯 MSE
- $\epsilon$-预测和 score-预测只差一个标量,两条线在此对接


{% details 附录1：怎么用贝叶斯化简后验分布 q %}
这个推导是"高斯运算"的经典练习——纯代数,但每一步都要清楚。我会用**配方法**(completing the square)系统推完。这是 diffusion 推导里最技术性的一段,但理解了它,后面所有"高斯之间的运算"你都会做。

---

### 一、起点:贝叶斯展开

我们要推:

$$q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1})\, q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$$

三个高斯都已知:

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{\alpha_t} x_{t-1},\, \beta_t I)$$

$$q(x_{t-1} \mid x_0) = \mathcal{N}(x_{t-1};\, \sqrt{\bar\alpha_{t-1}} x_0,\, (1-\bar\alpha_{t-1}) I)$$

$$q(x_t \mid x_0) = \mathcal{N}(x_t;\, \sqrt{\bar\alpha_t} x_0,\, (1-\bar\alpha_t) I)$$

(其中 $\alpha_t = 1-\beta_t$,$\bar\alpha_t = \prod_{s=1}^t \alpha_s$)

**总策略**:取对数,只看依赖 $x_{t-1}$ 的项,把它整理成"$x_{t-1}$ 的二次型 + 常数",对照高斯标准形式读出均值和方差。

为什么这个策略行?因为:**任何关于 $x_{t-1}$ 的表达,如果它的对数是 $x_{t-1}$ 的二次型,那它就是关于 $x_{t-1}$ 的高斯**。高斯被均值和方差完全确定,所以读出二次型的系数就够了。

---

### 二、写出对数

取对数(乘除 → 加减):

$$\log q(x_{t-1} \mid x_t, x_0) = \log q(x_t \mid x_{t-1}) + \log q(x_{t-1} \mid x_0) - \log q(x_t \mid x_0)$$

第三项 $\log q(x_t \mid x_0)$ **不依赖 $x_{t-1}$**,作为常数处理(吸收到归一化里)。

剩下两项,代入高斯密度公式 $\log \mathcal{N}(z; \mu, \sigma^2 I) = -\frac{1}{2\sigma^2}\| z - \mu \|^2 + \text{const}$(为了简洁我用了一维记号,多维各向同性同理):

**第一项**:

$$\log q(x_t \mid x_{t-1}) = -\frac{1}{2\beta_t}\| x_t - \sqrt{\alpha_t} x_{t-1} \|^2 + \text{const}$$

**第二项**:

$$\log q(x_{t-1} \mid x_0) = -\frac{1}{2(1-\bar\alpha_{t-1})}\| x_{t-1} - \sqrt{\bar\alpha_{t-1}} x_0 \|^2 + \text{const}$$

合并:

$$\log q(x_{t-1} \mid x_t, x_0) = -\frac{1}{2}\left[\frac{\| x_t - \sqrt{\alpha_t} x_{t-1} \|^2}{\beta_t} + \frac{\| x_{t-1} - \sqrt{\bar\alpha_{t-1}} x_0 \|^2}{1-\bar\alpha_{t-1}}\right] + \text{const}$$

---

### 三、展开 $x_{t-1}$ 的二次型

我们的目标是把方括号里的部分整理成关于 $x_{t-1}$ 的形式:

$$A \| x_{t-1} \|^2 - 2 B^\top x_{t-1} + (\text{不依赖 } x_{t-1} \text{ 的常数})$$

读出 $A$ 和 $B$ 后,均值方差就出来了。

### 第一块展开

$$\frac{\| x_t - \sqrt{\alpha_t} x_{t-1} \|^2}{\beta_t} = \frac{1}{\beta_t}\left(\| x_t \|^2 - 2\sqrt{\alpha_t}\, x_t^\top x_{t-1} + \alpha_t \| x_{t-1} \|^2\right)$$

只看依赖 $x_{t-1}$ 的项:

$$= \frac{\alpha_t}{\beta_t}\| x_{t-1} \|^2 - \frac{2\sqrt{\alpha_t}}{\beta_t}\, x_t^\top x_{t-1} + \text{const}$$

### 第二块展开

$$\frac{\| x_{t-1} - \sqrt{\bar\alpha_{t-1}} x_0 \|^2}{1-\bar\alpha_{t-1}} = \frac{1}{1-\bar\alpha_{t-1}}\left(\| x_{t-1} \|^2 - 2\sqrt{\bar\alpha_{t-1}}\, x_0^\top x_{t-1} + \bar\alpha_{t-1}\| x_0 \|^2\right)$$

只看依赖 $x_{t-1}$ 的项:

$$= \frac{1}{1-\bar\alpha_{t-1}}\| x_{t-1} \|^2 - \frac{2\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\, x_0^\top x_{t-1} + \text{const}$$

### 合并两块

$x_{t-1}$ 的**二次项系数 $A$**:

$$A = \frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}}$$

$x_{t-1}$ 的**一次项系数 $B$**(指 $-2 B^\top x_{t-1}$ 的 $B$):

$$B = \frac{\sqrt{\alpha_t}}{\beta_t}\, x_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\, x_0$$

---

### 四、配方法读出均值和方差

回忆标准高斯的对数(忽略常数):

$$\log \mathcal{N}(x_{t-1}; \tilde\mu, \tilde\sigma^2 I) = -\frac{1}{2\tilde\sigma^2}\| x_{t-1} - \tilde\mu \|^2 = -\frac{1}{2\tilde\sigma^2}\left(\| x_{t-1} \|^2 - 2 \tilde\mu^\top x_{t-1}\right) + \text{const}$$

所以标准形式中的二次项系数和一次项 $B$ 满足:

- 二次项系数 = $\frac{1}{\tilde\sigma^2}$
- 一次项系数 $B$ = $\frac{\tilde\mu}{\tilde\sigma^2}$

(注意我们的合并公式整体外面有一个 $-\frac{1}{2}$,所以提取的就是这个对应关系)

对照得到:

$$\frac{1}{\tilde\sigma^2} = A,\quad \frac{\tilde\mu}{\tilde\sigma^2} = B$$

所以:

$$\boxed{\tilde\sigma^2 = \frac{1}{A},\quad \tilde\mu = \frac{B}{A} = \tilde\sigma^2 \cdot B}$$

---

### 五、化简 $\tilde\sigma^2$

$$\tilde\sigma^2 = \frac{1}{\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}}}$$

通分:

$$\tilde\sigma^2 = \frac{1}{\frac{\alpha_t (1-\bar\alpha_{t-1}) + \beta_t}{\beta_t (1-\bar\alpha_{t-1})}} = \frac{\beta_t (1-\bar\alpha_{t-1})}{\alpha_t (1-\bar\alpha_{t-1}) + \beta_t}$$

化简分母。用 $\beta_t = 1 - \alpha_t$:

$$\alpha_t (1-\bar\alpha_{t-1}) + \beta_t = \alpha_t - \alpha_t \bar\alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t \bar\alpha_{t-1}$$

而 $\alpha_t \bar\alpha_{t-1} = \bar\alpha_t$(因为 $\bar\alpha_t = \prod_{s=1}^t \alpha_s = \alpha_t \prod_{s=1}^{t-1} \alpha_s = \alpha_t \bar\alpha_{t-1}$),所以分母是 $1 - \bar\alpha_t$。

得到:

$$\boxed{\tilde\sigma^2 = \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\, \beta_t}$$

完美——这就是 DDPM 公式里的方差。

---

### 六、化简 $\tilde\mu_t$

$$\tilde\mu_t = \tilde\sigma^2 \cdot B = \frac{\beta_t (1-\bar\alpha_{t-1})}{1 - \bar\alpha_t}\left[\frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}} x_0\right]$$

把 $\tilde\sigma^2$ 分配进去:

**$x_t$ 的系数**:

$$\frac{\beta_t (1-\bar\alpha_{t-1})}{1 - \bar\alpha_t} \cdot \frac{\sqrt{\alpha_t}}{\beta_t} = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$$

**$x_0$ 的系数**:

$$\frac{\beta_t (1-\bar\alpha_{t-1})}{1 - \bar\alpha_t} \cdot \frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}} = \frac{\sqrt{\bar\alpha_{t-1}}\, \beta_t}{1-\bar\alpha_t}$$

合起来:

$$\boxed{\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\, \beta_t}{1-\bar\alpha_t}\, x_0 + \frac{\sqrt{\alpha_t}\, (1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\, x_t}$$

完美——这就是 DDPM 公式里的均值。✓

---

### 七、检查一致性(sanity check)

数学推导要会自检,看这个结果合不合理。

**检查 1**:$x_0$ 和 $x_t$ 的系数加起来应该接近什么?

让我们看极端情况。当 $t = 1$ 时,$\bar\alpha_{t-1} = \bar\alpha_0 = 1$,$\bar\alpha_t = \alpha_1$:

- $x_0$ 系数:$\frac{\sqrt{1} \cdot \beta_1}{1-\alpha_1} = \frac{\beta_1}{\beta_1} = 1$
- $x_t$ 系数:$\frac{\sqrt{\alpha_1}(1-1)}{1-\alpha_1} = 0$

所以 $\tilde\mu_1 = x_0$ —— 当 $t = 1$ 时,逆向后验完全集中在 $x_0$,**没有不确定性**(方差也应该是 0,我们检查:$\tilde\beta_1 = \frac{1-1}{1-\alpha_1}\beta_1 = 0$ ✓)。

**直觉解释**:已知 $x_0$ 和 $x_1$,要问 $x_0$ 是什么——当然就是 $x_0$ 本身,完全确定。

**检查 2**:当 $t$ 很大、$\bar\alpha_t \to 0$ 时:

- $x_0$ 系数 $\approx \sqrt{\bar\alpha_{t-1}} \beta_t \to 0$
- $x_t$ 系数 $\approx \sqrt{\alpha_t}$

也就是 $\tilde\mu_t \approx \sqrt{\alpha_t}\, x_t$。当噪声非常大时,$x_0$ 信息被噪声完全淹没,逆向后验主要由 $x_t$ 决定——也合理。

**检查 3**:方差 $\tilde\beta_t < \beta_t$ 吗?

$\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t} \beta_t$。因为 $\bar\alpha_{t-1} > \bar\alpha_t$,有 $1-\bar\alpha_{t-1} < 1-\bar\alpha_t$,所以 $\tilde\beta_t < \beta_t$ ✓。

**直觉**:已知 $x_0$ 和 $x_t$ 之后,我们对 $x_{t-1}$ 的不确定性,比单纯看前向加噪 $\beta_t$ 要小——多了 $x_0$ 这个信息,后验就更窄。合理。

---

### 八、技巧总结:配方法的精髓

整个推导的核心是**配方法**——给一个二次型,读出均值方差。让我把这个"模板"提炼出来,因为它在 diffusion / VAE / 高斯过程里反复出现。

**模板**:如果一个分布的对数密度(忽略常数)长这样

$$-\frac{1}{2}\left(A \| z \|^2 - 2 B^\top z\right)$$

那这个分布就是高斯,均值 $\mu = B/A$,方差 $\sigma^2 = 1/A$。

**实践三步**:

1. 把所有"高斯乘积/比值"取对数,变成一堆二次型相加
2. 收集所有 $\| z \|^2$ 的系数得 $A$,所有 $z^\top$ 的系数得 $B$
3. 读出 $\mu = B/A$,$\sigma^2 = 1/A$,完成

整个过程**不需要任何创意**,纯机械。学会这个,你以后看 diffusion 各种高斯运算都不会怕。

#### 一个更几何的视角

更深一层:这个推导本质是"两个高斯的乘积还是高斯"——更准确地说,**两个关于 $x_{t-1}$ 的高斯的乘积,得到一个新的关于 $x_{t-1}$ 的高斯,新的精度(precision = 方差的倒数)是两个精度之和**。

具体到我们这里:

- $q(x_t \mid x_{t-1})$ 看作 $x_{t-1}$ 的函数,精度是 $\alpha_t / \beta_t$
- $q(x_{t-1} \mid x_0)$ 看作 $x_{t-1}$ 的函数,精度是 $1/(1-\bar\alpha_{t-1})$
- 乘积后的精度 = 两者之和 = $A$

这就是为什么贝叶斯推断里,**精度具有"可加性",而方差不具备**——后验精度 = 先验精度 + 似然精度。这个原理在贝叶斯线性回归、卡尔曼滤波里都是同一回事。

---

### 九、要点回顾

- 推导工具:**配方法**——把对数密度整理成关于变量的二次型,读出均值方差
- 三个高斯(前向 1 步、前向 $t-1$ 步、前向 $t$ 步)通过贝叶斯组合
- 第三个($q(x_t \mid x_0)$)不依赖 $x_{t-1}$,扔进常数
- 二次项系数 $A$ 和一次项系数 $B$ 决定 $\tilde\mu = B/A$,$\tilde\sigma^2 = 1/A$
- 关键化简:$1 - \alpha_t \bar\alpha_{t-1} = 1 - \bar\alpha_t$
- 检查极端情况($t=1$、$t$ 很大)能验证公式合理
- 几何视角:贝叶斯组合 = 精度相加

{% enddetails %}


{% details 附录2：怎么化简单步的 KL 散度 %}


### 一、先回顾我们在哪

DDPM 把负 ELBO 改写成:

$$-\mathcal{L} = \mathbb{E}_q\left[L_T + \sum_{t=2}^T L_{t-1} + L_0\right]$$

其中核心训练项是

$$L_{t-1} = D_{\text{KL}}\big(q(x_{t-1} \mid x_t, x_0) \,\big\|\, p_\theta(x_{t-1} \mid x_t)\big)$$

我们已经知道两个分布是什么:

- $q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde\mu_t(x_t, x_0),\, \tilde\beta_t I)$ ← 闭式高斯,刚推过
- $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t))$ ← 网络输出

**问题**:怎么计算两个高斯之间的 KL?

---

### 二、两个高斯的 KL 闭式公式

我们之前提过这个公式,但没真正用过。现在用最简化的版本:**两个各向同性、方差相同的高斯**。

设 $p = \mathcal{N}(\mu_1, \sigma^2 I)$,$q = \mathcal{N}(\mu_2, \sigma^2 I)$,都是 $d$ 维。则:

$$D_{\text{KL}}(p \| q) = \frac{\| \mu_1 - \mu_2 \|^2}{2\sigma^2}$$

**这是个奇迹般简洁的公式**——两个相同方差的高斯之间的 KL,**就是均值之差的平方,除以 $2\sigma^2$**。

为什么这么简单?因为方差相同时,KL 公式里的 $\log(\sigma_2/\sigma_1)$ 项和 $\sigma_1^2/(2\sigma_2^2) - 1/2$ 项都消掉了,只剩均值差。

---

### 三、DDPM 的简化:固定方差

要套用这个简洁公式,**两个高斯必须有相同的方差**。但:

- $q(x_{t-1} \mid x_t, x_0)$ 的方差是 $\tilde\beta_t I$(已知,闭式)
- $p_\theta(x_{t-1} \mid x_t)$ 的方差是 $\Sigma_\theta(x_t, t)$(网络输出)

DDPM 做了一个**简化设计**:**让 $p_\theta$ 的方差也固定**,不学,直接设成

$$\Sigma_\theta(x_t, t) = \sigma_t^2 I$$

其中 $\sigma_t^2$ 是预设的常数(论文实验了 $\sigma_t^2 = \beta_t$ 和 $\sigma_t^2 = \tilde\beta_t$,效果差不多)。

**为什么可以这样简化?**

- $\tilde\beta_t$ 已经是个**完全闭式**且**不依赖 $x_0, x_t$** 的量——它只依赖 $t$ 和 noise schedule
- 既然方差已经"事先知道",网络再去学一个就是浪费
- 理论上稍微损失一点表达力,但实践中效果反而更好(更稳定)

**简化之后**,两个高斯方差都是 $\sigma_t^2 I$(假设取 $\sigma_t^2 = \tilde\beta_t$),套上面的简洁公式:

$$L_{t-1} = D_{\text{KL}}\big(q \,\big\|\, p_\theta\big) = \frac{\| \tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t) \|^2}{2\sigma_t^2}$$

如果两边方差不完全相等(比如 $\sigma_t^2 = \beta_t$ 但 $q$ 的方差是 $\tilde\beta_t$),会多出一些常数项,但**不依赖 $\theta$**,可以塞进 const:

$$L_{t-1} = \frac{1}{2\sigma_t^2}\| \tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t) \|^2 + C$$

---

### 四、关键转折:KL 变成了 MSE

让我们停下来,欣赏这个时刻。

**之前**:训练 loss 是 $D_{\text{KL}}(q \| p_\theta)$,听起来是个抽象的"分布之间的距离"。

**现在**:训练 loss 是 $\| \tilde\mu_t - \mu_\theta \|^2$,**就是两个向量之间的均方误差**——和回归问题完全一样!

具体说,网络要做的事变成了:**给定 $x_t$ 和 $t$,输出一个向量 $\mu_\theta(x_t, t)$,让它逼近真实后验均值 $\tilde\mu_t(x_t, x_0)$**。

这就是为什么我说这一步是"魔法"——一个看起来很复杂的概率论目标,化成了最朴素的 MSE 回归。

---

### 五、$\tilde\mu_t$ 是什么 vs $\mu_\theta$ 是什么

为了让这一步更具体,我们看清楚两边到底是什么:

| | 表达式 | 来源 |
|---|---|---|
| $\tilde\mu_t(x_t, x_0)$ | $\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} x_t$ | 闭式公式,用 $x_0, x_t$ 算出 |
| $\mu_\theta(x_t, t)$ | 由神经网络输出 | 给定 $(x_t, t)$,网络预测 |

**目标**:训练 $\mu_\theta$,让它在所有 $(x_0, x_t)$ 组合下都接近 $\tilde\mu_t(x_t, x_0)$。

注意一个关键事实:**$\tilde\mu_t$ 依赖 $x_0$,而 $\mu_\theta$ 只能看到 $x_t$**。所以 $\mu_\theta$ 必须**从 $x_t$ 反推 $x_0$ 的信息**——这就是去噪。

---

### 六、训练这个 MSE 怎么操作?

来一遍完整的训练步骤,落实到代码层面:

```python
# 一次训练迭代
x_0 = sample_data()                                           # 从数据集采
t   = uniform(2, T)                                           # 采时刻
eps = randn_like(x_0)                                         # 采噪声

# 用一步公式造出 x_t(重参数化)
x_t = sqrt(α_bar[t]) * x_0 + sqrt(1 - α_bar[t]) * eps

# 算真实后验均值 μ̃_t(用闭式)
target_mu = (sqrt(α_bar[t-1]) * β[t]) / (1 - α_bar[t]) * x_0 \
          + (sqrt(α[t]) * (1 - α_bar[t-1])) / (1 - α_bar[t]) * x_t

# 网络输出 μ_θ
pred_mu = network(x_t, t)

# MSE loss
loss = ||target_mu - pred_mu||^2 / (2 * σ_t^2)

loss.backward()
```

这就是把 $L_{t-1}$ 落实到训练里的**全部内容**——一个标准的回归任务。

---

### 七、为什么 DDPM 还要再走一步?

到这里其实已经能训练了,而且原则上和最终的 DDPM 是等价的。但**实践中"预测 $\mu$"比"预测 $\epsilon$"差**,原因是:

#### 原因 1:数值尺度问题

$\tilde\mu_t$ 是 $x_0$ 和 $x_t$ 的加权和,**它的数值范围接近 $x_t$ 本身**——也就是说,大部分输出就是"把 $x_t$ 输出回去"。网络要做的"修正"只是一个小偏移。

让网络输出"几乎是输入本身 + 小修正" 是低效的——网络的大部分容量被花在"复刻输入"上。

#### 原因 2:$\epsilon$ 的尺度更"标准化"

对比一下,如果让网络预测 $\epsilon$:

$$\epsilon \sim \mathcal{N}(0, I)$$

——它的尺度永远是 0 均值、单位方差,**不依赖 $t$**。所有时刻 $t$ 的目标分布尺度统一,网络更好训练。

#### 原因 3:残差结构

预测 $\epsilon$ 在数学上等价于"网络预测残差"——即"$x_t$ 里的噪声成分是什么",然后从 $x_t$ 减掉。这是经典的残差学习思想,普遍比"直接学输入到输出的映射"更稳定。

所以 DDPM 做了**重参数化**:把网络的预测目标从 $\mu_\theta$ 换成 $\epsilon_\theta$,数学等价,但训练效果好得多。

---

### 八、所以这一步的本质是什么?

回到你问的:"$L_{t-1}$ 化简核心一步"到底在做什么?

**它做了三件事**:

1. **认识到 $L_{t-1}$ 是两个高斯的 KL**——两边都已经被设计成高斯了
2. **简化模型**:固定 $p_\theta$ 的方差为 $\sigma_t^2 I$,让两边方差相同
3. **套用"相同方差高斯 KL = 均值差平方"的闭式公式**——把 KL 变成 MSE

**结果**:训练 loss 从抽象的"两个分布的距离"变成了具体的"两个向量的 MSE"。

这是 DDPM 让训练变简单的**最关键一步**——但其实它的本质就是"利用我们之前讲过的'高斯 KL 有闭式',把分布问题降维成回归问题"。

---

### 九、和我们之前讨论的串起来

回头看,这一切都是之前讨论的"闭式 vs 蒙特卡洛"故事的最终兑现:

| 阶段 | loss 的形式 | 怎么算 |
|---|---|---|
| Sohl-Dickstein | $\log \frac{q(x_t \mid x_{t-1})}{p_\theta(x_{t-1} \mid x_t)}$ | 蒙特卡洛(方向不一致,无法 KL) |
| DDPM 改写后 | $D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \,\Vert\, p_\theta(x_{t-1} \mid x_t))$ | 闭式 KL(都是高斯,方差还相同) |
| DDPM 化简后 | $\Vert \tilde\mu - \mu_\theta \Vert^2$ | MSE,直接回归 |

每一步都在**降低 loss 的计算复杂度**:从需要采样估计,到闭式 KL,到 MSE 回归。

---

### 十、要点回顾

- **KL 变 MSE 的魔法**:两个相同方差高斯的 KL 公式只剩均值差平方,$D_{\text{KL}} = \| \mu_1 - \mu_2 \|^2 / (2\sigma^2)$
- DDPM 简化:**固定 $p_\theta$ 的方差**为 $\sigma_t^2 I$,让两边方差相同(代价小,收益大)
- 简化后,$L_{t-1}$ 变成 $\| \tilde\mu_t - \mu_\theta \|^2$ 的 MSE
- 训练目标:网络从 $x_t$ 出发,预测真实后验均值 $\tilde\mu_t$
- 但这还不是最终形式——$\mu$-预测有数值尺度问题,后面用重参数化换成 $\epsilon$-预测会更好

{% enddetails %}


{% details 附录3：采样公式的推导 %}


这是个非常好的问题——采样这一步看起来"显然就该这样写",但每一行背后其实都有具体的数学依据。我们一行一行拆解,你会看到它**根本不是凭空设计的,而是每一步都从训练时的数学严格推出来的**。

---

### 一、先把采样代码贴出来

```
x_T ~ N(0, I)                                 # 起点:纯噪声
for t = T, T-1, ..., 1:
    z = N(0, I) if t > 1 else 0               # 最后一步不加噪声
    μ = (1/sqrt(α_t)) * (x_t - β_t/sqrt(1-α_bar_t) * eps_θ(x_t, t))
    x_{t-1} = μ + σ_t * z
return x_0
```

四个值得问"为什么"的地方:

1. 为什么从 $\mathcal{N}(0, I)$ 出发?
2. 那个 $\mu$ 公式怎么来的?
3. 为什么再加一个 $\sigma_t z$ 噪声?
4. 为什么最后一步($t=1$)不加噪声?

我们逐个拆。

---

### 二、为什么从 $\mathcal{N}(0, I)$ 出发?

回忆训练目标 $L_T$:

$$L_T = D_{\text{KL}}\big(q(x_T \mid x_0) \,\big\|\, p(x_T)\big)$$

当 $T$ 大、$\bar\alpha_T \to 0$ 时,$q(x_T \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_T} x_0, (1-\bar\alpha_T) I) \to \mathcal{N}(0, I)$。

我们定义先验 $p(x_T) = \mathcal{N}(0, I)$。这两者几乎相同,所以 $L_T \approx 0$,我们不训练它。

**采样时**:既然真实的 $x_T$ 分布(无论 $x_0$ 是什么)都接近 $\mathcal{N}(0, I)$,那我们从 $\mathcal{N}(0, I)$ 采样作为起点,就是从训练分布采样——**不是任意选的"起点",而是数学上对应的那个分布**。

---

### 三、那个 $\mu$ 的公式怎么来的?

这是采样里最关键的一行。我们在训练时已经设计了:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(\mu_\theta(x_t, t),\, \sigma_t^2 I\big)$$

所以采样时,从这个高斯采 $x_{t-1}$。**问题是 $\mu_\theta$ 等于什么?**

回忆我们在训练阶段做了**重参数化**——让 $\mu_\theta$ 取以下形式:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\, \epsilon_\theta(x_t, t)\right)$$

**这个公式不是采样时凭空写出的——它是训练时网络的参数化定义本身**。我们就是这样让网络的输出(通过 $\epsilon_\theta$ 间接)给出 $\mu_\theta$ 的。

所以采样代码里那行

```
μ = (1/sqrt(α_t)) * (x_t - β_t/sqrt(1-α_bar_t) * eps_θ(x_t, t))
```

就是把训练时的参数化定义代入,**给定网络预测的 $\epsilon_\theta$,算出 $\mu_\theta$**。这一步纯粹是代数代入,没有新东西。

#### 这个公式的直观

让我们看清楚它在做什么。回忆 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$。如果网络完美预测 $\epsilon_\theta = \epsilon$,那么:

**第一步**:从 $x_t$ 反推 $x_0$:

$$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\, \epsilon_\theta}{\sqrt{\bar\alpha_t}}$$

这是把"加噪公式"反过来用。

**第二步**:算逆向后验均值 $\tilde\mu_t(x_t, \hat x_0)$。直接套真实后验的公式:

$$\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\hat x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} x_t$$

**两步合并**(代入 $\hat x_0$,化简——这就是 $\tilde\mu_t \to \epsilon$ 那步代数化简的逆向使用),得到那个 $\mu$ 公式。

**所以采样的 $\mu$ 步可以理解为**:

> "用网络预测噪声 → 反推一个 $\hat x_0$ 估计 → 用真实后验公式算 $x_{t-1}$ 的均值"

每次去噪一步,**间接地是在估计 $x_0$ 然后向 $x_0$ 走一步**。

---

### 四、为什么还要再加 $\sigma_t z$ 噪声?

这是最容易让初学者疑惑的一步——既然要去噪,为什么还要加噪声?

#### 4.1 数学原因

我们的逆向模型是

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 I)$$

——一个**有方差的高斯**,不是只输出均值。**从这个分布采样**就是"均值 + 标准差 × 标准高斯",即:

$$x_{t-1} = \mu_\theta + \sigma_t z,\quad z \sim \mathcal{N}(0, I)$$

代码里的 `μ + σ_t * z` 就是这个采样。所以这个噪声**本来就是逆向高斯分布的一部分**,没加就不是从 $p_\theta$ 采样了。

#### 4.2 直觉原因

为什么模型设计成有方差?因为**真实的逆向后验 $q(x_{t-1} \mid x_t, x_0)$ 也是有方差的高斯**($\tilde\beta_t I$)——不是确定性映射。

直观看:已知 $x_t$ 和 $x_0$,$x_{t-1}$ **不是唯一确定的**——它有一个分布的可能值。$\tilde\beta_t$ 衡量了这种不确定性。

更直观:前向加噪是随机过程,撤销它每一步也应该是随机的——给定 $x_t$,可能的"上一步" $x_{t-1}$ 有多种。如果逆向是确定的,我们就丢失了随机性,**最终生成的 $x_0$ 永远只有一个**(给定起点),失去多样性。

#### 4.3 物理类比

想象前向是"墨水扩散到水里"。如果你能逆转每个分子的运动,逆向也是确定的——但实际上,信息在扩散过程中**以热力学方式丢失**,逆向就必须有随机性来"补偿"丢失的信息。

每一步采样加的 $\sigma_t z$,正是"补偿"前向那一步丢的信息。

#### 4.4 一个常被问的细节:$\sigma_t$ 应该取什么?

DDPM 论文里实验了两种选择:

- $\sigma_t^2 = \beta_t$(前向方差)
- $\sigma_t^2 = \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$(真实后验方差)

两者效果相近。前者对应"假设 $x_0 \sim \mathcal{N}(0, I)$"的逆向后验,后者对应"假设 $x_0$ 是确定值"的逆向后验。实践中默认用前者(更简单)。

后来 DDIM(我们之后讲)发现:**$\sigma_t$ 可以是任意值,甚至 $\sigma_t = 0$**(完全确定性采样),仍然能生成正确分布的样本——这是个重要的发现,会让采样可以加速 10-50 倍。

---

### 五、为什么最后一步($t=1$)不加噪声?

代码里:

```
z = N(0, I) if t > 1 else 0
```

也就是当 $t = 1$ 时,$z = 0$,我们只取 $\mu$,不加噪声。

### 5.1 数学原因

最后一步是从 $x_1$ 生成 $x_0$。$x_0$ 是**最终输出**,我们希望它是图像本身,而不是"图像 + 一点点噪声"。

更精确地,DDPM 里 $L_0 = -\log p_\theta(x_0 \mid x_1)$ 是单独处理的——它对应真实数据的对数似然。在最后一步,我们直接取**最大似然估计**,即 $\mu_\theta$ 本身,不再加噪声。

### 5.2 直观原因

如果最后一步加噪声 $\sigma_1 z$,你的图像就会带一个最后的小噪声。$\sigma_1$ 虽然小,但对生成图像的清晰度还是有影响。

把这一步噪声去掉,相当于"在最后做个 MAP(maximum a posteriori)估计"——取分布的众数(高斯的众数就是均值),作为最佳点估计。

### 5.3 这是个工程选择

严格说,**完全按 $p_\theta$ 采样应该最后一步也加噪声**。"$t=1$ 时不加"是个**实践上的优化**——理论上稍微偏离了概率模型,但生成质量更好。

后来 Improved DDPM、DDIM 等都采用了这个约定。

---

### 六、把整个采样过程串起来

让我把所有"为什么"放在一起,画出完整图景:

```
起点 x_T = N(0, I)             ← 因为前向终点近似就是 N(0, I)

for t = T, ..., 1:
    
    用网络预测 ε_θ(x_t, t)
    
    算 μ = ...                  ← 训练时的参数化定义,等价于
                                  "反推 x_0 估计 → 算逆向后验均值"
    
    if t > 1:
        x_{t-1} = μ + σ_t z     ← 从 p_θ(x_{t-1}|x_t) 采样
                                  (有方差是因为真实后验也有方差)
    else:
        x_0 = μ                 ← 最后一步取最大似然估计

return x_0
```

每一步都不是"魔法",而是训练阶段数学结构的直接对应:

- 起点 ↔ $L_T$ 的训练目标
- $\mu$ 公式 ↔ 训练时的参数化设计
- 加噪声 ↔ $p_\theta$ 是高斯分布
- 最后不加噪 ↔ $L_0$ 的处理约定

---

### 七、采样到底在干什么?(更高层理解)

最后一句话总结整个采样的"灵魂":

**采样是把训练时学到的"逆向条件分布 $p_\theta(x_{t-1} \mid x_t)$"按马尔可夫链**反向运行**,从 $\mathcal{N}(0, I)$ 一步步走到数据分布 $p_{\text{data}}$。**

每一步本质上是:

> "网络看着 $x_t$,猜里面的噪声 $\epsilon$,然后:
>  - 大致知道'清掉 $\epsilon$ 后应该长什么样'
>  - 但不完全清掉,只走一小步,留一点噪声给下一步处理
>  - 这就是 $x_{t-1}$"

走 $T$ 步,最终噪声彻底清完,得到 $x_0$。

---

### 八、几个值得知道的细节

#### 8.1 为什么必须 $T$ 步?能不能跳步?

直接套这个采样过程,**必须走完整的 $T$ 步**——因为模型训练时假设了离散的 $t = 1, 2, \ldots, T$,逆向也按这个 schedule 走。

跳步不行,因为:

- 网络只在 $t \in \{1, \ldots, T\}$ 这些时刻"见过"
- 跳步意味着用错误的方差/均值组合

DDIM 后来重新设计了一个不依赖马尔可夫性的采样过程,**可以跳步,10-50 步搞定**。但那是另一个故事。

#### 8.2 这个采样方法叫什么?

正式名字叫 **Ancestral Sampling**(祖先采样)——按马尔可夫链定义的方向逐步采样。它是按字面意义"从 $p_\theta$ 采样"——所以是数学上"原汁原味"的方法。

#### 8.3 和 Langevin 动力学的对比

回忆第二讲里 NCSN 的 Langevin 采样:

$$x_{k+1} = x_k + \frac{\eta}{2}\, s_\theta(x_k) + \sqrt{\eta}\, z_k$$

DDPM 采样的形式:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t z$$

把 $\epsilon_\theta = -\sqrt{1-\bar\alpha_t}\, s_\theta$ 代入,展开后……

**两者形式上几乎一样!** 都是"当前点 + 学到的方向 + 一点噪声"。这绝非巧合——下一讲 Score SDE 会证明它们其实是**同一个 SDE 的不同离散化**。

---

### 九、要点回顾

- 起点 $\mathcal{N}(0, I)$:对应训练时 $L_T$ 的目标分布
- $\mu$ 公式:训练时的参数化定义,等价于"反推 $\hat x_0$ 后算后验均值"
- 加噪声 $\sigma_t z$:$p_\theta$ 是有方差的高斯,采样必须加
- 最后一步不加噪:工程优化,取 MAP
- 整个采样 = 按马尔可夫链反向运行 $p_\theta$,从噪声走到数据
- 每步直觉:**预测噪声 → 部分清除 → 留下少量给下一步**
- 形式上和 Langevin 动力学几乎一样——这预示了下一讲的统一框架

{% enddetails %}


{% details 附录4：为什么不直接一步预测 x_0 %}


这是个非常深刻的问题——你的直觉其实抓到了一个真实的"路径选择"。答案是:**确实可以这么做,而且这就是 DDIM 的核心思想**!但 DDPM 故意没这么做,原因值得仔细讲清楚。

---

### 一、先把你的想法精确化

你的直觉是:既然网络预测了 $\epsilon$,而且 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$,那直接反解就能拿到 $x_0$:

$$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$$

**一步到 $x_0$,为什么还要走 $T$ 步?**

这个想法完全对——而且实践中是可行的。但有微妙的问题。让我们看清楚。

---

### 二、第一种"直接"采样:一步式

### 2.1 直接公式

$$\hat x_0 = \frac{x_T - \sqrt{1-\bar\alpha_T}\, \epsilon_\theta(x_T, T)}{\sqrt{\bar\alpha_T}}$$

#### 2.2 它能 work 吗?

理论上,如果 $\epsilon_\theta$ 完美预测了 $\epsilon$,**这就是真实的 $x_0$**。

实际上,**几乎完全不能用**。原因如下。

#### 2.3 为什么不行?(这是关键)

回想训练时的损失:

$$L = \mathbb{E}_{x_0, t, \epsilon}\left[\| \epsilon - \epsilon_\theta(x_t, t) \|^2\right]$$

这个损失训练出的 $\epsilon_\theta$ 是 $\epsilon$ 的**条件均值估计**,即:

$$\epsilon_\theta(x_t, t) \approx \mathbb{E}[\epsilon \mid x_t]$$

(MSE 损失的最优解永远是条件均值——这是统计学的基本结论)

注意 $x_t$ 是**多个不同 $x_0$ 都能到达**的。给定一个 $x_t$,真实生成它的 $x_0$ **不唯一**——可能有许多个 $x_0$ 都"恰好"通过某个 $\epsilon$ 到达这个 $x_t$。$\epsilon$ 在这些可能性上**有一个分布**。

网络只能学这个分布的**均值**——也就是"在所有可能的 $\epsilon$ 中取平均"。

**当 $t$ 很大时,$x_t$ 和 $x_0$ 的关系几乎完全被噪声主导**——给定一个 $x_T \sim \mathcal{N}(0, I)$,**几乎任何 $x_0$ 都有可能生成它**(只是概率不同)。所以 $\mathbb{E}[\epsilon \mid x_T]$ 几乎不带 $x_0$ 的信息。

**结果**:从 $x_T$ 直接反推的 $\hat x_0$ ≈ "所有数据的平均"——一张模糊的、所有图像取平均的灰图。

这是个具体的、可观察的现象,Tweedie 公式给出了精确解释。我们看一下。

---

### 三、Tweedie 公式视角

Tweedie 公式说:

$$\mathbb{E}[x_0 \mid x_t] = \frac{1}{\sqrt{\bar\alpha_t}}\left(x_t + (1-\bar\alpha_t)\nabla_{x_t} \log q(x_t)\right)$$

把 score 和 $\epsilon$ 的关系 $\epsilon_\theta = -\sqrt{1-\bar\alpha_t}\, s_\theta$ 代入:

$$\mathbb{E}[x_0 \mid x_t] = \frac{x_t - \sqrt{1-\bar\alpha_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$$

**这正是你想"直接减"得到的 $\hat x_0$**!

所以**这个 $\hat x_0$ 不是真实的 $x_0$,而是 $x_0$ 在给定 $x_t$ 下的条件均值**——即"所有可能的 $x_0$ 的加权平均"。

| $t$ 时刻 | 含义 |
|---|---|
| $t$ 小($x_t$ 接近 $x_0$) | $\mathbb{E}[x_0 \mid x_t] \approx x_0$,反推几乎精确 |
| $t$ 大($x_t$ 接近噪声) | $\mathbb{E}[x_0 \mid x_t] \approx \mathbb{E}[x_0]$,得到"平均图像"——一团模糊 |

**所以从纯噪声 $x_T$ 一步反推不行,因为 $\mathbb{E}[x_0 \mid x_T]$ 没有信息**,只能给你"训练集的平均长相"。

---

### 四、为什么走 $T$ 步就 work?

关键洞察:**每一步只走一小步,$\epsilon_\theta$ 的预测就足够好用了**。

具体看:从 $x_t$ 走到 $x_{t-1}$,而不是直接到 $x_0$。

$\mathbb{E}[\epsilon \mid x_t]$ 仍然是条件均值——但这一步的"目标"是从 $x_t$ 走到 $x_{t-1}$,只前进 $\beta_t$ 那么一点点。在这一小步内:

- 即使条件均值"模糊",对应的"平均"也只是一小步范围内的平均
- 留下大量信息给后续步骤来精修

**多步走的物理直觉**:从 $x_T$ 想猜 $x_0$ 完全不可能,但猜"$x_T$ 上一步是什么样"是可行的——只要走得足够小。然后基于新的 $x_{T-1}$ 继续猜下一步,信息逐步注入。

每一步都是局部的、合理的预测,$T$ 步累积下来,把"完全没信息"的 $x_T$ 逐步变成"完全有信息"的 $x_0$。

---

### 五、一个更精确的图景

让我用一个比喻。设想你要画一只猫,但只能闭着眼"扫"一遍画布。

**方法 1(一步式)**:闭眼一次,猜整张画的所有像素。结果:平均起来"像猫"的概率密度——一团灰雾。

**方法 2(多步式)**:每次只扫一小块,基于已经画好的部分推下一块。每一步只决定一点点,但每一点都基于充分的局部信息。最终整张画连贯。

DDPM 是方法 2。每一步的"预测目标"很小,网络可以做出合理判断;$T$ 步累积出一张连贯的图像。

**核心数学事实**:从噪声分布到数据分布,**直接的概率映射**是高度非线性、多模态的——一个 $x_T$ 对应所有可能的 $x_0$,**不是函数**。但**逐步**的概率映射 $x_t \to x_{t-1}$ 几乎是线性的(高斯之间),网络容易学。

---

### 六、所以"直接减"完全不能用吗?

不是。让我们看几个**部分采用了你的想法**的方法。

#### 6.1 DDIM(Denoising Diffusion Implicit Models, 2020)

DDIM 的核心思想就是:**确实利用 $\hat x_0$,但不直接跳到 $x_0$**。

DDIM 的更新规则:

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\, \hat x_0(x_t) + \sqrt{1-\bar\alpha_{t-1}}\, \epsilon_\theta(x_t, t)$$

**含义**:

- 用网络反推一个 $\hat x_0$ 估计
- 然后**重新加噪**到 $t-1$ 时刻——用同一个 $\epsilon_\theta$
- 得到 $x_{t-1}$

这看起来很奇怪——估出 $\hat x_0$ 后又加回噪声?但这有两个好处:

1. **可以跳步**:不依赖马尔可夫性,$x_t$ 和 $x_{t-k}$ 之间可以直接跳
2. **确定性**:没有随机噪声 $z$(对比 DDPM 的 $\sigma_t z$),完全可逆

DDIM 让采样从 1000 步降到 20-50 步,质量几乎不变。

#### 6.2 一步生成:Consistency Models(2023)

Yang Song 等人的 **Consistency Models** 真的实现了"直接一步从 $x_T$ 跳到 $x_0$"——但代价是要**重新训练**一个网络,让它**学会**这个一步映射。

他们的训练目标是:让网络在所有 $t$ 上,直接预测一个固定的 $x_0$(而不是预测 $\epsilon$):

$$f_\theta(x_t, t) \approx x_0\quad \forall t$$

这绕开了"一步反推不准"的问题——网络从训练时就被强迫学这个一步映射,而不是从一个本来不擅长的目标(预测 $\epsilon$)拿来"硬反推"。

效果:CIFAR-10 上单步生成质量接近 DDPM 1000 步。

#### 6.3 其他

- **DPM-Solver**(2022):用高阶 ODE 数值求解器,4-10 步生成
- **Distillation 方法**:把 DDPM 1000 步的"教师"知识蒸馏到 1-4 步的"学生"

所有这些方法的共同点:**$\hat x_0$ 这个量是核心工具**,只是用法不同。

---

### 七、回到你最初的疑问:DDPM 为什么"绕一圈"?

现在你应该看清楚了:

1. **直接减一步反推**确实是个自然想法——$\hat x_0$ 这个量是真实存在的,而且有 Tweedie 公式的精确意义
2. **但它只是 $x_0$ 的"条件均值"**,不是真正的 $x_0$——所以一步直接到 $x_0$ 得到的是模糊的均值图像
3. **DDPM 选择走 $T$ 步**,每步只前进一点点,让"条件均值"足够准
4. **每一步内部**确实可以理解为"先反推 $\hat x_0$ 再算 $x_{t-1}$"——这就是 DDPM 公式的等价解读
5. **DDIM/Consistency** 用更聪明的方式利用 $\hat x_0$,大幅减少步数

所以你的直觉抓到的事:**$\hat x_0$ 是 diffusion 采样的核心量**——只是 DDPM 选择"间接使用"它来获得稳定性,DDIM 等改进在使用方式上做了文章。

---

### 八、一个直观演示

让我们想象一个具体的实验。设 $T = 1000$,你训练好了一个 DDPM。

**实验 1**:对一张测试图像 $x_0$,做前向到 $x_{500}$。然后:

- **直接反推**:$\hat x_0^{(500)} = \frac{x_{500} - \sqrt{1-\bar\alpha_{500}}\,\epsilon_\theta}{\sqrt{\bar\alpha_{500}}}$
- 结果:$\hat x_0^{(500)}$ 是个**比较模糊但能看出大致内容**的图像

**实验 2**:对 $x_{999}$(几乎纯噪声)做同样反推:

- 结果:$\hat x_0^{(999)}$ 是一团**几乎看不出任何具体内容**的雾

**实验 3**:从 $x_T \sim \mathcal{N}(0, I)$(任意噪声)直接反推:

- 结果:**所有 $x_T$ 反推都给出几乎一样的雾**——因为 $\mathbb{E}[x_0 \mid x_T] \approx \mathbb{E}[x_0]$ 不依赖 $x_T$

这就是为什么 DDPM 不能一步采样——但**它的多步采样里,每一步都隐含地包含了这个反推**。

---

### 九、要点回顾

- "直接减预测的噪声"得到的是 $\mathbb{E}[x_0 \mid x_t]$,**不是 $x_0$**(Tweedie 公式)
- 这个量在 $t$ 大时几乎没有 $x_t$ 的信息——是"训练集的平均图像"
- DDPM 的多步采样:每步只前进一点点,让条件均值足够准
- **每步内部确实在做"反推 $\hat x_0$"** ——只是这个 $\hat x_0$ 是临时估计,不直接输出
- DDIM、Consistency Models 等改进:更聪明地利用 $\hat x_0$,大幅减少步数
- 你的直觉抓到了 diffusion 加速研究的**核心入口**

{% enddetails %}


{% details 附录5：x_0 估计不准也可以用 %}

**是的,而且这正是为什么 diffusion 能 work 的根本原因**。让我把这个机制彻底讲清楚。

### 一、先精确表述你的洞察

你的问题翻译成数学:

> 即使 $\hat x_0(x_t) := \mathbb{E}[x_0 \mid x_t]$ 是个"模糊的、不准确的"估计,我们仍然把它代入 $\tilde\mu_t$ 的公式来算 $x_{t-1}$。这居然能 work?

**答案:能,而且这不是巧合,是数学上必然的**。

---

### 二、为什么"不准确的 $\hat x_0$"在概率意义下是对的

关键事实:**我们不需要 $\hat x_0$ 等于真实的 $x_0$,我们只需要 $p_\theta(x_{t-1} \mid x_t)$ 等于真实的 $q(x_{t-1} \mid x_t)$**。

这两件事是不同的!让我们分开看。

#### 2.1 我们想要的是什么?

我们想让模型分布 $p_\theta(x_{0:T})$ 等于真实分布 $q(x_{0:T})$,这等价于让每个**条件分布**都对:

$$p_\theta(x_{t-1} \mid x_t) = q(x_{t-1} \mid x_t)\quad \forall t$$

**注意是 $q(x_{t-1} \mid x_t)$,不是 $q(x_{t-1} \mid x_t, x_0)$**。前者是真实的逆向过程(没有 $x_0$ 信息),后者是我们推出闭式的那个。

#### 2.2 真实的 $q(x_{t-1} \mid x_t)$ 是什么?

它是对所有 $x_0$ 边际化:

$$q(x_{t-1} \mid x_t) = \int q(x_{t-1} \mid x_t, x_0)\, q(x_0 \mid x_t)\, dx_0$$

这是个**混合高斯**——以 $q(x_0 \mid x_t)$ 为权重,把所有 $q(x_{t-1} \mid x_t, x_0)$ 平均起来。

通常**不是高斯**——因为 $q(x_0 \mid x_t)$ 是一个复杂分布(后验数据分布)。

#### 2.3 但 DDPM 用单个高斯逼近它!

这正是 Sohl-Dickstein 那个 Feller 定理保证的事:**当 $\beta_t$ 足够小时,$q(x_{t-1} \mid x_t)$ 近似为高斯**。

具体地,可以证明:

$$q(x_{t-1} \mid x_t) \approx \mathcal{N}\left(\tilde\mu_t(x_t,\, \mathbb{E}[x_0 \mid x_t]),\, \tilde\beta_t I\right)$$

**这是核心**!真实的逆向后验,**用 $\mathbb{E}[x_0 \mid x_t]$(即条件均值,即"模糊的 $\hat x_0$")代入 $\tilde\mu_t$ 公式,得到的高斯就是它的最佳高斯逼近**。

也就是说:

> **"模糊的 $\hat x_0$"代入 $\tilde\mu_t$ 公式,得到的 $\mu$ 不是某个具体 $x_0$ 对应的逆向均值——它是真实逆向分布的均值,本身就是混合高斯的均值。**

---

### 三、关键洞察:训练目标实际上"自动追踪"这个最佳逼近

让我们看 DDPM 的训练目标究竟在做什么。

#### 3.1 训练目标的等价形式

我们之前推过:

$$L_{t-1} = \mathbb{E}_{x_0, x_t}\left[\| \tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t) \|^2\right] / (2\sigma_t^2)$$

注意这是**对 $x_0$ 求期望**——不同的 $x_0$ 给出不同的 $\tilde\mu_t$ 目标,网络要学的是这些目标的"平均"。

MSE 损失的最优解永远是条件均值:

$$\mu_\theta^*(x_t, t) = \mathbb{E}_{x_0 \mid x_t}[\tilde\mu_t(x_t, x_0)]$$

(给定 $x_t$,在所有可能的 $x_0$ 上,$\tilde\mu_t$ 的平均)

#### 3.2 这个条件均值等于什么?

由于 $\tilde\mu_t$ 是 $x_0$ 的**线性函数**(回忆公式:$\tilde\mu_t = c_1 x_0 + c_2 x_t$,$c_1, c_2$ 不依赖 $x_0$),**期望可以"穿过"**:

$$\mathbb{E}_{x_0 \mid x_t}[\tilde\mu_t(x_t, x_0)] = c_1\, \mathbb{E}[x_0 \mid x_t] + c_2 x_t = \tilde\mu_t(x_t, \mathbb{E}[x_0 \mid x_t])$$

**所以最优网络输出**:

$$\boxed{\mu_\theta^*(x_t, t) = \tilde\mu_t(x_t,\, \mathbb{E}[x_0 \mid x_t])}$$

——**正是把"模糊的条件均值 $\hat x_0$"代入 $\tilde\mu_t$ 公式得到的结果**。

#### 3.3 这意味着什么?

我们从来没有**显式**让网络学条件均值。我们只是让 $\mu_\theta$ 去回归各种 $\tilde\mu_t(x_t, x_0)$ 目标。但因为 MSE 损失的性质,**最优解自动等于"代入条件均值"的形式**。

这是 diffusion 数学最优雅的事实之一——**用 $\hat x_0 = \mathbb{E}[x_0 \mid x_t]$ 作为 $x_0$ 的"代理"不是近似,而是 ELBO 优化的精确闭式解**。

---

### 四、为什么这能 work?用一个简化例子

让我用一个极端简化的例子说清楚。

#### 4.1 设定

假设数据只有两个点:$x_0 = +1$ 或 $x_0 = -1$,各占一半。前向加噪到 $x_t$。

#### 4.2 一个具体的 $x_t$

假设我们采到 $x_t = 0$(正中间)。这个 $x_t$ 既可能来自 $x_0 = +1$,也可能来自 $x_0 = -1$,**几率各半**。

#### 4.3 真实的逆向后验 $q(x_{t-1} \mid x_t = 0)$

它是两个高斯的混合:

- 以概率 1/2 来自"$x_0 = +1$ 的链",对应 $q(x_{t-1} \mid x_t=0, x_0=+1)$
- 以概率 1/2 来自"$x_0 = -1$ 的链",对应 $q(x_{t-1} \mid x_t=0, x_0=-1)$

$q(x_{t-1} \mid x_t = 0)$ 是个**双峰分布**——不是单一高斯。

#### 4.4 DDPM 的近似:用单个高斯

DDPM 用 $p_\theta(x_{t-1} \mid x_t = 0) = \mathcal{N}(\mu_\theta, \sigma^2)$ ——单个高斯。它当然**不能完美匹配**双峰分布。

但 MSE 训练让 $\mu_\theta$ 最优地落在**两个峰的中点**——也就是

$$\mu_\theta^* = \mathbb{E}[x_{t-1} \mid x_t = 0] = \tilde\mu_t(0, \mathbb{E}[x_0 \mid x_t=0]) = \tilde\mu_t(0, 0)$$

(因为 $\mathbb{E}[x_0 \mid x_t=0] = 0$,正中间)

#### 4.5 这够好吗?

对于**单步**:不够——$p_\theta$ 是单峰高斯,而真实 $q$ 是双峰。如果直接从这个 $p_\theta$ 一步采样,得到的样本是均值 0 附近的高斯——**不是 $\pm 1$**。

**但走多步之后**:

随着 $t$ 减小($x_t$ 离真实数据越来越近),$\hat x_0(x_t) = \mathbb{E}[x_0 \mid x_t]$ 会**逐渐倾向于某一边**——如果当前的 $x_t$ 偏向 +1,$\hat x_0$ 就接近 +1;偏向 -1 类似。

具体说,逆向链每走一步,$x_t$ 会因为随机噪声 $\sigma_t z$ **稍微偏移**;偏移后 $\hat x_0$ 的"偏向"被放大;再下一步 $\hat x_0$ 更偏;以此类推。

**多步累积下来,初始的随机性会逐步把 $x_t$ 推向其中一个真实的 $x_0$**——形成"对称破缺"。

最终采样到的 $x_0$ 就在 $\pm 1$ 上,**符合真实的双峰分布**——尽管每一步都用单峰高斯逼近!

---

### 五、这就是 diffusion 多步采样的精髓

把上面这件事提炼成原则:

**单峰高斯不能一步表达多峰分布,但多步逐渐"破缺"可以。**

每一步:

- 用 $p_\theta = \mathcal{N}(\mu, \sigma^2)$ 单峰逼近真实的 $q(x_{t-1} \mid x_t)$
- $\mu$ 是在当前 $x_t$ 下"所有可能 $x_0$ 的平均"
- 每一步加的 $\sigma_t z$ 噪声把 $x_t$ 推向其中一个具体的 $x_0$ 的分支
- 下一步基于新 $x_t$ 重新算"平均",这个平均逐步偏向某一支

**$T$ 步累积下来,从一个对所有 $x_0$ 都"模糊兼容"的 $x_T$,逐步分化到具体的某个 $x_0$**。

---

### 六、回到你的问题:为什么"不准确"也能用?

现在最直接地回答:

> **$\hat x_0$ 不准确,但它是无偏的最佳估计**——具体地,它是**所有可能 $x_0$ 的概率加权平均**。

> **代入 $\tilde\mu_t$ 公式得到的 $\mu_\theta$,正是真实逆向后验的均值**——这是混合分布的均值,而不是某个具体分支的均值。

> **每一步用这个均值 + 一点随机噪声采样,等价于"平均地"探索所有可能的逆向路径**。

> **多步累积后,具体某条路径胜出**(被随机性放大),最终生成的 $x_0$ 是真实分布中的具体一个样本。

**关键不是 $\hat x_0$ 准不准,而是它在数学上是 ELBO 的最优解,这就够了**。

---

### 七、用一个更深的视角:Score 视角

回忆 $\hat x_0$ 和 score 的关系(Tweedie):

$$\hat x_0 = \frac{x_t + (1-\bar\alpha_t)\, s_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$$

也就是 $\hat x_0$ 本质上**就是 score 的一个等价表达**。

而 score $\nabla_{x_t} \log q(x_t)$ 的几何意义是"指向数据流形的方向"——它不是"指向某个具体的 $x_0$",而是"指向密度增大的整体方向"。

所以:

- 在大噪声时($x_t$ 离数据远),score "指向整团数据"——具体到不了某一个 $x_0$
- 在小噪声时($x_t$ 接近数据),score "指向局部具体的高密度区域"——逐渐锁定某个 $x_0$
- 多步采样 = 沿着 score 指引的方向逐步走 + 噪声扰动来"选择分支"

这是 score-based 视角,Score SDE 会让这件事更精确。

---

### 八、和你之前问题的串联

我们一路问下来的几个问题,在这里全部串起来了:

| 之前的问题 | 现在的答案 |
|---|---|
| "为什么改写 ELBO?" | 改写后 loss 变成 MSE,**MSE 最优解是条件均值**——这正是我们想要的 |
| "为什么 KL 要闭式?" | 闭式让 MSE 形式干净,自动追踪条件均值 |
| "$\hat x_0$ 不是 $x_0$,为什么用它?" | **$\hat x_0$ 在数学上就是 ELBO 最优 $\mu_\theta$ 的来源**——它是"模糊"的,但模糊正确 |
| "为什么走 $T$ 步?" | 单步无法表达多峰分布,多步用随机性破缺,逐步从模糊到具体 |
| "采样为什么加 $\sigma_t z$?" | 这个噪声驱动"对称破缺",让 $x_t$ 逐步偏向一个具体分支 |

**所有这些选择都不是孤立的,它们共同构成一个自洽的概率框架**——MSE 损失逼着网络学条件均值,条件均值给出最优单步逼近,多步采样用随机性把"模糊"分解成"具体",最终得到真实数据分布的样本。

---

### 九、要点回顾

- $\hat x_0$ 是 $\mathbb{E}[x_0 \mid x_t]$,**不是真实 $x_0$**——是所有可能 $x_0$ 的概率加权平均
- 但代入 $\tilde\mu_t$ 公式后,得到的 $\mu_\theta^*$ **正是真实逆向后验 $q(x_{t-1} \mid x_t)$ 的均值**——这是 MSE 损失的最优解性质保证的
- 单步 $p_\theta$ 是单峰高斯,**不能精确表达多峰真实后验**——但 ELBO 让它成为最优的单峰逼近
- 多步采样用随机噪声 $\sigma_t z$ **驱动对称破缺**——把"对所有 $x_0$ 模糊兼容"的状态逐步分化到具体某个 $x_0$
- $T$ 步累积后,采到的样本符合真实数据分布——尽管每一步都只是单峰高斯
- 这就是为什么 diffusion 能 work:**每步不准确不重要,$T$ 步联合起来精确**

{% enddetails %}


{% details 附录6：为什么要求KL散度有闭式解？ %}

---

### 一、问题的核心:有**两个**KL,你混淆了

让我先把问题挑明。在 diffusion 数学里出现的 KL **不是同一个**——有两个层次,作用完全不同。一个**不需要**闭式,一个**需要**闭式。

#### KL 类型 A:外层 KL(在恒等式里)

$$\log p_\theta(x_0) = \text{ELBO} + D_{\text{KL}}(q(x_{1:T} \mid x_0) \,\|\, p_\theta(x_{1:T} \mid x_0))$$

这个 KL 出现在"似然 = ELBO + KL"这个**恒等式**里。它的角色是:

- **理论分析**:解释 ELBO 是下界、解释下界的间隙是什么
- **不需要计算**:训练时我们**根本不算它**,只算 ELBO

所以这个 KL **不需要闭式**——它只是个理论概念,告诉我们 ELBO 优化的本质。我们无需关心它具体值多少。

#### KL 类型 B:内层 KL(ELBO 化简后出现)

DDPM 把 ELBO 重写成:

$$-\text{ELBO} = \mathbb{E}_q\left[\sum_{t=2}^T \underbrace{D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t))}_{\text{KL 类型 B}} + \cdots\right]$$

这个 KL 出现在 **ELBO 的内部**——是 ELBO 化简后的求和项。它的角色是:

- **每次训练都要算它**(因为它是 loss 的一部分)
- **必须能高效计算**

这个 KL **需要闭式**——闭式才能算得快、梯度精确。

---

### 二、所以两者的关系是

让我用一张图表示清楚:

```
log p(x_0)  =  ELBO  +  D_KL(q‖p)   ← 外层 KL(类型 A,理论用,不算)
              ↓
            ELBO 内部 = Σ E[ D_KL(...) ] + ...   ← 内层 KL(类型 B,训练算,要闭式)
                          ↑
                    这里的 KL 才需要闭式
```

**关键区分**:

| | 外层 KL(类型 A) | 内层 KL(类型 B) |
|---|---|---|
| 出现位置 | 恒等式 $\log p = \text{ELBO} + \text{KL}$ | ELBO 化简后的求和项 |
| 是 谁 vs 谁 的 KL | 完整变分分布 $q(x_{1:T}\mid x_0)$ vs 真实后验 $p_\theta(x_{1:T}\mid x_0)$ | 单步前向后验 $q(x_{t-1}\mid x_t,x_0)$ vs 单步逆向 $p_\theta(x_{t-1}\mid x_t)$ |
| 是否需要计算 | 不需要 | 每次训练都要 |
| 是否需要闭式 | 不需要 | 需要(否则训练慢/方差大) |
| 性质 | 高维($T$ 维联合分布之间) | 低维($d$ 维单步分布之间) |

---

### 三、为什么内层 KL 必须高效计算?

回到训练流程。每次训练一步:

1. 采一个数据点 $x_0$
2. 采一个时刻 $t$
3. 采一个噪声 $\epsilon$,得到 $x_t$
4. **计算 loss**(就是某个 $t$ 的内层 KL 项)
5. 反向传播

第 4 步每次迭代都要做几百万次(整个训练过程)。如果它不闭式:

- 要嵌套蒙特卡洛(里面再采样估计)→ 慢
- 估计有方差 → 训练不稳
- 梯度不精确 → 收敛慢

**所以内层 KL 必须闭式**——这是工程必需。

---

### 四、那"蒙特卡洛估计 ELBO"到底估的是什么?

回到你的疑问:既然要蒙特卡洛估 ELBO,为什么还要闭式?

答案是:**蒙特卡洛和闭式估的是不同维度的随机性**。我们**两者都在用**——把它们组合起来才得到最终的训练 loss。

#### ELBO 的完整结构

DDPM 改写后:

$$\text{ELBO} = -\mathbb{E}_{q}\left[\sum_{t=2}^T D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t)) + \cdots\right]$$

注意有**两层结构**:

- **外层**:对 $q$ 的期望 $\mathbb{E}_q[\cdots]$ —— 这层用蒙特卡洛
- **内层**:每个 KL —— 这层用闭式

#### 训练时怎么算?

**外层(蒙特卡洛)**:从 $q$ 采样 $x_t$、随机选 $t$,得到一个具体的 KL 项。这是单样本 MC 估计。

**内层(闭式)**:对这个具体的 $(x_t, x_0, t)$,直接用高斯 KL 闭式公式算出 KL 的精确值——不需要再采样。

具体地,对一个 batch 里的样本 $(x_0, t, \epsilon)$:

$$\text{loss} = \underbrace{\frac{\| \tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t) \|^2}{2\sigma_t^2}}_{\text{KL 闭式公式,无需采样}}$$

整个 loss 对**这一组** $(x_0, t, \epsilon)$ 而言是个**确定性的、可微的、精确的**值。多个样本平均就是对外层期望的 MC 估计。

---

### 五、对比 Sohl-Dickstein:为什么"内层闭式"是关键

现在你能精确看懂 Sohl-Dickstein 和 DDPM 的差异了:

#### Sohl-Dickstein 的形式

$$\text{ELBO} = \mathbb{E}_q\left[\sum_t \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1})}\right]$$

每一项 $\log \frac{p_\theta}{q}$ **依赖于** $x_{t-1}$ 和 $x_t$ **的具体值**——它本身就是个随机变量。

**没有"内层闭式"——每一项都是采样得到的随机数**。

外层蒙特卡洛要估这个随机变量的期望,但它方差大,估计困难。

#### DDPM 的形式

$$\text{ELBO} = \mathbb{E}_q\left[\sum_t D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))\right]$$

每一项是个 **KL**,不依赖于具体的 $x_{t-1}$ 取值——它只依赖 $x_t, x_0$,且**有闭式**。

**有"内层闭式"——每一项是确定性的精确值**(给定 $x_t, x_0$)。

外层蒙特卡洛只对 $x_t$ 的随机性估,因为内层已经被解析积掉了——方差大幅降低。

---

### 六、用一个具体类比讲清楚

想象你要估这个嵌套期望:

$$J = \mathbb{E}_X\left[\mathbb{E}_Y[f(X, Y) \mid X]\right]$$

**方法 1**:不用闭式,纯蒙特卡洛。采 $X_i$、再采 $Y_{i,j}$,平均所有 $f(X_i, Y_{i,j})$。两层采样,方差大。

**方法 2**:如果内层期望 $g(X) := \mathbb{E}_Y[f(X, Y) \mid X]$ **有闭式**,直接采 $X_i$,平均 $g(X_i)$。一层采样,方差小。

这就是 **Rao-Blackwell 化**——能解析积的就解析积,不要乱采样。

**回到 diffusion**:外层期望(对 $x_0, x_t$ 的随机性)用蒙特卡洛——这没办法,只能采;内层期望(每个 KL 项中"对 $x_{t-1}$ 的随机性")**用闭式直接算**——这是降方差的关键。

---

### 七、再用代码看一眼

```python
# 训练一步
x_0 = sample_data()
t   = uniform(1, T)
eps = randn_like(x_0)

# 第一层随机性 → 蒙特卡洛(就这一组采样)
x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * eps

# 第二层"随机性"原本要积掉 → 用闭式 KL 公式直接算
mu_q     = posterior_mean(x_0, x_t, t)    # 真实后验均值,闭式
mu_theta = network(x_t, t)                 # 网络输出
loss     = 0.5 * ||mu_q - mu_theta||^2 / sigma**2   # 高斯 KL 闭式

loss.backward()
```

看到了吗?

- 外层(`x_0, t, eps`):蒙特卡洛
- 内层(`loss`):闭式公式,**不需要再采样估计**

如果内层不闭式,这行 `loss = ...` 就要变成"再做一次蒙特卡洛估计",训练会慢得多、噪声大得多。

---

### 八、总结:你的混淆是怎么解开的

回到你最初的问题:

> "我们不是用蒙特卡洛估计去估计 ELBO 的吗?为什么要求 KL 有闭式解?"

**答案**:

1. ELBO 里**有两层结构**:外层期望 + 内层每项
2. **外层**:用蒙特卡洛估(对 $x_0, t, x_t$ 的随机性)
3. **内层每项**:**最好是闭式**,这样不用再嵌套一层蒙特卡洛
4. 闭式 + 蒙特卡洛是**配合关系,不是二选一**——把外层的"无法避免的随机性"留给 MC,把内层的"能算的部分"用闭式精确算掉
5. "$\log p = \text{ELBO} + \text{KL}$"中的 KL(**外层 KL**)只是理论恒等式,不计算,不需要闭式
6. ELBO 化简后出现的 KL(**内层 KL**)才是训练时真正算的,**必须闭式**

---

### 九、要点回顾

- diffusion 数学里**有两个 KL**,一定要分清:外层(理论用)和内层(训练算)
- 蒙特卡洛 ≠ 闭式的对立面,**两者是配合关系**
- Rao-Blackwell 化:能闭式就闭式,降方差靠这个
- DDPM 比 Sohl-Dickstein 强的本质:**让内层有闭式**,把"两层蒙特卡洛"减少到"一层蒙特卡洛"
- 训练 loss 对单个样本而言是**确定性、精确、可微**的——多样本平均才是 MC 估计

---

清楚了吗?这个区分非常微妙,但理解了它你就抓到了 diffusion 数学优雅性的真正命门。

{% enddetails %}

继续推进的话,正好可以看一下 DDPM 的贝叶斯改写——你会**亲眼看到**"外层 KL"如何在等式中出现,"内层 KL"如何被造出来,以及高斯 KL 闭式如何把内层化简成 MSE。要不要现在就推这一段?