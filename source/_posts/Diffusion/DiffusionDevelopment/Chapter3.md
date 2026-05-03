---
title: Chapter 3 DDPM 模型
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

---

## 九、复盘:为什么这一切如此优雅?

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

## 十、要点回顾

- DDPM 没有改 Sohl-Dickstein 的模型,只改了**表达方式**
- 关键引理:$q(x_{t-1} \mid x_t, x_0)$ 是闭式高斯,均值 $\tilde\mu_t$ 是 $x_0, x_t$ 的线性组合
- 贝叶斯重写:把 ELBO 改成"$L_T + \sum L_{t-1} + L_0$"的形式
- $L_T$ 是常数,$L_0$ 单独处理,**核心训练**是 $L_{t-1}$
- $L_{t-1}$ 是两个高斯 KL → 闭式 MSE on $\mu_\theta$
- 重参数化 $\mu_\theta \to \epsilon_\theta$,变成噪声预测
- $L_{\text{simple}}$ 去掉权重,纯 MSE
- $\epsilon$-预测和 score-预测只差一个标量,两条线在此对接


## 附录1：
好,这个推导是"高斯运算"的经典练习——纯代数,但每一步都要清楚。我会用**配方法**(completing the square)系统推完。这是 diffusion 推导里最技术性的一段,但理解了它,后面所有"高斯之间的运算"你都会做。

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



## 附录2：


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
| DDPM 改写后 | $D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))$ | 闭式 KL(都是高斯,方差还相同) |
| DDPM 化简后 | $\| \tilde\mu - \mu_\theta \|^2$ | MSE,直接回归 |

每一步都在**降低 loss 的计算复杂度**:从需要采样估计,到闭式 KL,到 MSE 回归。

---

### 十、要点回顾

- **KL 变 MSE 的魔法**:两个相同方差高斯的 KL 公式只剩均值差平方,$D_{\text{KL}} = \| \mu_1 - \mu_2 \|^2 / (2\sigma^2)$
- DDPM 简化:**固定 $p_\theta$ 的方差**为 $\sigma_t^2 I$,让两边方差相同(代价小,收益大)
- 简化后,$L_{t-1}$ 变成 $\| \tilde\mu_t - \mu_\theta \|^2$ 的 MSE
- 训练目标:网络从 $x_t$ 出发,预测真实后验均值 $\tilde\mu_t$
- 但这还不是最终形式——$\mu$-预测有数值尺度问题,后面用重参数化换成 $\epsilon$-预测会更好







## 附录3：为什么要求KL散度有闭式解？

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
| 是 谁 vs 谁 的 KL | 完整变分分布 $q(x_{1:T}\|x_0)$ vs 真实后验 $p_\theta(x_{1:T}\|x_0)$ | 单步前向后验 $q(x_{t-1}\|x_t,x_0)$ vs 单步逆向 $p_\theta(x_{t-1}\|x_t)$ |
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

继续推进的话,正好可以看一下 DDPM 的贝叶斯改写——你会**亲眼看到**"外层 KL"如何在等式中出现,"内层 KL"如何被造出来,以及高斯 KL 闭式如何把内层化简成 MSE。要不要现在就推这一段?