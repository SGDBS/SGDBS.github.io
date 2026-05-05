---
title: Chapter 3. Sohl-Dickstein Model
categories: Diffusion Models
date: 2026-05-01 12:00:00
mathjax: true
tags:
    - AI
    - Diffusion Models
---


## 1. 模型框架的全貌


这是 diffusion model 真正的"鼻祖"论文:"Deep Unsupervised Learning using Nonequilibrium Thermodynamics"。

核心思想来自统计物理:任何复杂分布 q(x0)q(x_0)
q(x0​),只要慢慢加噪声,最终都会变成简单分布(高斯)。如果能学会逆转这个过程,就能从高斯采样回到数据分布。

Sohl-Dickstein 的模型有两个过程,**一个固定、一个学习**。

### 1.1. 前向过程(固定,不学习)

把数据 $x_0 \sim q(x_0)$ 逐步加噪,经过 $T$ 步变成纯噪声:

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I)$$

整个前向链:

$$q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})$$

**关键性质**:

- $\beta_t \in (0, 1)$ 是预设的 noise schedule(比如线性从 $10^{-4}$ 到 $0.02$)
- 系数 $\sqrt{1-\beta_t}$ 保证方差不爆炸:若 $\text{Var}(x_{t-1}) = 1$,则 $\text{Var}(x_t) = 1$
- $T$ 足够大时,$x_T \approx \mathcal{N}(0, I)$,与 $x_0$ 无关
- **没有任何参数**——这一步设计完就锁死

### 1.2. 逆向过程(要学习)

从纯噪声 $x_T \sim \mathcal{N}(0, I)$ 出发,逐步去噪生成 $x_0$:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t))$$

整个逆向链:

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)$$

其中 $p(x_T) = \mathcal{N}(0, I)$ 是固定的。

**为什么逆向也用高斯?** 这是 Sohl-Dickstein 引用 Feller (1949) 的关键洞察:**当前向每步 $\beta_t$ 足够小时,真实的逆向条件分布 $q(x_{t-1} \mid x_t)$ 也近似为高斯**。所以用高斯参数化是合理的。

直觉解释:每步加的噪声很小,所以"撤销"这一步只需要小幅修正,小幅修正的不确定性也是高斯的。如果一步加大量噪声,逆向分布就不再是高斯(可能多峰),那就学不动了。

**这一点决定了 diffusion 必须用很多步**——这是它和 GAN 一步生成的本质区别。

{% details 为什么要用高斯分布 %}

**理论上不必须,但高斯是被多重约束逼出来的几乎唯一选择**。让我把这些约束一条条摊开。

### 一、先回答"能不能不用高斯"

**理论上完全可以**。Sohl-Dickstein 2015 原论文里就讨论了两种:

- **高斯扩散**:用于连续数据(图像、音频)
- **二项扩散**:用于二值数据(每像素 0/1)

后来还有人做过:

- **分类扩散**(Categorical Diffusion):用于离散数据(token、文本)——D3PM (2021)、Argmax Flow 等
- **泊松扩散**:用于计数数据
- **流形扩散**(Manifold Diffusion):在球面、SO(3) 等流形上
- **泡沫扩散、混合扩散**:研究性质

**所以高斯不是必须的——它是一个选择。**

但当你看大多数主流 diffusion 模型(DDPM、Stable Diffusion、FLUX、Sora 等)时,几乎全是高斯。这是因为高斯有**一系列其他分布凑不齐的好性质**。

### 二、为什么选高斯?七大理由

### 理由 1:高斯之间的 KL 有闭式

这是最关键的工程理由。回忆 DDPM 训练目标的核心化简:

$$D_{\text{KL}}(\mathcal{N}(\mu_1, \sigma^2 I) \| \mathcal{N}(\mu_2, \sigma^2 I)) = \frac{\|\mu_1 - \mu_2\|^2}{2\sigma^2}$$

这个公式让 ELBO 化简成 MSE。**换成其他分布族**:

| 分布 | KL 闭式? | 简洁度 |
|------|---------|--------|
| 高斯 | ✓ | 极简(就是均值差的平方) |
| 伯努利、分类 | ✓ | 还行(熵 + 交叉熵) |
| 学生 t、Cauchy、混合分布 | ✗ | 没有闭式 |
| 高斯 vs 非高斯 | ✗ | 没有闭式 |

很多分布有 KL 闭式,但**没有哪个能像高斯一样让 KL 简化成"均值差平方"这么干净**。

### 理由 2:高斯之和还是高斯(可加性)

如果 $X_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$ 和 $X_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)$ 独立,那么

$$X_1 + X_2 \sim \mathcal{N}(\mu_1 + \mu_2,\, \sigma_1^2 + \sigma_2^2)$$

**这个性质让 diffusion 能算闭式边际**——多步累积的噪声"塌缩"成一个等效的单步高斯。

如果换成别的分布:

- 伯努利之和是二项分布(不再是伯努利) ✗
- Cauchy 之和还是 Cauchy ✓ 但 Cauchy 没有方差,无法度量"信号"
- 拉普拉斯之和不是拉普拉斯 ✗
- 学生 t 之和不是学生 t ✗

**几乎只有高斯同时满足"和封闭"且"有有限方差"**——这就是它在概率论里的特殊地位。

### 理由 3:重参数化超级干净

$$z \sim \mathcal{N}(\mu, \sigma^2) \iff z = \mu + \sigma \epsilon,\quad \epsilon \sim \mathcal{N}(0, 1)$$

这是个**线性、可逆、可微**的变换。**线性重参数化让所有 diffusion 公式保持简洁**——比如 $x_t$ 是 $x_0$ 和 $\epsilon$ 的**线性**组合,这个事实在推导反向公式、分析理论性质时都至关重要。

### 理由 4:逆向条件分布也是高斯(Feller 定理)

Sohl-Dickstein 论文的关键洞察来自 Feller 1949:

> 对一个连续马尔可夫扩散过程,**当时间步 $\Delta t \to 0$ 时,逆向条件分布 $q(x_{t-1} \mid x_t)$ 也是高斯的**(且与前向同族)。

这个性质让我们可以**用高斯参数化逆向过程** $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta, \Sigma_\theta)$,而且这个选择**不会损失表达力**——只要前向步够小,真实的逆向就是高斯,我们就能精确学到。

### 理由 5:中心极限定理的"普适性"

中心极限定理:**任何独立随机变量(满足温和条件)的和,经过归一化后趋向高斯**。

物理意义:把数据反复加各种小扰动,最终收敛到的极限分布**一定是高斯**——这是物理世界的普遍规律。所以"用高斯加噪"不只是数学方便,**它本身就是"信息消失"过程的自然终态**。

### 理由 6:和 SDE 框架的优雅对应

VP 扩散其实是 OU 过程的精确离散化:

$$dx = -\tfrac{1}{2}\beta(t) x\, dt + \sqrt{\beta(t)}\, dw$$

这里 $dw$ 是布朗运动,**布朗运动的增量本身就是高斯的**。整个 SDE 理论(Itô 微积分、Fokker-Planck 方程、Anderson 反向 SDE)都建立在高斯噪声上——换成别的噪声,你失去整套 SDE 工具箱。

### 理由 7:和 Score Matching 的天然对接

Score 是 $\nabla_x \log p_t(x)$。对高斯加噪,score 有特别简单的形式:

$$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar\alpha_t}\, x_0}{1-\bar\alpha_t} = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

**预测噪声 $\epsilon$ 等价于预测 score**——这才让 DDPM(变分视角)和 NCSN(score 视角)被统一。

### 三、不用高斯的代价(具体看几个例子)

### 例 1:D3PM(分类扩散,Austin et al. 2021)

为离散 token 设计。前向过程用一个**转移矩阵** $Q_t$。代价:每一步是矩阵乘法、$q(x_t \mid x_0)$ 的闭式需要矩阵幂、没有重参数化技巧。效果能用,但比连续 diffusion 难调,实践中常用的方法反而是把 token 嵌入到连续空间再做高斯 diffusion。

### 例 2:流形扩散

研究方向之一,用于分子构象、机器人姿态。前向过程用流形上的布朗运动。代价:没有简单的"加性噪声"、$q(x_t \mid x_0)$ 通常没有简洁闭式、需要重新推导 score、loss、实现复杂。

### 例 3:Cold Diffusion(去掉随机性试试,Bansal et al. 2022)

**用确定性变换替换高斯加噪**(比如逐步模糊、像素化、遮罩)。也能 work,但失去了 SDE/score 的理论框架,只能从损失上 motivate。

**结论**:不是不能不用高斯,但你失去的远多于得到的。

### 四、Flow Matching 算什么?

Flow Matching(Lipman et al. 2023)、Rectified Flow 是 diffusion 的现代推广,FLUX、SD3 都在用。

**它仍然在用高斯**——只是把"高斯到数据的路径"从扩散过程推广到任意概率路径。比如可以用直线插值:

$$x_t = (1-t)\, \epsilon + t\, x_0,\quad \epsilon \sim \mathcal{N}(0, I)$$

仍然是高斯噪声,只是路径变了。这反而进一步证明了**高斯的核心地位**。

### 五、一个更深的视角:为什么物理世界偏爱高斯?

**为什么高斯在概率论里这么特殊?**

1. **最大熵分布**:在固定均值和方差的约束下,**高斯是熵最大的分布**——也就是"最不偏见"的分布
2. **CLT 终态**:任何独立扰动的和都趋向高斯——它是"扰动累积"的吸引子
3. **平移不变 + 旋转不变 + 独立性**:满足"$X, Y$ 独立 ⇒ $X+Y, X-Y$ 独立"的连续分布**只有**高斯(Bernstein 定理)
4. **指数族 + 二阶充分统计**:统计量只需要均值和方差,所有信息都浓缩了

所以高斯不是数学家"选出来"的,**它是从一些极简的对称性公理里"长出来"的**。

**diffusion 用高斯,本质上是在"借用物理世界已经验证好的最优噪声形式"**。

### 六、要点回顾

- 高斯**不是必须的**——可以做分类、二项、流形等其他扩散
- 但高斯**有一系列其他分布凑不齐的好性质**:
  1. KL 有简洁闭式
  2. 高斯之和还是高斯(支持闭式边际)
  3. 重参数化是线性的
  4. Feller 定理保证逆向也是高斯
  5. CLT 让它成为"信息消失"的自然终态
  6. 和 SDE/Itô 微积分天然兼容
  7. Score 形式简单,统一变分和 score-based 两条线
- **diffusion 不是"先选了高斯然后推出整套理论",而是"在追求闭式 + 简洁 + 普适的多重约束下,被逼到只能选高斯"**

{% enddetails %}

### 1.3. 模型参数化:$\mu_\theta$ 和 $\Sigma_\theta$ 是什么?

在 2015 年原版,$\mu_\theta$ 和 $\Sigma_\theta$ 都是神经网络的输出。给定 $(x_t, t)$,网络输出均值向量和协方差矩阵(实践中通常对角)。

网络架构在原论文里比较朴素(MLP 或简单 CNN)——这也是它当年效果不够惊艳的原因之一。后来 DDPM 用 UNet 才让效果上去。

{% details 附录:逆向条件分布的密度怎么算 %}

### 一、先澄清:不是"计算 $p_\theta(x_{t-1} \mid x_t)$",而是"计算这个分布在某点的密度值"

先把概念分开:

- $p_\theta(x_{t-1} \mid x_t)$ 作为**分布**,本身就是 $\mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$——它是一个高斯,由网络输出 $\mu_\theta, \Sigma_\theta$ 完全定义
- **训练时我们要算的不是这个分布,而是"已经采到的 $x_{t-1}$ 在这个分布下的概率密度值"** $p_\theta(x_{t-1} \mid x_t)$ 在那个特定点的取值

也就是说,损失里的 $\log p_\theta(x_{t-1} \mid x_t)$ 是一个**数**,通过把具体的 $x_{t-1}$ 代入高斯密度公式得到。

### 二、具体计算流程

### 步骤 1:前向采样得到 $x_{t-1}$ 和 $x_t$

从数据 $x_0$ 出发,跑前向链:

$$x_1 = \sqrt{1-\beta_1}\, x_0 + \sqrt{\beta_1}\, \epsilon_0$$
$$x_2 = \sqrt{1-\beta_2}\, x_1 + \sqrt{\beta_2}\, \epsilon_1$$
$$\vdots$$
$$x_{t-1} = \sqrt{1-\beta_{t-1}}\, x_{t-2} + \sqrt{\beta_{t-1}}\, \epsilon_{t-2}$$
$$x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_{t-1}$$

现在我们手上有两个**具体的向量**:$x_{t-1}$ 和 $x_t$(都是 $d$ 维向量,比如 28×28=784 维)。

它们都是 $\theta$-无关的——纯粹由前向链和随机种子决定。

### 步骤 2:把 $x_t$ 喂给网络,得到 $\mu_\theta$ 和 $\Sigma_\theta$

神经网络 $f_\theta$ 的输入:$(x_t, t)$
神经网络的输出:两个量

$$\mu_\theta(x_t, t) \in \mathbb{R}^d \quad (\text{一个 } d \text{ 维向量})$$
$$\Sigma_\theta(x_t, t) \in \mathbb{R}^{d \times d} \quad (\text{一个协方差矩阵,实践中通常对角})$$

实践中协方差用对角矩阵 $\Sigma_\theta = \text{diag}(\sigma_\theta^2)$,网络输出 $d$ 个方差值(或对数方差),这样:

- 网络输出 $2d$ 个数
- 前 $d$ 个是 $\mu_\theta$
- 后 $d$ 个是 $\sigma_\theta^2$(对角元素)

### 步骤 3:把 $x_{t-1}$ 代入高斯密度公式

现在 $p_\theta(\cdot \mid x_t) = \mathcal{N}(\mu_\theta, \Sigma_\theta)$ 这个分布被完全定义了。我们要算这个分布在 $x_{t-1}$ 这个点的密度值。

多维高斯密度公式:

$$\mathcal{N}(x; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)\right)$$

取 $\log$:

$$\log \mathcal{N}(x; \mu, \Sigma) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log|\Sigma| - \frac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)$$

把 $x = x_{t-1}$、$\mu = \mu_\theta(x_t, t)$、$\Sigma = \Sigma_\theta(x_t, t)$ 代入:

$$\log p_\theta(x_{t-1} \mid x_t) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log|\Sigma_\theta| - \frac{1}{2}(x_{t-1}-\mu_\theta)^\top \Sigma_\theta^{-1} (x_{t-1}-\mu_\theta)$$

如果 $\Sigma_\theta = \text{diag}(\sigma_{\theta,1}^2, \ldots, \sigma_{\theta,d}^2)$ 是对角的,这进一步简化为:

$$\log p_\theta(x_{t-1} \mid x_t) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\sum_{i=1}^d \log \sigma_{\theta,i}^2 - \frac{1}{2}\sum_{i=1}^d \frac{(x_{t-1,i} - \mu_{\theta,i})^2}{\sigma_{\theta,i}^2}$$

**这就是一个具体可算的标量**。每一项都是当前张量的简单运算——对应到代码里就是几行 `torch` 操作。

### 三、伪代码版本

让流程更具体一些:

```python
# 训练一个数据点
x_0 = sample_from_dataset()
t = random.randint(1, T)

# 步骤 1: 前向采样到 x_{t-1} 和 x_t
x_prev = forward_sample(x_0, t-1)   # 即 x_{t-1}
eps = randn_like(x_prev)
x_t = sqrt(1 - beta[t]) * x_prev + sqrt(beta[t]) * eps

# 步骤 2: 网络输出 mu 和 sigma^2
mu_theta, log_var_theta = network(x_t, t)
var_theta = exp(log_var_theta)

# 步骤 3: 算 log p(x_{t-1} | x_t),代高斯密度公式
diff = x_prev - mu_theta
log_p = -0.5 * d * log(2*pi) \
        - 0.5 * sum(log_var_theta) \
        - 0.5 * sum(diff**2 / var_theta)

# 损失里的对应项:-log p
loss_term = -log_p
```

### 四、关键澄清点

### 4.1 训练时的 $x_{t-1}$ 不是网络生成的

**初学者最容易混淆的一点**:训练时,$x_{t-1}$ 不是从 $p_\theta$ 采样得到的——它是**前向链事先采好的"标签"**。

我们做的事情是:

- 给网络看 $x_t$
- 让网络输出一个分布 $p_\theta(\cdot \mid x_t)$
- 评估这个分布在"真实"的 $x_{t-1}$ 上的概率密度
- 优化网络让这个密度尽量大

**这是经典的最大似然训练**——网络输出一个分布,我们让它在真实数据上的概率最大。和分类任务里"输出 softmax 概率,让真实类别的概率最大"完全同构,只是这里"类别"换成了连续的 $x_{t-1}$,概率换成了高斯密度。

### 4.2 梯度怎么流?

梯度路径:

$$\theta \xrightarrow{\text{网络}} (\mu_\theta, \Sigma_\theta) \xrightarrow{\text{密度公式}} \log p_\theta(x_{t-1} \mid x_t) \xrightarrow{} \text{loss}$$

注意 $x_t, x_{t-1}$ 是常量(对梯度而言),只有 $\mu_\theta, \Sigma_\theta$ 是 $\theta$ 的函数。所以梯度只通过这两个量回流——**不需要重参数化穿过任何采样动作**(这就是上一讲说的"Sohl-Dickstein 不必须用重参数化"的具体含义)。

### 4.3 直观上网络在学什么

最关键的一项是:

$$-\frac{1}{2}\| x_{t-1} - \mu_\theta(x_t, t) \|^2 / \sigma^2$$

最大化它 = 让 $\mu_\theta(x_t, t)$ 尽量靠近 $x_{t-1}$。

也就是说:**网络看到 $x_t$,要预测出"这个 $x_t$ 上一步应该长啥样"**——一种"去噪"行为。这就是 diffusion 网络在做的事的本质。

DDPM 后来做的"预测 $\epsilon$"其实是这个目标的等价变换——既然 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$,与其让网络预测 $\mu_\theta$,不如让它预测 $\epsilon$,数学上完全等价但优化更稳定。

### 五、要点回顾

- $p_\theta(x_{t-1} \mid x_t)$ 是个高斯分布,由网络输出 $\mu_\theta, \Sigma_\theta$ 完全定义
- "计算 $p_\theta(x_{t-1} \mid x_t)$" 实际指**算高斯密度公式在 $x_{t-1}$ 这个点的取值**
- $x_{t-1}$ 是从前向链采样得到的"标签",不是网络生成的
- 梯度只通过 $\mu_\theta, \Sigma_\theta$ 回流,不穿过采样
- 本质上是最大似然:让网络输出的分布在真实 $x_{t-1}$ 上概率最大

{% enddetails %}

---

## 2. 损失函数:从 ELBO 到具体形式

### 2.1. 起点:ELBO

我们已经推过:

$$\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T} \mid x_0)}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right] =: \mathcal{L}$$

在训练时我们最小化 $-\mathcal{L}$(负 ELBO)。

### 2.2. 展开

把 $p_\theta(x_{0:T})$ 和 $q(x_{1:T} \mid x_0)$ 的乘积形式代入:

$$-\mathcal{L} = \mathbb{E}_q\left[-\log \frac{p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)}{\prod_{t=1}^T q(x_t \mid x_{t-1})}\right]$$

把 $\log$ 拆开:

$$= \mathbb{E}_q\left[-\log p(x_T) - \sum_{t=1}^T \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1})}\right]$$

整理成更标准的形式:

$$\boxed{-\mathcal{L} = \mathbb{E}_q\left[-\log p(x_T) + \sum_{t=1}^T \log \frac{q(x_t \mid x_{t-1})}{p_\theta(x_{t-1} \mid x_t)}\right]}$$

这就是 **Sohl-Dickstein 2015 的训练目标**。

### 2.3. 这个目标怎么训练?

具体训练步骤:

1. 从数据集采 $x_0$
2. 前向采样整条链 $x_1, x_2, \ldots, x_T$(每一步都按 $q(x_t \mid x_{t-1})$ 采)
3. 计算每一项 $\log \frac{q(x_t \mid x_{t-1})}{p_\theta(x_{t-1} \mid x_t)}$
4. 求和,反向传播,更新 $\theta$

这就是**完整的 Sohl-Dickstein 训练流程**。

### 2.4. 为什么这个目标"能算但不好用"?

每一项里:

- **分子** $q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I)$ 是已知的高斯,$\log$ 有闭式
- **分母** $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t))$ 是网络输出的高斯,$\log$ 也有闭式

所以每一项的对数比值都能算。问题是**方差**。

让我具体分析一下方差从哪儿来。在前向采样时:

- 你从 $x_0$ 走到 $x_t$,中间每步都是随机的
- $x_t$ 的随机性会"传染"给所有后续步骤

于是这个估计量 $\log \frac{q(x_t \mid x_{t-1})}{p_\theta(x_{t-1} \mid x_t)}$ 在不同的随机轨迹下取值跳得很厉害,$T = 1000$ 步求和后方差会累积得很大。

### 2.5. 一个不那么明显的问题:**方向不一致**

更深的问题是:

- 分子 $q(x_t \mid x_{t-1})$ 是**前向**的:已知 $x_{t-1}$,问 $x_t$
- 分母 $p_\theta(x_{t-1} \mid x_t)$ 是**逆向**的:已知 $x_t$,问 $x_{t-1}$

两者条件方向相反,导致:

- 这不是一个"自然"的 KL 散度形式
- 没法用闭式 KL 简化
- 每一项必须用蒙特卡洛估计

DDPM 后来的关键改写就是**把分子也变成逆向条件分布**,让每项都变成两个高斯之间的 KL,有闭式解,直接消除蒙特卡洛噪声。


## 3. Sohl-Dickstein 论文里的另外两件事

为了完整,提一下论文里的另外两个细节:

### 3.1. noise schedule 的选择

原论文用了几种:

- 二项扩散(用于二值数据)
- 高斯扩散(用于连续数据)——也就是我们上面讨论的

他们做了一些手工设计,后来 DDPM 用了简单的线性 schedule,效果就很好。

### 3.2. 一个有意思的小细节:乘性 vs 加性

注意前向 $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\, x_{t-1}, \beta_t I)$ 里的 $\sqrt{1-\beta_t}$。这个**乘性收缩**有什么用?

如果只用加性噪声 $q(x_t \mid x_{t-1}) = \mathcal{N}(x_{t-1}, \beta_t I)$,方差会一直涨,$T \to \infty$ 时 $x_T$ 是无限大方差——不收敛到标准高斯。

加上 $\sqrt{1-\beta_t}$ 这个收缩,信号每一步都向 $0$ 靠拢一点,噪声也加一点,**总方差保持不变**。这种设计叫 **Variance Preserving (VP)**,是 DDPM 沿用的形式。

后来 NCSN/Score SDE 用了另一种思路 **Variance Exploding (VE)**:不收缩,直接加大量噪声让方差爆炸。两种都行,只是参数化不同。

---

## 4. 用一张图总结

```
                  前向(固定,加噪)
   x_0  ────→  x_1  ────→  x_2  ────→ ... ────→  x_T ≈ N(0, I)
        q(x_1|x_0)    q(x_2|x_1)              q(x_T|x_{T-1})
   
   
   x_0  ←────  x_1  ←────  x_2  ←──── ... ←────  x_T
         p_θ(x_0|x_1)  p_θ(x_1|x_2)         p_θ(x_{T-1}|x_T)
                  逆向(学习,去噪)


训练目标(Sohl-Dickstein 2015):
  最小化 −ELBO
       = E_q [−log p(x_T) + Σ_t log q(x_t|x_{t-1}) / p_θ(x_{t-1}|x_t) ]


采样:
  x_T ~ N(0, I)
  for t = T, ..., 1:
      x_{t-1} ~ p_θ(x_{t-1} | x_t)   # 用学到的网络
  return x_0
```

---

## 5. 和后续工作的承接

理解了原版,后续每一步"改进"你都能看清:

| 问题 | 谁解决的 | 怎么解决 |
|------|---------|---------|
| ELBO 里 $q$ 和 $p_\theta$ 方向不一致 | DDPM (2020) | 用贝叶斯把 ELBO 重写成 KL 之和 |
| KL 没闭式 | DDPM | 高斯之间的 KL 有闭式 |
| 损失里要预测 $\mu_\theta$ | DDPM | 重参数化为预测噪声 $\epsilon$ |
| 网络架构弱 | DDPM | UNet + 自注意力 + 时间嵌入 |
| 采样要 $T$ 步太慢 | DDIM (2020) / DPM-Solver (2022) | 跳步采样 / ODE 数值解 |
| 直接在像素空间太贵 | LDM / Stable Diffusion (2022) | 在 VAE 潜空间做 diffusion |

---

## 6. 要点回顾

- Sohl-Dickstein 的框架 = **固定的高斯加噪前向 + 学习的高斯去噪逆向**
- 损失函数 = ELBO,展开后是一串"前向条件 / 逆向条件"的对数比值之和
- 这个目标**理论上正确、实际上方差大**,导致原始版本训练效果一般
- 关键卡点:前向和逆向条件方向不一致,无法化简成闭式 KL
- DDPM 的本质改进就是**通过贝叶斯把这个不一致修好**

---

到这里,**第一讲(Sohl-Dickstein 起源)**就讲完整了。我们已经覆盖:
- 物理动机和 ELBO 的推导
- 蒙特卡洛采样估计的本质
- 完整的模型框架和损失函数
- 它为什么 work、为什么不够好

## 7. 为什么 2015 年没火?

Sohl-Dickstein 的论文方向完全正确,但有几个问题:

1. **算力不够**:2015 年训练一个像样的 diffusion 模型代价很大
2. **架构不够好**:UNet + attention 的组合是后来才成熟的
3. **目标函数没简化**:直接的 ELBO 训练信号弱
4. **采样太慢**:几百步采样在当时不可接受

所以这篇论文沉寂了 5 年,直到 2020 年 DDPM 才让它重新焕发生机。
