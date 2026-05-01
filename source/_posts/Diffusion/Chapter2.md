---
title: Chapter 1 Sohl-Dickstein 模型
categories: Diffusion Models
date: 2026-05-01 12:00:00
mathjax: true
tags:
    - AI
    - Diffusion Models
---


## 1. 模型框架的全貌

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

### 1.3. 模型参数化:$\mu_\theta$ 和 $\Sigma_\theta$ 是什么?

在 2015 年原版,$\mu_\theta$ 和 $\Sigma_\theta$ 都是神经网络的输出。给定 $(x_t, t)$,网络输出均值向量和协方差矩阵(实践中通常对角)。

网络架构在原论文里比较朴素(MLP 或简单 CNN)——这也是它当年效果不够惊艳的原因之一。后来 DDPM 用 UNet 才让效果上去。

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

## 8. 这一讲的要点回顾

- diffusion 的本质思想:**定义简单可解析的破坏过程,学习其逆过程**
- 前向用马尔可夫链 + 高斯加噪,$T$ 大时趋于标准高斯
- 逆向也用高斯参数化($\beta_t$ 小时这是合理的)
- 训练目标是 ELBO
- 这套框架在 2015 年提出,但要到 2020 年才"开花"