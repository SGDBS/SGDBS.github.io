---
title: Chapter 7. Score SDE
categories: Diffusion Models
date: 2026-05-04 12:00:00
mathjax: true
tags:
    - AI
    - Diffusion Models
---

这一讲是 diffusion 数学的"加冕时刻"——把变分线(DDPM 那一讲)和 score 线(NCSN 那一讲)正式统一在一个连续时间的随机微分方程框架下。Song et al. 2021 的论文 "Score-Based Generative Modeling through Stochastic Differential Equations" 是这条线的集大成者。

我会按这个顺序展开:

1. 为什么需要连续时间视角
2. SDE 的基础(Itô 微积分最小集)
3. 前向 SDE:DDPM 和 NCSN 都是它的特例
4. **Anderson 反向 SDE**:这一讲的"心脏"
5. Probability Flow ODE:确定性视角
6. 训练:连续时间的 score matching
7. 统一图景

---

## 一、动机:从离散到连续

回顾我们走过的路:

- DDPM:离散链 $x_0 \to x_1 \to \cdots \to x_T$,$T = 1000$
- NCSN:离散噪声尺度 $\sigma_1 > \cdots > \sigma_L$

每个都是"离散时间"的过程。但思考一下:

- $T$ 是**任意选的**——为什么是 1000?500 行不行?2000 呢?
- $\beta_t$ 是个**离散 schedule**,它是某种连续函数的"采样"
- DDPM 和 NCSN 看起来是不同的过程,但**最终训练目标几乎相同**($\epsilon$-预测 vs score-预测,差一个标量)

这强烈暗示:**它们是同一个连续过程的两种离散化**。让 $T \to \infty$、$\Delta t \to 0$,会得到一个连续时间的 SDE,所有特例都能从它推出来。

这就是 Score SDE 的核心思想——**连续时间是 diffusion 的"自然语言",离散都是它的近似**。

---

## 二、SDE 基础(最小必备)

我假设你只接触过 ODE,我们快速过 SDE 的核心概念。

### 2.1 什么是 SDE?

ODE:$\frac{dx}{dt} = f(x, t)$,确定性演化。

SDE:加上随机扰动:

$$dx = f(x, t)\, dt + g(t)\, dw$$

- $f(x, t)$:**漂移**(drift)——确定性的"流向"
- $g(t)$:**扩散系数**——随机扰动的强度
- $dw$:**布朗运动的增量**——核心的"随机种子"

### 2.2 布朗运动(Wiener 过程)

$w_t$ 是布朗运动,意思是:

- $w_0 = 0$
- 增量 $w_{t+\Delta t} - w_t \sim \mathcal{N}(0, \Delta t)$,且与过去独立
- 路径连续但处处不可微(这是关键技术点)

直观:**$dw$ 是"无穷小的高斯增量"**,标准差 $\sqrt{dt}$。

### 2.3 Euler-Maruyama 离散化

把 SDE 离散化到步长 $\Delta t$:

$$x_{t+\Delta t} \approx x_t + f(x_t, t)\, \Delta t + g(t)\sqrt{\Delta t}\, z,\quad z \sim \mathcal{N}(0, I)$$

注意:漂移项是 $\Delta t$,扩散项是 $\sqrt{\Delta t}$ ——这是 SDE 和 ODE 的**关键区别**。$\Delta t$ 小时,$\sqrt{\Delta t} \gg \Delta t$,**随机扰动主导短时行为**。

### 2.4 Fokker-Planck 方程

我们关心的不是某条具体路径,而是分布 $p_t(x)$ 怎么演化。**Fokker-Planck 方程**给出答案:

$$\frac{\partial p_t(x)}{\partial t} = -\nabla_x \cdot [f(x,t)\, p_t(x)] + \frac{1}{2} g(t)^2 \nabla_x^2 p_t(x)$$

这是个偏微分方程(PDE),描述密度的时间演化:

- 第一项:漂移导致的输运
- 第二项:扩散导致的"模糊化"(就是热方程里的拉普拉斯算子)

我们用不到这个 PDE 的解析解,但要知道它的存在——它是反向 SDE 推导的关键。

---

## 三、前向 SDE:DDPM 和 NCSN 是特例

### 3.1 一般前向 SDE

定义前向 SDE($t \in [0, T]$):

$$dx = f(x, t)\, dt + g(t)\, dw$$

我们要选 $f, g$ 使得:

- $t = 0$:$x_0 \sim p_{\text{data}}$
- $t = T$:$x_T$ 接近某个简单分布(高斯)

### 3.2 VP-SDE(对应 DDPM)

**Variance Preserving** SDE:

$$dx = -\frac{1}{2}\beta(t)\, x\, dt + \sqrt{\beta(t)}\, dw$$

让我们验证它对应 DDPM。Euler-Maruyama 离散化($\Delta t = 1$,$\beta_t = \beta(t)$):

$$x_{t} \approx x_{t-1} - \frac{1}{2}\beta_t\, x_{t-1} + \sqrt{\beta_t}\, z = (1 - \tfrac{1}{2}\beta_t)\, x_{t-1} + \sqrt{\beta_t}\, z$$

而 $\sqrt{1 - \beta_t} \approx 1 - \frac{1}{2}\beta_t$(泰勒展开,$\beta_t$ 小)。所以这正是 DDPM 的离散公式:

$$x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, z$$

VP-SDE **就是 DDPM 的连续时间极限**——精确到一阶,但 $\sqrt{1-\beta_t}$ 用了精确形式而非泰勒近似(这保证了方差严格守恒)。

VP-SDE 的边际分布也是闭式:

$$p_t(x \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha(t)}\, x_0,\, (1-\bar\alpha(t))I)$$

其中 $\bar\alpha(t) = \exp\left(-\int_0^t \beta(s)\, ds\right)$ ——离散版的 $\bar\alpha_t$ 的连续推广。

### 3.3 VE-SDE(对应 NCSN)

**Variance Exploding** SDE:

$$dx = \sqrt{\frac{d\sigma^2(t)}{dt}}\, dw$$

(没有漂移项)边际分布:

$$p_t(x \mid x_0) = \mathcal{N}(x_0, \sigma^2(t) I)$$

——直接给 $x_0$ 加方差为 $\sigma^2(t)$ 的高斯。这正是 NCSN 的多尺度噪声,只是从离散尺度 $\sigma_l$ 变成连续函数 $\sigma(t)$。

### 3.4 统一视角

DDPM 和 NCSN 都是**前向 SDE 的特例**——它们看起来是不同的过程,但在 SDE 框架下只是参数化的差异:

| | $f(x, t)$ | $g(t)$ | $x_T$ 分布 |
|---|---|---|---|
| VP (DDPM) | $-\frac{1}{2}\beta(t) x$ | $\sqrt{\beta(t)}$ | $\mathcal{N}(0, I)$ |
| VE (NCSN) | $0$ | $\sqrt{d\sigma^2/dt}$ | $\mathcal{N}(0, \sigma^2(T) I)$ |

第一个统一:**前向过程不是问题,关键是怎么逆向**。

---

## 四、Anderson 反向 SDE:这一讲的心脏

这是 Score SDE 论文最关键的引用——Anderson (1982) 的一个经典定理。

### 4.1 定理陈述

给定前向 SDE:

$$dx = f(x, t)\, dt + g(t)\, dw$$

它**有一个对应的反向 SDE**:

$$\boxed{dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t)\, d\bar w}$$

时间从 $T$ 走到 $0$,$d\bar w$ 是反向时间的布朗运动。

**关键点**:**只要知道每个时刻的 score $\nabla_x \log p_t(x)$,就能精确反向运行 SDE,从 $p_T$ 采样回 $p_0$**。

### 4.2 这是什么意思?

让我们消化这个公式的威力:

- **前向 SDE**:把数据加噪到高斯——已知,简单
- **反向 SDE**:从高斯回到数据——只需要 score!
- **score 怎么得到**:用 score matching 训练 → 连续版的 DSM

**这就是 score-based generative modeling 的完整框架**:

1. 选一个前向 SDE(VP/VE/任意)
2. 用 score matching 学 $s_\theta(x, t) \approx \nabla_x \log p_t(x)$
3. 把 $s_\theta$ 代入反向 SDE,从 $\mathcal{N}(0, I)$ 数值求解到 $t = 0$

### 4.3 反向 SDE 的简单证明思路

让我给你一个非严谨但直观的推导,看清楚 score 为什么会出现。

考虑前向 SDE 的 Fokker-Planck 方程:

$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (f p_t) + \frac{1}{2} g^2 \nabla^2 p_t$$

我们想找一个反向 SDE,它的 FPE 给出**同样的 $p_t$**(只是时间倒着走)。

技巧:把 $\nabla^2 p_t = \nabla \cdot (\nabla p_t)$,而 $\nabla p_t = p_t \nabla \log p_t$(对数导数公式!)。所以:

$$\frac{1}{2} g^2 \nabla^2 p_t = \frac{1}{2} g^2 \nabla \cdot (p_t \nabla \log p_t)$$

代回 FPE:

$$\frac{\partial p_t}{\partial t} = -\nabla \cdot \left[(f - g^2 \nabla \log p_t) p_t\right] + \frac{1}{2} g^2 \nabla^2 p_t$$

注意我们做了一个**代数恒等变形**:把 $\frac{1}{2} g^2 \nabla \cdot (p_t \nabla \log p_t)$ 拆成两份,一份移到第一项里(变成 $-g^2 \nabla \log p_t$ 加进漂移),另一份留在原位置。

这个新的 FPE 形式对应的 SDE:

$$dx = (f - g^2 \nabla \log p_t)\, dt + g\, dw$$

**这就是反向 SDE 的形式**(细节上时间方向需要正式处理,但核心逻辑就是这个对数导数变换)。

### 4.4 关键技术工具:对数导数公式

注意整个推导的核心又是:

$$\nabla p_t = p_t\, \nabla \log p_t$$

——同一个公式,在 score matching 推导里用过,在分部积分里用过,在 DSM 等价性证明里用过。

**这是 score 视角下反复出现的"瑞士军刀"**。

### 4.5 一个直觉解释

反向 SDE 的形式是:

$$dx = \left[\underbrace{f(x, t)}_{\text{原前向漂移}} - \underbrace{g(t)^2 \nabla \log p_t(x)}_{\text{score 修正}}\right] dt + g(t)\, d\bar w$$

直觉:

- **保留扩散项 $g\, d\bar w$**:反向也加噪声(对应 DDPM 采样里的 $\sigma_t z$)
- **漂移项变了**:原本是 $f$,现在是 $f - g^2 \nabla \log p_t$
- **$- g^2 \nabla \log p_t$ 把粒子推向高密度区域**——score 指向数据密度,这一项是"反向的拉力"

物理图景:前向是"扩散+漂移",反向是"反扩散(沿 score 方向走)+ 同样的漂移修正 + 噪声"。score 是反向过程的"导航星"。

---

## 五、Probability Flow ODE:确定性视角

Anderson 反向 SDE 是 score-based 生成的"标准答案",但 Score SDE 论文还做了一件神奇的事:**给出对应的确定性 ODE**。

### 5.1 神奇的事实

可以构造一个**ODE**(没有随机项):

$$\boxed{\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)}$$

它的解 $x(t)$ 与反向 SDE **有相同的边际分布** $p_t$。

注意系数:从反向 SDE 的 $g^2$ 变成 $\frac{1}{2}g^2$,扩散项消失。

### 5.2 这意味着什么?

**两件事**:

**(1) 采样可以是确定性的**

不需要任何随机性,只要数值求解 ODE。给定 $x_T \sim \mathcal{N}(0, I)$,确定性地走到 $x_0$ ——同一个 $x_T$ 永远给同一个 $x_0$。这是 DDIM 的连续版本(DDIM 其实就是这个 ODE 的特定离散化)。

**(2) 可以精确算似然**

ODE 是可逆变换,通过**瞬时变量替换公式**(continuous normalizing flows 的核心):

$$\log p_0(x_0) = \log p_T(x_T) + \int_0^T \nabla \cdot \tilde f(x_t, t)\, dt$$

(其中 $\tilde f$ 是 ODE 的右端)

这给出 $p_0(x_0)$ 的**精确**对数似然——对比 ELBO 只是下界。这让 diffusion 成为真正的概率模型,能做密度估计、anomaly detection 等。

### 5.3 SDE vs ODE 的对比

| | 反向 SDE | Probability Flow ODE |
|---|---|---|
| 形式 | $dx = (f - g^2 s)\,dt + g\, d\bar w$ | $dx = (f - \frac{1}{2}g^2 s)\, dt$ |
| 随机性 | 有 | 无 |
| 同一起点 | 不同采样得不同结果 | 唯一解 |
| 边际分布 | 等于真实 $p_t$ | 等于真实 $p_t$ |
| 似然计算 | ELBO 下界 | 精确似然 |
| 采样多样性 | 高(随机性贡献) | 低(完全由 $x_T$ 决定) |
| 数值求解 | Euler-Maruyama 等 | RK4 等高阶 ODE solver |

**两者的边际分布相同——但路径不同**。ODE 给了一条"最短路径",SDE 给了一族"加噪路径"。

### 5.4 为什么会有这个 ODE?

简单的直觉:Fokker-Planck 方程

$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (f p_t) + \frac{1}{2} g^2 \nabla^2 p_t$$

可以重写为输运方程(连续性方程)的形式:

$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (\tilde f p_t)$$

其中 $\tilde f = f - \frac{1}{2}g^2 \nabla \log p_t$。

输运方程对应的就是 ODE $dx/dt = \tilde f(x, t)$ ——粒子按这个速度场运动,密度自动满足 FPE。

**所以 SDE 和 ODE 是"边际分布等价"的两个版本**——一个有随机性,一个没有,但它们描述的概率密度演化完全一样。

---

## 六、训练:连续时间的 score matching

我们要学 $s_\theta(x, t) \approx \nabla_x \log p_t(x)$ 对所有 $t \in [0, T]$ 同时成立。

### 6.1 连续 score matching 损失

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim U[0, T]}\left[\lambda(t)\, \mathbb{E}_{p_t(x)}\left[\| s_\theta(x, t) - \nabla_x \log p_t(x) \|^2\right]\right]$$

其中 $\lambda(t)$ 是个权重函数(比如 $\lambda(t) = g(t)^2$,这选择对应 ELBO 优化)。

但 $\nabla \log p_t(x)$ 还是不知道。用 DSM 的连续版:

$$\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{t}\left[\lambda(t)\, \mathbb{E}_{x_0, x_t}\left[\| s_\theta(x_t, t) - \nabla_{x_t} \log p_t(x_t \mid x_0) \|^2\right]\right]$$

对 VP-SDE,$p_t(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha(t)} x_0, (1-\bar\alpha(t)) I)$,条件 score 是 $-\epsilon/\sqrt{1-\bar\alpha(t)}$,所以:

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_t\left[\frac{\lambda(t)}{1-\bar\alpha(t)}\, \mathbb{E}_{x_0, \epsilon}\left[\| \sqrt{1-\bar\alpha(t)}\, s_\theta(x_t, t) + \epsilon \|^2\right]\right]$$

定义 $\epsilon_\theta := -\sqrt{1-\bar\alpha(t)}\, s_\theta$,选 $\lambda(t) = (1-\bar\alpha(t))$:

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_t \mathbb{E}_{x_0, \epsilon}\left[\| \epsilon - \epsilon_\theta(x_t, t) \|^2\right]$$

**正是 DDPM 的 $L_{\text{simple}}$**!

第二个统一:**DDPM 的训练目标是连续 score matching 的特定离散化 + 特定权重选择**。

---

## 七、统一图景:把所有线索串起来

现在我们终于可以画出 diffusion 数学的完整地图。

### 7.1 三条线索的对照

| 视角 | 来源 | 核心数学对象 |
|---|---|---|
| **变分** | Sohl-Dickstein 2015, DDPM 2020 | 离散马尔可夫链 + ELBO |
| **Score** | Hyvärinen 2005, NCSN 2019 | score $\nabla \log p$ + Langevin |
| **SDE** | Song 2021 | 连续 SDE + 反向 SDE + ODE |

### 7.2 它们如何统一

**前向**:
- DDPM 的离散链 = VP-SDE 的 Euler-Maruyama 离散化
- NCSN 的多尺度噪声 = VE-SDE 的 Euler-Maruyama 离散化

**训练**:
- DDPM 的 $L_{\text{simple}}$ = 连续 DSM 的离散化(特定 $\lambda$)
- NCSN 的加权 DSM = 同样的连续 DSM(不同 $\lambda$)
- $\epsilon_\theta = -\sqrt{1-\bar\alpha(t)}\, s_\theta$ —— 一个简单标量变换

**采样**:
- DDPM 的 ancestral sampling = 反向 SDE 的特定离散化
- DDIM = Probability Flow ODE 的特定离散化
- NCSN 的 annealed Langevin = 反向 SDE 的另一种离散化

### 7.3 SDE 视角带来的新东西

不只是统一,SDE 视角还**带来新工具**:

1. **更好的采样器**:能用高阶 ODE/SDE 数值求解器(RK4、predictor-corrector)
2. **精确似然**:Probability Flow ODE 给出精确 $\log p_0$
3. **可控生成**:Conditional reverse SDE 提供框架,classifier guidance、CFG 都从这里推
4. **任意路径**:启发了 Flow Matching、Rectified Flow 等用任意概率路径的方法
5. **在流形上**:可以推广到 Riemannian SDE,用于分子、机器人

---

## 八、一个总览图

让我把整个框架画一遍:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                       连续 SDE 框架
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

前向 SDE:    dx = f(x,t) dt + g(t) dw
                     ↓
            数据 x_0  ──────→  噪声 x_T

         参数选择:VP(DDPM),VE(NCSN),Sub-VP,...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

训练:    用 DSM 学 s_θ(x, t) ≈ ∇log p_t(x)

         L = E_t,x_0,ε [ ‖ε - ε_θ(x_t, t)‖² ]
              ↑
         (epsilon 和 score 等价,差一个标量)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

采样选项 1:Anderson 反向 SDE(随机)
         dx = [f - g²s_θ] dt + g dw̄
         离散化 → DDPM ancestral sampling, annealed Langevin, ...

采样选项 2:Probability Flow ODE(确定)
         dx/dt = f - ½g² s_θ
         离散化 → DDIM, DPM-Solver, ...
         附加能力 → 精确似然计算

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 九、一个深刻的反思

走到这里,值得停下来问:**这一切为什么如此优雅?**

我的理解是:**diffusion 的"连续时间本质"反映了一个深刻的物理事实**——数据流形在欧氏空间中的"加噪/去噪"过程,本质就是一个被 score 引导的随机过程。

- 前向(扩散) = 自然的热力学第二定律,信息丢失,趋向平衡分布
- 反向(去噪) = 用 score 这个"信息梯度"逆转扩散,从平衡回到结构
- $T \to \infty$、$\Delta t \to 0$ 是这个过程的"自然单位"——离散步数只是数值近似的产物

所以 diffusion 不是一个被设计出来的算法——**它是一个被发现的物理过程**,不同的研究者从不同入口(变分、score、SDE)进入同一个数学结构。

这种"被多个独立路径同时发现"是数学和科学里很罕见的——通常只发生在你触到了真正的 deep structure。

---

## 十、要点回顾

- **连续时间 SDE 是 diffusion 的自然语言**,DDPM 和 NCSN 是它的离散化
- **Anderson 反向 SDE**:只要有 score,就能反向运行任何前向 SDE
- $$dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)] dt + g(t)\, d\bar w$$
- **Probability Flow ODE**:同样的边际分布,确定性版本,系数从 $g^2$ 变 $\frac{1}{2}g^2$
- 训练用连续 DSM,等价于 DDPM 的 $L_{\text{simple}}$(差一个时间权重)
- 采样:反向 SDE 离散化(DDPM,Langevin)或 ODE 离散化(DDIM,DPM-Solver)
- ODE 视角带来精确似然
- **三条线在 SDE 框架下完全统一**——DDPM 是 VP-SDE 的 Euler 步,NCSN 是 VE-SDE 的 Langevin 离散
- score $\nabla \log p$ 是 diffusion 的"灵魂"——所有视角最终都依赖它
