---
title: Chapter 8. DDIM
categories: Diffusion Models
date: 2026-05-05 12:00:00
mathjax: true
tags:
    - AI
    - Diffusion Models
---

DDPM 把 diffusion 的训练目标做漂亮了,但留下了一个工程上的"痛":**采样要走 1000 步**,生成一张图比 GAN 慢两个数量级。DDIM(Song et al. 2020,"Denoising Diffusion Implicit Models")在不重新训练的前提下,用同一个网络把采样步数压到 **20-50 步,质量几乎不变**。这一讲讲清楚它怎么做到。

我会按这个顺序展开:

1. DDPM 慢的根源 + DDIM 的关键洞察
2. 非马尔可夫前向过程
3. 重新推导逆向公式
4. $\eta$ 参数:DDPM 和确定性 ODE 的连续族
5. 跳步采样:为什么能加速
6. 和 Probability Flow ODE 的关系
7. DDIM 的额外能力:确定性、插值、编辑
8. PyTorch 实现
9. 和 DDPM 的对照

---

## 一、DDPM 慢在哪?

回忆 DDPM 的采样:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t z$$

要从 $t=T$ 走到 $t=0$ 整 $T$ 次,每次过一次 UNet。$T=1000$ 时:

- 一张图 = 1000 次 UNet 前向
- 一张 256×256 的图,在单卡 V100 上要约 30 秒
- 训练完一个模型再生成一万张评测图,要 80+ 小时

**为什么必须走 $T$ 步?** 表面上的原因:DDPM 是**马尔可夫链**,$x_t \to x_{t-1}$ 的转移由训练时假设的离散 schedule 决定,跳步意味着用错误的 $\beta$ 组合。

**深一层的原因**:DDPM 的采样过程是"祖先采样"(ancestral sampling)——严格按马尔可夫链定义运行。每一步加的 $\sigma_t z$ 是必要的(让生成有多样性),但也限制了步长——如果步太大,中间引入的噪声方差不再匹配真实分布。

DDIM 的核心发现是:**马尔可夫性根本不是必要的**。

---

## 二、关键洞察:训练只决定边际,不决定联合

这是整个 DDIM 的核心思想,值得仔细看。

回忆 DDPM 的训练目标 $L_{\text{simple}}$:

$$L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

其中 $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$。

**注意**:这个目标只用到了 $q(x_t \mid x_0)$——也就是**单步的边际**。它**完全不依赖**于"中间状态 $x_1, \ldots, x_{t-1}$ 怎么连起来"——也就是**联合分布** $q(x_{1:T} \mid x_0)$。

换句话说:

> **任何前向过程,只要它的边际 $q(x_t \mid x_0)$ 等于 DDPM 的 $\mathcal{N}(\sqrt{\bar\alpha_t}\, x_0, (1-\bar\alpha_t) I)$,训练出的网络都一样**。

这是个惊人的事实。它告诉我们:

- 训练好的 $\epsilon_\theta$ **不绑死**于 DDPM 那条特定的马尔可夫链
- 我们可以**事后**换一条不同的"前向过程"——只要边际相同——然后用同一个 $\epsilon_\theta$ 采样
- 如果新的前向过程对应的逆向"刚好"可以跳步,那就能加速

DDIM 就是在**重新设计这条前向过程**——选一个**非马尔可夫**的 $q_\sigma(x_{1:T} \mid x_0)$,边际还是 DDPM 那个,但联合的结构更灵活,采样可以快得多。

---

## 三、非马尔可夫前向过程

### 3.1 反过来定义:从 $q(x_{t-1} \mid x_t, x_0)$ 出发

DDPM 里我们是先定义前向 $q(x_t \mid x_{t-1})$,然后推出 $q(x_{t-1} \mid x_t, x_0)$。DDIM 反过来——**直接钦定 $q_\sigma(x_{t-1} \mid x_t, x_0)$ 的形式**,只要它满足:

1. 边际正确:$q_\sigma(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\, x_0, (1-\bar\alpha_t) I)$
2. 是高斯(便于推导)

DDIM 选的形式:

$$\boxed{q_\sigma(x_{t-1} \mid x_t, x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_{t-1}}\, x_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\cdot \frac{x_t - \sqrt{\bar\alpha_t}\, x_0}{\sqrt{1-\bar\alpha_t}},\ \sigma_t^2 I\right)}$$

这看起来吓人,但拆开就清楚:

- 均值由两部分组成:**指向 $x_0$ 方向**的项 + **指向 $x_t$ 方向**(通过 $x_0$ 减去得到的"噪声方向")的项
- 方差是**自由参数** $\sigma_t^2$——这就是 DDIM 比 DDPM 多出的旋钮

注意 $(x_t - \sqrt{\bar\alpha_t}\, x_0) / \sqrt{1-\bar\alpha_t}$ 正好是从 $x_0, x_t$ 反推出的噪声 $\epsilon$:由 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ 得 $\epsilon = (x_t - \sqrt{\bar\alpha_t} x_0)/\sqrt{1-\bar\alpha_t}$。

所以可以更简洁地写:

$$q_\sigma(x_{t-1} \mid x_t, x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_{t-1}}\, x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot \epsilon,\ \sigma_t^2 I\right)$$

其中 $\epsilon$ 是把 $x_0, x_t$ 反推出的那个噪声。

### 3.2 为什么这个构造能保住边际?

需要证明:**对任意 $\sigma_t \geq 0$,$q_\sigma(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\, x_0, (1-\bar\alpha_t) I)$ 恒成立**。

这是个非平凡的事实——构造的精妙之处就在这里。**完整证明用归纳法**(详见附录1),关键是:

$$q_\sigma(x_t \mid x_0) = \int q_\sigma(x_t \mid x_{t-1}, x_0)\, q_\sigma(x_{t-1} \mid x_0)\, dx_{t-1}$$

如果归纳假设 $q_\sigma(x_{t-1} \mid x_0)$ 是 DDPM 那个边际,可以解出 $q_\sigma(x_t \mid x_{t-1}, x_0)$,验证 $q_\sigma(x_t \mid x_0)$ 也是 DDPM 那个。**关键:$q_\sigma(x_t \mid x_{t-1}, x_0)$ 显式依赖 $x_0$**——这是非马尔可夫的体现。

{% details 附录1:DDIM 边际守恒的归纳证明 %}

我们要证:对任意 $\sigma_t \geq 0$,有

$$q_\sigma(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\, x_0, (1-\bar\alpha_t) I)$$

### 归纳基

$t = T$:这一步直接由前向定义保证(我们让 $q_\sigma(x_T \mid x_0)$ 仍然是 DDPM 的最终分布)。

### 归纳步:假设 $t-1$ 成立,证 $t$ 成立

假设 $q_\sigma(x_{t-1} \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_{t-1}}\, x_0, (1-\bar\alpha_{t-1}) I)$。

由我们对 $q_\sigma(x_{t-1} \mid x_t, x_0)$ 的定义,反推得 $q_\sigma(x_t \mid x_{t-1}, x_0)$ 也是高斯(贝叶斯组合两个高斯)。具体地,由

$$q_\sigma(x_t, x_{t-1} \mid x_0) = q_\sigma(x_{t-1} \mid x_t, x_0)\, q_\sigma(x_t \mid x_0) = q_\sigma(x_t \mid x_{t-1}, x_0)\, q_\sigma(x_{t-1} \mid x_0)$$

由于左边是关于 $(x_t, x_{t-1})$ 的高斯(因为两边都是高斯),展开二次型可以验证:**只要 $q_\sigma(x_t \mid x_0)$ 取 DDPM 那个边际,等式两边一致**。

直接验证:对任意 $x_0$,从 $q_\sigma(x_{t-1} \mid x_0)$ 采 $x_{t-1}$,再从 $q_\sigma(x_t \mid x_{t-1}, x_0)$ 采 $x_t$,边际:

$$x_t = \underbrace{\sqrt{\bar\alpha_t}\, x_0}_{\text{均值}} + \underbrace{\sqrt{1-\bar\alpha_t}\, \epsilon}_{\text{噪声,方差 }(1-\bar\alpha_t)}$$

通过线性高斯计算可验证均值方差正确。完整的代数推导比较繁琐,DDIM 论文的 Lemma 1 给出了详细形式。

### 关键启示

这个构造**留出了 $\sigma_t$ 这个自由度**——它只影响**联合分布**(也就是 $x_{t-1}$ 和 $x_t$ 的相关性),不影响**边际**。我们后面会看到,$\sigma_t$ 从 0 到 DDPM 后验方差是一个连续族,DDPM 是这个族的特定一员。

{% enddetails %}

---

## 四、重新推导逆向过程

有了 $q_\sigma(x_{t-1} \mid x_t, x_0)$ 的非马尔可夫构造,我们如何用训练好的 $\epsilon_\theta$ 采样?

### 4.1 套路一样:用 $\epsilon_\theta$ 反推 $\hat x_0$

训练时网络学的是 $\epsilon_\theta(x_t, t) \approx \mathbb{E}[\epsilon \mid x_t]$。给定 $x_t$,反推 $x_0$ 的估计(Tweedie 公式):

$$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$$

这一步和 DDPM 的逻辑完全一样——网络给的不是真实 $x_0$,而是给定 $x_t$ 下 $x_0$ 的**条件均值**(也叫"模糊版的 $x_0$",越靠近 $t=0$ 越准)。

### 4.2 DDIM 采样公式

把 $\hat x_0$ 代入 $q_\sigma(x_{t-1} \mid x_t, x_0)$ 的均值公式,从中采一个 $x_{t-1}$:

$$\boxed{x_{t-1} = \underbrace{\sqrt{\bar\alpha_{t-1}}\, \hat x_0}_{\text{(1) 朝 }x_0\text{ 方向迈一步}} + \underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t, t)}_{\text{(2) 重新加合适比例的噪声方向}} + \underbrace{\sigma_t\,\epsilon}_{\text{(3) 额外随机扰动}}}$$

$\epsilon \sim \mathcal{N}(0, I)$ 是新采的噪声。

每一项都有清晰的物理含义:

- **(1)** 朝估计的 $x_0$ 走一步。$\sqrt{\bar\alpha_{t-1}}$ 控制"走多近"——越接近 $t=0$,$\bar\alpha$ 越大,这一项越主导
- **(2)** 重新引入"指向当前噪声方向"的成分。注意系数从 $\sqrt{1-\bar\alpha_{t-1}}$ 变成了 $\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}$——**为额外的随机性 $\sigma_t \epsilon$ 让出方差**
- **(3)** 额外随机扰动。$\sigma_t = 0$ 时这一项消失,采样**完全确定**

---

## 五、$\eta$ 参数:DDPM 和 ODE 的连续族

DDIM 论文用一个统一参数 $\eta \in [0, 1]$ 把 $\sigma_t$ 写成:

$$\sigma_t = \eta\, \tilde\beta_t^{1/2} = \eta\, \sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}\,\sqrt{1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}}$$

($\tilde\beta_t$ 是 DDPM 后验方差)

| $\eta$ 取值 | 行为 | 名字 |
|---|---|---|
| $\eta = 1$ | $\sigma_t = \tilde\beta_t^{1/2}$,DDPM 的后验方差 | **DDPM**(还原) |
| $\eta = 0$ | $\sigma_t = 0$,完全确定性 | **DDIM** |
| $\eta \in (0, 1)$ | 中间值,部分确定性 | DDIM 家族 |

**所以 DDPM 和 DDIM 不是两个对立的算法,而是同一个连续族的两个端点**——同一个训练好的 $\epsilon_\theta$,采样时切换 $\eta$ 就在两者之间过渡。

### 5.1 $\eta = 0$:确定性映射

当 $\eta = 0$、$\sigma_t = 0$,采样公式简化为:

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\, \hat x_0 + \sqrt{1-\bar\alpha_{t-1}}\,\epsilon_\theta(x_t, t)$$

这是个**完全确定**的映射——给定起点 $x_T$,逐步迭代得到的 $x_0$ 是唯一的。这带来两个好处:

1. **可逆**:理论上可以把图反向推回噪声(image inversion)
2. **可插值**:在噪声空间做插值,生成连续的图像过渡

### 5.2 $\eta = 1$:还原 DDPM

当 $\eta = 1$,代入会发现 DDIM 公式恰好等价于 DDPM 的 ancestral sampling 公式(经过一些代数化简)。**所以 DDPM 是 DDIM 在 $\eta = 1$ 时的特例**——DDIM 是更一般的框架。

---

## 六、跳步采样:为什么 DDIM 能加速

到这里,DDIM 在 $\eta = 0$ 下变成了一个**确定性迭代映射**。但它还是要走 $T$ 步——加速从哪来?

### 6.1 关键:公式不依赖马尔可夫性

回头看 DDIM 的采样公式:

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\, \hat x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t, t) + \sigma_t \epsilon$$

它**只依赖** $\bar\alpha_t$ 和 $\bar\alpha_{t-1}$——这两个累积量。**它根本不要求 $t-1$ 是 $t$ 的前一个 timestep**!

我们可以**任选**一个时间子序列 $\tau_1 < \tau_2 < \cdots < \tau_S$($S \ll T$),把公式里的 $(t, t-1)$ 替换成 $(\tau_{i+1}, \tau_i)$,采样过程变成 $S$ 步:

$$x_{\tau_i} = \sqrt{\bar\alpha_{\tau_i}}\, \hat x_{0,\tau_{i+1}} + \sqrt{1-\bar\alpha_{\tau_i}-\sigma_{\tau_{i+1}}^2}\,\epsilon_\theta(x_{\tau_{i+1}}, \tau_{i+1}) + \sigma_{\tau_{i+1}} \epsilon$$

其中 $\hat x_{0,\tau_{i+1}}$ 是用 $x_{\tau_{i+1}}$ 反推的 $x_0$ 估计。

### 6.2 例子:1000 步 → 50 步

最常用的子序列是**等间隔**(uniform 或 quadratic):

```
T = 1000, S = 50
均匀: τ = [20, 40, 60, ..., 1000]
平方: τ = [round((i/S)² * T) for i in 1..S]  # 起步密、后段稀
```

只要训练好的 $\bar\alpha$ schedule 在这些点的值有了,就能用——不需要重训。

### 6.3 为什么 DDPM 不能这么跳?

DDPM 的祖先采样要求**严格按 $\beta_t$ 的 schedule 一步步走**——因为它的逆向过程是按马尔可夫链定义的,跳步意味着用了错误的 $\beta$ 组合,导致采样的方差不对。

DDIM 因为是**直接定义 $q_\sigma(x_{t-1} \mid x_t, x_0)$**(非马尔可夫),公式中只出现 $\bar\alpha$ 这种累积量,**任意子序列都自洽**——这是它能跳步的根本原因。

### 6.4 步数 vs 质量

DDIM 论文实验:

| 步数 | DDPM (η=1) FID | DDIM (η=0) FID |
|---|---|---|
| 1000 | 4.04 | 4.16 |
| 100 | 11.0 | 4.16 |
| 50 | ~30 | 4.62 |
| 20 | 崩坏 | 6.8 |
| 10 | 崩坏 | 13.4 |

**关键观察**:

- DDPM 在步数减少时**质量崩坏**——因为方差累积不对
- DDIM 即使 50 步也保持质量——快 20 倍,几乎免费

这就是 DDIM 在工程上影响力巨大的原因——同一个训练好的模型,采样从 30 秒压到 1.5 秒。

---

## 七、和 Probability Flow ODE 的关系

回忆 Score SDE 那一讲:VP-SDE 对应的 Probability Flow ODE 是

$$\frac{dx}{dt} = -\frac{1}{2}\beta(t)\, x - \frac{1}{2}\beta(t)\,\nabla_x \log p_t(x)$$

定理:**DDIM($\eta = 0$)恰好是这个 ODE 的特定离散化**(具体地,做一个变量替换 $\bar x = x/\sqrt{\bar\alpha}$ 后的 Euler 方法)。

这个对应解释了几件事:

1. **为什么 DDIM 是确定性的**:它本质上是 ODE,ODE 当然是确定性的
2. **为什么能跳步**:ODE 的数值解可以用大步长(只是误差大);diffusion ODE 解的轨迹相对光滑,$50$ 步的 Euler 已经很准
3. **为什么可以做 image inversion**:ODE 是可逆的——从 $x_0$ 反向跑就能回到 $x_T$
4. **后续加速器的天花板**:DPM-Solver(2022)是更高阶的 ODE 数值方法(Heun、RK4),对同样的 ODE 用更高阶离散化,**10 步**就能达到 DDIM 50 步的质量

所以 DDIM 是连接"DDPM 离散概率视角"和"Score SDE 连续 ODE 视角"的重要桥梁——它本质上是同一个 ODE 的一个具体离散化。

---

## 八、DDIM 的额外能力

确定性带来一些 DDPM 做不到的事。

### 8.1 一致性:同噪声 → 同图

DDIM($\eta=0$)是确定性的,所以 $x_T$ 决定 $x_0$。这意味着可以把 $x_T$ 当成图的"语义编码"——同一个 $x_T$ 永远生成同一张图。

### 8.2 噪声空间插值

给定两个噪声 $x_T^{(1)}, x_T^{(2)}$,在它们之间做球面线性插值(slerp):

$$x_T^{(\lambda)} = \frac{\sin((1-\lambda)\theta)}{\sin\theta}\, x_T^{(1)} + \frac{\sin(\lambda\theta)}{\sin\theta}\, x_T^{(2)}$$

(其中 $\theta = \arccos(\hat x_T^{(1)} \cdot \hat x_T^{(2)})$,$\hat\cdot$ 是单位化)

每个 $x_T^{(\lambda)}$ 经 DDIM 采样得到一张图,$\lambda$ 从 0 到 1,生成一系列**平滑过渡**的图像。这是 DDPM 做不到的(DDPM 的 $\sigma_t z$ 把过渡破坏掉)。

为什么用 slerp 而不是 lerp(线性)?因为 $x_T \sim \mathcal{N}(0, I)$,在球面附近概率最大;线性插值的中间点会"塌进球内",离 $\mathcal{N}(0, I)$ 高密度区远,生成图模糊。slerp 沿球面走,中间点也在高密度区。

### 8.3 Image inversion:从图反推噪声

DDIM 的更新公式是确定性映射 $f_\theta(x_t, t) \to x_{t-1}$。我们可以**反过来求逆**——把 $x_{t-1}$ 映射回 $x_t$:

$$x_t = \sqrt{\bar\alpha_t}\,\hat x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_{t-1}, t-1)$$

(用 $\epsilon_\theta(x_{t-1}, t-1)$ 而不是 $\epsilon_\theta(x_t, t)$——这是个近似,但实践中很有效)

这给了我们**给定图 → 找到对应噪声**的能力,叫 **DDIM Inversion**。它是后来很多图像编辑工作的基础(Prompt-to-Prompt、Null-text Inversion 等)——拿到噪声后,通过修改 prompt 重新采样,实现"保留结构,改变内容"的编辑。

---

## 九、PyTorch 实现

DDIM 的实现极简——**网络和 DDPM 用同一个**(UNet + 时间嵌入 + ε 预测),只改采样函数。训练代码完全不变。

### 9.1 跳步索引

| 项 | 内容 |
|---|---|
| **输入** | 总步数 `T`、采样步数 `S` |
| **输出** | 长度 `S` 的子序列 `tau`,从大到小排 |

```python
def get_ddim_timesteps(T, S, schedule='uniform'):
    if schedule == 'uniform':
        # 均匀间隔
        tau = torch.linspace(0, T - 1, S).long()
    elif schedule == 'quadratic':
        # 平方间隔(DDIM 论文常用,前期稀疏,后期密集)
        tau = (torch.linspace(0, math.sqrt(T - 1), S) ** 2).long()
    return tau   # (S,)
```

### 9.2 核心采样循环

| 项 | 内容 |
|---|---|
| **输入** | 训练好的 `model`、形状 `shape`、步数 `S`、$\eta$ |
| **输出** | $x_0$,形状 `shape` |

```python
@torch.no_grad()
def ddim_sample(model, shape, T=1000, S=50, eta=0.0, device='cuda'):
    """
    eta=0:  纯 DDIM(确定性)
    eta=1:  还原 DDPM(随机)
    eta∈(0,1): 中间状态
    """
    model.eval()
    tau = get_ddim_timesteps(T, S).to(device)        # (S,)
    # 在前面补一个 -1 表示"采到 x_0 那步对应的虚拟 t",方便循环
    tau_prev = torch.cat([torch.tensor([-1], device=device), tau[:-1]])

    x = torch.randn(shape, device=device)             # x_T

    for i in reversed(range(S)):                       # 从大 t 到小 t
        t = tau[i].item()
        t_prev = tau_prev[i].item()                    # 上一个采样点(可能 < 0)

        # 取 ᾱ_t 和 ᾱ_{t_prev}
        a_t = alpha_bar[t]
        a_prev = alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

        # 1. 网络预测 ε
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps_pred = model(x, t_batch)                   # (B, C, H, W)

        # 2. 反推 x_0 估计(Tweedie)
        x0_pred = (x - torch.sqrt(1 - a_t) * eps_pred) / torch.sqrt(a_t)
        # 通常 clamp 到 [-1, 1] 增强稳定性
        x0_pred = x0_pred.clamp(-1, 1)

        # 3. 算 σ_t (按 η 缩放)
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t)) * torch.sqrt(1 - a_t / a_prev)

        # 4. DDIM 更新公式
        dir_xt = torch.sqrt(1 - a_prev - sigma ** 2) * eps_pred  # 噪声方向项
        if eta > 0 and i > 0:
            noise = sigma * torch.randn_like(x)
        else:
            noise = 0
        x = torch.sqrt(a_prev) * x0_pred + dir_xt + noise

    return x
```

逐行对应公式:

1. **`eps_pred = model(x, t)`**:网络估当前噪声
2. **`x0_pred = ...`**:Tweedie 反推 $\hat x_0$
3. **`sigma = ...`**:按 $\eta$ 缩放出噪声系数
4. **`dir_xt + noise`**:朝 $x_0$ 走 + 重新加噪声方向 + 随机扰动

整个函数大约 25 行。

### 9.3 Image Inversion(反推噪声)

```python
@torch.no_grad()
def ddim_invert(model, x_0, T=1000, S=50, device='cuda'):
    """
    给定一张图 x_0,反推对应的 x_T(η=0 假设)
    """
    model.eval()
    tau = get_ddim_timesteps(T, S).to(device)
    tau_prev = torch.cat([torch.tensor([-1], device=device), tau[:-1]])

    x = x_0.to(device)

    for i in range(S):                                 # 从小 t 到大 t,反过来跑
        t = tau[i].item()
        t_prev = tau_prev[i].item()

        a_t = alpha_bar[t]
        a_prev = alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

        # 在 x(代表 x_{t_prev})处估噪声,作为整段的 ε 近似
        t_prev_batch = torch.full((x.size(0),), max(t_prev, 0), device=device, dtype=torch.long)
        eps_pred = model(x, t_prev_batch)

        # 反推 x_0
        x0_pred = (x - torch.sqrt(1 - a_prev) * eps_pred) / torch.sqrt(a_prev)
        x0_pred = x0_pred.clamp(-1, 1)

        # 反向更新到 x_t
        x = torch.sqrt(a_t) * x0_pred + torch.sqrt(1 - a_t) * eps_pred

    return x   # 此时 x ≈ x_T
```

注意**不能完全精确反推**——因为我们用的是 $\epsilon_\theta(x_{t_\text{prev}})$ 近似 $\epsilon_\theta(x_t)$。误差很小但累积会有偏差,这是后续 Null-text Inversion 等改进要解决的问题。

### 9.4 噪声空间 slerp 插值

```python
def slerp(z1, z2, lam):
    """球面线性插值,适合高斯噪声"""
    z1_flat = z1.flatten(1)
    z2_flat = z2.flatten(1)
    cos_theta = (z1_flat * z2_flat).sum(-1) / (z1_flat.norm(dim=-1) * z2_flat.norm(dim=-1))
    cos_theta = cos_theta.clamp(-1, 1)
    theta = torch.arccos(cos_theta)
    sin_theta = torch.sin(theta)
    w1 = torch.sin((1 - lam) * theta) / sin_theta
    w2 = torch.sin(lam * theta) / sin_theta
    return (w1.view(-1, *([1] * (z1.dim() - 1))) * z1
          + w2.view(-1, *([1] * (z1.dim() - 1))) * z2)


# 使用:生成 11 张过渡图
z1 = torch.randn(1, 3, 32, 32)
z2 = torch.randn(1, 3, 32, 32)
imgs = []
for lam in torch.linspace(0, 1, 11):
    z = slerp(z1, z2, lam.item())
    img = ddim_sample(model, z.shape, S=50, eta=0)  # 注意要把 z 当起点喂进去
    imgs.append(img)
```

---

## 十、和 DDPM 的对照

| 维度 | DDPM | DDIM($\eta=0$) |
|---|---|---|
| 训练目标 | $L_{\text{simple}}$ | **完全相同**,网络可共用 |
| 前向过程 | 马尔可夫 | 非马尔可夫(显式条件 $x_0$) |
| 逆向 | 祖先采样 | 直接定义 $q_\sigma(x_{t-1} \mid x_t, x_0)$ |
| 随机性 | 每步加 $\sigma_t z$ | 完全确定 |
| 步数 | $T = 1000$ | 通常 50,可低至 10 |
| 跳步 | 不能跳 | 任意子序列都自洽 |
| 同输入相同输出 | 否(随机) | 是(确定) |
| 插值 / Inversion | 难 | 自然 |
| 与 ODE 关系 | SDE 离散化 | Probability Flow ODE 离散化 |

**最重要的事实**:**同一个训练好的 $\epsilon_\theta$,既能用 DDPM 采也能用 DDIM 采**。研究者训练时不需要选,推理时按需切换。这种"训练目标统一,推理灵活"的特性是 DDIM 工程上极受欢迎的根本原因。

---

## 十一、复盘:DDIM 的精神

DDIM 教给我们的核心方法论是:

> **当你有一个训练目标时,检查它实际依赖了什么。如果它只依赖于"边际",那"联合"就是自由的——你可以重新设计联合,得到一个新算法,而不需要重新训练。**

具体到 diffusion:

- DDPM 训练**只依赖** $q(x_t \mid x_0)$
- DDIM 重新设计前向 $q_\sigma(x_{1:T} \mid x_0)$,边际不变,联合改了
- 同一个 $\epsilon_\theta$ 在新的联合下,采样可以跳步、确定化

这种"训练时锁定边际,推理时改造路径"的思路,后来被 DPM-Solver、Consistency Models、Flow Matching 等大量复用——它们都是在不同程度上**重新利用同一个训练目标对应的 score**。

整个 diffusion 加速研究的"祖宗"是 DDIM,它揭示了一个本来不显眼的自由度。

---

## 十二、要点回顾

- DDPM 慢的根源:1000 步祖先采样,马尔可夫性强制每步只能挪一格
- 关键洞察:**$L_{\text{simple}}$ 只依赖边际 $q(x_t \mid x_0)$**,前向过程的具体马尔可夫结构无关紧要
- DDIM 直接钦定非马尔可夫的 $q_\sigma(x_{t-1} \mid x_t, x_0)$,边际守恒,联合更灵活
- 采样公式拆为三项:**朝 $\hat x_0$ 走 + 噪声方向项 + 随机扰动 $\sigma_t \epsilon$**
- $\eta$ 参数把 DDPM($\eta=1$)和确定性 DDIM($\eta=0$)统一在一个连续族
- 跳步采样:公式只用 $\bar\alpha$ 的累积量,任意子序列 $\tau_1 < \cdots < \tau_S$ 都自洽
- 50 步的 DDIM 质量接近 1000 步的 DDPM,**~20 倍加速且不需重训**
- $\eta = 0$ 的 DDIM 等价于 Probability Flow ODE 的特定离散化——这解释了它的确定性、可逆性、加速空间
- 额外能力:噪声空间 slerp 插值、image inversion(被后续编辑方法广泛使用)
- 训练目标完全没动——DDIM 只是改了**怎么用**这个训练好的网络
