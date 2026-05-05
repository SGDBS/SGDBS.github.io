---
title: Chapter 5. NCSN Models
categories: Diffusion Models
date: 2026-05-01 13:00:00
mathjax: true
tags:
    - AI
    - Diffusion Models
---

## 一、NCSN 是什么

**NCSN = Noise Conditional Score Network**(噪声条件 score 网络)

是 Yang Song(就是 Score SDE 的作者)和 Stefano Ermon 在 2019 年提出的方法,论文 "Generative Modeling by Estimating Gradients of the Data Distribution"。

它和 DDPM 是 diffusion 两条独立线索的"成熟代表":

- **DDPM(2020)**:从变分推断和马尔可夫链出发
- **NCSN(2019)**:从 score matching 和 Langevin 动力学出发

最终它们在 Score SDE 框架下被统一,但**NCSN 比 DDPM 早一年**——它其实是更早的"现代 diffusion"工作。

---

## 二、它要解决什么问题

回顾我们讲 score matching 时讲的两个**核心障碍**:

### 障碍 1:原版 score matching 算不动

Hyvärinen 2005 原版的损失里有 trace 项 $\text{tr}(\nabla_x s_\theta)$,需要 $d$ 次反向传播,高维下不可行。

### 障碍 2:低密度区域 score 估不准

即使用 DSM(Denoising Score Matching)绕开了 trace 问题,**还有一个更根本的问题**:

- 真实数据集中在低维流形上,$\mathbb{R}^d$ 的绝大部分空间是低密度区域
- 训练数据稀疏,网络在低密度区域学不准 score
- Langevin 采样从噪声 $\mathcal{N}(0, I)$ 出发,**初始点几乎肯定在低密度区域**——score 给的方向是错的,采样不收敛

这导致单尺度 DSM + Langevin 完全不能 work——理论可行,实际从噪声出发就走不动。

**NCSN 的核心贡献:用多尺度噪声 + 退火 Langevin 解决这个问题**。

---

## 三、关键想法:多尺度噪声

NCSN 的核心洞察:

> **同一个 score 网络,在多个噪声尺度上同时学**。大噪声让低密度区域有足够数据,小噪声让结果接近真实数据。

具体地,选 $L$ 个噪声尺度,几何递减:

$$\sigma_1 > \sigma_2 > \cdots > \sigma_L$$

通常 $\sigma_1 \approx$ 数据的最大距离尺度(很大),$\sigma_L \approx 0.01$(很小)。

对每个尺度 $\sigma_i$,定义加噪分布:

$$q_{\sigma_i}(\tilde x \mid x) = \mathcal{N}(\tilde x; x, \sigma_i^2 I)$$

$$q_{\sigma_i}(\tilde x) = \int q_{\sigma_i}(\tilde x \mid x)\, p_{\text{data}}(x)\, dx$$

然后训练**一个**网络 $s_\theta(x, \sigma)$,对每个 $\sigma_i$ 都学对应的 score $\nabla_{\tilde x} \log q_{\sigma_i}(\tilde x)$。

### 为什么"多尺度"能解决问题?

**关键观察**:

- **大尺度 $\sigma_1$**:$q_{\sigma_1}$ 把数据"涂抹"到整个空间——任何点都有显著密度,score 处处可学
- **小尺度 $\sigma_L$**:$q_{\sigma_L} \approx p_{\text{data}}$,score 接近真实数据 score
- **中间尺度**:平滑过渡

**采样时**:从大尺度开始(那里 score 准),逐步退火到小尺度(那里数据精细)。这样每一步 Langevin 都用着"准确的 score",不会卡住。

这就是"退火"的精髓——**用大尺度先把样本"拉到正确的大方向上",再用小尺度精修细节**。

---

## 四、训练目标

### 4.1 加权 DSM 损失

对每个尺度 $\sigma_i$,DSM 损失是:

$$\ell(\theta; \sigma_i) = \frac{1}{2}\mathbb{E}_{x \sim p_{\text{data}}, \tilde x \sim q_{\sigma_i}(\tilde x \mid x)}\left[\left\| s_\theta(\tilde x, \sigma_i) - \nabla_{\tilde x} \log q_{\sigma_i}(\tilde x \mid x) \right\|^2\right]$$

代入条件 score $\nabla_{\tilde x} \log q_{\sigma_i}(\tilde x \mid x) = -(\tilde x - x)/\sigma_i^2$:

$$\ell(\theta; \sigma_i) = \frac{1}{2}\mathbb{E}\left[\left\| s_\theta(\tilde x, \sigma_i) + \frac{\tilde x - x}{\sigma_i^2} \right\|^2\right]$$

总损失是所有尺度的加权求和:

$$\mathcal{L}(\theta) = \frac{1}{L}\sum_{i=1}^L \lambda(\sigma_i)\, \ell(\theta; \sigma_i)$$

### 4.2 权重选择 $\lambda(\sigma_i) = \sigma_i^2$

直接平均会出问题:小尺度的 $1/\sigma^2$ 项会让损失数值非常大,大尺度的项相对很小。**不同尺度的损失数值不在一个量级**,训练时小尺度主导,大尺度学不好。

NCSN 的方案:取 $\lambda(\sigma_i) = \sigma_i^2$,这样每个尺度的损失数值大致相同。

代入后:

$$\sigma_i^2 \cdot \ell = \frac{1}{2}\mathbb{E}\left[\left\| \sigma_i\, s_\theta(\tilde x, \sigma_i) + \frac{\tilde x - x}{\sigma_i} \right\|^2\right]$$

注意 $\frac{\tilde x - x}{\sigma_i} = \epsilon$(因为 $\tilde x = x + \sigma_i \epsilon$),所以:

$$\boxed{\mathcal{L}(\theta) = \frac{1}{2L}\sum_{i=1}^L \mathbb{E}_{x, \epsilon}\left[\left\| \sigma_i\, s_\theta(\tilde x, \sigma_i) + \epsilon \right\|^2\right]}$$

或定义 $\epsilon_\theta := -\sigma_i\, s_\theta$:

$$\mathcal{L} = \frac{1}{2L}\sum_i \mathbb{E}\left[\| \epsilon - \epsilon_\theta(\tilde x, \sigma_i) \|^2\right]$$

**和 DDPM 的 $L_{\text{simple}}$ 几乎完全一样**!唯一的区别是用 $\sigma_i$ 替代 $t$ 作为时间参数。这不是巧合——下面会讲它们的精确对应。

---

## 五、采样:退火 Langevin 动力学

回忆基础 Langevin 动力学:

$$x_{k+1} = x_k + \frac{\eta}{2}\, s_\theta(x_k) + \sqrt{\eta}\, z_k$$

NCSN 的退火版本:

```
初始化:x ~ N(0, σ_1² I)        # 从最大噪声分布采

for i = 1, 2, ..., L:           # 从大尺度到小尺度
    η_i = ε · σ_i² / σ_L²       # 步长(随尺度衰减)
    
    for k = 1, ..., K:          # 在尺度 σ_i 上跑 K 步 Langevin
        z ~ N(0, I)
        x ← x + (η_i / 2) · s_θ(x, σ_i) + sqrt(η_i) · z

return x
```

**关键设计**:

- **初始化**:从 $\mathcal{N}(0, \sigma_1^2 I)$ 出发,这是 $q_{\sigma_1}$ 的高密度区域(因为 $q_{\sigma_1}$ 也接近 $\mathcal{N}(0, \sigma_1^2 I)$,当 $\sigma_1$ 远大于数据尺度时)。所以 score 在这个起点学得准
- **退火**:逐步从 $\sigma_1$ 到 $\sigma_L$,每个尺度跑 $K$ 步 Langevin
- **步长缩放**:$\eta_i \propto \sigma_i^2$,保证不同尺度的"信噪比"恒定

**直觉图景**:

```
σ_1 (大噪声):
  数据被涂抹成几乎均匀的雾
  score 处处合理
  样本被拉到"数据团块的大方向"

σ_中间:
  数据轮廓显现
  score 引导样本走向具体区域
  样本被精修到"具体的子团块"

σ_L (小噪声):
  数据接近真实分布
  score 接近真实 score
  样本被精修到"具体的数据点"
```

每个尺度做的事 = "在当前模糊度下做局部 Langevin 采样"。退火是在做**多尺度优化**。

---

## 六、PyTorch 实现要点

NCSN 的网络架构和 DDPM 的 UNet **几乎一样**——都是带条件输入(NCSN 是 σ,DDPM 是 t)的 noise predictor。所以我们重点说**和 DDPM 不同的几个关键点**,完整 UNet/数据预处理参考 [DDPM 那章的实现部分](Chapter6/)。

### 6.1 多尺度噪声表

| 量 | 形状 | 含义 |
|---|---|---|
| `sigmas` | `(L,)` | $\sigma_1 > \sigma_2 > \cdots > \sigma_L$,几何递减,$\sigma_1$ 远大于数据尺度 |

```python
import torch

L = 232                              # 噪声尺度数量
sigma_max = 50.0                     # 远大于数据尺度,让 q_{σ_1} ≈ N(0, σ_1² I)
sigma_min = 0.01                     # 接近 0,让 q_{σ_L} ≈ p_data

# 几何递减:σ_l = σ_max · (σ_min/σ_max)^((l-1)/(L-1))
sigmas = torch.exp(torch.linspace(
    math.log(sigma_max), math.log(sigma_min), L
))
```

### 6.2 网络:把 σ 作为条件输入

NCSN 的网络叫 **NCSN(Noise Conditional Score Network)**,接受 `(x, σ_idx)` 而不是 DDPM 的 `(x, t)`。架构基本一致——把"时间嵌入"换成"噪声尺度嵌入":

```python
class NoiseConditionalUNet(nn.Module):
    def __init__(self, base_ch=128, n_sigmas=L):
        super().__init__()
        # 直接学一个 σ_idx → 嵌入的 lookup table(或者用正弦嵌入也行)
        self.sigma_emb = nn.Embedding(n_sigmas, 4 * base_ch)
        # ... 其余 ResBlock / 下采样 / 上采样和 DDPM 的 UNet 完全一样
        # 只是网络的最后输出尺度上,常见做法是除以 σ:
        # 即网络输出 raw,然后 score = raw / σ —— 让网络输出维持单位方差

    def forward(self, x, sigma_idx, sigmas):
        emb = self.sigma_emb(sigma_idx)              # (B, 4C)
        # ... UNet 主体 ...
        raw = unet_body(x, emb)                       # (B, C, H, W)
        # 关键:把 score 除以 σ,等价于输出"标准化的 score"
        sigma = sigmas[sigma_idx].view(-1, 1, 1, 1)
        return raw / sigma
```

**为什么除以 σ?** 真实 score 在不同 σ 下尺度差异巨大($\nabla \log q_\sigma \sim 1/\sigma$),让网络直接输出原始 score 数值不稳。**让网络输出 $\sigma \cdot s_\theta$,即"归一化的预测",再除以 $\sigma$ 还原**——这是 NCSN 的工程稳健性技巧。

### 6.3 训练:加权 DSM 损失

回忆训练目标(第四节推过):

$$\mathcal{L} = \frac{1}{2L}\sum_l \mathbb{E}_{x_0, \epsilon}\left[\| \sigma_l\, s_\theta(\tilde x_l, \sigma_l) + \epsilon \|^2\right]$$

```python
device = 'cuda'
model = NoiseConditionalUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
sigmas = sigmas.to(device)

for epoch in range(epochs):
    for x_0, _ in train_loader:                       # x_0: (B, 3, 32, 32)
        x_0 = x_0.to(device)
        B = x_0.size(0)

        # 1. 每个样本随机选一个噪声尺度
        sigma_idx = torch.randint(0, L, (B,), device=device)
        sigma = sigmas[sigma_idx].view(B, 1, 1, 1)

        # 2. 加噪
        eps = torch.randn_like(x_0)
        x_tilde = x_0 + sigma * eps

        # 3. 网络预测 score
        score = model(x_tilde, sigma_idx, sigmas)     # (B, 3, 32, 32)

        # 4. 加权 DSM 损失:权重 λ(σ) = σ²,展开后形式如下
        # ‖ σ · s_θ + ε ‖² = ‖ σ · score + ε ‖²
        target = -eps / sigma                         # 真实条件 score
        loss = ((score - target) ** 2 * sigma ** 2).mean()
        # 或等价写法:
        # loss = ((sigma * score + eps) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 6.4 退火 Langevin 采样

| 项 | 内容 |
|---|---|
| **输入** | 形状 `shape = (n, 3, 32, 32)` |
| **输出** | $x_0$,从数据分布采的样本 |
| **过程** | 从 $\mathcal{N}(0, \sigma_1^2 I)$ 出发,每个尺度跑 $K$ 步 Langevin,从大尺度到小尺度退火 |

```python
@torch.no_grad()
def annealed_langevin(model, shape, sigmas, n_steps_per_sigma=100, eps_start=2e-5):
    """
    eps_start: 最小尺度的步长,其它按 σ²/σ_L² 缩放
    """
    model.eval()
    L = len(sigmas)
    x = torch.randn(shape, device='cuda') * sigmas[0]    # 从 N(0, σ_1² I) 开始

    for l in range(L):
        sigma_l = sigmas[l]
        sigma_idx = torch.full((shape[0],), l, device='cuda', dtype=torch.long)
        eta = eps_start * (sigma_l ** 2) / (sigmas[-1] ** 2)   # 步长缩放

        for k in range(n_steps_per_sigma):
            score = model(x, sigma_idx, sigmas)               # 学到的 score
            z = torch.randn_like(x)
            x = x + 0.5 * eta * score + torch.sqrt(eta) * z

    return x
```

每个 σ 上跑 K 步 Langevin,做的事:

1. **`score = model(x, ...)`**:在当前 noisy 样本处,网络给出 score 估计
2. **漂移项 `0.5 * eta * score`**:把样本拉向 $q_{\sigma_l}$ 的高密度区
3. **扩散项 `√eta * z`**:加随机性保证遍历

### 6.5 NCSN vs DDPM 的代码差异速查

| 维度 | NCSN | DDPM |
|---|---|---|
| 噪声参数 | $\sigma_l$(连续值) | $t$(整数 timestep) |
| 嵌入方式 | `Embedding(L, dim)` 或 σ 的正弦嵌入 | `t` 的正弦嵌入 |
| 加噪公式 | `x̃ = x + σ · ε`(VE,加性) | `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε`(VP,收缩+加噪) |
| 损失 | `‖ σ · score + ε ‖²` | `‖ ε - ε_θ ‖²` |
| 采样 | annealed Langevin,每个 σ K 步 | ancestral sampling,T 步 |
| 起点 | `N(0, σ_1² I)`,σ_1 远大于数据 | `N(0, I)`,直接标准高斯 |
| 步数 | $L \times K$,如 232 × 100 = 23200(更多) | $T = 1000$ |

**核心算法不变,只是参数化不同**——这就是为什么下一讲 Score SDE 能把它们统一在 SDE 框架下。

---

## 七、和 DDPM 的精确对应

让我们把 NCSN 和 DDPM 摆在一起:

| | NCSN | DDPM |
|---|---|---|
| 噪声参数 | $\sigma_i$(尺度,几何递减) | $\beta_t$(schedule,通常线性) |
| 加噪过程 | $\tilde x = x + \sigma_i \epsilon$ | $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$ |
| 终态分布 | $\mathcal{N}(0, \sigma_1^2 I)$,**方差大** | $\mathcal{N}(0, I)$,**方差守恒** |
| 网络预测 | score $s_\theta$ | 噪声 $\epsilon_\theta$ |
| 训练损失 | 加权 DSM($\lambda = \sigma^2$) | $L_{\text{simple}}$ |
| 采样 | 退火 Langevin | Ancestral sampling |
| SDE 对应 | VE-SDE | VP-SDE |

### 6.1 关键差异 1:VE vs VP

**NCSN 是 Variance Exploding(VE)**:噪声直接加,不收缩信号,$\tilde x$ 的方差随尺度增大。

**DDPM 是 Variance Preserving(VP)**:同时收缩信号 + 加噪,方差守恒。

之前我们讨论过为什么 DDPM 选 VP——朴素加噪让 $x_T$ 不收敛到固定分布。但 NCSN 用 VE,它怎么处理这个问题?

NCSN 的处理:**让 $\sigma_1$ 足够大**,使得 $\sigma_1$ 远大于数据尺度。这时

$$q_{\sigma_1}(\tilde x) = \int q_{\sigma_1}(\tilde x \mid x) p_{\text{data}}(x)\, dx \approx \mathcal{N}(0, \sigma_1^2 I)$$

(因为 $\sigma_1^2 I \gg \text{Var}(x)$,加噪后数据信息几乎被完全淹没)

所以**实际上 $q_{\sigma_1}$ 也近似是 $\mathcal{N}(0, \sigma_1^2 I)$**,只是尺度大了——本质和 VP 没差很多。

### 6.2 关键差异 2:网络预测什么

NCSN 直接预测 **score**,DDPM 预测 **noise**。两者关系:

$$s_\theta(\tilde x, \sigma) = -\frac{\epsilon_\theta(\tilde x, \sigma)}{\sigma}$$

(对 NCSN)和

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}}$$

(对 DDPM)

只差一个标量,**完全等价**。两个网络做的"事情"一样,只是输出的"单位"不同。

### 6.3 关键差异 3:采样方式

DDPM:严格按 $p_\theta(x_{t-1} \mid x_t)$ 采样,$T$ 步走到底。

NCSN:退火 Langevin,每个尺度跑 $K$ 步,共 $L \times K$ 步。

两者都是迭代去噪,具体公式略有不同,但都是同一个 SDE 的不同离散化(下面再说)。

---

## 八、SDE 视角:统一两者

按上一讲的 Score SDE 框架:

### NCSN 对应 VE-SDE

$$dx = \sqrt{\frac{d\sigma^2(t)}{dt}}\, dw$$

边际:$p_t(x \mid x_0) = \mathcal{N}(x_0, \sigma^2(t) I)$。

NCSN 的离散尺度 $\sigma_i$ 就是 $\sigma(t)$ 的离散采样,退火 Langevin 是这个 SDE 反向 SDE 的一种特定离散化(用 score 项做 Langevin,忽略漂移因为 VE 没有漂移)。

### DDPM 对应 VP-SDE

$$dx = -\frac{1}{2}\beta(t) x\, dt + \sqrt{\beta(t)}\, dw$$

边际:$p_t(x \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha(t)}\, x_0,\, (1-\bar\alpha(t))I)$。

DDPM 的离散步是 VP-SDE 的 Euler-Maruyama 离散化,ancestral sampling 是反向 VP-SDE 的离散化。

### 一图对照

```
                       Score SDE 框架
                            │
              ┌─────────────┴─────────────┐
              │                           │
          VE-SDE                       VP-SDE
              │                           │
              ↓                           ↓
        离散化(NCSN)              离散化(DDPM)
              │                           │
        多尺度 σ_i                    schedule β_t
        VE 加噪                       VP 加噪
        预测 score                    预测 noise
        退火 Langevin                 Ancestral sampling
```

---

## 九、NCSN 的历史意义

NCSN 是 score-based 路线的"**第一次成功**":

- **2019** NCSN(v1):在 CIFAR-10 上 FID ≈ 25,质量略差于当时 GAN
- **2020** NCSN++(v2):改进版,FID ≈ 10,接近 SOTA GAN
- **2021** Score SDE(同作者):统一框架,质量超过 GAN

它的影响:

1. **第一次证明 score-based 方法在图像生成上能 work**——之前 score matching 只在低维玩具数据上演示过
2. **多尺度噪声**这一关键设计被 DDPM 间接吸收(DDPM 的 $T$ 步本质就是多尺度,只是从 VP 角度参数化)
3. **退火 Langevin**启发了后续大量采样器设计
4. **作者 Yang Song 的 Score SDE 论文**直接接在 NCSN 之后,把整个 score-based 路线推到顶峰

可以说,**NCSN 是现代 diffusion 的"另一个奠基人"**——和 DDPM 一起,定义了这条路线的所有核心元素。

---

## 十、为什么 DDPM 比 NCSN 出名?

公平地说,DDPM 在工程实践中传播得更广。原因:

1. **VP 比 VE 更稳定**:方差守恒让数值更可控,训练更顺
2. **预测噪声比 score 更直观**:$\epsilon \sim \mathcal{N}(0, I)$ 是标准化的,$s$ 的尺度依赖 $\sigma$
3. **Ancestral sampling 比退火 Langevin 简单**:每步只需一次网络调用,不需要 $K$ 步 Langevin
4. **DDPM 的论文写作更面向工程**:有清晰的算法、伪代码,容易复现
5. **后续的 DDIM、Stable Diffusion 都基于 DDPM 路线**:因为 VP 的闭式 $\bar\alpha_t$ 更方便

但**理论上 NCSN/Score SDE 是更通用的框架**——DDPM 是它的一个特例。Yang Song 的 Score SDE 论文实际上把 NCSN 升级成了一个能 cover DDPM 的统一理论。

---

## 十一、要点回顾

- **NCSN = Noise Conditional Score Network**,Song & Ermon 2019
- 解决 score-based 方法的两个老问题:trace 算不动(用 DSM)、低密度 score 不准(用多尺度噪声)
- 核心设计:**一个网络 + 多个尺度 + 退火 Langevin**
- 训练:加权 DSM,权重 $\lambda(\sigma) = \sigma^2$,最终形式和 DDPM 的 $L_{\text{simple}}$ 几乎一样
- 采样:从大尺度到小尺度退火,每个尺度跑 $K$ 步 Langevin
- 是 **VE-SDE** 的离散化(对比 DDPM 是 VP-SDE)
- 预测 score(对比 DDPM 预测 noise),两者只差一个标量
- 历史地位:**score-based 路线在图像生成上的第一次成功**,直接催生了 Score SDE 的统一框架
