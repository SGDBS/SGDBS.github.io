---
title: Chapter0 LLM 量化总览
categories: 学习笔记- AI Infra
date: 2026-06-17 14:00:00
mathjax: true
tags:
    - AI
    - AI Infra
    - 模型量化
---


**LLM 量化要把推理系统里最贵的显存、HBM 带宽和矩阵计算，转化成一套可控的数值近似问题。**

在大模型推理里，参数要从 HBM 读，activation 要在层之间流动，decode 阶段还要持续维护 KV cache。模型越大、上下文越长、batch 越高，瓶颈越不像“算不过来”，而更像“数据搬不动、显存放不下”。量化的价值就在这里：用更低 bit 的数近似原来的 FP16/BF16/FP32 权重和中间结果，从而降低显存、带宽和部分计算成本，同时尽量保持模型质量。

但量化不是免费午餐。它的核心矛盾是：

```
更低 bit
  -> 更省显存 / 更低带宽 / 更适合推理硬件
  -> 数值表示更粗糙
  -> rounding error 和 clipping error 变大
  -> 可能导致困惑度上升、输出变差、长上下文不稳定
```

所以量化不是一个孤立 dtype 选择，而是一套系统工程判断：哪些 tensor 可以低精度，哪些层必须保留高精度，scale 用什么粒度，activation outlier 怎么处理，kernel 和硬件收益能不能覆盖精度损失。

## 一、起因：为什么 LLM 推理需要量化

### 1.1 第一层压力是显存和带宽

如果只从模型代码看，量化像是在改数据类型；但从推理系统看，量化首先是在改成本结构。

Transformer block 里真正反复消耗显存带宽和算力的，是一层层 Linear：attention 里的 `q_proj`、`k_proj`、`v_proj`、`o_proj`，MLP 里的 `gate_proj`、`up_proj`、`down_proj`，以及最后的 `lm_head`。

这些矩阵权重在推理时是固定的，但每次 forward 都要被读取。权重越大，HBM 读带宽压力越大；batch 越高，KV cache 和 activation 也会进一步吃显存。对于 serving 系统来说，显存不是只影响“能不能放下模型”，还会影响：

- 能不能开更大的 batch。
- 能不能支持更长上下文。
- 能不能减少 HBM traffic。
- 能不能让 Tensor Core / INT 矩阵单元真正吃满。

这也是为什么 LLM 推理里最常见的第一步通常是 weight-only quantization：先把静态权重压下去，收益稳定，风险相对可控。

### 1.2 量化改变的是精度预算

真实系统里很少把所有 tensor 一刀切成低精度，更常见的是 mixed precision：

- weight 可以是 INT8 / INT4。
- activation 可能保留 FP16/BF16，也可能在 W8A8 里动态量化。
- KV cache 可能使用 INT8 / INT4 / FP8。
- softmax、norm、部分累加仍然保留高精度。

背后的判断是：不同 tensor 对误差的敏感度不同。weight 是静态的，可以离线分析和校准；activation 和输入相关，运行时动态变化，更容易遇到 outlier；KV cache 会贯穿整个 decode 过程，误差可能随生成长度累积。

量化真正要做的是分配精度预算，而不是机械地把所有 FP16 都换成 INT8。

## 二、主线：量化系统由哪些维度组成

### 2.1 数值格式：用什么数表示

浮点格式更擅长表达动态范围，整数格式更依赖显式 scale。

| 类型 | 常见位置 | 工程特点 |
| --- | --- | --- |
| FP32 | 训练、参考精度、敏感累加 | 精度高，动态范围大，成本最高 |
| BF16 | LLM 训练和部分推理 | 动态范围接近 FP32，训练稳定 |
| FP16 | GPU 推理/训练常用 | 硬件友好，范围比 BF16 小 |
| FP8 | 新硬件上的训练/推理加速 | 省带宽，但依赖 scaling 策略 |
| INT8 | 成熟推理量化 | 精度和速度平衡较好 |
| INT4 | LLM weight-only 常用 | 极省显存，但更依赖算法和 kernel |

如果只记一个判断：**浮点低精度更像保留范围、牺牲部分精度；整数低精度更像用 scale 重新定义数轴。**

### 2.2 映射公式：整数如何还原浮点

整数没有指数位，所以 INT8 / INT4 必须依赖 `scale / zero_point` 来表达真实浮点值。

最基础的公式是：

$$
x_{\mathrm{float}} \approx scale \cdot (x_{\mathrm{int}} - zero\_point)
$$

其中 `scale` 决定一个整数步长代表多大的真实值，`zero_point` 决定浮点 0 在整数空间里的位置。

量化误差主要来自两类：

- rounding error：浮点数映射到整数时要四舍五入。
- clipping error：超过表示范围的值会被截断。

前者来自步长不够细，后者来自范围不够大。几乎所有量化方法，本质上都在这两个误差之间重新分配预算。

### 2.3 映射方式：symmetric 还是 asymmetric

symmetric quantization 把 0 放在整数中心：

$$
x_{\mathrm{float}} \approx scale \cdot x_{\mathrm{int}},\quad zero\_point = 0
$$

它计算简单、硬件友好，常用于 weight quantization，尤其是 Linear 权重。

asymmetric quantization 允许 `zero_point != 0`：

$$
x_{\mathrm{float}} \approx scale \cdot (x_{\mathrm{int}} - zero\_point)
$$

它更适合非对称分布，比如 ReLU 后的非负 activation，但 zero_point 会给 GEMM 带来额外修正项，kernel 实现更复杂。

### 2.4 scale 粒度：一个 scale 管多少数

scale 不是越多越好，因为 scale metadata 本身也有存储和带宽开销；但 scale 太粗，又容易被 outlier 拉大。

| 粒度 | 常见对象 | 优点 | 代价 |
| --- | --- | --- | --- |
| per-tensor | 简单 activation / 传统 PTQ | 最简单，metadata 最少 | 最怕 outlier |
| per-channel | Linear / Conv weight | 精度和成本平衡好 | kernel 需要按列/通道读 scale |
| group-wise | INT4 weight-only | 适合 GPTQ/AWQ/W4A16 | scale 更多，解包和 dequant 更复杂 |
| per-token | LLM activation | 更适应动态输入 | runtime 要算 scale，kernel 更复杂 |
| per-head | KV cache | 适配 attention head 分布 | 长上下文质量和开销要权衡 |

scale 粒度是量化里最容易被低估的一层：同样是 INT8，per-tensor static activation INT8 和 per-token dynamic activation INT8 可能完全不是一个东西。

### 2.5 scale 来源：static、dynamic 和 calibration

`scale / zero_point` 并不是模型自动“学出来”的，它们要么离线估计，要么运行时计算。

- static quantization：用 calibration 数据提前统计 `scale / zero_point`，部署时固定使用。
- dynamic quantization：运行时根据当前输入动态计算 scale。

weight 几乎都是 static，因为权重固定、可离线统计、可离线打包。activation 可以 static，也可以 dynamic，取决于精度目标、kernel 能力和输入分布变化。KV cache 介于两者之间，因为它随 decode 长度增长，也随 token 分布变化。

calibration 的本质是：用一小批代表性数据，统计每一层 activation 的分布范围，从而确定 `scale / zero_point`。常见方法包括 `min / max`、`absmax`、`percentile`、`histogram`、`KL divergence` 和 `MSE search`。

calibration 真正难的地方不是跑一遍 forward，而是数据是否代表真实推理分布。短 prompt 校准出来的范围，不一定适合长上下文；通用文本校准出来的范围，也不一定适合代码、数学或特定业务输入。

## 三、和 LLM 模块的关系

### 3.1 Linear：量化的主战场

Linear 是：

$$
Y = XW + b
$$

LLM 里大量计算都落在 Linear 上：attention projection、MLP projection 和 `lm_head`。它同时占参数量、计算量和权重读取带宽的大头，所以绝大多数权重量化方法最终都要落到 Linear kernel 上。

常见组合：

- W8A16：weight INT8，activation FP16/BF16。
- W8A8：weight INT8，activation INT8。
- W4A16：weight INT4，activation FP16/BF16。

这些名字只描述 bit 数，不描述 scale 粒度、scale 来源、zero_point 策略和 kernel 路径。真实工程里必须继续问：weight 是 per-channel 还是 group-wise？activation 是 static 还是 dynamic？dequant 是否融合进 GEMM？

### 3.2 Attention：误差最容易扩散

Attention 结构可以概括为：

$$
Q = XW_q,\quad K = XW_k,\quad V = XW_v
$$

$$
P = \mathrm{softmax}(QK^T / \sqrt{d}),\quad O = PV
$$

可以量化的位置很多：`Wq / Wk / Wv / Wo` 权重、`Q / K / V` activation、KV cache、attention score、softmax 输入输出。

但工程上通常比较谨慎：QKV projection 可以量化，KV cache 可以量化，softmax 通常保留 FP16/BF16，attention score 一般不轻易低比特整数化。原因是 softmax 对数值误差敏感，`QK^T` 的误差会改变 attention 分布，而 attention 分布一变，后面的 V 加权求和也会跟着变。

### 3.3 MLP：收益大，但 outlier 常见

LLaMA/Qwen 类 MLP 常见形式是：

$$
\mathrm{MLP}(x) = down\_proj(\mathrm{SiLU}(gate\_proj(x)) \cdot up\_proj(x))
$$

MLP 计算量很大，非常值得量化。但它也容易出现 activation outlier，尤其是 `gate_proj / up_proj` 后的激活，以及 SiLU / GELU 后的中间值。

所以 MLP 量化经常需要 per-channel weight quant、per-token activation quant、SmoothQuant 或 outlier-aware method。

### 3.4 KV cache：decode 阶段的显存账

LLM 推理分两阶段：prefill 一次性处理 prompt，decode 一个 token 一个 token 生成。decode 阶段会保存每层的 K/V cache，形状大致是：

$$
[num\_layers,\ batch,\ num\_kv\_heads,\ seq\_len,\ head\_dim]
$$

上下文越长，KV cache 越大。KV cache quantization 就是把 K/V 从 FP16/BF16 压到 INT8、INT4 或 FP8。

收益很直接：减少显存、降低显存带宽、支持更长上下文、提升 batch size。难点也明确：K 影响 attention score，V 影响加权求和结果，误差会随生成长度积累，长上下文下更敏感。

## 四、LLM 量化真正难在哪里

### 4.1 activation outlier

LLM activation 经常是“大部分值很小，少数 channel 或 token 特别大”。例如：

`[0.1, -0.2, 0.05, 0.3, 73.0]`

如果用 per-tensor INT8，scale 会被 `73.0` 拉大。结果是 `73.0` 可以表示，但 `0.1`、`0.2`、`0.3` 这些普通值会变得非常粗糙。

这就是为什么 LLM 里经常说：**weight quantization 容易，activation quantization 难。** weight 可以离线观察，可以 per-channel，可以搜索 scale；activation 是运行时出现的，输入一变分布就可能变。

### 4.2 static 分布不等于线上分布

static quantization 的本质是用 calibration 的固定分布去近似未来线上动态分布。calibration 数据如果不代表真实输入，scale 就会偏。

这类风险在 LLM 里尤其明显：聊天、代码、数学、长上下文的 activation 分布可能差异很大；同一个模型不同层的敏感度也不同。

### 4.3 低 bit 不等于更快

很多人以为 FP16 变成 INT8 一定更快，实际不一定。量化引入了额外操作：quantize activation、dequantize output、乘 scale、处理 zero_point、读取 scale metadata、layout conversion。

量化是否更快取决于：

$$
\mathrm{saved\ compute/bandwidth}
>
\mathrm{quant/dequant/layout/scale\ overhead}
$$

这也是为什么某些 ONNX 量化后反而更慢：图上 dtype 变低了，但后端没有把低比特计算、scale、layout、dequant 融合成一条高效路径。

## 五、系统地图

量化的概念很多，但不要背成散点。更好的方式是按系统维度归类：

| 维度 | 对应概念 | 关注问题 |
| --- | --- | --- |
| 数值格式 | FP32, FP16, BF16, FP8, INT8, INT4 | 用什么数表示 |
| 量化对象 | weight, activation, KV cache | 量化谁 |
| 量化组合 | W8A16, W8A8, W4A16 | 权重和激活分别几 bit |
| 映射公式 | scale, zero_point | 整数如何还原浮点 |
| 映射方式 | symmetric, asymmetric | 0 点怎么放 |
| 粒度 | per-tensor, per-channel, per-token, group-wise | 一个 scale 管多少数 |
| scale 来源 | static, dynamic, calibration | scale 从哪里来 |
| LLM 难点 | activation outlier, long-context error, KV cache error | 误差在哪里放大 |
| 解决方法 | SmoothQuant, AWQ, GPTQ, mixed precision | 如何补偿误差 |
| 工程后端 | TensorRT-LLM, vLLM, SGLang, kernel, NPU backend | 如何真正跑快 |

## 六、学习路线

这条线最适合按下面顺序走：

```
1. 数值格式
   FP32 / FP16 / BF16 / FP8 / INT8 / INT4

2. 量化映射
   scale / zero_point / symmetric / asymmetric

3. 量化对象与组合
   weight / activation / KV cache
   W8A16 / W8A8 / W4A16

4. scale 粒度与来源
   per-tensor / per-channel / group-wise / per-token
   static / dynamic / calibration

5. LLM 特有难点
   activation outlier / SmoothQuant / AWQ / GPTQ

6. 工程落地
   TensorRT-LLM / vLLM / SGLang / quantized kernel / dequant fusion
```

## 七、一句话总结

量化不是 dtype 替换，而是在 **数值误差、scale 设计、kernel 路径、显存带宽和模型质量** 之间做系统权衡。

读完这一章，应该先建立一个总心智模型：

$$
X_{\mathrm{fp16}}W_{\mathrm{fp16}}
\quad\rightarrow\quad
Q(X)Q(W)
\quad\rightarrow\quad
\mathrm{scale\ restore}
$$

后面真正要细讲的是 Linear 量化，因为 W8A16 / W8A8 可以把 `scale`、`zero_point`、activation、weight、kernel、精度损失和性能收益全部串起来。
