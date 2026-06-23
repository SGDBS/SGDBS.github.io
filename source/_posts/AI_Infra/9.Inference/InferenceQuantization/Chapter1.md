---
title: Chapter1 Linear 量化全景
categories: 学习笔记- AI Infra
date: 2026-06-17 14:00:00
mathjax: true
tags:
    - AI
    - AI Infra
    - 模型量化
---



**理解 LLM 量化需要盯住 Linear 层。** 大部分参数在这里，大部分 GEMM 在这里，大部分低比特 kernel 的收益和风险也在这里。

Transformer 里反复消耗显存带宽和算力的，主要是 attention 里的 `q_proj`、`k_proj`、`v_proj`、`o_proj`，MLP 里的 `gate_proj`、`up_proj`、`down_proj`，以及最后的 `lm_head`。所以 W8A16、W8A8、W4A16 这些名字，不能只理解成 dtype 标签。它们真正改变的是一条 Linear 数据路径：

- 权重从 HBM 读多少字节。
- activation 是否也被量化。
- GEMM 能不能走低精度硬件路径。
- `scale`、`zero_point`、layout 转换和 dequant 是否被融合进 kernel。

这一讲把原来分散在几节里的内容收束到 Linear：从 W8A16 / W8A8 讲到 `scale`、`zero_point`、粒度、calibration，再回到 kernel 和性能账本。

## 一、问题定义：Linear 层到底在消耗什么

一个普通 Linear 层是：

$$
Y = XW + b
$$

在 LLM 里，`X` 是输入 activation，`W` 是权重，`Y` 是输出 activation。为了看清矩阵乘，通常把 `batch` 和 `seq_len` 压成一个维度：

$$
X \in \mathbb{R}^{M \times K},\quad
W \in \mathbb{R}^{K \times N},\quad
Y \in \mathbb{R}^{M \times N}
$$

其中：

$$
M = batch \times seq\_len,\quad K = hidden\_dim,\quad N = out\_dim
$$

FP16 推理里的计算可以写成：

$$
Y_{\mathrm{fp16}} = X_{\mathrm{fp16}} W_{\mathrm{fp16}}
$$

如果只从数学看，这就是一个 GEMM；但从推理系统看，它同时有两本账：

| 账本 | 含义 | 量化影响 |
| --- | --- | --- |
| 计算账 | $M \times N \times K$ 次乘加 | 低精度 Tensor Core / integer GEMM 可能提高吞吐 |
| 带宽账 | 读取 `X`、读取 `W`、写回 `Y` | 低 bit 能减少 HBM 读写 |
| 元数据账 | 读取 `scale`、`zero_point`、layout metadata | 粒度越细，元数据和 kernel 复杂度越高 |
| 转换账 | quantize、dequant、unpack、rescale | 没有融合时会吞掉收益 |

在 prefill 阶段，$M$ 往往较大，GEMM 更容易接近 compute-bound；在 decode 阶段，每次只生成少量 token，$M$ 很小，权重读取、kernel launch、layout 和 dequant 开销更容易暴露。量化是否有收益，必须放在这个阶段差异里看。

## 二、W8A16：先压权重，保住 activation

W8A16 的含义是：**Weight INT8 + Activation FP16/BF16**。

也就是 `X` 仍然是 FP16/BF16，`W` 被压成 INT8：

$$
W_{\mathrm{fp16}} \approx s_w W_{\mathrm{int8}}
$$

代入 Linear：

$$
Y \approx X_{\mathrm{fp16}}(s_w W_{\mathrm{int8}})
$$

这类方法也常被叫做 weight-only quantization。它最重要的价值不是让所有计算都变成 INT8，而是减少权重存储和读取带宽。权重是静态的，部署前就知道，所以可以离线统计范围、选择 `scale`、量化、评估误差，再按后端需要的 layout 打包。

以一组权重为例：

$$
W = [-0.8,\ -0.3,\ 0.1,\ 0.6]
$$

使用 symmetric INT8：

$$
s_w = \frac{\max(|W|)}{127} = \frac{0.8}{127} \approx 0.0063
$$

量化和反量化分别是：

$$
W_{\mathrm{int8}} = \mathrm{round}(W / s_w)
$$

$$
W_{\mathrm{dequant}} = s_w W_{\mathrm{int8}}
$$

大致得到：

- 原始权重：`[-0.8, -0.3, 0.1, 0.6]`
- INT8 权重：`[-127, -48, 16, 95]`
- 反量化权重：`[-0.800, -0.302, 0.101, 0.599]`

误差较小，更关键的是这件事可以离线完成，不会随着 prompt、token 或 batch 改变。

## 三、W8A16 的真实计算路径

W8A16 在工程里有两种常见路径。它们数学上接近，性能差别很大。

### 3.1 先 dequant，再 GEMM

最直观的做法是先把 `W_int8` 反量化回 FP16，再做普通 GEMM：

```python
W_fp16_approx = scale_w * W_int8
Y = X_fp16 @ W_fp16_approx
```

这种方式容易理解，但如果完整的 `W_fp16_approx` 落回显存，低 bit 权重带来的带宽收益会被抵消一大块。它更像是“存储压缩”，不一定是高性能推理路径。

### 3.2 kernel 内部融合 dequant

高性能实现更希望把解包和反量化放进 GEMM kernel：

- 读入 `X_fp16`。
- 读入 `W_int8` 或 packed `W_int4`。
- 在 tile 内部乘以对应 `scale`。
- 直接参与累加和 epilogue。

可以理解为：

$$
Y = X_{\mathrm{fp16}} \cdot \mathrm{dequant}(W_{\mathrm{int8}}, s_w)
$$

但 `dequant(W_int8, s_w)` 不生成一个完整大 tensor。它在寄存器、shared memory 或 MMA 数据路径附近完成。W8A16 能不能快，核心就看后端是否把“读低比特权重、解包、乘 scale、累加”融合得足够好。

## 四、W8A8：让主 GEMM 进入 INT8 路径

W8A8 的含义是：**Weight INT8 + Activation INT8**。

它比 W8A16 激进，因为 activation 也要量化：

$$
X_{\mathrm{fp16}} \approx s_x X_{\mathrm{int8}},\quad
W_{\mathrm{fp16}} \approx s_w W_{\mathrm{int8}}
$$

代入 Linear：

$$
Y \approx (s_x X_{\mathrm{int8}})(s_w W_{\mathrm{int8}})
$$

如果 `scale` 可以在 epilogue 处理：

$$
Y \approx s_x s_w (X_{\mathrm{int8}} W_{\mathrm{int8}})
$$

理想路径是：

```
X_fp16
  -> quantize
X_int8

W_fp16
  -> offline quantize
W_int8

X_int8 @ W_int8
  -> int32 accumulate
Y_int32
  -> multiply scale_x * scale_w
Y_fp16 / Y_bf16
```

W8A8 的理论收益更大：权重和 activation 都更小，主 GEMM 也可能走 INT8 Tensor Core 或整数矩阵单元。但它的风险也更集中：activation 是运行时变量，分布会随 prompt、token、batch、上下文长度和层数变化。

## 五、activation 为什么比 weight 难量化

weight 是静态的，可以离线处理；activation 是动态的，只能在真实前向里出现。

例如几个 token 的 activation：

- `token 1 = [0.1, 0.2, -0.1, 0.3]`
- `token 2 = [3.2, -2.5, 0.4, 7.8]`
- `token 3 = [0.01, -0.02, 0.03, 120.0]`

第三个 token 出现了一个巨大 outlier。如果整块 activation 共用一个 scale，普通值会被严重压扁。以 symmetric INT8 为例：

$$
s = \frac{\max(|X|)}{127}
$$

当：

$$
X = [0.1,\ 0.2,\ -0.1,\ 0.3,\ 120.0]
$$

则：

$$
s = \frac{120}{127} \approx 0.945
$$

普通值的量化结果会接近 0：

$$
0.1 / 0.945 \approx 0.106 \rightarrow 0
$$

$$
0.2 / 0.945 \approx 0.212 \rightarrow 0,\quad
0.3 / 0.945 \approx 0.317 \rightarrow 0
$$

真正的问题不是 `120.0` 表示不了，而是为了表示 `120.0`，`scale` 被迫变大，普通值落进了同一个量化格子里。这就是 activation outlier 对 W8A8 的破坏。

## 六、scale 和 zero_point：整数数轴怎么贴到浮点数轴

量化的核心公式是：

$$
x_{\mathrm{float}} \approx scale \cdot (x_{\mathrm{int}} - zero\_point)
$$

其中：

- `scale` 决定整数每走一格，对应浮点世界多长一段距离。
- `zero_point` 决定浮点 0 映射到整数空间里的哪个位置。
- `symmetric / asymmetric` 决定整数数轴是否围绕 0 对称。

从浮点到整数：

$$
x_{\mathrm{int}} =
\mathrm{clamp}(\mathrm{round}(x_{\mathrm{float}} / scale + zero\_point), q_{\min}, q_{\max})
$$

从整数回到浮点：

$$
x_{\mathrm{float}} \approx scale \cdot (x_{\mathrm{int}} - zero\_point)
$$

这里有两类误差：

| 误差 | 来源 | 典型症状 |
| --- | --- | --- |
| rounding error | 连续浮点值落到离散整数格子 | 小值被粗糙表示，低 bit 下更明显 |
| clipping error | 超出可表示范围后被截断 | outlier 或校准范围偏小时特别危险 |

`scale` 不是越小越好。太大，小值容易被 round 成 0；太小，大值容易被 clamp。量化质量本质是在普通值分辨率和 outlier 覆盖之间做取舍。

## 七、symmetric 和 asymmetric：精度与 kernel 的取舍

symmetric quantization 令：

$$
zero\_point = 0
$$

公式变成：

$$
x_{\mathrm{float}} \approx scale \cdot x_{\mathrm{int}}
$$

它的好处是计算路径干净，主 GEMM 不需要额外 zero_point 修正，所以更硬件友好，也更适合 weight。

asymmetric quantization 允许：

$$
zero\_point \neq 0
$$

公式保留为：

$$
x_{\mathrm{float}} \approx scale \cdot (x_{\mathrm{int}} - zero\_point)
$$

它更适合偏移明显的分布，例如 ReLU 后的非负 activation，但 kernel 更复杂。把量化矩阵乘展开就能看出代价：

$$
X_{\mathrm{float}} \approx s_x (X_{\mathrm{int}} - zp_x),\quad
W_{\mathrm{float}} \approx s_w (W_{\mathrm{int}} - zp_w)
$$

$$
Y \approx s_x s_w ((X_{\mathrm{int}} - zp_x)(W_{\mathrm{int}} - zp_w))
$$

展开后不只剩主 GEMM：

$$
X_{\mathrm{int}}W_{\mathrm{int}}
- zp_w \sum X_{\mathrm{int}}
- zp_x \sum W_{\mathrm{int}}
+ K \cdot zp_x \cdot zp_w
$$

如果 $zp_x = 0$ 且 $zp_w = 0$，这些修正项就消失了。这就是 symmetric 在高性能 INT8 GEMM 里常见的原因。

| 维度 | symmetric | asymmetric |
| --- | --- | --- |
| `zero_point` | 0 | 不一定是 0 |
| 整数数轴 | 围绕 0 对称 | 贴合真实范围 |
| 计算复杂度 | 低 | 高 |
| 硬件友好度 | 更好 | 取决于后端 |
| 常见对象 | weight | 部分 activation |
| 风险 | 非对称分布会浪费范围 | zero_point 修正项增加 kernel 成本 |

## 八、scale 粒度：一个 scale 管多少个数

同样是 INT8 或 INT4，真正拉开精度差距的经常不是 bit 数，而是 `scale` 粒度。粒度越细，outlier 影响范围越小，但 `scale` 元数据、带宽和 kernel 复杂度也会上升。

| 粒度 | scale 数量 | 精度 | 工程复杂度 | 常见对象 |
| --- | ---: | ---: | ---: | --- |
| per-tensor | 最少 | 最低 | 最低 | 简单 activation / 传统 PTQ |
| per-channel | 较少 | 较好 | 中等 | Linear / Conv weight |
| per-token | 随 token 动态变化 | 较好 | 较高 | LLM activation |
| group-wise | 中等 | 很好 | 较高 | INT4 weight |

### 8.1 per-tensor：最简单，也最怕 outlier

整个 tensor 共用一个 scale。例如：

$$
X \in \mathbb{R}^{M \times K},\quad s_x \in \mathbb{R}
$$

它实现简单、metadata 少、kernel 友好，但一个 outlier 会影响整个 tensor。LLM activation 如果粗暴使用 per-tensor，很容易让大量普通值被压扁。

### 8.2 per-channel：Linear weight 的常见折中

对于：

$$
W \in \mathbb{R}^{K \times N}
$$

per-channel weight quantization 通常让每个输出通道一个 scale：

$$
s_w \in \mathbb{R}^{N}
$$

也就是：

$$
W[:, j] \approx s_w[j] \cdot W_{\mathrm{int}}[:, j]
$$

它适合 weight，因为权重固定、可离线统计，不同输出通道的分布差异又可能很大。per-channel 比 per-tensor 精度好很多，scale 存储开销仍然很小。

### 8.3 per-token：LLM activation 的动态尺度

对 activation，常见做法是每个 token 一套 scale：

$$
X[i, :] \approx s_x[i] \cdot X_{\mathrm{int8}}[i, :]
$$

这样不同 token 的动态范围互不污染。配合 per-channel weight，输出可以写成：

$$
Y[i, j] \approx s_x[i] \cdot s_w[j] \cdot acc_{\mathrm{int32}}[i, j]
$$

其中：

$$
acc_{\mathrm{int32}}[i, j] = \sum_k X_{\mathrm{int8}}[i, k] \cdot W_{\mathrm{int8}}[k, j]
$$

这个公式直接对应 kernel epilogue：`scale_x[i]` 和 `scale_w[j]` 要在哪里加载、什么时候乘、能不能和输出 cast 融合，都会影响最终性能。

### 8.4 group-wise：INT4 weight-only 的核心形态

INT4 只有 4 bit，表示范围太窄，per-channel 往往还不够细。group-wise quantization 会把一个输出通道沿 `K` 方向切成多个 group，每个 group 一个 scale。常见 `group_size` 是 `32`、`64`、`128`。

假设 128 个 FP16 权重原始存储为：

$$
128 \times 16\ \mathrm{bit} = 2048\ \mathrm{bit}
$$

若用 INT4，并让这 128 个权重共享一个 FP16 scale：

$$
128 \times 4\ \mathrm{bit} + 16\ \mathrm{bit} = 528\ \mathrm{bit}
$$

压缩比约为：

$$
\frac{2048}{528} \approx 3.88\times
$$

如果 `group_size = 32`，128 个权重需要 4 个 scale：

$$
128 \times 4\ \mathrm{bit} + 4 \times 16\ \mathrm{bit} = 576\ \mathrm{bit}
$$

压缩比约为：

$$
\frac{2048}{576} \approx 3.56\times
$$

粒度更细，精度通常更好，但 scale 开销和 kernel 复杂度也更高。这就是 INT4 量化里 group size 很关键的原因。

## 九、scale 来源：static、dynamic 和 calibration

`scale` 和 `zero_point` 不是模型自动给出的，它们要么提前估计，要么运行时计算。

### 9.1 static quantization：部署前固定 scale

static quantization 的定义是：`scale / zero_point` 在部署前确定，运行时不再变化。

典型流程是：

```
calibration dataset
  -> run model forward
  -> collect activation statistics
  -> choose clipping threshold / scale / zero_point
  -> export quantized model
```

它的工程价值是运行时路径干净：

- 不需要实时统计 activation range。
- kernel 更简单。
- 更适合 ONNX、TensorRT、NPU compiler 这类静态图或编译型后端。
- 更容易做 layout 预打包和 kernel fusion。

风险也很明确：用一批 calibration 数据去预测未来线上输入分布。如果校准数据不代表真实请求，scale 就会偏。

### 9.2 dynamic quantization：运行时计算 scale

dynamic quantization 会根据当前输入实时计算 scale。以 per-token activation 为例：

$$
s_x[i] = \frac{\max(|X[i, :]|)}{127}
$$

它更适合 LLM activation，因为不同 prompt、token、上下文长度会让分布不断变化。代价是运行时多了 `absmax` reduction、quantize、scale 传递和 epilogue rescale。如果这些步骤没有融合，dynamic 可能更稳但不一定更快。

### 9.3 calibration 方法本质是在处理 outlier

常见 calibration 方法可以放在一张表里：

| 方法 | 机制 | 优点 | 风险 |
| --- | --- | --- | --- |
| min-max / absmax | 覆盖观测到的最大范围 | 简单，不容易 clipping | 极易被 outlier 拉大 scale |
| percentile | 忽略极端分位外的值 | 普通值分辨率更好 | 一定会 clipping 部分 outlier |
| KL divergence | 搜索量化后分布最接近原分布的阈值 | 考虑分布形状 | 实现和搜索更复杂 |
| MSE search | 最小化量化前后数值误差 | 直接贴近数值误差 | 不一定等价于模型质量最优 |

percentile 的取舍可以写成：

$$
\mathrm{rounding\ error\ for\ common\ values}
\quad vs \quad
\mathrm{clipping\ error\ for\ outliers}
$$

MSE search 的目标则是：

$$
\min_s \|x - \mathrm{dequant}(\mathrm{quant}(x, s), s)\|_2^2
$$

calibration 的难点不是跑一遍 forward，而是数据分布。短文本、聊天、代码、数学、长上下文、业务真实请求，对 LLM activation 的覆盖差异很大。

## 十、常见组合：不要只说 INT8 / INT4

一个完整量化方案至少要说明对象、bit 数、粒度、scale 来源、`zero_point` 和 kernel 路径。只说“INT8 量化”信息量太低。

| 方案 | weight | activation | scale 常见选择 | 核心收益 | 核心风险 |
| --- | --- | --- | --- | --- | --- |
| W8A16 | INT8 | FP16/BF16 | weight static per-channel | 省权重显存和带宽 | 速度依赖 fused dequant kernel |
| W8A8 | INT8 | INT8 | weight per-channel + activation per-token | 可能走 INT8 GEMM | activation outlier 和 dynamic overhead |
| W4A16 | INT4 | FP16/BF16 | weight group-wise | 显存压缩明显 | unpack / dequant / scale apply 依赖 kernel |
| KV cache INT8/INT4/FP8 | K/V 低精度 | 当前计算仍常用 FP16/BF16 混合 | per-head / per-token / hybrid | 长上下文显存下降 | attention score 和 value 聚合误差积累 |

工程上更完整的配置会像这样描述：

```yaml
weight:
  dtype: int4
  granularity: group-wise
  group_size: 128
  symmetric: true

activation:
  dtype: fp16

kv_cache:
  dtype: int8
  granularity: per-token-per-head
```

这比“INT4 量化”更接近真实部署约束。

## 十一、性能账本：低 bit 不自动等于更快

量化真正能加速，需要满足一个简单不等式：

$$
\mathrm{saved\ bandwidth} + \mathrm{saved\ compute}
>
\mathrm{quantize} + \mathrm{dequantize} + \mathrm{unpack} + \mathrm{scale} + \mathrm{layout\ overhead}
$$

很多量化方案看起来 dtype 变低了，但没有真正变快，通常原因在这里：

- activation quantize 单独成了额外 kernel。
- dequant 生成了完整 FP16 中间 tensor。
- `scale` 加载和广播破坏了主 kernel 吞吐。
- INT4 需要 unpack，但 unpack 没有和 matmul 融合。
- zero_point 修正项让 epilogue 变复杂。
- 后端只支持图上的 INT8，不支持高性能的 INT8 GEMM layout。

所以判断一个 Linear 量化方案，不是看论文或配置里写了多少 bit，而是看数据路径是否真的短了：

```
bad path:
  low-bit tensor -> dequant to fp16 tensor -> fp16 gemm -> write output

good path:
  low-bit tensor -> unpack/dequant inside matmul tile -> fused epilogue -> write output
```

量化的系统价值，不在于把 dtype 名字换掉，而在于让 HBM、Tensor Core、register、shared memory 和 epilogue 这条链路变短。

## 十二、工程判断清单

看到一个 LLM Linear 量化方案时，先问这些问题：

1. 量化对象是谁：weight、activation，还是 KV cache？
2. bit 数是什么：INT8、INT4，还是 FP8？
3. `scale` 粒度是什么：per-tensor、per-channel、per-token，还是 group-wise？
4. `scale` 来源是什么：static calibration，还是 runtime dynamic？
5. `zero_point` 是否为 0：symmetric 还是 asymmetric？
6. 计算路径是什么：真 INT8 GEMM，还是低比特存储后 dequant 到 FP16？
7. dequant、unpack、rescale、output cast 是否融合进主 kernel？
8. 精度风险来自哪里：activation outlier、clipping、rounding、KV cache，还是某些敏感层？
9. prefill 和 decode 阶段是否都收益，还是只优化了其中一个阶段？

这些问题比“是不是 INT8”更接近真实系统里的性能和精度边界。

## 十三、写在最后：这一讲要留下的心智模型

W8A16 只量化权重，activation 仍是 FP16/BF16。它容易落地，主要省显存和权重读取带宽，但速度收益依赖 fused dequant / mixed GEMM kernel。

W8A8 让权重和 activation 都进入 INT8 路径。它理论收益更大，但 activation outlier、dynamic scale、zero_point 修正和 epilogue 融合会决定它能不能真正跑快。

W4A16 更激进地压权重，常见于 GPTQ、AWQ 和 bitsandbytes 4bit。它的关键不只是 INT4，而是 group-wise scale、权重打包和 kernel 内部 unpack/dequant。

一句话总结：

$$
\text{quantization quality}
= f(\text{object},\ \text{bit},\ \text{scale granularity},\ \text{scale source},\ \text{kernel path})
$$

Linear 量化不是单个公式，而是一条数据路径的系统优化。只有把数值误差和 kernel 账本放在一起看，才能判断一个量化方案到底是在省显存、提吞吐，还是只是把复杂度转移到了别的地方。
