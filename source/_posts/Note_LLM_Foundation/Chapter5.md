---
title: Chapter5 SFT 与参数高效微调：MLE、LoRA、QLoRA
categories: 学习笔记-大模型
date: 2026-03-30 16:03:34
mathjax: true
tags:
    - AI
    - AI面试知识
---

> **本章定位**：从表示学习（Ch1–Ch4）跨入生成模型对齐（Ch5–Ch8）的第一站。SFT 是把"只会预测下一个词"的 Base Model 转变为"能听懂指令"的 Assistant 的关键步骤；LoRA/QLoRA/DoRA 是当今所有微调任务的工程标配。

> **承上**：Ch1 §6 的交叉熵 = SFT 的损失函数本身。
> **启下**：Ch6 起的所有 RL 对齐方法都以 SFT 模型为初始化。

---

# §A 数学原理

## 1. SFT 的本质：最大似然估计 (MLE)

给定数据集 $D_{\text{SFT}} = \{(x_i, y_i)\}$，其中 $x$ 是 prompt、$y$ 是 response，目标是最大化条件概率 $P_\theta(y \mid x)$：

$$\max_\theta \mathbb{E}_{(x, y) \sim D_{\text{SFT}}} \log P_\theta(y \mid x)$$

由于 $y = (y_1, y_2, \dots, y_T)$ 是 token 序列，自回归模型把它分解为：
$$P_\theta(y \mid x) = \prod_{t=1}^{T} P_\theta(y_t \mid x, y_{<t})$$

取负对数得到 SFT 的损失：
$$\boxed{\mathcal{L}_{\text{SFT}}(\theta) = - \mathbb{E}_{(x, y) \sim D_{\text{SFT}}} \left[ \sum_{t=1}^{|y|} \log P_\theta(y_t \mid x, y_{<t}) \right]}$$

这就是 Ch1 §6 的交叉熵：让模型分布逼近"标注员的分布"。

## 2. 为什么只对 Response 计算 Loss？

SFT 时，整条序列 `[Prompt][Response]` 都喂给模型，但**Loss Mask** 让 prompt 部分不参与 loss 计算：

$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=|x|+1}^{|x|+|y|} \log P_\theta(y_t \mid x, y_{<t})$$

**原因**：
- 我们不希望模型学习"用户怎么提问"（prompt 是外部输入）
- 只要求模型学习"给定 prompt 如何生成正确 response"

**实现方式**：构造 `labels` 张量，prompt 位置设为 `-100`（PyTorch 默认 `ignore_index`，自动跳过）。

## 3. LoRA：低秩适配的数学

LoRA（Low-Rank Adaptation, Microsoft 2021）的核心假设：**预训练模型在下游任务上的更新量 $\Delta W$ 是低秩的**。

### 3.1 参数化方式

原始权重 $W_0 \in \mathbb{R}^{d \times d}$ 冻结，引入两个低秩矩阵：
- $A \in \mathbb{R}^{r \times d}$，初始化为高斯分布
- $B \in \mathbb{R}^{d \times r}$，**初始化为零矩阵**

更新形式：
$$W = W_0 + \Delta W = W_0 + BA$$

前向传播：
$$h = Wx = W_0 x + BAx$$

### 3.2 为什么 $B$ 初始化为零？

如果 $A, B$ 都随机初始化，训练初期 $BA \neq 0$，模型从一开始就偏离了预训练。$B = 0$ 保证 $BAx = 0$，模型初始状态 = 预训练状态，**训练从一个已知的好起点出发**。

### 3.3 参数量节省

原始权重参数量：$d \times d = d^2$
LoRA 参数量：$d \times r + r \times d = 2dr$

设 $d = 4096, r = 8$：
$$\frac{2 \times 4096 \times 8}{4096^2} \approx 0.4\%$$

**LoRA 把可训练参数压缩到原始的 1% 以下**。

### 3.4 缩放因子 $\alpha$

实际的 LoRA 公式还有一个缩放：
$$\Delta W = \frac{\alpha}{r} BA$$

$\alpha$ 通常设为 $r$ 的 2–4 倍（如 $r=8, \alpha=32$）。**作用**：当增大 $r$ 时，$\frac{\alpha}{r}$ 自动缩小，避免重新调整学习率。

## 4. QLoRA：4-bit 量化 + LoRA

QLoRA（Dettmers 2023）让 65B 模型可以**在单张 48GB 显卡上微调**：

| 技术 | 作用 |
| :--- | :--- |
| **4-bit NF4 量化** | Base model 权重压缩到 4-bit（NormalFloat 4，对正态分布权重最优） |
| **Double Quantization** | 量化常数本身再量化，节省 ~0.4 bit/参数 |
| **Paged Optimizer** | 用 NVIDIA 统一内存，optimizer state 临时换出到 CPU |
| **LoRA on top** | 量化的 base model 不动，只训练 16-bit 的 LoRA 矩阵 |

**核心数学**：base model 用 4-bit 存储但**前向传播时反量化为 BF16 计算**，loss 和梯度都是 BF16，所以训练精度几乎不损失。

## 5. DoRA：Weight Decomposed LoRA

DoRA（NVIDIA 2024）把权重分解为"幅值 + 方向"：
$$W = m \cdot \frac{V}{\|V\|}$$
其中 $m$ 是幅值（每列一个标量），$V/\|V\|$ 是方向。LoRA 只更新方向部分 $V$，幅值 $m$ 全量训练。

**直觉**：预训练模型已经学到了大致的"权重幅值分布"，下游任务主要在调整"方向"。DoRA 在多个任务上比 LoRA 高 1–2 个点。

---

# §B 模型结构（PyTorch 实现）

## B.1 SFT 数据处理：构造 `labels = -100` 的 mask

这是 SFT 工程上最容易写错的地方。

```python
def build_sft_labels(input_ids, prompt_length):
    """
    input_ids:     [B, L]  完整序列 (prompt + response)
    prompt_length: [B]     每条样本的 prompt 长度
    返回 labels: prompt 部分为 -100，response 部分保留 token id
    """
    labels = input_ids.clone()
    for i, plen in enumerate(prompt_length):
        labels[i, :plen] = -100                                  # ⭐ Prompt 不算 loss
    return labels


# 用法
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    labels.view(-1),
    ignore_index=-100,                                            # 自动跳过 -100
)
```

**TRL 库的现成实现**：
```python
from trl import DataCollatorForCompletionOnlyLM

collator = DataCollatorForCompletionOnlyLM(
    response_template="### Response:",                            # 标志 response 开始
    tokenizer=tokenizer,
)
```

## B.2 LoRA 的 PyTorch 手写实现

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """替换原始 Linear 层：W_0 + BA"""
    def __init__(self, base_layer: nn.Linear, r=8, alpha=32, dropout=0.0):
        super().__init__()
        self.base_layer = base_layer
        # ⭐ 冻结原始权重
        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # ⭐ 引入低秩矩阵 A 和 B
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))    # ⭐ B 初始化为 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 原始权重的输出
        result = self.base_layer(x)
        # LoRA 旁路：x @ A^T @ B^T，乘以 scaling
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + self.scaling * lora_out


def replace_with_lora(model, target_modules=("q_proj", "v_proj"), r=8):
    """递归替换 model 中所有 target_modules 为 LoRALinear"""
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, LoRALinear(module, r=r))
    return model
```

## B.3 PEFT 库的工业级使用

实际工程中用 PEFT 库就够了：

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",                  # Attention 全层
        "gate_proj", "up_proj", "down_proj",                     # FFN 全层（现代趋势）
    ],
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
model = get_peft_model(model, config)
model.print_trainable_parameters()
# 输出示例: trainable params: 41,943,040 || all params: 8,030,261,248 || trainable%: 0.52%
```

**target_modules 的演进**：
- 早期 LoRA 论文：只加在 $W_Q, W_V$
- 现代趋势：**全层加 LoRA**（包括 FFN 和输出投影），效果显著提升

## B.4 推理时合并 LoRA 权重

LoRA 训练完后，**推理时可以把 $BA$ 合并回 $W_0$**，零额外开销：

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$$

```python
# PEFT 一键合并
merged_model = model.merge_and_unload()                          # 合并 LoRA + 卸载 PEFT 包装
merged_model.save_pretrained("./merged_model")
```

合并后模型与原始模型结构完全一致，**推理时无任何性能损失**——这是 LoRA 比 Adapter（需推理时额外计算）优越的关键。

---

# §C 训练与推理

## C.1 训练流程：完整的 SFT 训练循环

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from trl import DataCollatorForCompletionOnlyLM

# 1. 加载模型 + 配置 LoRA
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, config)

# 2. 数据 collator：自动 mask prompt 部分
collator = DataCollatorForCompletionOnlyLM(
    response_template="### Response:", tokenizer=tokenizer,
)

# 3. 训练参数
training_args = TrainingArguments(
    output_dir="./output_sft",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,                                # 等价 batch=16
    learning_rate=2e-4,                                            # LoRA 学习率比全量大
    num_train_epochs=3,
    bf16=True,
    save_strategy="epoch",
    logging_steps=10,
    warmup_ratio=0.03,
)

# 4. 启动
trainer = Trainer(
    model=model, args=training_args,
    train_dataset=dataset, data_collator=collator,
)
trainer.train()
```

## C.2 显存对比：Full FT vs LoRA vs QLoRA

以 Llama-3 8B 为例（粗略估算）：

| 方式 | 模型权重 | 梯度 | Adam state | 总计 (~) |
| :--- | :---: | :---: | :---: | :---: |
| **Full FT** (BF16) | 16 GB | 16 GB | 32 GB (FP32) | **64+ GB** |
| **LoRA** (BF16) | 16 GB | 0.08 GB | 0.16 GB | **~17 GB** |
| **QLoRA** (4-bit + LoRA) | 4 GB | 0.08 GB | 0.16 GB | **~5 GB** |

> QLoRA 让单张 24GB 卡能微调 65B 模型，单张 48GB 卡能微调 70B 模型。

## C.3 SFT 解决了什么 / 还差什么

**解决了**：
- ✅ 格式对齐：模型学会"问答"形式
- ✅ 能力激活：预训练知识被"提取"出来用于回答

**还差什么**：
- ❌ **多解性**：写代码、写诗有多种正确答案，SFT 强迫模型只学某一种
- ❌ **错误积累**：本质是模仿学习，标注里的错误会被学得很扎实
- ❌ **没学到"什么是不好的"**：SFT 只展示正例，没有负例对比

→ 这就是为什么需要后续的 RLHF/DPO（Ch6, Ch7）。

## C.4 SFT 后的推理：解码策略

SFT 训出的模型在推理时**和 base model 形式上没有区别**，都是自回归生成。但因为模型分布变得更"尖锐"（向标注者的语气和格式聚集），常用的解码策略需要调整：

| 策略 | 公式/做法 | 适用场景 |
| :--- | :--- | :--- |
| **Greedy** | $\arg\max_y P(y \mid \cdot)$ | 确定性任务（代码、数学） |
| **Top-k** | 只从概率最高的 $k$ 个 token 中采样 | 一般生成 |
| **Top-p (nucleus)** | 从累积概率 $\geq p$ 的最小集合中采样 | 创意生成、对话 |
| **Temperature $T$** | $\text{softmax}(\text{logits} / T)$，$T < 1$ 更尖锐 | 控制确定性 |
| **Repetition Penalty** | 已生成 token 的 logits 减去惩罚 | 避免循环输出 |

> **常见组合**：对话默认 `temperature=0.7, top_p=0.9`；代码任务用 `temperature=0.0`（即 greedy）。

---

# §D 章末速查

## D.1 LoRA 的关键 Q&A

**Q1：LoRA 训练多少参数？**
- 取决于 r 和 target_modules。典型 7B 模型：
  - 只加 q,v：~4M（0.05%）
  - 加全 7 个 module：~40M（0.5%）

**Q2：LoRA 学习率多大？**
- 通常 2e-4 到 5e-4，比全量微调（2e-5）大 10 倍。
- 因为只训练少量参数，需要更大的步长。

**Q3：LoRA 推理时有额外开销吗？**
- 训练时：有（额外 BA 计算）
- 推理时：合并后零开销，与原模型完全一致

**Q4：LoRA 合并后能再继续训练吗？**
- 可以。但通常做法是保存多个 LoRA adapter，按需加载，**不合并**。这样一个 base model 可以服务多个任务。

## D.2 SFT 数据规模与质量

| 数据规模 | 适用场景 | 代表 |
| :--- | :--- | :--- |
| **1k–10k** 高质量 | 概念验证、风格定制 | LIMA（"少即是多"） |
| **10k–100k** | 通用 SFT | InstructGPT |
| **100k–1M** | 全方位强化 | Llama 3 等 |
| **>1M** | 边际收益递减 | 商业大模型 |

> **LIMA 论文核心结论**：1000 条精心筛选的 SFT 数据，能让 65B 模型达到 GPT-4 的对话质量。**质量远比数量重要**。

---

## 承上启下

SFT 让模型学会了"听话"，但只学到了"标注者怎么写"，无法理解"什么是更好的"。下一章 **Ch6** 引入奖励模型（RM）和 PPO，让模型从**人类偏好**中学习——这是从 GPT-3.5 到 ChatGPT 的关键一跳。

> Ch6 中的 SFT 模型在三个地方用到：
> 1. **Policy 初始化**：SFT 模型直接作为 PPO 的初始 Policy
> 2. **Reference Model**：SFT 模型的冻结副本（呼应 Ch4 §D 的 BYOL Target）
> 3. **RM 初始化**：RM 也用 SFT 模型改造（去掉 LM Head，加 Scalar Head）
