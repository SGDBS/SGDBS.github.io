---
title: Chapter4 强化学习 RLHF
date: 2026-03-30 16:03:34
categories: AI大学习
mathjax: true
tags:
    - AI
    - AI面试知识
---


我们可以把 RLHF 看作是将人类的“价值观”和“偏好”量化为奖励信号，并指导模型进化的过程。

## RLHF (Reinforcement Learning from Human Feedback)的三大阶段

### 1. 监督微调 (SFT - Supervised Fine-Tuning)

* 目标：让预训练模型学会“听话”，掌握对话的基本格式。
* 做法：使用高质量的“指令-回答”数据（由人类编写）对预训练模型进行有监督学习。
* 数学本质：最大化似然估计 $\max_\theta \mathbb{E}_{(x, y) \sim D} [\log P_\theta(y|x)]$。
* 产物：SFT 模型（这是后续阶段的起点）。

### 2. 训练奖励模型 (RM - Reward Modeling)

这是 RLHF 的核心。人类并不擅长给模型打分（打 80 分还是 82 分？），但非常擅长做排序。
* 做法：
    1. 让 SFT 模型针对同一个问题 $x$ 生成多个回答 $\{y_1, y_2, \dots\}$。
    2. 人类根据偏好对回答进行排序（例如 $y_{win} > y_{lose}$）。训练一个奖励模型 $r_\phi(x, y)$，使其对人类喜欢的回答打高分。
* 损失函数 (Pairwise Ranking Loss)：$$L(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim D} [\log(\sigma(r_\phi(x, y_w) - r_\phi(x, y_l)))]$$
* 直觉：两个回答的奖励分差值经过 Sigmoid 后，应该尽可能接近 1。
* 产物：奖励模型 RM（它代表了人类的审美）。

### 3. 强化学习对齐 (PPO - Proximal Policy Optimization)

这一步是利用 RM 作为“裁判”，通过强化学习算法（通常是 PPO）来榨取模型的潜力。
* 做法：
    1. 将 SFT 模型初始化为 Policy 网络。
    2. 模型生成回答，RM 打分。
    3. 根据得分更新模型参数。
* 关键约束：**KL 散度**：
    为了防止模型为了骗高分而变得“油腔滑调”或产生乱码（即模型坍缩），会在 Reward 中加入一个惩罚项：$$\text{Reward} = r_\phi(x, y) - \beta \cdot \text{KL}(P_{RL}(y|x) \| P_{SFT}(y|x))$$
    * 意义：强迫对齐后的模型不要偏离原始 SFT 模型太远。
* 产物：最终的 RLHF 模型。

### 常见问题

1. 为什么 RLHF 很重要？
* **不可定义性**：人类对“幽默”、“安全”、“有用”的定义很难用传统的 Loss Function 写出来，但 RM 可以通过学习排序规律隐式地捕获这些特征。

* **泛化能力**：RM 训练好后，可以给海量的模型生成结果打分，这比纯人工标注（SFT）的规模要大得多，效率更高。

* **解决幻觉**：通过偏好引导，模型可以学会“知之为知之，不知为不知”，减少一本正经胡说八道。

2. 为什么要加 KL 散度约束？

答：防止 Reward Hacking（奖励作弊）。如果没有约束，模型可能会发现某些特定字符串（如“！！！！”）在 RM 看来分数很高，从而大量生成这类无意义内容来刷分。
KL 散度保证了模型在对齐的同时，依然保持语言模型的本色。

3. 问：RM 模型和 Policy 模型一定要一样大吗？

答：不一定。通常 RM 会稍微小一点（例如 Policy 是 175B，RM 可能是 6B），因为它只需要判断好坏，不需要生成复杂的文本。

4. 问：RLHF 有什么局限性？

答：成本极高（需要大量人工排位）；存在“人类偏好偏见”（人类可能更倾向于字数多、语气委婉但内容错误的回答）；PPO 算法训练极度不稳定，对超参数敏感。


## 第一阶段：SFT (Supervised Fine-Tuning)

这是将一个“只会预测下一个词”的**预训练语言模型（Base Model）转变为“能够听懂指令”的对话模型（Assistant Model）**的关键步骤。

### 1. 数据准备
* 这一阶段需要的是 (Prompt, Response) 对，即“问题-答案”对。

* 来源：通常由专业标注人员编写，或者从高质量的种子任务中衍生。

* 规模：通常在几万条左右（相对于预训练的万亿级数据，这非常小，但质量要求极高）。

> 例子：
    >* Prompt: “请帮我写一首关于春天的五言绝句。”
    >* Response: “春色满园开，红花映绿苔。微风吹柳面，燕子衔泥来。”

### 2. 训练逻辑
* 模型状态：加载预训练好的 Base Model（如 Llama-3 或 GPT-3 的原始权重）。

* 训练方式：标准的自回归语言建模（Autoregressive Language Modeling）。

* 核心细节：在训练时，我们只对 Response（回答） 部分计算损失，而不对 Prompt 部分计算。

### 3. 数学本质 

SFT 的本质是最大似然估计 (MLE)。给定输入的上下文 $x$，模型输出人类标注答案 $y$ 的概率：
$$\mathcal{L}_{SFT}(\theta) = - \mathbb{E}_{(x, y) \sim D_{SFT}} \left[ \sum_{t=1}^{|y|} \log P_\theta(y_t | x, y_{<t}) \right]$$
* $\theta$：模型参数。$y_t$：回答中的第 $t$ 个 token。
* 直观理解：让模型生成的分布尽可能地贴近人类标注员写的那个分布。

### 4. 更新参数

$$L_{SFT} = - \sum_{t=|x|+1}^{L} \log P(y_t | x, y_{<t})$$

#### 全参数微调 (Full Fine-Tuning)
这是最原始、最暴力的方法。
* 训练哪些参数：模型的所有权重矩阵。以 Transformer 架构为例，包括所有的 $W_Q, W_K, W_V, W_O$（注意力权重），$W_1, W_2$（前馈网络层），以及 Embedding 层和最后的 LM Head（分类输出层）。
* 状态：所有参数的梯度（Gradients）都会被计算，优化器（如 AdamW）会更新所有参数的状态。
* 缺点：显存压力极大（通常需要模型参数量 4-8 倍的显存）。

#### 高效参数微调 (PEFT - Parameter-Efficient Fine-Tuning)
LoRA 的参数更新逻辑LoRA 的核心思想是：冻结原模型，只训练旁路的小矩阵。
* 不训练哪些：冻结预训练模型（Base Model）的所有原始参数 $\mathbf{W}_0$。这些参数在反向传播时不计算梯度。
* 训练哪些参数：只训练新增的两个低秩矩阵 $\mathbf{A}$ 和 $\mathbf{B}$。
    * 假设原始矩阵是 $d \times d$，LoRA 引入 $A (d \times r)$ 和 $B (r \times d)$，其中秩 $r$ 通常很小（如 8 或 16）。
    * 更新公式：$h = \mathbf{W}_0 x + \Delta \mathbf{W} x = \mathbf{W}_0 x + \mathbf{BA} x$
* 在 SFT 阶段，通常只在 Attention 层（$W_Q, W_V$）添加 LoRA 模块，但现在的趋势是全层（包括 MLP）都加，效果更好。

#### 为什么 SFT 只训练 Response 部分？
这是工程实现上的一个细节。在训练时，虽然整条序列 [Prompt] [Response] 都会喂给模型，但：

* Mask 操作：我们会构造一个 Loss Mask。对于 Prompt 对应的 Token，其 Loss 被设为 0；只有 Response 对应的 Token 才会计算 Cross-Entropy Loss。

* 原因：我们不希望模型去“学习”用户是怎么提问的（Prompt 是外部输入的），我们只要求模型学习在给定 Prompt 下，如何生成正确率最高的 Response。

### 5. 这一阶段解决了什么？
* 格式对齐：模型学会了“问答”这种形式。
* 能力激活：预训练阶段学到的海量知识被“提取”出来，用于回答问题。

### 6. 为什么 SFT 还不够？

* 标注成本：写出高质量的回答非常累，人类很难覆盖所有的边界情况。

* 多解性：对于“写个代码”这种任务，答案有很多种。SFT 强迫模型只学某一个标注员的写法，会限制模型的创造力。

* 错误积累：SFT 属于“模仿学习”，如果标注数据里有一点点幻觉或错误，模型会学得非常扎实。


### 关键代码

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# 1. 加载预训练模型和分词器
model_id = "your-base-model-path"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# 2. 定义 LoRA 配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # 任务类型：自回归语言模型
    inference_mode=False,         # 训练模式
    r=8,                          # LoRA 的秩 (Rank)，决定了参数量
    lora_alpha=32,                # 缩放因子，通常为 r 的 2-4 倍
    lora_dropout=0.1,             # 防止过拟合
    target_modules=["q_proj", "v_proj"] # 关键：指定 LoRA 作用于哪些权重矩阵
)

# 3. 将原始模型包装为 PeftModel
# 此时 model 内部的原始参数会被 freeze，只有 A 和 B 矩阵可训练
model = get_peft_model(model, config)

# 4. 打印可训练参数量（面试时常被问到：LoRA 训练多少参数？）
model.print_trainable_parameters()

# 5. 训练配置 (SFT 关键点：只计算 Response 损失通过 DataCollator 实现)
training_args = TrainingArguments(
    output_dir="./output_sft",
    per_device_train_batch_size=4,
    learning_rate=2e-4,           # LoRA 学习率通常比全量微调稍大
    num_train_epochs=3,
    logging_steps=10,
    bf16=True                     # 如果显卡支持，bf16 效果更好
)

# 6. 启动训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,        # 预先处理好的数据集
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True) # 自动 mask prompt 部分
)

trainer.train()
```


## 第二阶段：RM (Reward Modeling)

### 1. 数据构造：从“写答案”到“排高下”
人类很难给一个回答打出精确的分数（比如 79 分 vs 81 分），但人类非常擅长做对比选择。
* **采样**：给出一个问题（Prompt）$x$，让第一阶段得到的 SFT 模型 生成多个不同的回答 $\{y_1, y_2, \dots, y_k\}$（通常使用不同的采样随机种子）。
* **标注**：标注员会对这些回答进行两两比较（Pairwise），选出更好的一个。结果格式：$(x, y_w, y_l)$，其中 $y_w$ (win) 是较好的回答，$y_l$ (lose) 是较差的回答。
* **数据规模**：通常需要数十万甚至上百万对这种对比数据，比 SFT 的数据量大得多。

### 2. 模型结构：从“概率分布”到“标量分数”
奖励模型通常由 SFT 模型改造而来，但其输出层发生了本质变化：
* **结构调整**：去掉 SFT 模型最后的 LM Head（原本是输出词表大小 $V$ 的概率分布），换成一个 Scalar Head（输出一个单一的实数值 $r$）。
* **输入输出**：输入一个序列 $(x, y)$，模型不再预测下一个词，而是对整条序列吐出一个分值 $r_\phi(x, y)$。
* **参数量**：RM 通常比生成模型稍小或对等。例如 175B 的 GPT-3，其 RM 可能是 6B 的


### 3. 数学本质：排序损失函数 (Pairwise Ranking Loss)
我们希望对于同一问题 $x$，好的回答 $y_w$ 得分高于差的回答 $y_l$。损失函数定义如下：
$$L(\phi) = -\frac{1}{K} \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \left( \sigma \left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right) \right]$$
* $\sigma$ (Sigmoid 函数)：将分数的差值映射到 $(0, 1)$ 区间。
* 逻辑：如果 $r(y_w) \gg r(y_l)$，则 $\sigma(\text{diff}) \approx 1$，$\log(1) = 0$，Loss 极小。如果 $r(y_w) \ll r(y_l)$，则 $\sigma(\text{diff}) \approx 0$，Loss 极大。* 直觉：通过拉大正负样本之间的分差，迫使模型学习人类偏好的底层特征（如安全性、事实性、语气等）。

### 4. 训练时的参数更新

* 初始化：使用 SFT 模型的权重初始化 RM。这是因为 RM 首先得“读懂”文本，SFT 的预训练知识对它至关重要。
* 训练参数：
    * 全量微调：更新 RM 的所有参数。
    * LoRA 模式：同样可以使用 LoRA 只训练旁路。
* 注意：训练 RM 时，不使用 Mask。我们需要模型对整个序列（Prompt + Response）产生一个全局的理解。


### 5. 具体训练过程

#### 1. 模型架构：
通常由第一阶段训练好的 SFT 模型（Supervised Fine-Tuned Model）改造而来。
* **基座 (Backbone)**：保持 GPT 的 Transformer Decoder 结构不变
* **结构改动**：
    * 移除 LM Head：去掉原来输出维度为词表大小 $V$（如 50,257）的线性层。
    * 添加 Scalar Head：增加一个随机初始化的线性层（Linear Layer），其输出维度固定为 1。
* **输入限制**：RM 通常具有与 SFT 相同的上下文窗口（如 4096 或 8192），以便能够处理长对话。

#### 2. 详细训练流程：Pairwise RankingRM 的训练本质上是一个排序问题。
##### 步骤 A：数据采集与标注：
1. 采样：从 Prompt 池中取一个问题 $x$，让 SFT 模型生成 $k$ 个不同的回答（$k \ge 2$）。
2. 标注：人类标注员对这 $k$ 个回答进行排序。我们从中提取出二元对比组：$(x, y_w, y_l)$。
    * $y_w$ (winning)：人类更喜欢的回答。
    * $y_l$ (losing)：人类较不喜欢的回答。
##### 步骤 B：前向传播 (Forward Pass)
模型需要分别计算 $y_w$ 和 $y_l$ 在同一个问题 $x$ 下的分数。
1. 拼接序列：构造两个输入序列 。
    * $s_w = [x; y_w]$
    * $s_l = [x; y_l]$
2. 提取特征：将 $s_w$ 和 $s_l$ 分别喂入 Transformer。取最后一个 Token：提取 Transformer 最后一层在序列末尾位置（$last\_token$）的隐藏状态向量。

##### 步骤 C：计算 Loss 与反向传播
使用**排序损失（Pairwise Ranking Loss）**更新参数。


### 6. 关键代码
```py
import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel

class GPTRewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.config = base_model.config
        self.backbone = base_model # 这里的 base_model 是 SFT 后的 GPT
        
        # 这里的 d_model 对应 GPT 的 hidden_size (如 4096)
        # 输出维度为 1，代表奖励分
        self.v_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        # 1. 经过 Transformer 得到所有 hidden states
        # 输出维度: [batch, seq_len, hidden_size]
        transformer_outputs = self.backbone(input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs.last_hidden_state
        
        # 2. 关键点：取每一个序列中【最后一个非 padding】的 token
        # 找到每个 batch 中最后一个有效 token 的索引
        last_token_indices = attention_mask.sum(dim=1) - 1
        
        # 提取该 token 的向量: [batch, hidden_size]
        batch_size = input_ids.shape[0]
        last_token_hidden_states = hidden_states[torch.arange(batch_size), last_token_indices]
        
        # 3. 映射为标量分数: [batch, 1]
        values = self.v_head(last_token_hidden_states)
        return values

def compute_ranking_loss(chosen_rewards, rejected_rewards):
    """
    chosen_rewards: [batch, 1] - 胜出回答的分数
    rejected_rewards: [batch, 1] - 落败回答的分数
    """
    # 公式: Loss = -log(sigmoid(r_w - r_l))
    # 我们希望分差越大越好
    loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
    return loss



# 假设 batch 包含: 
# input_ids_chosen (Prompt + 好回答)
# input_ids_rejected (Prompt + 差回答)

def train_step(model, batch, optimizer):
    model.train()
    
    # 1. 计算两组回答的分数
    # 为了节省显存，可以合并到一个 batch 里跑，然后拆开
    rewards_chosen = model(batch['input_ids_chosen'], batch['attention_mask_chosen'])
    rewards_rejected = model(batch['input_ids_rejected'], batch['attention_mask_rejected'])
    
    # 2. 计算排序损失
    loss = compute_ranking_loss(rewards_chosen, rewards_rejected)
    
    # 3. 反向传播
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()
```


## 第三阶段：PPO (Proximal Policy Optimization，近端策略优化)

这一步的目标是：利用第二阶段训练好的 RM (奖励模型) 作为判官，采用强化学习算法来更新 SFT 模型，使其生成的回答能够最大化奖励分数。

### 1. PPO 系统的“四模并行”架构

* **Actor Model (Policy)**： 主角，接收 Prompt，生成 Response。最终我们要的成品。更新参数
* **Critic Model (Value)**： 教练。预估当前状态的价值（Value），辅助 Actor 计算“收益是否超预期”。更新参数
* **Reference Model**	标杆。SFT 模型的冻结副本，用来限制 Actor 不要跑太远。不更新参数
* **Reward Model**	判官。阶段二训好的模型，给 Actor 的成品打分。不更新参数

