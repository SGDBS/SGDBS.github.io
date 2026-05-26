---
title: Chapter3 多模态与文本对比学习：CLIP、SimCSE、BGE 与 RAG 检索器
categories: 学习笔记-大模型
date: 2026-03-28 16:03:34
mathjax: true
tags:
    - AI
    - AI面试知识
---

> **本章定位**：把 Ch2 的 InfoNCE 推广到 (1) 跨模态——CLIP 用图文对训练，成为多模态 LLM 的视觉骨干；(2) 文本——SimCSE 把 InfoNCE 引入 NLP，BGE/E5 训出现代 RAG 检索器。

> **承上**：Ch2 §A 的 InfoNCE 损失、§B 的 In-batch 负样本机制。
> **启下**：Ch4 转向无负样本路线（BYOL/SimSiam/DINO）。

---

# §A 数学原理

## 1. CLIP 的对称 InfoNCE

CLIP（OpenAI 2021）的核心是**双塔对比**：图像塔与文本塔分别将各自模态映射到同一个表示空间。

设一个 batch 含 $N$ 个图文对 $\{(I_i, T_i)\}_{i=1}^N$：
- Image Encoder（ViT 或 ResNet）$\to v_i \in \mathbb{R}^d$
- Text Encoder（Transformer）$\to t_i \in \mathbb{R}^d$
- 二者 L2 归一化到单位球面

**对称损失**：构造 $N \times N$ 相似度矩阵 $S_{ij} = v_i^T t_j / \tau$。对角线为正样本，其余 $N-1$ 为负样本。两个方向的交叉熵相加：

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2}\left( \mathcal{L}_{\text{img→txt}} + \mathcal{L}_{\text{txt→img}} \right)$$

$$\mathcal{L}_{\text{img→txt}} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(S_{ii})}{\sum_j \exp(S_{ij})}, \quad \mathcal{L}_{\text{txt→img}} \text{ 同理}$$

**关键工程细节**：
- **温度 $\tau$ 是可学习参数**：CLIP 让 $\log(1/\tau)$ 直接参与训练，初始化为 $\log(1/0.07)$。这样模型自己决定相似度尺度。
- **训练规模**：4 亿图文对（WIT 数据集），batch size 32768——巨大的 batch 等价于巨大的负样本池

## 2. SimCSE：NLP 对比学习的"SimCLR 时刻"

文本是离散 token，无法做 Crop/Color Jitter。SimCSE（EMNLP 2021）发现一个**简单到震惊**的方案：**两次不同的 Dropout 当作正样本对**。

### 2.1 Unsupervised SimCSE

同一句话 $x$ 两次过同一个 BERT，由于 Dropout mask 不同，得到 $h_1, h_2$，作为正样本对。Batch 内其他句子作负样本。损失就是标准 InfoNCE：

$$\mathcal{L}_i = -\log \frac{\exp(\text{sim}(h_i^{(1)}, h_i^{(2)}) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(h_i^{(1)}, h_j^{(2)}) / \tau)}$$

### 2.2 Supervised SimCSE

用 NLI（自然语言推理）数据：
- 正样本：(premise, entailment)（蕴含）
- **Hard Negative**：(premise, contradiction)（矛盾）
- 加 hard negative 后效果显著提升

### 2.3 为什么 Dropout 这么简单的方案有效？

Dropout 在表示空间施加了**最小但语义保持**的扰动：
- 保留 anchor 的语义（输入文本完全不变）
- 制造足够的随机性让模型学到鲁棒特征
- 复杂的文本增强（同义词替换、删词）反而可能改变语义，引入噪声标签

> **核心洞察**：对比学习的成功核心是"**语义保持的扰动 + 互斥的负样本池**"。增强的强度只要够引入随机性即可，不必复杂。

## 3. In-batch Negatives 的数学

对 retrieval 任务，batch 内 $N$ 个 (query, positive) 对，构造 $N \times N$ 相似度矩阵：
- 对角线 $S_{ii}$ = (query_i, positive_i) 得分
- **非对角线 $S_{ij}, j \neq i$** = (query_i, positive_j) 得分 → **当作 query_i 的负样本**

一次前向就拿到了 $N-1$ 个免费负样本。损失：
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(S_{ii} / \tau)}{\sum_j \exp(S_{ij} / \tau)}$$

这是 DPR、BGE、E5 等所有 retriever 的标配做法。**Batch size 越大，负样本越多，效果越好**——这与 SimCLR 的逻辑一致。

---

# §B 模型结构（PyTorch 实现）

## B.1 CLIP 完整 forward + loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        # ⭐ 可学习温度（log scale 训练更稳）
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))  # = log(1/0.07)

    def forward(self, images, text_tokens):
        # 1. 各塔提取特征 + 归一化
        v = F.normalize(self.image_encoder(images), dim=-1)         # [N, D]
        t = F.normalize(self.text_encoder(text_tokens), dim=-1)     # [N, D]

        # 2. 相似度矩阵 + 温度缩放
        logit_scale = self.logit_scale.exp().clamp(max=100)         # 上限防爆
        logits_per_image = logit_scale * v @ t.T                    # [N, N]
        logits_per_text = logits_per_image.T

        # 3. 对称 InfoNCE：对角线是正样本
        labels = torch.arange(v.size(0), device=v.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2
```

**为什么用 `logit_scale.exp()` 而非直接学 $\tau$？**
- $\tau > 0$ 是硬约束，直接学 $\tau$ 会导致优化时跑出可行域
- 学 $\log(1/\tau)$ 然后取 exp 自动满足正性约束（参数化技巧，类似 reparameterization）

## B.2 SimCSE Dropout 增强 + 损失

```python
class SimCSE(nn.Module):
    """Unsupervised SimCSE：同一文本两次 forward，Dropout 制造正样本"""
    def __init__(self, bert, temperature=0.05):
        super().__init__()
        self.bert = bert                                            # 默认 dropout=0.1
        self.temperature = temperature

    def encode(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask=attention_mask)
        return out.pooler_output                                    # [N, D]

    def forward(self, input_ids, attention_mask):
        # ⭐ 两次 forward，Dropout mask 不同，得到不同表示
        z1 = self.encode(input_ids, attention_mask)
        z2 = self.encode(input_ids, attention_mask)

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # In-batch negatives：z1[i] 的正样本是 z2[i]，其余 z2[j] 都是负样本
        sim = z1 @ z2.T / self.temperature                          # [N, N]
        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)
```

**两个易错点**：
1. **必须开启 model.train()**：否则 dropout 不生效，两次 forward 完全一样，loss 永远为 0
2. **In-batch 负样本依赖 batch size**：常见 64–256，太小效果差

## B.3 BGE/E5 Retrieval 模型 + Hard Negative

实际 RAG 检索器训练时，每个 (query, pos) 还要配 $K$ 个 hard negatives：

```python
class RetrievalModel(nn.Module):
    def forward(self, q_ids, pos_ids, neg_ids, q_mask, pos_mask, neg_mask):
        """
        q_ids:   [B, L_q]      pos_ids: [B, L_p]
        neg_ids: [B, K, L_n]  K = hard negatives per query
        """
        B, K, L_n = neg_ids.shape
        q_emb = self.encode(q_ids, q_mask)                           # [B, D]
        pos_emb = self.encode(pos_ids, pos_mask)                     # [B, D]
        neg_emb = self.encode(
            neg_ids.view(B*K, L_n), neg_mask.view(B*K, L_n)
        ).view(B, K, -1)                                             # [B, K, D]

        q_emb = F.normalize(q_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        neg_emb = F.normalize(neg_emb, dim=-1)

        # 1. 当前 query 与自己的 positive
        l_pos = (q_emb * pos_emb).sum(dim=-1, keepdim=True)          # [B, 1]

        # 2. 与自己的 K 个 hard negatives
        l_hard_neg = (q_emb.unsqueeze(1) * neg_emb).sum(dim=-1)      # [B, K]

        # 3. ⭐ In-batch negatives：与其他 query 的 positive
        l_in_batch = q_emb @ pos_emb.T                                # [B, B]
        l_in_batch.fill_diagonal_(float('-inf'))                      # 排除自己

        # 拼起来：[正样本, K 个 hard neg, B-1 个 in-batch neg]
        logits = torch.cat([l_pos, l_hard_neg, l_in_batch], dim=-1) / self.temperature
        labels = torch.zeros(B, dtype=torch.long, device=q_emb.device)
        return F.cross_entropy(logits, labels)                       # 正样本永远在第 0 位
```

**这是 BGE / E5 训练的"压舱石"代码**——理解这段代码就理解了现代 RAG 检索器的训练。

---

# §C 训练与推理

## C.1 训练视角：BGE / E5 / GTE 的三阶段训练

当前主流开源 Embedding 模型几乎都遵循：

| 阶段 | 数据 | 损失 | 数据规模 |
| :--- | :--- | :--- | :--- |
| **1. 弱监督对比预训练** | 爬虫"标题-正文"、"问题-答案"等天然配对 | In-batch InfoNCE | **百亿级** |
| **2. 监督对比微调** | MS MARCO、NLI 等高质量标注 | InfoNCE | 千万级 |
| **3. Hard Negative 蒸馏** | 用 cross-encoder 教师挖难负例 | InfoNCE + KL 蒸馏 | 百万级 |

**Hard Negative Mining 的四种方法**：

| 方法 | 思路 | 代价 |
| :--- | :--- | :--- |
| **BM25 Hard Negatives** | 用 BM25 召回 top-k，去掉真 positive | 便宜，但负样本质量一般 |
| **ANCE** (Microsoft) | 用上一版本模型自挖负例，定期刷新 | 中等，需多轮训练 |
| **RocketQA** | Cross-encoder 二次过滤"伪负样本" | 贵，但效果最好 |
| **MoCHi** (NeurIPS 2020) | 在特征空间 mixup 合成 hard negatives | 中等 |

## C.2 推理视角一：FAISS 索引构建 + 召回

训练完 embedding 模型后，**RAG 系统的推理流程**：

```python
import faiss
import numpy as np

# ============ Index 构建（离线一次）============
# 1. 全量文档向量化
doc_embeddings = []
for batch in doc_loader:
    with torch.no_grad():
        emb = retriever.encode(batch).cpu().numpy()
    doc_embeddings.append(emb)
doc_embeddings = np.concatenate(doc_embeddings).astype(np.float32)
faiss.normalize_L2(doc_embeddings)                           # ⭐ 归一化

# 2. 构建 IVF + PQ 索引（亿级文档常用）
d = doc_embeddings.shape[1]
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist=4096, m=8, nbits=8)
index.train(doc_embeddings)
index.add(doc_embeddings)

# ============ Query 召回（每次请求）============
def retrieve(query, k=10):
    with torch.no_grad():
        q_emb = retriever.encode([query]).cpu().numpy().astype(np.float32)
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, k)              # 内积索引
    return indices[0]                                         # top-k 文档编号
```

**关键工程细节**：
- **必须 L2 归一化**：embedding 模型训练时归一化的，索引也必须归一化
- **用 IndexFlatIP 而非 IndexFlatL2**：内积比余弦快 30%（呼应 Ch1 §C.2）
- **大规模用 IVF + PQ**：暴力 IndexFlatIP 在亿级数据上太慢，IVF + PQ 牺牲少量精度换速度

## C.3 推理视角二：完整 RAG 流程

```
用户 Query
   ↓
[Embedding 模型] 编码 query → q_emb
   ↓
[FAISS 索引] 召回 top-100 文档（粗排）
   ↓
[Cross-Encoder Reranker] 重排 → top-5（精排，更准但更慢）
   ↓
[LLM] 把 top-5 文档塞进 prompt 生成答案
```

**为什么要两阶段（召回 + 重排）？**
- 召回阶段：双塔模型，离线编码所有文档，向量检索极快但只看"语义相似"
- 重排阶段：cross-encoder（query 和 doc 拼起来过一次模型），考虑细粒度交互，但只能处理少量候选

## C.4 推理视角三：CLIP 在多模态 LLM 中扮演什么角色

多模态 LLM（LLaVA、GPT-4V、Qwen-VL）的标准做法：

```
图像 ──→ [CLIP Vision Encoder] ──→ 视觉 token ──┐
                       (冻结)                     ├──→ [LLM] ──→ 答案
文字 ──→ [Tokenizer] ──→ 文本 token ──────────────┘
```

**关键工程点**：
1. **冻结 CLIP**：训练时 vision encoder 通常不动，只训练投影层 + LLM 微调
2. **投影层（Projection）**：一个轻量 MLP，把 CLIP 的视觉 token 维度对齐到 LLM 的词嵌入空间
3. **训练成本**：相比从零训练 vision encoder，CLIP 已经把"看图"能力打包好了

**为什么 CLIP 这么"通用"？**
- 4 亿图文对训练 → 几乎覆盖了人类所有视觉概念的语言描述
- 对比学习 → 视觉特征天然与"语言描述"对齐，LLM 理解起来天然顺畅

---

# §D 章末速查

## D.1 三种 InfoNCE 变体对比

| 方法 | 正样本来源 | 负样本来源 | 温度 | 典型 batch |
| :--- | :--- | :--- | :--- | :---: |
| **SimCLR**（Ch2） | 同图像两次增强 | Batch 内其他图像 | 固定 0.5 | 4096+ |
| **CLIP** | 配对的图文 | Batch 内其他图文对 | **可学习** | 32768 |
| **SimCSE** | 同句子两次 Dropout | Batch 内其他句子 | 固定 0.05 | 64–256 |
| **BGE / E5** | 配对的 (q, pos) | In-batch + Hard Neg | 固定 0.02 | 数百到上千 |

## D.2 关键工程要点回顾

- ✅ **L2 归一化** + **内积索引** = 余弦相似度的工程化（Ch1 §C.2）
- ✅ **可学习温度**：CLIP 的关键技巧，让模型自决定相似度尺度
- ✅ **Dropout = 最小增强**：SimCSE 的洞察，复杂增强反而引入噪声
- ✅ **In-batch + Hard Negative**：BGE/E5 训练范式
- ✅ **召回 + 重排**：RAG 标准两阶段架构

---

## 承上启下

本章和 Ch2 一起，把对比学习的"有负样本路线"讲完了：
- Ch2：视觉、SimCLR/MoCo
- Ch3：跨模态/文本、CLIP/SimCSE/BGE

下一章 **Ch4** 进入完全不同的路线：**没有负样本如何训表征模型？** BYOL/SimSiam/DINO 给出了惊人的答案——只要打破对称性（Stop-gradient + Predictor），模型就不会塌缩。这套机制后来直接被 RLHF 借鉴（Reference Policy ≈ Target Network）。
