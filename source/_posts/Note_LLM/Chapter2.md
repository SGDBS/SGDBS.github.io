---
title: Chapter2 对比学习（Contrastive Learning）的核心机理
categories: 学习笔记-大模型
date: 2026-03-27 16:03:34
mathjax: true
tags:
    - AI
    - AI面试知识
---

# 深度进阶：对比学习（Contrastive Learning）的核心机理

对比学习的核心思想是 **"Instance Discrimination" (个体判别)**：通过构造正负样本对，在无标注数据下学习"物以类聚，人以群分"的特征表示。

> 对 LLM 而言，对比学习是**两条关键路径**的核心训练范式：
> 1. **Embedding 模型训练**（BGE、E5、text-embedding-3）—— RAG 的基石
> 2. **多模态对齐**（CLIP）—— GPT-4V、LLaVA 等视觉语言模型的视觉编码器来源

---

## 1. 样本生成：数据增强（Data Augmentation）

对比学习不依赖标签，通过对同一样本 $x$ 应用两次随机增强 $t, t' \sim \mathcal{T}$，生成正样本对 $(x_i, x_j)$。

### 1.1 视觉领域的核心增强算子
* **Random Resized Crop（最关键）**：强迫模型学习局部与整体、不同尺度下的语义一致性。
* **Color Jitter & Grayscale**：打破模型对色彩统计特性的依赖，防止模型通过颜色直方图"作弊"。
* **Gaussian Blur**：模糊纹理细节，促使模型关注高层轮廓。

### 1.2 NLP 领域的核心增强算子

文本是离散符号，无法直接做 Crop/Blur，因此 NLP 对比学习的"增强"思路完全不同：

* **Dropout 作为最小增强（SimCSE 的核心洞察）**：同一句话两次过同一个 BERT，但**两次 Dropout mask 不同**，得到两个略有差异的 embedding，作为正样本对。这个看似"简单到离谱"的方案在 STS 任务上吊打所有复杂增强。
* **Back-translation（回译）**：中→英→中，得到语义等价但表达不同的句子。
* **EDA (Easy Data Augmentation)**：同义词替换、随机插入、随机交换、随机删除。
* **T5 / LLM Paraphrasing**：用大模型生成同义改写。
* **Token Masking / Cutoff**：随机遮盖部分 token 或连续片段（类似图像 Cutout）。

> **直觉**：图像增强保留"语义不变性"靠像素级扰动；文本增强保留"语义不变性"靠**模型内部噪声（Dropout）或外部改写**。

---

## 2. 数学本质：InfoNCE 损失函数

对比学习本质上是在高维球面上进行 **$K+1$ 路分类**。

### 2.1 公式定义
$$\mathcal{L}_{q, k_+} = -\log \frac{\exp(q \cdot k_+ / \tau)}{\exp(q \cdot k_+ / \tau) + \sum_{i=1}^{K} \exp(q \cdot k_i / \tau)}$$

形式上就是一个以 $q \cdot k_+$ 为正类 logit、$K$ 个负样本 logit 的 softmax 交叉熵。

### 2.2 与互信息（Mutual Information）的关系

InfoNCE 是互信息 $I(q; k_+)$ 的**下界**：
$$I(q; k_+) \geq \log K - \mathcal{L}_{\text{InfoNCE}}$$

* **结论**：负样本数 $K$ 越大，下界越紧，模型对特征空间分布的刻画越精准。
* **直觉**：让模型在 $K+1$ 个候选里挑出正样本越难，$q$ 必须包含越多关于 $k_+$ 的信息。

### 2.3 温度参数 $\tau$ 的双面性

$\tau$ 控制 softmax 分布的**尖锐度**：

| $\tau$ 大小 | softmax 形状 | 效应 |
| :--- | :--- | :--- |
| **大 $\tau$** (如 1.0) | 平滑、接近均匀 | 各样本权重接近，训练稳定但难以聚焦 hard negatives；可避免梯度爆炸 |
| **小 $\tau$** (如 0.05) | 尖锐 | 极大放大"长得像但不是"的困难负样本的 loss，迫使编码器学到细致判别特征；但太小会导致梯度集中在少数样本上，训练不稳 |

* SimCLR 经验值：$\tau = 0.1 \sim 0.5$
* MoCo / SimCSE 经验值：$\tau = 0.05 \sim 0.07$
* CLIP：$\tau$ 设为**可学习参数**（$\log \tau$ 直接参与训练）

---

## 3. 视觉对比学习算法

### 3.1 SimCLR：端到端对称对比 (End-to-End)
SimCLR 的核心逻辑是 **"在大 Batch 中寻找自己"**。

**具体步骤：**
1.  **输入分配**：取一个 Batch 的原始图像 $\{x_k\}_{k=1}^N$。
2.  **双路增强**：对每张图 $x_k$ 进行两次随机增强，生成 $2N$ 张图。其中 $x_{2k-1}$ 和 $x_{2k}$ 互为**正样本对**。
3.  **前向传播**：
    * **提取特征**：所有图片通过同一个编码器 $f(\cdot)$，得到特征 $h = f(x)$。
    * **非线性投影**：通过投影头 $g(\cdot)$（MLP）映射到对比空间：$z = g(h)$。
4.  **计算相似度矩阵**：计算这 $2N$ 个向量两两之间的余弦相似度，形成一个 $2N \times 2N$ 的矩阵。
5.  **损失计算**：对于每一个向量 $z_i$，其正样本只有一个（对应的增强版本），其余 $2N-2$ 个向量均为负样本。
6.  **更新**：梯度同时流经 $g$ 和 $f$，同步更新全网参数。

---

### 3.2 MoCo：动量字典查询 (Dictionary Lookup)
MoCo 的核心逻辑是 **"维护一个平滑演变的负样本库"**。

**具体步骤：**
1.  **双编码器输入**：
    * 输入 $x$，增强得到 $x_q$ 和 $x_k$。
    * $x_q$ 进入 **Query Encoder ($\theta_q$)**，得到向量 $q$。
    * $x_k$ 进入 **Key Encoder ($\theta_k$)**，得到向量 $k_+$。
2.  **归一化 (L2 Norm)**：将 $q, k_+$ 以及队列中所有的 $k$ 映射到单位超球面上。
3.  **度量计算 (Logits)**：
    * **正对**：计算 $q \cdot k_+$。
    * **负对**：计算 $q$ 与 **Queue (队列)** 中存储的 $K$ 个历史特征的点积。
4.  **对比损失**：将正对和 $K$ 个负对看作一个 $K+1$ 类的分类问题，计算 InfoNCE。
5.  **动量更新 (核心)**：
    * **梯度更新**：只对 $\theta_q$ 进行反向传播。
    * **平滑跟随**：$\theta_k \leftarrow m\theta_k + (1-m)\theta_q$，$m$ 通常取 0.999。保证了 Key Encoder 生成的特征在队列中具有时空一致性。
6.  **队列维护**：将当前 Batch 的 $k_+$ 加入队列（Enqueue），并剔除最老的特征（Dequeue, FIFO）。

### 3.3 SimCLR vs MoCo

| 特性 | SimCLR | MoCo |
| :--- | :--- | :--- |
| **负样本规模** | 受 Batch Size 限制 (需超大 Batch) | 由 Queue 决定 (可达 65536) |
| **负样本一致性** | 实时计算，一致性完美 | 依靠**动量更新**保证一致性 |
| **解决模型坍塌** | 显式负样本排斥 | 显式负样本排斥 |
| **硬件要求** | 极高 (TPU/多卡集群) | 友好 (单卡即可训练大模型) |

> **面试 Tip**：为什么 SimCLR 需要 Projection Head？
> 因为对比学习任务可能会由于过于关注"不变性"而损害特征的语义。非线性投影层 $g(h)$ 可以作为一个"防火墙"，让信息损失发生在 $z$ 层，从而保护 $h$ 层保留更多的下游任务有用信息（如颜色、形状）。

---

## 4. CLIP：跨模态对比学习 —— 多模态 LLM 的基石

CLIP（Contrastive Language–Image Pretraining，OpenAI 2021）是对比学习史上最具影响力的工作之一，也是 **GPT-4V / LLaVA / Qwen-VL / Gemini** 等多模态 LLM 视觉编码器的标准来源。

### 4.1 双塔架构

* **Image Encoder**：ViT 或 ResNet，输出图像向量 $v \in \mathbb{R}^d$
* **Text Encoder**：Transformer，输出文本向量 $t \in \mathbb{R}^d$
* 两个向量 L2 归一化到单位球面

### 4.2 对称 InfoNCE 损失

设一个 batch 含 $N$ 个图文对 $\{(I_i, T_i)\}_{i=1}^N$，构造 $N \times N$ 相似度矩阵 $S_{ij} = v_i \cdot t_j / \tau$。

对角线 $S_{ii}$ 是正样本，其余为负样本。同时计算"图找文"和"文找图"两个方向的交叉熵：

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2}\left( \mathcal{L}_{i \to t} + \mathcal{L}_{t \to i} \right)$$

### 4.3 关键工程细节
* **训练规模**：4 亿图文对（WIT 数据集），batch size 32768
* **温度可学习**：$\tau$ 不是超参数，而是一个可训练标量，初始化为 $\log(1/0.07)$
* **零样本分类**：训练后可直接将类别名（如 "a photo of a dog"）编码成文本向量，与图像向量比相似度 → 不需要任何分类头

> **CLIP 在 LLM 中扮演什么角色？**
> 多模态 LLM（如 LLaVA）的标准做法是：**冻结 CLIP 的 Image Encoder** 提取图像特征，再接一个轻量的 projection layer 把视觉特征对齐到 LLM 的词嵌入空间。也就是说，你看到的所有"GPT-4 看图说话"，背后都是 CLIP 学到的对比表示在打底。

---

## 5. NLP 中的对比学习：Sentence Embedding 与 RAG 检索器

这是把 Chapter1 的余弦相似度落地到 LLM 实际应用的关键一环。

### 5.1 SimCSE（EMNLP 2021）—— NLP 对比学习的"SimCLR 时刻"

**核心思想**：不需要任何复杂数据增强，**只用 Dropout 当作噪声**。

* **Unsupervised SimCSE**：同一句话 $x$ 两次过同一个 BERT，由于 Dropout mask 不同，得到 $h_1, h_2$，作为正样本对；batch 内其他句子作为负样本。
* **Supervised SimCSE**：用 NLI 数据集，"蕴含 (entailment)" 关系当正样本，"矛盾 (contradiction)" 关系当 hard negative。

> **为什么 Dropout 这么简单的方案有效？**
> Dropout 在表示空间施加了**最小但语义保持**的扰动——既保留了 anchor 的语义，又制造了足够的随机性让模型学到鲁棒表征。复杂的文本增强（如同义词替换）反而可能改变语义，引入噪声标签。

### 5.2 In-batch Negatives：检索训练的标配

设 batch 内有 $N$ 个 (query, positive) 对，构造 $N \times N$ 相似度矩阵：
* 对角线 $S_{ii}$ = 正样本得分
* 非对角线 $S_{ij}, j \neq i$ = **同 batch 内其他样本的 positive 充当 query $i$ 的负样本**

这样一次前向就拿到了 $N-1$ 个免费负样本，是 DPR、BGE、E5 等所有 retriever 的标配做法。

### 5.3 Hard Negative Mining：让训练真正"难"起来

In-batch 随机负样本太简单，模型很快就能区分。需要主动挖**长得像但不相关**的样本：

| 方法 | 思路 |
| :--- | :--- |
| **BM25 Hard Negatives** | 用 BM25 召回 top-k，去掉真正的 positive，剩下的当 hard negatives |
| **ANCE** (Microsoft) | 用上一版本模型自己挖难负例，定期刷新 |
| **RocketQA** | Cross-encoder 二次过滤，去除"伪负样本"（其实是正样本但没标注） |
| **MoCHi** (NeurIPS 2020) | 在特征空间通过混合（mixup）合成 hard negatives |

### 5.4 BGE / E5 / GTE 的三阶段训练范式

当前主流开源 Embedding 模型几乎都遵循：
1. **大规模弱监督对比预训练**：用爬虫抓的"标题-正文"、"问题-答案"等天然配对（百亿规模）
2. **监督对比微调**：用 MS MARCO、NLI 等高质量标注数据
3. **Hard negative 蒸馏**：用 cross-encoder 教师挖难负例并重训

---

## 6. 不需负样本的方向：BYOL / SimSiam / Barlow Twins

> 完全不要负样本！这是对比学习一度被认为"不可能"的方向。

### 6.1 BYOL（Bootstrap Your Own Latent）
* **架构不对称**：Online network（含 Predictor）+ Target network（EMA 更新，无 Predictor）
* **损失**：让 Online 的 prediction 去回归 Target 的输出
* **关键技术**：**Stop-gradient**，梯度不流向 Target 网络

### 6.2 为什么不会塌缩？（核心问题）

如果两个网络一样，最优解就是输出常数。但 BYOL/SimSiam **打破了对称性**：
* **Stop-gradient + EMA**：让 Target 成为 Online 网络的"过去自己"——一个移动平均。Online 想匹配的不是另一份当前自己，而是历史快照。
* **Predictor MLP**：把 Online 的输出再经过一层非线性，相当于在表示空间中引入额外的"目标空间"，进一步打破对称。
* **数学解释（SimSiam 论文）**：这种结构隐式实现了一种 EM 算法——交替优化"表征"和"目标"，类似 K-means 不会塌缩到单点。

### 6.3 Barlow Twins：从协方差矩阵入手

让正样本对的特征**互相关矩阵**趋近单位矩阵：
* 对角线 → 1：同一样本不同视图的对应维度强相关
* 非对角线 → 0：特征维度之间去相关，避免冗余

这种方法不需要负样本，也不需要不对称架构，靠的是显式的"特征去相关"约束。

---

## 7. 总结表：CV 与 NLP 的对照

| 视觉方法 | NLP / LLM 对应 | 关键差异 |
| :--- | :--- | :--- |
| **SimCLR** | **SimCSE** | 增强：Crop/Color → Dropout |
| **MoCo** | **MoCo for Sentence Embedding** | 队列机制相同 |
| **CLIP** | **CLIP / BLIP / SigLIP** | 图文双塔 → 多模态 LLM 视觉编码器 |
| **Hard Negative Mining (MoCHi)** | **ANCE / RocketQA** | 检索训练的核心技术 |
| **BYOL / SimSiam** | （NLP 中较少用，因 Dropout 增强已足够） | — |

---

## 8. 对比学习负样本是否重要？负样本构造成本过高怎么解决？

### 8.1 负样本为什么重要？（数学本质）

从信息论角度，对比学习的目标是最大化正样本对之间的**互信息**：
$$I(q; k_+) \geq \log K - \mathcal{L}_{\text{InfoNCE}}$$

* **防止模型崩溃 (Model Collapse)**：没有负样本时，模型最简单的"偷懒"方式是输出常数，相似度永远最大。负样本提供"推开"的力，迫使模型寻找区分性特征。
* **InfoNCE 紧致度**：负样本数 $K$ 越大，下界越紧——模型对特征空间的刻画就越精准。

### 8.2 负样本成本过高的四种解法

* **A. 缓存机制（代表作：MoCo）**
    * 用队列存储历史 batch 的特征 + 动量更新解决"特征陈旧"问题
    * 极小显存即可获得上万负样本

* **B. 寻找"硬"负样本（Hard Negative Mining）**
    * 与其要一万个无用负样本，不如要十个真正难的
    * 代表方法：**MoCHi**（特征空间合成）、**ANCE**（自挖）、**RocketQA**（cross-encoder 过滤）

* **C. 改变对称性 / 预测机制（代表作：BYOL, SimSiam）**
    * 通过 Stop-gradient + Predictor 打破对称，无需负样本也不塌缩

* **D. 特征去相关（代表作：Barlow Twins）**
    * 让特征互相关矩阵趋近单位矩阵，从协方差层面防止塌缩
