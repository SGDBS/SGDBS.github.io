---
title: Chapter0 全景导读：LLM 训练全链路学习路径
categories: 学习笔记-大模型
date: 2026-05-07 12:00:00
mathjax: true
tags:
    - AI
    - AI面试知识
---

# 全景导读

这套笔记的目标是从**数学原理 → 模型结构 → 训练与推理**三个视角，串起一条清晰的 LLM 训练全链路：从最底层的距离/散度数学，到表示学习的各种范式，再到现代大模型对齐方法。

---

## 学习路径

```
                            ┌─────────────────────┐
                            │  Ch1 数学工具箱      │  距离 · 散度 · Attention 点积 · CE Loss
                            └──────────┬──────────┘
                                       │ 提供基础度量
                ┌──────────────────────┼──────────────────────┐
                ↓                      ↓                      ↓
   ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
   │ Ch2 视觉对比学习    │  │ Ch3 多模态/文本对比 │  │ Ch4 自监督新范式    │
   │ SimCLR · MoCo      │  │ CLIP · SimCSE      │  │ BYOL · SimSiam     │
   │ InfoNCE 数学        │  │ BGE/E5 · RAG       │  │ DINO · EM 推导      │
   └─────────┬──────────┘  └─────────┬──────────┘  └─────────┬──────────┘
             │                       │                       │
             │   产出 Embedding /     │                       │  产出 stop-grad + EMA 思想
             │   多模态视觉骨干       │                       │  迁移到 RLHF reference
             └───────────────────────┴────────┬──────────────┘
                                              ↓
                            ┌──────────────────────────────┐
                            │  Ch5 SFT 与参数高效微调       │  MLE · LoRA · QLoRA
                            └──────────────┬───────────────┘
                                           ↓
                            ┌──────────────────────────────┐
                            │  Ch6 经典 RLHF：RM + PPO     │  BT · GAE · clip · KL
                            └──────────────┬───────────────┘
                                           ↓ 离线化简
                            ┌──────────────────────────────┐
                            │  Ch7 离线对齐：DPO 家族       │  DPO · IPO · KTO · ORPO
                            └──────────────┬───────────────┘
                                           ↓ 推理时代演进
                            ┌──────────────────────────────┐
                            │  Ch8 推理时代 + AI Feedback   │  GRPO · PRM · RLAIF · CAI
                            └──────────────────────────────┘
```

---

## 三视角统一框架

每一章都按以下三个固定 section 组织：

| 视角 | 核心问题 | 内容形式 |
| :--- | :--- | :--- |
| **§A 数学原理** | "**为什么**这样设计？" | 完整推导，不跳步 |
| **§B 模型结构** | "**长什么样**？" | 架构图 + PyTorch 关键代码 |
| **§C 训练与推理** | "**怎么用**？" | 训练循环、损失计算、推理时如何使用产物 |

> 这三个视角对应工业界三类常见的工作场景：
> - **数学** → 算法/研究：理解为什么 work，知道什么时候不 work
> - **结构** → 模型/工程：能复现、能改、能调
> - **训练与推理** → 部署/应用：知道训练耗多少卡、推理多快、生成质量与超参的关系

---

## 章节对照与依赖关系

| 章节 | 内容核心 | 数学难度 | 工程深度 | 前置依赖 |
| :--- | :--- | :---: | :---: | :--- |
| **Ch1** 数学工具箱 | 相似度 · 散度 · CE/KL | ★★ | ★ | — |
| **Ch2** 视觉对比学习 | InfoNCE · SimCLR · MoCo | ★★★ | ★★ | Ch1 |
| **Ch3** 多模态/文本对比 | CLIP · SimCSE · BGE | ★★ | ★★★ | Ch1, Ch2 |
| **Ch4** 自监督新范式 | BYOL · SimSiam · DINO · EM | ★★★★ | ★★★ | Ch2 |
| **Ch5** SFT + PEFT | MLE · LoRA 数学 | ★★ | ★★★★ | Ch1 |
| **Ch6** 经典 RLHF | BT · GAE · PPO clip | ★★★★ | ★★★★ | Ch1, Ch5 |
| **Ch7** DPO 家族 | KL-RL 闭式解 · BT 反推 | ★★★★ | ★★ | Ch6 |
| **Ch8** 推理时代 | GRPO · PRM · RLAIF | ★★★ | ★★★★ | Ch6, Ch7 |

---

## 推荐阅读顺序

### 顺序一：完整学习（推荐）
**Ch0 → Ch1 → Ch2 → Ch3 → Ch4 → Ch5 → Ch6 → Ch7 → Ch8**

按依赖顺序，不留盲区。

### 顺序二：直奔 LLM 对齐
**Ch1（速读）→ Ch5 → Ch6 → Ch7 → Ch8**

只关心 SFT 之后的事，跳过表示学习。但 **Ch1 的 KL/CE 必读**，否则后面 PPO 的 KL 惩罚和 DPO 的 KL 闭式解都看不懂。

### 顺序三：RAG/Embedding 工程师
**Ch1 → Ch2（速读）→ Ch3（精读）→ Ch4（速读 DINO 部分）**

聚焦 embedding 模型训练 + 多模态视觉骨干。

### 顺序四：面试速查
按章节末尾的 "**常见问题**" / "**核心结论**" 速过，每章用 10 分钟。

---

## 关键概念跨章索引

某些概念在多章出现，这里给出索引：

| 概念 | 首次出现 | 深度展开 | 落地应用 |
| :--- | :--- | :--- | :--- |
| **点积 / Scaled Dot-Product** | Ch1 §3 | Ch1 §B (PyTorch) | Ch2 InfoNCE / Ch6 注意力 |
| **KL 散度** | Ch1 §6 | Ch1 §C (estimator) | Ch6 PPO penalty / Ch7 DPO 闭式解 |
| **交叉熵** | Ch1 §6 | Ch1 §C (LM loss) | Ch5 SFT 损失 / Ch6 RM ranking |
| **InfoNCE** | Ch2 §A | Ch2 §A (MI 下界) | Ch3 CLIP / SimCSE |
| **L2 归一化** | Ch1 §2 | Ch2 §B | Ch3 CLIP / Ch4 BYOL |
| **Stop-gradient + EMA** | Ch4 §A | Ch4 §B (PyTorch) | Ch6 reference policy / Ch7 DPO |
| **Bradley-Terry 模型** | Ch6 §A | Ch6 §A | Ch7 DPO 推导起点 |
| **Reward Hacking** | Ch6 §C | Ch6 §C | Ch7 DPO length bias / Ch8 GRPO |

---

## 写作约定

- **数学符号**：向量用粗体 $\mathbf{x}$，标量用 $x$，参数用 $\theta, \phi, \xi$。
- **PyTorch 代码**：所有代码默认 `import torch; import torch.nn as nn; import torch.nn.functional as F`。
- **超参数**：经验值会标注但不绝对，请结合具体任务调整。
- **引文**：关键论文会在首次出现时给出年份和会议（如 NeurIPS 2020），便于查阅。

---

## 学完后你应该能回答

- ✅ 为什么 Transformer 注意力要除 $\sqrt{d_k}$？（Ch1）
- ✅ SimCLR / MoCo / CLIP 的 InfoNCE 公式分别长什么样？为什么 CLIP 的温度是可学习的？（Ch2, Ch3）
- ✅ BYOL 没有负样本为什么不会塌缩？SimSiam 又怎么去掉了 EMA？（Ch4）
- ✅ LoRA 的低秩分解为什么有效？为什么 SFT 要把 prompt 部分 label 设为 -100？（Ch5）
- ✅ PPO 的 clip 为什么叫 "Proximal"？per-token KL penalty 怎么计算？（Ch6）
- ✅ DPO 怎么从 KL 约束 RL 闭式解推出来的？$\log Z(x)$ 为什么会消去？（Ch7）
- ✅ GRPO 为什么能去掉 Critic？DeepSeek-R1 怎么靠纯 RL 学会 long CoT？（Ch8）

如果有任何一题答不上，回到对应章节。
