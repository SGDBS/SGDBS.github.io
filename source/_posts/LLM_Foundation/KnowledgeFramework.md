---
title: 工业界 LLM 完整知识框架清单（2025–2026）—— 跨 AI_Model / AI_Infra / LLM_Foundation 全景版
categories: 学习笔记-大模型
date: 2026-05-26 12:00:00
mathjax: true
tags:
    - AI
    - AI面试知识
    - LLM Foundation
---

> 你的笔记已经按三个文件夹组织：
> - **`source/_posts/AI_Model/`** —— 模型架构本体（Transformer、RoPE…）
> - **`source/_posts/AI_Infra/`** —— 训练/推理基础设施（DDP、ZeRO、Pipeline、Kernel…）
> - **`source/_posts/LLM_Foundation/`** —— 算法（表示学习、SFT、PPO、DPO、GRPO…）
>
> 这份文档做三件事：
> 1. 给三个文件夹**划清边界**，每块写什么、不写什么
> 2. 对照 2025–2026 工业界完整知识树，**列出每个模块的归属和当前覆盖情况**
> 3. 给出**真正还缺的内容清单**和补完优先级

---

# Part 1：三大文件夹的边界与定位

| 文件夹 | 关注什么 | 不要碰什么 | 一句话定位 |
| :--- | :--- | :--- | :--- |
| **AI_Model** | 模型架构本体、组件、组装方式 | 训练算法（SFT/PPO）、并行/Kernel | "**这个网络长什么样**" |
| **AI_Infra** | 训练并行、显存账、通信、Kernel、推理服务 | 模型架构、算法目标函数 | "**这个网络怎么跑起来**" |
| **LLM_Foundation** | 训练目标 / 损失 / 对齐 / RL 算法 / 表示学习 | 架构细节、底层并行 | "**这个网络要优化什么**" |

边界判断小技巧：
- 看到"如何切"、"显存怎么省"、"通信路径"、"Kernel" → AI_Infra
- 看到"网络结构"、"Block 怎么组装"、"位置编码"、"Norm 选哪个" → AI_Model
- 看到"Loss 怎么定义"、"对齐怎么做"、"RM/PPO/DPO/GRPO"、"采样策略与 reward" → LLM_Foundation

---

# Part 2：现有覆盖盘点

## 2.1 AI_Model 已覆盖

| 主题 | 文件 | 状态 |
| :--- | :--- | :--- |
| Transformer 完整结构 | `Transformer.md` | ✅ Encoder-Decoder / MHA / FFN / Add&Norm（含 RMSNorm）/ Pre/Post-Norm |
| 训练机制 | `Transformer.md` §7 | ✅ Teacher Forcing / Label Smoothing / Warmup LR |
| 推理解码 | `Transformer.md` §8 | ✅ Greedy / Beam / Top-k / Top-p |
| 位置编码（正弦） | `Transformer.md` §2.2–2.4 | ✅ |
| **RoPE** | `RoPE.md` | ✅ 复数旋转 + 相对位置推导 |

## 2.2 AI_Infra 已覆盖

| 主题 | 文件 | 状态 |
| :--- | :--- | :--- |
| 显存账 / 7B 模型显存构成 | `1.DataParallel/MemoryBudget.md` | ✅ |
| 混合精度 BF16/FP16 + Loss Scaling | `1.DataParallel/MixedPrecision.md` | ✅ |
| DDP / Ring AllReduce | `1.DataParallel/DDP.md` | ✅ |
| ZeRO-1/2/3 | `1.DataParallel/ZeRO.md` | ✅ |
| 多机 NCCL + 网络拓扑 | `1.DataParallel/MultiNode.md` | ✅ |
| 异步计算 / CUDA Stream | `1.DataParallel/AsyncCompute.md` | ✅ |
| Checkpoint（含 FSDP shard） | `1.DataParallel/Checkpoint.md` | ✅ |
| GPipe / 1F1B / Interleaved 1F1B | `2.PipelineParallel/*.md` | ✅ |
| **FlashAttention** | `4.GPUKernel/FlashAttention.md` | ✅ Tiling+Online Softmax+Recompute |
| **KV Cache** | `4.GPUKernel/KVCache.md` | ✅ Prefill/Decode 全流程 |
| 总规划 | `0.Introduction/Chapter0.md` | ✅ 12 章蓝图 |
| Triton/CUDA Kernel 路线 | `0.Introduction/KernelRoadmap.md` | ✅ 9 章规划 |

## 2.3 AI_Infra 已规划但还没写

| 模块 | 目录 | 你的规划里包含 |
| :--- | :--- | :--- |
| **Tensor Parallel + Sequence Parallel** | `3.TensorParallel/` | Megatron TP 切法、SP、ColumnParallel/RowParallel |
| **Expert Parallel（MoE 训练）** | （Chapter0 Ch4） | Top-k routing、load balance、EP 通信 |
| **Context Parallel（长上下文训练）** | （Chapter0 Ch5） | Ring Attention、序列切分 |
| **Inference 完整栈** | `5.Inference/` | vLLM/SGLang、PagedAttention、Continuous Batching、Speculative、Quantization |
| **Triton/CUDA Kernel 实战** | KernelRoadmap | GPU 硬件心智模型、矩阵乘、Reduction、Fused Kernel、FlashAttn 实战 |

## 2.4 LLM_Foundation 已覆盖

| Ch | 主题 | 状态 |
| :--- | :--- | :--- |
| Ch1 | 距离/散度/CE/KL | ✅ |
| Ch2 | SimCLR / MoCo / InfoNCE | ✅ |
| Ch3 | CLIP / SimCSE / BGE / RAG retriever 训练 | ✅ |
| Ch4 | BYOL / SimSiam / DINO（含 EM 推导） | ✅ |
| Ch5 | SFT + LoRA/QLoRA/DoRA | ✅ |
| Ch6 | RM + PPO + 失败模式 + Infra 视角 | ✅✅（质量很高） |
| Ch7 | DPO / IPO / KTO / ORPO / SimPO | ✅ |
| Ch8 | GRPO / PRM / RLAIF / CAI | ✅ |
| tmp.md | PPO IS 草稿 | ⚠️ 建议删（已并入 Ch6 §A.3） |

---

# Part 3：完整工业知识树 × 你的目录归属

> 每个模块标注：**已覆盖位置**、**还缺什么**、**该写在哪个文件夹**。

## 模块 1：数学与基础 → `LLM_Foundation/Ch1`

✅ 你的 Ch1 已经覆盖了所有 LLM 必需的数学工具。可微调补充：
- *（小补充）* JS 散度、Wasserstein（在 GAN/分布对齐场景偶尔考）
- *（小补充）* 互信息 / PMI（虽然 Ch2 用 InfoNCE 间接讲了 MI 下界）

## 模块 2：Tokenizer & Chat Template → **缺**，建议放 `AI_Model`

工业 LLM 必备：
- BPE / BBPE / WordPiece / Unigram / SentencePiece 算法差异
- Tiktoken / Llama-3 / Qwen / DeepSeek 各家 tokenizer 对比
- 特殊 token + Chat Template（ChatML、Llama-3、Qwen2）
- Tokenizer 边界 bug（SolidGoldMagikarp、leading-space）
- 多语言压缩率与训练成本

📁 建议：`AI_Model/Tokenizer.md`（这是模型本体的一部分）

## 模块 3：Transformer 架构本体 → `AI_Model/Transformer.md`

✅ 你的 Transformer.md 已经把核心讲透了（含 RMSNorm、Pre/Post-Norm）。

**还缺的现代点**（建议补到 Transformer.md 后半段或独立文件）：
- **SwiGLU / GeGLU / GLU 家族**（Llama 系 FFN 标配）
- **GQA / MQA / MHA 切换**（Llama-3、Qwen 都用 GQA）
- LM Head Tied / Untied、logit soft-capping（Gemma 2）
- µP 参数化（超参跨规模迁移）

📁 建议：把这些追加到 `AI_Model/Transformer.md`，或新建 `AI_Model/ModernTransformer.md`

## 模块 4：位置编码与长上下文 → `AI_Model` + `AI_Infra`

✅ RoPE 数学：`AI_Model/RoPE.md`

**还缺**：
- ALiBi、T5 relative bias（对照组）
- **RoPE 外推**：NTK-aware、**YaRN**、LongRoPE、Position Interpolation、θ-scaling
- **StreamingLLM**（attention sink 现象）
- 长上下文评测：Needle / RULER / LongBench

📁 建议：
- 外推数学 → `AI_Model/RoPEScaling.md`（接 RoPE.md）
- Ring Attention / Context Parallel 系统层面 → `AI_Infra/5.Inference/` 或 `AI_Infra/6.ContextParallel/`（你 Chapter0 Ch5 已规划）

## 模块 5：现代架构变体 → **缺**，分两边写

| 主题 | 归属 | 说明 |
| :--- | :--- | :--- |
| **MoE 架构本身**（top-k routing、shared experts） | `AI_Model` | 网络结构 |
| **MoE 训练 Infra**（Expert Parallel、load balance） | `AI_Infra`（Chapter0 Ch4 已规划） | 切分与通信 |
| **MLA**（Multi-head Latent Attention，DeepSeek-V2） | `AI_Model` | KV 压缩到 latent，结构改造 |
| GQA / MQA / MHA | `AI_Model`（已在 Transformer 提了但可加深） | 结构与显存权衡 |
| Sliding Window / Sparse / Global+Local | `AI_Model` | Mistral、Longformer |
| **SSM / Mamba / Mamba-2 / RWKV / RetNet** | `AI_Model`（如要写实现）+ `AI_Infra`（硬件感知 scan） | 线性注意力替代 |
| Hybrid（Jamba/Samba/Zamba） | `AI_Model` | Transformer⊕Mamba |

## 模块 6：预训练（Pretraining）→ **缺**，建议新建专门系列

数据 + Scaling Laws + 训练目标这套属于"训练方法学"，三个文件夹都不完全对口。建议新建：

📁 **`source/_posts/Pretraining/`**（新文件夹），内容：
- 数据：CommonCrawl / FineWeb / RedPajama / Dolma、dedup / quality filter / mixture
- 训练目标：CLM / FIM（代码必备）/ UL2 / Mixture-of-Denoisers
- **Scaling Laws**：Kaplan、Chinchilla（20× tokens）、over-training、inference-aware、µP
- LR schedule：cosine / **WSD**（Warmup-Stable-Decay）/ Inverse-Sqrt、batch ramp
- Curriculum、continual / mid-training（DeepSeek-Math、Llama-3 long-context stage）
- 训练稳定性：loss spike、Z-loss、QK-LayerNorm、router z-loss
- 优化器：AdamW / **Lion** / **Muon** / Shampoo / SOAP
- 数值精度战略：BF16 / FP16 / **FP8**（H100/B200）

⚠️ 注意分工：
- **算法层**（AdamW 数学、WSD 形状、Scaling Laws 公式）→ Pretraining
- **系统层**（FP8 怎么在硬件上跑、optimizer state offload）→ AI_Infra/MixedPrecision

## 模块 7：分布式训练 → `AI_Infra`（你已规划完整）

✅ DDP / ZeRO（1.DataParallel）  
✅ GPipe / 1F1B / Interleaved（2.PipelineParallel）  
⏳ **Tensor Parallel + Sequence Parallel**（3.TensorParallel 待写）  
⏳ **Expert Parallel**（Chapter0 Ch4 已规划）  
⏳ **Context Parallel / Ring Attention**（Chapter0 Ch5 已规划）  
✅ Mixed Precision、AsyncCompute、Checkpoint、MultiNode  

**还可以补**：3D 并行（DP×TP×PP 组合策略）、TorchTitan / Megatron-LM / DeepSpeed 框架对比、Elastic training。

## 模块 8：Post-training（对齐）→ `LLM_Foundation`（你的核心，已覆盖）

✅ SFT（Ch5）  
✅ PEFT：LoRA/QLoRA/DoRA（Ch5）  
✅ PPO/RLHF（Ch6）  
✅ DPO 家族（Ch7）  
✅ GRPO / PRM / RLAIF（Ch8）

**建议补**（在 Ch8 后或新增 Ch9）：
- **RFT（Rejection Fine-Tuning）/ STaR / Best-of-N → SFT**
- **蒸馏**：黑盒 vs 白盒、top-k logit、**on-policy distillation（MiniLLM、DistiLLM）**
- 合成数据：Self-Instruct、Evol-Instruct、Magpie、persona-hub
- **Reasoning 训练栈细节**：long CoT 训练、Self-Play（SPIN、Self-Rewarding、SPPO）
- 新 PG 变体：**RLOO、ReMax、REINFORCE++、DAPO、VAPO**（GRPO 的近亲）

## 模块 9：推理与服务 → `AI_Infra/5.Inference/`（你已规划）

⏳ 这块你 Chapter0 列了规划，但 5.Inference 目录还是空的。建议写：

| 子主题 | 内容 |
| :--- | :--- |
| FlashAttention v2/v3 | ✅ `4.GPUKernel/FlashAttention.md` 已有；可补 v3 / FlashDecoding |
| Quantization | **GPTQ / AWQ / SmoothQuant / FP8 / GGUF / Marlin kernel** |
| 批处理调度 | **Continuous Batching、PagedAttention、Chunked Prefill、Prefix Caching** |
| **Disaggregated Prefill/Decode** | Splitwise、DistServe、Mooncake |
| 投机解码 | Vanilla Spec / **Medusa / EAGLE / Lookahead** |
| 推理框架 | **vLLM / SGLang / TensorRT-LLM / llama.cpp / LMDeploy** |
| Multi-LoRA serving | LoRAX / Punica |
| 解码策略 | *已在 `AI_Model/Transformer.md` §8 覆盖*；可补 Min-p / DRY / Contrastive / Constrained（Outlines） |

⚠️ 注意：解码算法（Top-k/p/Beam）放 AI_Model，**服务系统**（vLLM/调度/量化 kernel）放 AI_Infra。

## 模块 10：评测（Evaluation）→ **缺**，建议放 `LLM_Foundation`

评测和对齐紧密耦合（Reward Model 训练数据来自评测），但单独一章更好：

📁 建议：`LLM_Foundation/Ch9_Evaluation.md`，覆盖：
- 综合：MMLU / MMLU-Pro / AGIEval / BBH / GPQA / SuperGPQA
- 数学：GSM8K / MATH / **AIME 2024-2025** / OlympiadBench
- 代码：HumanEval+ / MBPP+ / **LiveCodeBench / SWE-Bench Verified / Aider**
- 指令：IFEval / FollowBench
- 对话：MT-Bench / **Arena Hard / Chatbot Arena ELO** / AlpacaEval 2 LC
- 长文本：Needle / RULER / LongBench / InfiniteBench
- Agent：BFCL / τ-bench / GAIA / WebArena / OSWorld
- 多模态：MMMU / MathVista / VideoMME
- 安全：HarmBench / AdvBench / WildJailbreak
- **方法学**：LLM-as-judge 偏差（position/length/self-preference）、Pairwise vs Single、数据污染、可复现性

## 模块 11：RAG 完整栈 → `LLM_Foundation` 或新建 `RAG/`

你的 Ch3 只讲了 retriever 训练。RAG 工程栈还缺：

📁 建议：`LLM_Foundation/Ch10_RAG.md` 或独立文件夹
- 分块：fixed / recursive / semantic / late chunking
- 向量索引：FAISS（IVF/HNSW/PQ）/ Milvus / Qdrant / pgvector
- **混合检索**：BM25 + dense + RRF / **ColBERT v2**（late interaction）
- **Reranker**：bge-reranker、Cohere、Jina、LLM-as-reranker
- Query 增强：**HyDE、Multi-query、Step-back、Query decomposition**
- 高级范式：**GraphRAG（Microsoft）、RAPTOR、Self-RAG、Corrective RAG、Agentic RAG**
- 评测：**RAGAS、ARES**、Faithfulness / Answer Relevance / Context Precision
- 长上下文 vs RAG：lost-in-the-middle、需求场景分流

## 模块 12：Agent / Tool Use → **缺**，建议新建 `Agent/`

📁 建议：`source/_posts/Agent/`
- Function Calling（OpenAI / Anthropic / Qwen-Agent）
- **MCP（Model Context Protocol）** —— 2024-2026 工业标准
- 范式：ReAct / Reflexion / Plan-and-Solve / Self-Ask
- Code Interpreter / Sandbox（E2B）
- Multi-Agent：AutoGen / CrewAI / LangGraph / MetaGPT / Swarm
- Browser/Desktop Agent：WebArena / SeeAct / **Anthropic Computer Use**
- RL for Agents：Tool-Star / ToolRL / Agent Tuning
- 评测：BFCL / τ-bench / SWE-bench / OSWorld

## 模块 13：多模态 VLM → 多个文件夹

| 子主题 | 归属 |
| :--- | :--- |
| 视觉骨干（CLIP/DINOv2/SigLIP） | ✅ 已在 `LLM_Foundation/Ch3, Ch4` |
| **投影方式**：MLP / Q-Former / Perceiver Resampler | `AI_Model/VLM.md`（新增） |
| **训练阶段**：alignment → instruction tuning | `LLM_Foundation`（属于 post-training） |
| 高分辨率：AnyRes / tile / native res（Qwen2-VL） | `AI_Model` |
| 原生多模态：Chameleon / Gemini / GPT-4o / Janus | `AI_Model` |
| 视频 / 音频 | `AI_Model` |
| **多模态对齐**：RLHF-V / VLFeedback / SPA-VL | `LLM_Foundation` |

## 模块 14：安全 / 红队 → **缺**，建议放 `LLM_Foundation`

📁 建议：`LLM_Foundation/Ch11_Safety.md`
- Jailbreak：DAN / role-play / Many-shot / **PAIR / AutoDAN / GCG**
- Prompt Injection（间接注入 via tool/retrieval）
- 防御：system prompt hardening、output filtering、Constitutional Classifier
- Refusal training、harmlessness vs helpfulness 平衡
- 红队工具：HarmBench / JailbreakBench / PyRIT
- 水印（Kirchenbauer / SynthID-Text）
- 隐私：成员推断、数据萃取

## 模块 15：工具链与生态 → 各处自然渗透即可

不需要单独写章。在用到的地方自然提到：
- 训练：HF Transformers / TRL / Accelerate / PEFT / Axolotl / LLaMA-Factory / Unsloth / TorchTune / DeepSpeed / Megatron / NeMo / TorchTitan / **veRL**
- 推理：vLLM / SGLang / TensorRT-LLM / Ollama / llama.cpp / LMDeploy
- RAG：LangChain / LlamaIndex / DSPy
- Agent：LangGraph / AutoGen / CrewAI / Smol-Agents

---

# Part 4：你目前的全景图（视觉化）

```
┌─────────────────────────────────────────────────────────────────────┐
│                      已完成（深度 ★★★★+）                            │
├─────────────────────────────────────────────────────────────────────┤
│  AI_Model:          Transformer 主体 + RoPE                          │
│  AI_Infra:          DP/ZeRO/Pipeline/FlashAttn/KV/MixedPrec/NCCL    │
│  LLM_Foundation:    数学 + 表示学习 + SFT + PPO + DPO + GRPO         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                  已规划但未写（你已经知道要补）                       │
├─────────────────────────────────────────────────────────────────────┤
│  AI_Infra:                                                           │
│    • 3.TensorParallel（Megatron TP/SP）                              │
│    • Expert Parallel（MoE 训练）                                     │
│    • Context Parallel（Ring Attention）                              │
│    • 5.Inference（vLLM/Quant/Spec/Disaggregated）                    │
│    • Triton/CUDA Kernel 实战路线                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    全空白（建议补的高优先级）                         │
├─────────────────────────────────────────────────────────────────────┤
│  AI_Model:                                                           │
│    P0  • Tokenizer & Chat Template                                   │
│    P0  • SwiGLU/GLU + GQA/MQA 现代细节（追加到 Transformer.md）       │
│    P0  • RoPE Scaling（YaRN/NTK/PI）                                 │
│    P1  • MoE 架构（DeepSeek-V3、Mixtral）                            │
│    P1  • MLA（Multi-head Latent Attention）                          │
│    P2  • Mamba / SSM / Hybrid                                        │
│    P2  • VLM 架构（投影方式、原生多模态）                            │
│                                                                      │
│  新建文件夹 Pretraining/:                                            │
│    P1  • 数据 / Scaling Laws / WSD                                   │
│    P1  • 优化器（AdamW/Lion/Muon）/ FP8 训练                         │
│    P1  • 训练稳定性 / loss spike                                     │
│                                                                      │
│  LLM_Foundation/（在现有基础上加章）:                                │
│    P0  • Ch9 评测体系                                                │
│    P1  • Ch10 RAG 完整栈                                             │
│    P1  • Ch11 安全 / Jailbreak                                       │
│    P2  • Ch12 蒸馏 / 合成数据 / RFT                                  │
│    P2  • Ch13 Reasoning 训练栈深化                                   │
│                                                                      │
│  新建文件夹 Agent/:                                                  │
│    P2  • Function Calling / MCP / ReAct / Multi-Agent                │
└─────────────────────────────────────────────────────────────────────┘
```

---

# Part 5：补完优先级（按 ROI 排序）

如果按"补完后立刻能上面试 / 工业实战"的回报排：

### 第一梯队（必补，2-3 周）

1. **`AI_Model/Transformer.md` 补**：SwiGLU、GQA/MQA → 1 天
2. **`AI_Model/Tokenizer.md` 新增**：BPE/Chat Template/Template 漏洞 → 2 天
3. **`AI_Model/RoPEScaling.md` 新增**：NTK/YaRN/LongRoPE → 2 天
4. **`AI_Infra/3.TensorParallel/`**（你已规划）：Megatron 切法、SP → 3 天
5. **`AI_Infra/5.Inference/`**（你已规划）：vLLM/PagedAttn/Continuous Batching/Speculative/GPTQ-AWQ → 1 周
6. **`LLM_Foundation/Ch9_Evaluation.md`**：评测体系全图 → 3 天

### 第二梯队（强烈推荐，3-4 周）

7. **`AI_Model/MoE.md`**：MoE 架构 + DeepSeek-V3 / Mixtral 解析 → 3 天
8. **`AI_Model/MLA.md`**：DeepSeek-V2 MLA → 2 天
9. **`AI_Infra/Chapter0` 里 Ch4 Expert Parallel + Ch5 Context Parallel**（你已规划）：MoE 训练 / Ring Attention → 1 周
10. **新建 `Pretraining/`**：数据 / Scaling Laws / WSD / Lion-Muon / FP8 训练 → 1 周
11. **`LLM_Foundation/Ch10_RAG.md`**：完整 RAG 栈 → 1 周

### 第三梯队（应用层，2-3 周）

12. **`LLM_Foundation/Ch11_Safety.md`** → 3 天
13. **`LLM_Foundation/Ch12_Distillation.md`**（含 RFT / 合成数据）→ 3 天
14. **`AI_Model/VLM.md` + `LLM_Foundation` VLM 对齐** → 1 周
15. **新建 `Agent/`** → 1 周

### 暂可不写

- Mamba / SSM / Hybrid（工业占比低）
- µP（小众，研究向）
- 多机训练调度（Slurm/K8s 细节）

---

# Part 6：现有 LLM_Foundation 内部整理建议

具体回到你最初问的——LLM_Foundation 文件夹内部要不要动？

### 删

- `tmp.md` —— Ch6 §A.3 已经写完整了，删

### 微调

- **Ch4 SimSiam EM 推导**：保留 stop-grad + EMA 与 RLHF reference 的连接（§D 那段），EM 数学推导（§A.2）建议挪到附录或单独博客，因为它对 LLM 工程几乎没用

### 可以合并

- Ch2 + Ch3 + Ch4 三章对比学习内容，如果不想压缩，至少在 Ch0 全景导读里说清楚："这三章是表示学习史，对 LLM 的真正贡献只是给 Ch6 RLHF 提供了'stop-grad + EMA'的思想来源 + Ch3 RAG retriever 训练知识"

### 加

- Ch9 评测
- Ch10 RAG 完整栈
- Ch11 安全
- Ch12 蒸馏 / 合成数据 / RFT
- Ch13 Reasoning 训练栈深化（long CoT 训练细节、Self-Play、新 PG 变体）

---

# Part 7：一句话总结

你的现状是 **"对齐算法（LLM_Foundation）+ 训练 Infra（AI_Infra）+ Transformer 本体（AI_Model）"** 这三大支柱已经有了相当不错的底子。真正的高优先级缺口是：

> **Tokenizer + RoPE 外推 + 现代架构变体（MoE/MLA/GQA-SwiGLU）+ Tensor Parallel + Inference 服务栈 + 评测体系 + RAG 完整栈**

这一波补完，你的整套笔记就能称得上"完整工业级 LLM Foundation"。
