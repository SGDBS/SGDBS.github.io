---
title: AI Infra学习计划
categories: 日记
date: 2026-04-23 22:48:00
mathjax: true
tags:
    - 日记
---


# 字节 27 届 AI Infra 秋招 3 个月冲刺计划

> **制定日期**:2026 年 4 月 24 日
> **目标岗位**:字节跳动 27 届 AI Infra / Diffusion / 大模型相关岗位
> **入职时间**:2027 年春/夏
> **时间窗口**:2026.4.25 ~ 2026.7.25(约 13 周)
> **每日投入**:4-6 小时(学习 3-4h + 求职 1-2h)

---

## 📋 个人情况诊断

| 项目 | 现状 | 策略影响 |
|---|---|---|
| 算法基础 | ICPC 铜牌 | ✅ 核心竞争力,恢复手感即可 |
| 毕业时间 | 2026 年 9 月 | 研二下 + 毕业论文同步进行 |
| 实习状态 | **无,在找** | ⚠️ **第 1 个月最大任务** |
| CUDA / 分布式训练 | 概念懂,未写过代码 | 需要 3-4 周动手训练 |
| Infra 项目经验 | 暂无 | 3 个月内做 2 个能讲的项目 |
| 每日可投入 | 4-6 小时 | 中等强度,需合理分配 |

---

## 🎯 三个月总体策略

```
第 1 月(4.25 - 5.25)   求职冲刺 + CUDA 启动        60% 求职 + 40% 学习
第 2 月(5.26 - 6.25)   实习 + 技术深化             依情况调整
第 3 月(6.26 - 7.25)   面试冲刺 + 项目打磨          80% 面试准备
```

**核心原则:**

1. **求职永远优先于学习** — 一段好实习 > 读 10 篇 paper
2. **CUDA 不追求精通,够用就行** — 能读能改能写简单 kernel 即可
3. **项目要能讲深** — 宁可做 2 个深的,也不做 5 个浅的
4. **算法手感 ICPC 底子够用** — 不需要疯狂刷题,每天少量保持即可
5. **每周留 1 天完全休息** — 防崩溃

---

## 📅 第 1 月:求职冲刺 + 技术启动

### 第 1 月核心 KPI

- [ ] 投递 **20+ 家**公司
- [ ] 获得至少 **3 个面试机会**
- [ ] 理想情况:**拿到 1 个实习 offer**
- [ ] CUDA:写出 matmul 并做 2-3 次优化
- [ ] 读完 4 篇核心 paper:Megatron-LM、ZeRO、FlashAttention、vLLM/PagedAttention
- [ ] GitHub 项目 #1 完成雏形

---

### 📌 第 1 周(4.25 - 5.1):启动周 — 简历 + 投递 + 环境

**本周重点:求职准备工作 + CUDA 入门**

| 日期 | 上午 | 下午 | 晚上 |
|---|---|---|---|
| 周一 4/28 | 简历 v1(Infra 方向) | LinkedIn / 脉脉资料完善 | PMPP 第 1 章 |
| 周二 4/29 | 简历 v2(Diffusion 方向)+ 找校友内推 | 配置 CUDA 开发环境 | PMPP 第 2 章 + Hello CUDA |
| 周三 4/30 | **投出前 5 家公司** | 手写 vector add kernel | FlashAttention paper 第 1 遍 |
| 周四 5/1 | 投 5 家(累计 10) | 手写 naive matmul kernel | Megatron-LM paper 第 1 遍 |
| 周五 5/2 | 跟进已投公司 | matmul 优化:加 tiling | PMPP 第 3-4 章 |
| 周六 5/3 | matmul shared memory 版 + 测性能 | 写博客草稿 | 继续 |
| 周日 5/4 | 🏖️ **休息日** | | |

**本周交付物:**
- ✅ 2 份简历(Infra + Diffusion 方向)
- ✅ 投出 10 家公司
- ✅ CUDA 环境就绪
- ✅ 手写 matmul 三版(naive / tiled / shared mem)
- ✅ 读完 FlashAttention + Megatron-LM 第一遍

---

### 📌 第 2 周(5.5 - 5.11):CUDA 深化 + 分布式训练基础

**本周重点:动手 + 读 ZeRO/FSDP**

| 日期 | 上午 | 下午 | 晚上 |
|---|---|---|---|
| 周一 5/5 | 投 5 家(累计 15) | 实现 reduction kernel | PMPP 第 5-6 章(warp / bank conflict) |
| 周二 5/6 | 面试准备或继续投递 | 实现 softmax kernel | ZeRO paper(重点 1/2/3 区别) |
| 周三 5/7 | 优先处理面试邀约 | PyTorch DDP 训 GPT-2 small | FSDP 官方 tutorial |
| 周四 5/8 | 继续求职 | FSDP 训同样模型 对比性能 | GPipe paper |
| 周五 5/9 | 继续求职 | 开项目 #1:Triton fused attention | Triton 官方 tutorial 1-3 |
| 周六 5/10 | 项目 #1:完成 forward pass | 继续 | |
| 周日 5/11 | 🏖️ **休息日** | | |

**本周交付物:**
- ✅ 投出累计 15 家
- ✅ CUDA:写过 matmul / reduction / softmax
- ✅ 跑通 DDP + FSDP,会对比性能
- ✅ 读完 ZeRO、GPipe
- ✅ Triton 项目雏形

---

### 📌 第 3 周(5.12 - 5.18):推理入门 + 持续投递

**本周重点:vLLM 生态 + 推理加速**

| 日期 | 上午 | 下午 | 晚上 |
|---|---|---|---|
| 周一 5/12 | 投递 + 跟进(累计 20+) | 读 PagedAttention / vLLM paper | 本地部署 vLLM + 跑 Qwen-7B |
| 周二 5/13 | 求职跟进 | 读 vLLM 源码:scheduler.py | FlashAttention-2 paper |
| 周三 5/14 | 求职跟进 | 读 vLLM 源码:block_manager.py | SGLang / RadixAttention |
| 周四 5/15 | 求职跟进 | vLLM 跑 throughput 对比实验 | Speculative Decoding |
| 周五 5/16 | 求职跟进 | 项目 #1 继续 | 整理笔记到 Notion |
| 周六 5/17 | 项目 #1 冲刺 forward 性能 | 继续 | |
| 周日 5/18 | 🏖️ **休息日** | | |

**本周交付物:**
- ✅ 投递累计 20+
- ✅ vLLM 源码关键模块读过
- ✅ 读完 PagedAttention、FlashAttention-2、SGLang、Spec Decoding
- ✅ 项目 #1 forward pass 性能达标

---

### 📌 第 4 周(5.19 - 5.25):项目收尾 + 月度复盘

| 日期 | 上午 | 下午 | 晚上 |
|---|---|---|---|
| 周一 5/19 | 求职 | 项目 #1 性能测试 + 对比 | 写 README + 技术文档 |
| 周二 5/20 | 求职 | 写博客:Triton FlashAttention | 整理投递状态 |
| 周三 5/21 | 求职 | 面试准备或深化项目 | 读 DDPM paper |
| 周四 5/22 | 求职 | 扩投日本 AI 公司 | 读 Latent Diffusion |
| 周五 5/23 | 月末复盘 | 更新简历加项目 #1 | 规划第 2 月 |
| 周六 5/24 | 🛌 学校事项 / 论文 | | |
| 周日 5/25 | 🏖️ **休息日** | | |

**第 1 月末 Checkpoint:**
- [ ] 投递 20+ 家,至少 3 个面试机会
- [ ] 理想:1 个实习 offer
- [ ] CUDA:能写 kernel,理解 memory hierarchy
- [ ] 分布式:能跑 DDP/FSDP,理解 TP/PP/ZeRO
- [ ] 推理:理解 vLLM 调度,读过源码
- [ ] GitHub 项目 #1 完成并开源
- [ ] 1 篇技术博客

⚠️ **如果第 1 月末没拿到任何面试 → 立即调整策略**(降低公司门槛 / 改简历 / 换求职渠道)

---

## 📅 第 2 月:技术深化 + 实习启动

### 第 2 月核心 KPI

- [ ] **开始实习**(理想字节 / 头部模型公司 / 日本 AI;保底 SwanLab 类)
- [ ] 完成 GitHub 项目 #2(更有深度)
- [ ] Megatron-LM 源码精读 TP 部分
- [ ] 补齐 MoE / Long Context / Diffusion 系统化知识
- [ ] 毕业论文推进 30%+

### 场景分支

**场景 A:已拿到实习 offer**
- 实习 = 最好的学习,每天在工作中自然吸收
- 业余 2-3 小时继续深化知识点 + 做项目 #2
- 本月末实习应该产出可写入简历的具体成果

**场景 B:尚未拿到实习**
- 继续投递 + 把能接的保底 offer 接上(如 SwanLab 远程)
- 加大自学强度,用项目弥补实习空缺
- 第 2 月末仍无 offer → 考虑日本本地公司 / 开源项目贡献

---

### 📌 第 5 周(5.26 - 6.1):分布式训练实操深化

| 日期 | 主要任务 |
|---|---|
| 周一 5/26 | 读 Megatron-LM 源码(`model_parallel_utils`、`ColumnParallelLinear`) |
| 周二 5/27 | 读 `RowParallelLinear` 实现 + 理解 forward/backward 通信 |
| 周三 5/28 | 用 FSDP 在云 GPU 上训 1B 模型(Llama 架构) |
| 周四 5/29 | Pipeline Parallel 深入:读 1F1B 和 PipeDream |
| 周五 5/30 | 读 Flash Attention v3(Hopper 架构优化) |
| 周六 5/31 | 开始项目 #2:实现支持 TP 的简化训练框架(300M 模型级别) |
| 周日 6/1 | 🏖️ **休息日** |

---

### 📌 第 6 周(6.2 - 6.8):MoE + 长文本训练

| 日期 | 主要任务 |
|---|---|
| 周一 6/2 | 读 Switch Transformer + GShard |
| 周二 6/3 | 读 DeepSpeed-MoE + Expert Parallel 原理 |
| 周三 6/4 | 读 Ring Attention(长上下文训练) |
| 周四 6/5 | 读 Context Parallel(Megatron 实现) |
| 周五 6/6 | 项目 #2 继续:加入 TP 通信逻辑 |
| 周六 6/7 | 项目 #2:训起一个小模型,验证正确性 |
| 周日 6/8 | 🏖️ **休息日** |

---

### 📌 第 7 周(6.9 - 6.15):推理深入 + 量化

| 日期 | 主要任务 |
|---|---|
| 周一 6/9 | 读 GPTQ paper + 跑 GPTQ 量化 Llama |
| 周二 6/10 | 读 AWQ + SmoothQuant |
| 周三 6/11 | 读 vLLM V1 架构 / 新调度器设计 |
| 周四 6/12 | 学 TensorRT-LLM 基础,跑一个推理示例 |
| 周五 6/13 | FP8 / INT4 量化对比实验 |
| 周六 6/14 | 项目 #2 继续:加推理部分(可选) |
| 周日 6/15 | 🏖️ **休息日** |

---

### 📌 第 8 周(6.16 - 6.22):Diffusion 集中攻克

| 日期 | 主要任务 |
|---|---|
| 周一 6/16 | 复习 DDPM 数学推导 + 实现简单 DDPM |
| 周二 6/17 | 读 DDIM / DPM-Solver(采样加速) |
| 周三 6/18 | 读 DiT(Transformer-based Diffusion) |
| 周四 6/19 | 读 Classifier-Free Guidance 理论 + 工程实现 |
| 周五 6/20 | Stable Diffusion 推理加速实验(xFormers / TensorRT) |
| 周六 6/21 | 读视频 Diffusion(Sora / HunyuanVideo 论文) |
| 周日 6/22 | 🏖️ **休息日** |

---

### 📌 第 9 周(6.23 - 6.25):第 2 月收尾 + 月度复盘

- 项目 #2 收尾 + 开源 + 写博客
- 第 2 月技术笔记整理,形成面试"知识图谱"
- 更新简历 v3(加入项目 #2 + 实习经历)
- 为第 3 月面试冲刺做准备

**第 2 月末 Checkpoint:**
- [ ] 已在实习(或明确即将实习)
- [ ] 2 个 GitHub 项目,至少 1 个有 star
- [ ] 核心知识点能 30 分钟讲清楚(TP/PP/ZeRO/FlashAttn/PagedAttn/量化/Diffusion)
- [ ] Megatron-LM 源码读过关键模块
- [ ] 简历 v3 就绪

---

## 📅 第 3 月:面试冲刺 + 秋招开战

### 第 3 月核心 KPI

- [ ] 算法手感恢复(LeetCode Hot 100 过一遍)
- [ ] 系统设计能应对 3 类题目(训练平台 / 推理服务 / MLOps 系统)
- [ ] 项目能 3 种深度讲解(3 分钟版 / 10 分钟版 / 30 分钟追问版)
- [ ] 字节秋招开闸立即投递
- [ ] 收到字节面试 + 至少通过 1 轮

---

### 📌 第 10 周(6.26 - 7.2):算法恢复 + 面经研究

**算法(每天 1.5-2 小时):**

- LeetCode Hot 100 过一遍,ICPC 底子下只需 5-7 天
- 每天 4-6 题,重点:DP / 图 / 二分 / Trie / 单调栈
- 字节高频题集中刷(牛客 / 代码随想录 可找到整理)

**面经研究(每天 1 小时):**

- 牛客 + 一亩三分地 + 知乎搜"字节 AML 面经"、"字节 Seed 面经"
- 按主题整理高频问题到 Notion(≥ 50 道)
- 分类:CUDA / 分布式 / 推理 / PyTorch / 算法 / 系统设计

**技术复习(每天 1 小时):**

- 按主题快速过自己的笔记
- 每个主题能 10 分钟讲完核心概念

---

### 📌 第 11 周(7.3 - 7.9):系统设计 + 项目打磨

**系统设计三大题:**

1. **设计一个分布式训练平台**
   - 资源调度(K8s / Slurm)、checkpoint 管理、fault tolerance
   - 日志收集、指标监控(这里能结合 SwanLab 类工具讨论)

2. **设计一个 LLM 推理服务**
   - 多租户、动态 batching、KV cache、自动扩缩容
   - SLA 保障(P99 延迟)、GPU 利用率优化

3. **设计一个实验跟踪系统**(相对简单但可能会问)
   - 高并发指标写入、时序数据库、多维度查询

**项目打磨:**

- 每个项目写出 3 种讲解版本(3 分钟 / 10 分钟 / 深度追问)
- 准备至少 10 个可能的追问(为什么选 X、改进方向、局限性)
- 练习用英文讲一遍(日本公司 / 外企备用)

---

### 📌 第 12 周(7.10 - 7.16):模拟面试 + 短板补强

**模拟面试:**

- 找 ICPC 队友或同方向朋友互相面(每周 2-3 场)
- 每场后立即复盘,记录卡点

**短板补强:**

- 根据模拟反馈精准补弱点
- 常见卡点:系统设计不知道怎么起手、Diffusion 细节说不清、CUDA 性能分析工具不熟

**字节投递准备:**

- 所有字节组的 JD 研究过一遍
- 内推联系人最终确认
- 投递文案准备

---

### 📌 第 13 周(7.17 - 7.25):秋招开战

**字节秋招通常 7 月下旬开投,此时状态必须拉满:**

- 简历最终版锁定(中英文各一份)
- 投递字节多个 BU(AML / Seed / Doubao / Flow / 即梦 等)
- 同步投递:阿里通义、腾讯混元、Moonshot、DeepSeek、MiniMax、智谱
- 日本公司:PFN / Sakana AI / Rakuten 做保底
- **每天 1 道算法题保持手感**
- **每天复习一个核心知识点**
- **调整作息** — 秋招是马拉松,别一开始就崩

---

## 🎯 目标公司清单

### 核心目标(必投)

**字节跳动多个 BU:**
- AML(Applied Machine Learning)— 训练引擎、大规模 Infra
- Seed — 大模型研究 + Infra
- Doubao — 豆包大模型训练/推理
- Flow — AI 产品
- 即梦 — 图像/视频生成(Diffusion 方向)
- ByteDance Japan — 东京办公室(地理优势)

### 一线国内大模型公司

- Moonshot AI(月之暗面)
- DeepSeek
- 智谱 AI
- MiniMax
- 阶跃星辰
- 百川智能
- 零一万物

### AI Infra 专精独角兽

- 硅基流动(推理服务)
- 潞晨科技(ColossalAI)
- 无问芯穹
- 清程极智

### Diffusion 方向

- 生数科技(Vidu)
- Stability AI Japan
- 爱诗科技(PixVerse)
- HeyGen

### 日本本地(地理优势 + 签证便利)

- **Preferred Networks**(东京,日本最强 AI Infra)
- **Sakana AI**(东京,David Ha + Llion Jones)
- **Rakuten / LY Corp / CyberAgent / DeNA** 的 AI 组
- **NVIDIA Japan / Google Japan / Meta Japan**(英语 OK 的话)
- ELYZA / Stability AI Japan

### 互联网大厂 Infra 组

- 阿里通义 / PAI
- 腾讯混元
- 百度 PaddlePaddle
- 美团大模型组
- 小红书大模型
- 快手 Kuaishou AI Lab

---

## 📚 必读 Paper 清单

### Tier 1(必读,面试高频)

- [ ] **Megatron-LM**(2019)— Tensor Parallel
- [ ] **ZeRO**(2020)— 显存优化
- [ ] **FlashAttention v1**(2022)— 注意力优化
- [ ] **FlashAttention v2**(2023)— 改进版
- [ ] **PagedAttention / vLLM**(2023)— 推理革命
- [ ] **GPipe**(2019)— Pipeline Parallel

### Tier 2(次高频)

- [ ] **Megatron-LM 3D Parallelism**(2021)
- [ ] **FSDP**(PyTorch 文档)
- [ ] **SGLang / RadixAttention**(2024)
- [ ] **Speculative Decoding**(2023)
- [ ] **GPTQ**(2022)— 量化
- [ ] **Switch Transformer**(2021)— MoE

### Tier 3(加分项)

- [ ] **Ring Attention**(长上下文)
- [ ] **FlashAttention v3**(Hopper)
- [ ] **AWQ / SmoothQuant**(量化)
- [ ] **DDPM / DDIM / LDM / DiT**(Diffusion 四件套)
- [ ] **DPM-Solver**(Diffusion 采样加速)

---

## 🛠️ 项目规划

### 项目 #1(第 1 月):Triton 实现 FlashAttention(简化版)

**目标:** 证明你能写 GPU kernel,理解 attention 优化
**技术栈:** Python + Triton + PyTorch
**交付:**
- GitHub 开源
- README 清晰(原理 + 实现 + benchmark)
- 性能:forward 达到 PyTorch SDPA 同量级
- 技术博客 1 篇

### 项目 #2(第 2 月):简化版分布式训练框架(支持 TP)

**目标:** 证明你理解分布式训练,能实现而非只调用
**技术栈:** PyTorch + NCCL
**交付:**
- GitHub 开源
- 支持 300M 模型的 Tensor Parallel 训练
- 和 Megatron 在同等配置下对比正确性
- README + 技术博客

### (可选)项目 #3:Diffusion 推理加速

**目标:** Diffusion 方向加分
**技术栈:** Diffusers + TensorRT / torch.compile
**交付:**
- Stable Diffusion XL 推理加速(测 speedup)
- 技术博客对比不同加速方案

---

## ⏰ 每日时间模板

### 平日(有课/论文日,每天 4 小时)

```
08:00 - 09:00  算法 1-2 道(保持手感)
09:00 - 12:00  学校事项(课 / 论文 / 导师任务)
13:00 - 14:30  核心学习(读 paper 或写代码)⭐
14:30 - 15:30  项目动手时间
15:30 - 16:30  求职(投简历 / 回邮件 / 内推沟通)
晚上          自由 / 休息 / 看视频
```

### 深度学习日(周末或空档,每天 6 小时)

```
09:00 - 12:00  项目深度开发 ⭐
13:30 - 15:00  读 paper + 做笔记
15:00 - 16:30  系统设计 / 源码阅读
16:30 - 17:30  求职跟进
晚上          博客写作 / 复习 / 休息
```

---

## 📊 每周复盘模板

每周日晚上填写(留 20 分钟):

```markdown
## 第 X 周复盘(日期)

### 学习
- 读完的 paper:
- 写完的代码/项目:
- 学到的最重要的一件事:

### 求职
- 本周投递:X 家(累计 X 家)
- 新增面试邀约:X 个
- 面试进行:X 个
- 拿到 offer:X 个

### 问题
- 本周最大的卡点:
- 下周需要解决的:

### 下周计划调整
- 保持:
- 改进:
```

---

## ⚠️ 风险预案

| 风险 | 触发条件 | 应对策略 |
|---|---|---|
| 5 月底仍无面试 | 第 4 周末 | 降低公司门槛 / 大改简历 / 换求职渠道 / 找大厂校友模拟面试 |
| 6 月底仍无实习 | 第 9 周末 | 接 SwanLab 类保底 + 扩投日本本地公司 |
| 毕业论文拖延严重 | 任何时候 | **每周固定 1-1.5 天处理论文**,保护毕业优先 |
| 身心疲惫 | 任何时候 | 立即休息 2-3 天,**毕业就业是长期战不是短跑** |
| 错过字节投递窗口 | 7 月后 | 走社招通道 / 走内推直推 / 锁定后续补录 |

---

## 🎓 心法总结

1. **时间最贵** — 不要 2 周才投完简历,一周内投完 15 家
2. **闭环最重要** — 读 paper → 写代码 → 写博客 → 能讲出来 = 一个完整闭环
3. **简历有两个关键** — 项目的深度 + 能量化的结果(N% 加速、N 倍吞吐)
4. **面试不是技能比拼,是沟通能力** — 同样的知识,能讲清楚的人拿 offer
5. **ICPC 铜牌是你的王牌** — 面试开场介绍必提,是大厂 Infra 招聘的硬通货
6. **早稻田 + 日本经历是差异化** — 投日本公司 / 字节海外 / NVIDIA Japan 都有独特优势
7. **不要完美主义** — 70 分的计划坚持执行 > 100 分的计划半途而废

---

**Good luck! 秋招见字节 offer 🚀**