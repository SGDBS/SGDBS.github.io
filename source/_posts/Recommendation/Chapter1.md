---
title: Chapter1 
date: 2026-03-27 12:48:00
categories: 面试-推荐算法
mathjax: true
tags:
    - AI
    - AI面试知识
---

## 推荐系统面试笔记：用户行为日志字段深度解析

在推荐系统中，数据决定了模型效果的上限。用户行为日志（User Behavior Logs）不仅是训练数据的来源，更是理解用户意图的唯一窗口。

---

### 1. 核心字段总览表

| 字段类别 | 关键字段 | 含义 | 面试深度视角 (Deep Dive) |
| :--- | :--- | :--- | :--- |
| **主体标识** | `User_ID` | 唯一标识用户 | **ID 稀疏性**：高维稀疏特征，通常通过 Embedding 映射到低维空间。 |
| **客体标识** | `Item_ID` | 唯一标识物品 | **冷启动问题**：对于新 Item_ID，需降级到使用 Category/Tag 等属性特征。 |
| **行为类型** | `Action_Type` | 点击、购买、点赞等 | **多目标建模**：不同行为代表不同强度的意图（点击是兴趣，购买是决策）。 |
| **时间维度** | `Timestamp` | 行为发生的精确时间 | **兴趣衰减**：用户 1 分钟前的行为比 1 年前的重要得多。 |
| **环境上下文** | `Context` | 设备、地理位置、网络 | **场景化差异**：Wi-Fi 下用户倾向看长视频，4G 下倾向看图文。 |
| **附加属性** | `Attributes` | 播放时长、评分、金额 | **负反馈挖掘**：播放时长极短的“点击”通常被视为强负反馈（误点）。 |

---

### 2. 知识点深度挖掘

#### A. 行为权重的数学处理 (Action Weighting)
在计算用户偏好得分时，不能简单地加总行为。通常会根据业务逻辑赋予权重。
> **公式示例**：用户的综合兴趣得分 $S$ 可以表示为不同行为的加权和：
$$S_u = \sum_{i \in \text{Actions}} w_i \cdot f(t_i) \cdot \text{Action\_Value}_i$$
其中 $w_i$ 是行为权重（如：购买=5.0, 点击=1.0），$f(t_i)$ 是随时间衰减的函数。

#### B. 时序模式与行为序列 (Behavior Sequence)
现在的 SOTA 模型（如 DIN, BST）不再把用户行为看作独立的点，而是一串**序列**。
* **用途**：通过 `Timestamp` 排序，提取用户最近 50/100 个点击记录。
* **面试加分**：提到 **Target Attention** 机制。即利用当前候选物品（Candidate Item）去与历史行为序列做 Attention，找出历史中真正与当前相关的兴趣点。

#### C. 隐藏的字段：位置偏置 (Position Bias)
日志中通常还包含一个未被提起的关键字段：`Position`（物品在页面上的排名）。
* **知识点**：用户点击第一名的物品，可能仅仅是因为它排在第一，而不是因为最喜欢。
* **工程对策**：在离线训练时将 `Position` 作为特征输入，在线推理时将其设为默认值（如 0），以此消除位置偏差。

---

### 3. 面试官连环炮 (Predictive Q&A)

#### Q1: 如果日志中 `User_ID` 缺失（游客模式），推荐系统如何工作？
* **回答**：
    1.  利用 **Context（上下文）** 特征：如地理位置、热门趋势。
    2.  利用 **Session-based 推荐**：根据该用户在当前会话（Session）内的前几次点击，利用 RNN 或 Transformer 预测下一次行为。
    3.  利用 **Device_ID**：在合规前提下，通过设备指纹进行跨会话追踪。

#### Q2: 为什么“曝光未点击”的数据在日志中比“点击”的数据更多，且更难处理？
* **回答**：
    * **数据倾斜**：正负样本极度不平衡（1:100 甚至更高）。
    * **真假负样本**：曝光未点击不代表不喜欢，可能只是用户没看到。处理时通常会采用 **负采样（Negative Sampling）** 策略，或在 Loss 函数中对正负样本进行加权平衡。

#### Q3: 播放时长 (Duration) 这个字段在短视频推荐里怎么用？
* **回答**：
    * 不能直接用绝对时长，因为视频长度不同。
    * 通常使用 **完播率 (Completion Rate)** 或 **时长分位数**。
    * **深度应用**：将时长作为回归目标（Regression），与点击率（CTR）进行多任务融合（Multi-task Learning），解决“标题党”问题。

---

### 4. 避坑指南：特征穿越 (Feature Leakage)
在处理行为日志时，最严重的工程错误是**使用了未来的数据预测过去**。
* **反思**：在构建特征时，必须确保特征的生成时间戳早于预测行为的时间戳。
* **实践**：严格执行 `Point-in-time Join`（时点关联）。


```py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import math

# 1. 初始化 Spark
spark = SparkSession.builder.appName("TimeDecayFeatures").getOrCreate()

# 2. 模拟原始日志数据 (用户ID, 行为类型, 时间戳)
data = [
    ("user_A", "click", "2023-10-01 10:00:00"),
    ("user_A", "click", "2023-10-05 10:00:00"),
    ("user_A", "click", "2023-10-07 09:00:00"), # 最近的一次
    ("user_B", "click", "2023-09-01 10:00:00"), # 很久以前的一次
    ("user_B", "click", "2023-10-07 08:00:00"),
]

df = spark.createDataFrame(data, ["user_id", "action", "timestamp"]) \
    .withColumn("timestamp", F.to_timestamp("timestamp"))

# 3. 设置超参数
# 假设当前时间是 2023-10-07 10:00:00
current_time = "2023-10-07 10:00:00"
# 衰减因子 lambda: 决定衰减速度。
# 例如：设置半衰期为 3 天，则 lambda = ln(2) / 3 ≈ 0.231
decay_lambda = math.log(2) / 3 

# 4. 计算时间衰减得分
# 公式: Score = exp(-lambda * delta_t)
# 其中 delta_t 是当前时间与行为时间的差值（单位：天）
decay_df = df.withColumn("current_ts", F.to_timestamp(F.lit(current_time))) \
    .withColumn("delta_t", 
        (F.unix_timestamp("current_ts") - F.unix_timestamp("timestamp")) / (3600 * 24) # 换算成天
    ) \
    .withColumn("decay_score", F.exp(F.lit(-decay_lambda) * F.col("delta_t")))

# 5. 按用户聚合，得到最终的活跃度特征
user_activity_features = decay_df.groupBy("user_id").agg(
    F.count("action").alias("total_click_count"),              # 原始总计数
    F.sum("decay_score").alias("weighted_activity_score"),    # 衰减后的加权得分
    F.max("timestamp").alias("last_action_time")               # 最近一次行为时间
)

user_activity_features.show()
```

## 特征工程之用户活跃度与物品热度

在推荐系统面试中，特征工程是决定模型上线效果的关键。简单的统计量（如点击次数）往往不足以打动面试官，他们更看重你对**数据分布、统计噪声以及工程落地**的深度思考。

---

### 一、 用户活跃度特征 (User Activity Features)

用户活跃度旨在刻画用户的“粘性”与“意图强度”。

#### 1. 三维提取框架 (RFM 逻辑)
* **强度 (Intensity)**：总点击、总购买、总收藏。
    * **进阶：** 计算“行为转化比”，如 `购买次数 / 点击次数`，识别高转化精准用户。
* **频度 (Frequency)**：日均/周均行为次数、访问天数比例。
    * **进阶：** 计算 **行为方差**。波动大的用户可能是受营销活动驱动，波动小的用户是忠实老客。
* **近期性 (Recency)**：距今时间、最近 1h/24h 行为数。
    * **进阶：时间衰减 (Exponential Decay)**。
        相比滑动窗口，工业界更倾向于使用指数衰减公式来更新用户分数：
        $$Score_{new} = Score_{old} \cdot e^{-\lambda \cdot \Delta t} + \text{Value}_{current}$$
        *其中 $\lambda$ 是衰减因子，$\Delta t$ 是时间差。这种方式无需存储历史明细，对 Flink 流式计算极其友好。*

#### 2. 会话特征 (Session Features)
* **会话长度**：单次打开 App 的点击序列长度。
* **会话频率**：每天触发多少个 Session。
* **用途**：识别用户是“闲逛型”还是“目的明确型”。

---

### 二、 物品热度特征 (Item Popularity Features)

物品热度决定了系统处理“马太效应”和“冷启动”的能力。

#### 1. 全局与局部热度
* **全局热度**：历史累计点击、购买。反映物品的长期生命力。
* **时效热度**：过去 1 小时、1 天的活跃度。捕捉“突发爆款”。

#### 2. 核心指标：点击率 (CTR) 的平滑处理
直接计算 $Clicks / Impressions$ 在小样本下会失效（例如 1 投 1 中）。
* **面试加分：贝叶斯平滑 (Bayesian Smoothing)**
    给分子分母加上先验参数 $\alpha$ 和 $\beta$：
    $$CTR_{smoothed} = \frac{Clicks + \alpha}{Impressions + \alpha + \beta}$$
    *这能让展示量极低的新物品 CTR 趋向于全局平均值，防止其通过高噪声虚假占榜。*

#### 3. 物品冷启动 (Cold Start)
* **类目平滑**：当新物品无数据时，取其所属叶子类目（Category）或标签（Tag）的平均热度。
* **相似度传递**：利用内容向量（Embedding）寻找相似物品，借鉴老物品的热度。

---

### 三、 深度思考：分布处理与工程实现

#### 1. 长尾分布与数据归一化
用户活跃度和物品热度通常服从**长尾分布（Power Law）**。
* **技巧**：直接输入原始大数值会导致模型梯度爆炸。通常使用 **Log 变换** 处理：
    $$X_{feature} = \ln(X_{original} + 1)$$
    

#### 2. 工程落地：Lambda 架构
* **离线层 (Spark)**：利用 `Window Functions` 或 `GroupBy` 计算 T+1 的长期统计特征（如过去 30 天点击量）。
* **实时层 (Flink)**：维护 `State` 计算实时特征（如过去 5 分钟热度）。
* **存储层**：特征存入 Redis 或 Feature Store，供模型推理服务（Inference）毫秒级查询。

---

### 四、 面试高频追问预测

* **Q：如果热度特征非常强，模型总是推荐热门物品怎么办？**
    * **A**：这叫“马太效应”。可以在训练时对热门样本降采样，或者在特征中引入 `Position Bias`（位置偏置）并进行校准，或者在排序后增加 `Diversity`（多样性）打散算子。
* **Q：如何定义“活跃用户”？**
    * **A**：不能只看点击。通常结合 **活跃天数 (DAU)** 和 **核心动作（如播放完、购买）**。在短视频中，完播率高于 80% 的用户行为权重大于单纯的点击。