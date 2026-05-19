---
title: 0. KV Cache
categories: 学习笔记- AI Infra
date: 2026-05-14 22:48:00
mathjax: true
tags:
    - AI
    - AI Infra
---

## 设定一个具体例子

为了让矩阵形状清晰，我们用一个小模型：

```
d (hidden dim)     = 8
h (num heads)      = 2
d_h (head dim)     = 4    （= d / h）
B (batch size)     = 1    （为简化，省略 batch 维）
```

Prompt：`"The cat sat"`，假设 tokenize 后是 3 个 token，记为 `x₁, x₂, x₃`。

**目标**：先 prefill 处理 prompt，然后 decode 逐个生成 `x₄, x₅, x₆`。

---

## 阶段 1：Prefill（一次性处理 3 个 token）

### Step 1.1：Embedding

输入 token ids `[t₁, t₂, t₃]` 经过 embedding 层：

```
X = [ x₁ ]    形状 [3, 8]，即 [seq_len=3, d=8]
    [ x₂ ]
    [ x₃ ]
```

每一行是一个 token 的 8 维向量。

### Step 1.2：QKV 投影

用三个权重矩阵 W_Q, W_K, W_V，每个形状 `[d=8, d=8]`：

```
Q = X @ W_Q    →  [3, 8] @ [8, 8]  =  [3, 8]
K = X @ W_K    →  [3, 8] @ [8, 8]  =  [3, 8]
V = X @ W_V    →  [3, 8] @ [8, 8]  =  [3, 8]
```

具体地（每行代表一个 token 的 Q/K/V 向量）：

```
Q = [ q₁ ]   K = [ k₁ ]   V = [ v₁ ]
    [ q₂ ]       [ k₂ ]       [ v₂ ]
    [ q₃ ]       [ k₃ ]       [ v₃ ]
```

### Step 1.3：Reshape 成多头

把最后一维 `d=8` 拆成 `h=2` 个 head，每个 head `d_h=4`，然后转置让 head 维度在前：

```
Q: [3, 8] → [3, 2, 4] → [2, 3, 4]   ([h, seq, d_h])
K: [3, 8] → [3, 2, 4] → [2, 3, 4]
V: [3, 8] → [3, 2, 4] → [2, 3, 4]
```

现在每个 head 独立处理。**为了讲清楚，只看 head 0**（head 1 完全对称）：

```
Q^(0) = [ q₁ ]   形状 [3, 4]
        [ q₂ ]   （每行是一个 token 在 head 0 上的 4 维 query）
        [ q₃ ]

K^(0) = [ k₁ ]   形状 [3, 4]
        [ k₂ ]
        [ k₃ ]

V^(0) = [ v₁ ]   形状 [3, 4]
        [ v₂ ]
        [ v₃ ]
```

### Step 1.4：Attention Score = Q @ K^T

这是关键的一步。**Q @ K^T** 计算每个 query 和每个 key 的相似度：

```
Q^(0) @ (K^(0))^T :  [3, 4] @ [4, 3]  =  [3, 3]
```

形象地画出来：

```
                  K^T (transposed)
                 ┌────────────────┐
                 │ k₁ᵀ  k₂ᵀ  k₃ᵀ │   每列是一个 key
                 └────────────────┘
        Q              ↓ 矩阵乘
    ┌──────┐    ┌──────────────────────┐
    │  q₁  │ →  │ q₁·k₁  q₁·k₂  q₁·k₃ │  ← q₁ 与所有 key 的点积
    │  q₂  │ →  │ q₂·k₁  q₂·k₂  q₂·k₃ │  ← q₂ 与所有 key 的点积
    │  q₃  │ →  │ q₃·k₁  q₃·k₂  q₃·k₃ │  ← q₃ 与所有 key 的点积
    └──────┘    └──────────────────────┘
                       S 形状 [3, 3]
```

`S[i][j] = qᵢ · kⱼ`，表示 token i 关注 token j 的"原始注意力分数"。

### Step 1.5：Causal Mask + Softmax

由于是 decoder，**token i 不能看 token j > i**，所以右上三角要被 mask 成 -∞：

```
S 加 mask 后:
┌─────────────────────┐
│ q₁·k₁   -∞    -∞   │   ← token 1 只能看 token 1
│ q₂·k₁  q₂·k₂  -∞   │   ← token 2 只能看 token 1, 2
│ q₃·k₁  q₃·k₂  q₃·k₃│   ← token 3 可以看 token 1, 2, 3
└─────────────────────┘

除以 √d_h = 2，然后 softmax（按行）：

A = ┌──────────────────────┐
    │ 1.00   0     0      │   token 1 完全 attend 自己
    │ 0.4   0.6    0      │   token 2 在 token 1, 2 间分布
    │ 0.2   0.3   0.5     │   token 3 在 token 1, 2, 3 间分布
    └──────────────────────┘
       形状仍是 [3, 3]
```

每行和为 1。这就是 attention weights。

### Step 1.6：加权求和 A @ V

用注意力权重对 V 加权求和：

```
A @ V^(0):  [3, 3] @ [3, 4]  =  [3, 4]
```

形象画法：

```
            V
       ┌──────┐
       │ v₁   │  ← 4 维
       │ v₂   │
       │ v₃   │
       └──────┘
   A        ↓ 加权求和
┌──────────────┐    ┌──────────────────────┐
│ 1.0  0   0  │ →  │ 1.0·v₁ + 0·v₂ + 0·v₃│ = output₁
│ 0.4 0.6  0  │ →  │ 0.4·v₁ + 0.6·v₂     │ = output₂
│ 0.2 0.3 0.5 │ →  │ 0.2·v₁ + 0.3·v₂+0.5·v₃│=output₃
└──────────────┘    └──────────────────────┘
                       O^(0) 形状 [3, 4]
```

**每一行 outputᵢ 是 token i 的 context-aware 表示**——它是历史所有 token 的 value 向量的加权混合。

### Step 1.7：合并多头 + 输出投影

```
O^(0): [3, 4]     ┐
                  ├→ concat → [3, 8] → @ W_O [8, 8] → [3, 8]
O^(1): [3, 4]     ┘                                      ↑
                                                    最终 attention 输出
```

然后经过 residual、LayerNorm、FFN、再 residual……到下一层。

### Step 1.8：Prefill 的关键产物——KV Cache

**这一层结束时，把 K, V 存进 cache**：

```
KV_Cache[layer_i]:
  K_cache = [ k₁ ]   形状 [h=2, seq=3, d_h=4]
            [ k₂ ]
            [ k₃ ]

  V_cache = [ v₁ ]   形状 [h=2, seq=3, d_h=4]
            [ v₂ ]
            [ v₃ ]
```

每一层都有自己独立的 KV cache。

### Step 1.9：取最后一个 token 预测 x₄

经过所有层后得到 `[3, 8]`，取最后一行 `[1, 8]`，过 LM head `[8, vocab]`，softmax 采样得到 `x₄`。

**Prefill 完成。计算特点：3 个 token 一次性算，是一个标准的矩阵乘 GEMM，计算密集。**

---

## 阶段 2：Decode（逐 token 生成 x₄）

现在我们要算 x₄ 经过 attention 的输出。**关键问题：x₄ 的 query 需要 attend 到 x₁, x₂, x₃, x₄ 的 key/value。** 但 x₁, x₂, x₃ 的 K/V 已经在 cache 里了！

### Step 2.1：只输入 1 个新 token

```
X_new = [ x₄ ]    形状 [1, 8]
```

注意：**只有 1 行**，不是 4 行。这是和 prefill 最大的区别。

### Step 2.2：QKV 投影（只算新 token）

```
q_new = X_new @ W_Q   →  [1, 8] @ [8, 8]  =  [1, 8]
k_new = X_new @ W_K   →  [1, 8] @ [8, 8]  =  [1, 8]
v_new = X_new @ W_V   →  [1, 8] @ [8, 8]  =  [1, 8]
```

**注意 FLOPs 上的省力**：投影从 `[3, 8] @ [8, 8]` 变成 `[1, 8] @ [8, 8]`，省了 3 倍。
更重要的是：**这不再是矩阵-矩阵乘（GEMM），而是矩阵-向量乘（GEMV）**——这是 decode 变成 memory-bound 的根因。

### Step 2.3：Reshape 多头

```
q_new: [1, 8] → [1, 2, 4] → [2, 1, 4]   ([h, 1, d_h])
k_new: [1, 8] → [1, 2, 4] → [2, 1, 4]
v_new: [1, 8] → [1, 2, 4] → [2, 1, 4]
```

只看 head 0：

```
q_new^(0) = [ q₄ ]    形状 [1, 4]
k_new^(0) = [ k₄ ]    形状 [1, 4]
v_new^(0) = [ v₄ ]    形状 [1, 4]
```

### Step 2.4：Cache 更新——把 k₄, v₄ 拼接进去

```
更新前：
K_cache^(0) = [ k₁ ]    形状 [3, 4]
              [ k₂ ]
              [ k₃ ]

更新后（concat 在 seq 维）：
K^(0) = [ k₁ ]    形状 [4, 4]   ← 多了一行 k₄
        [ k₂ ]
        [ k₃ ]
        [ k₄ ]   ← 刚算出的

V^(0) 同理：[4, 4]
```

实际实现中，cache 是预分配的 `[h, max_len, d_h]`，写入是 `K_cache[:, current_pos, :] = k_new`，没有真的 concat。

### Step 2.5：Attention Score——形状的关键变化

```
q_new^(0) @ (K^(0))^T :  [1, 4] @ [4, 4]  =  [1, 4]
```

**注意！这里左操作数只有 1 行**：

```
                   K^T
                 ┌────────────────────┐
                 │ k₁ᵀ  k₂ᵀ  k₃ᵀ  k₄ᵀ│   4 列
                 └────────────────────┘
   q_new              ↓
   ┌──────┐    ┌──────────────────────────┐
   │  q₄  │ →  │ q₄·k₁  q₄·k₂  q₄·k₃  q₄·k₄│
   └──────┘    └──────────────────────────┘
                    s 形状 [1, 4]
```

**只有 1 行 score**，因为我们只关心 q₄ 对所有历史 key 的 attention。

### Step 2.6：不需要 Causal Mask！

为什么？因为 q₄ 本来就是"最新的 token"，它能看到所有历史是合法的，而 cache 里**根本不存在 k₅, k₆**（未来的 token 还没生成）。**Mask 的作用是阻止看未来——但未来此刻还不存在。**

```
直接除以 √d_h，softmax：

a = [ a₁  a₂  a₃  a₄ ]    形状 [1, 4]，和为 1
```

### Step 2.7：加权求和 a @ V

```
a^(0) @ V^(0) :  [1, 4] @ [4, 4]  =  [1, 4]
```

形象画法：

```
              V (包括 cache 和 v₄)
         ┌──────┐
         │ v₁   │
         │ v₂   │
         │ v₃   │
         │ v₄   │
         └──────┘
   a            ↓
┌─────────────────┐    ┌────────────────────────────────┐
│ a₁ a₂ a₃ a₄    │ →  │ a₁·v₁ + a₂·v₂ + a₃·v₃ + a₄·v₄│
└─────────────────┘    └────────────────────────────────┘
                            output₄ 形状 [1, 4]
```

只得到 **1 个输出向量**——就是 x₄ 经过这一层 attention 后的表示。

### Step 2.8：后续 + 下一步

`output^(0), output^(1)` concat 成 `[1, 8]` → W_O → 后续层 → LM head → 采样得到 x₅。

然后回到 Step 2.1，把 x₅ 作为新的 X_new，cache 长度变成 5，继续……

---

## 用一张图对比 Prefill vs Decode 的矩阵形状

```
              ┌───────────────────────┬───────────────────────┐
              │       PREFILL         │        DECODE         │
              │     (3 tokens 一起)    │     (1 token 一步)     │
              ├───────────────────────┼───────────────────────┤
              │                       │                       │
   输入 X     │      [3, 8]           │      [1, 8]           │
              │                       │                       │
   Q          │      [2, 3, 4]        │      [2, 1, 4]        │
              │      ↑全部新算          │      ↑只算新的         │
              │                       │                       │
   K          │      [2, 3, 4]        │      [2, 4, 4]        │
              │      ↑全部新算→存cache  │      ↑3 来自 cache+1新│
              │                       │                       │
   V          │      [2, 3, 4]        │      [2, 4, 4]        │
              │                       │                       │
   Q @ K^T    │  [3,4]@[4,3] = [3,3]  │ [1,4]@[4,4] = [1,4]   │
              │   方阵，需要 causal mask│  长条，不需要 mask     │
              │                       │                       │
   A @ V      │  [3,3]@[3,4] = [3,4]  │ [1,4]@[4,4] = [1,4]   │
              │                       │                       │
   计算类型    │  GEMM（矩阵×矩阵）     │  GEMV（向量×矩阵）     │
   瓶颈       │  Compute-bound        │  Memory-bound         │
              └───────────────────────┴───────────────────────┘
```

---

## 几个最容易混淆的点

**1. 为什么 Q 不需要 cache？**
因为生成时每步只用"最新位置的 query"——历史 token 的 query 算出来后只用于一次性产生那一步的 output，之后不再需要。而 K, V 会被未来所有 step 反复 attend 到，所以必须缓存。

**2. 为什么 Decode 的 Q@K^T 不需要 mask？**
Prefill 时是 `[3, 3]` 的方阵，每行代表不同的 query，要阻止靠前的 query 看到靠后的 key，所以要 mask。Decode 时 `[1, L]` 只有一行 query，是"最新的"，本来就该看所有历史，而未来的 key 还不存在，所以天然不需要 mask。

**3. Cache 长度何时增加？**
每个 decode step 增加 1。生成 100 个 token，cache 从 L₀ 增长到 L₀+100。这就是为什么长生成会越来越慢——`q@K^T` 中 K 的长度在变长。

**4. 一次只算一行 Q——是不是浪费 GPU？**
是的！这正是 decode memory-bound 的根本原因。GPU 擅长 GEMM 但 GEMV 算术强度低。所以才有 **continuous batching**（把多个请求的 q_new 堆成 `[batch, 1, d]`，让矩阵长起来）和 **speculative decoding**（一次猜多个 token 让 q 行数变大）这些优化。

---

## 把整个过程串起来的伪代码

```python
def generate(prompt_ids, max_new=100):
    # ===== Prefill =====
    X = embed(prompt_ids)              # [L₀, d]
    kv_cache = []
    for layer in layers:
        Q = X @ layer.W_Q              # [L₀, d]
        K = X @ layer.W_K              # [L₀, d]
        V = X @ layer.W_V              # [L₀, d]
        # ... reshape to [h, L₀, d_h]
        
        S = Q @ K.transpose() / sqrt(d_h)        # [h, L₀, L₀]
        S = S + causal_mask                       # 必须 mask
        A = softmax(S)                            # [h, L₀, L₀]
        out = A @ V                               # [h, L₀, d_h]
        
        kv_cache.append((K, V))                  # 存 cache
        X = post_attention(out, layer)
    
    next_token = sample(lm_head(X[-1]))
    
    # ===== Decode loop =====
    for step in range(max_new):
        X = embed([next_token])                  # [1, d]
        for i, layer in enumerate(layers):
            q = X @ layer.W_Q                    # [1, d]
            k = X @ layer.W_K                    # [1, d]
            v = X @ layer.W_V                    # [1, d]
            # reshape to [h, 1, d_h]
            
            K_full = concat(kv_cache[i].K, k)   # [h, L_cur, d_h]
            V_full = concat(kv_cache[i].V, v)   # [h, L_cur, d_h]
            kv_cache[i] = (K_full, V_full)      # 更新 cache
            
            s = q @ K_full.transpose() / sqrt(d_h)   # [h, 1, L_cur]
            a = softmax(s)                            # 不需要 mask！
            out = a @ V_full                          # [h, 1, d_h]
            
            X = post_attention(out, layer)
        
        next_token = sample(lm_head(X[-1]))
```