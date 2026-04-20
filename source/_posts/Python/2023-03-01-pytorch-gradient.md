---
layout:     post
title:      Pytorch学习笔记#1：拟合函数/梯度下降
categories: Python
subtitle:
date:       2023-03-01
author:     BY 水蓝
header-img: img/sv1.jpg
catalog: true
mathjax: true
tags:
    - Python
    - PyTorch
    - 深度学习
---

# Pytorch学习笔记#1：拟合函数/梯度下降

学习自https://pytorch.org/tutorials/beginner/pytorch\_with\_examples.html

### 概念

Pytorch Tensor在概念上和Numpy的array一样是一个nnn维向量的。不过Tensor可以在GPU中进行计算，且可以跟踪计算图（computational graph）和**梯度**（gradients）。

### 手动梯度下降拟合函数

我们用三次函数去拟合任意函数。  
$y^=a+bx+cx2+dx3\hat{y}=a+bx+cx^2+dx^3y^=a+bx+cx2+dx3$  
那么梯度为:  
$L:2∗∑(y−y^)L:2\*\sum(y-\hat{y})L:2∗∑(y−y^)$  
$d:2∗x3∗∑(y−y^)d:2\*x^3\*\sum(y-\hat{y})d:2∗x3∗∑(y−y^)$

代码

```python
import torch
import math

dtype = torch.float
device = torch.device("cuda:0") # Run on GPU

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device,dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x **2 + d *x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)
    
    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
```python

### 自动梯度下降拟合函数

通过PyTorch: nn构建神经网络，如果我们需要一个三次函数来拟合，那么我们就需要一个隐藏层为1层，节点为3个的神经网络。  
$即y^=∑i=13(wixi+bi)\hat{y}=\sum\_{i=1}^3(w\_ix^i+b\_i)y^=∑i=13(wixi+bi)$

```python
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1), #三个节点
    torch.nn.Flatten(0, 1) # 把三个节点的结果加起来
)
```

由于我们的神经网络第一层有三个输入（x,x2,x3x,x^2,x^3x,x2,x3）,所以我们需要把数据预处理一下

```python
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)
```python

然后我们预测输出就可以直接调用model了

```python
y_pred = model(xx) # y_pred也是一个tensor
```

**损失函数**

```python
loss_fn = torch.nn.MSELoss(reduction='sum') # 定义,使用均方误差
loss = loss_fn(y_pred, y) # 计算均方误差
model.zero_grad() # 先把原先模型的梯度信息清零
loss.backward() # 计算反向传播的梯度
```python

完整代码

```python
import torch
import math

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)


model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):
    y_pred = model(xx)
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    model.zero_grad()

    loss.backward()

    with torch.no_grad(): # 进行梯度下降
        for param in model.parameters():
            param -= learning_rate * param.grad

linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
```