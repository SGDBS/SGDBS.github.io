---
layout:     post
title:      Pytorch学习笔记#2: 搭建神经网络训练MNIST手写数字数据集
subtitle:
date:       2023-03-23
author:     BY 水蓝
header-img: img/sv1.jpg
catalog: true
categories: Python
mathjax: true
tags:
    - Python
    - PyTorch
    - 深度学习
---

# Pytorch学习笔记#2: 搭建神经网络训练MNIST手写数字数据集

学习自https://pytorch.org/tutorials/beginner/basics/quickstart\_tutorial.html

### 导入并预处理数据集

pytorch中数据导入和预处理主要用torch.utils.data.DataLoader 和 torch.utils.data.Dataset  
Dataset 存储样本及其相应的标签，DataLoader在数据上生成一个可迭代对象(Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset.)

```
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```

将数据集作为参数传递给 DataLoader。 这在我们的数据集上包装了一个可迭代对象，并支持自动批处理、采样、混洗和多进程数据加载。并且每一个batch大小为64。

```
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```

### 搭建神经网络

MNIST手写数字数据集的图片是28∗2828\*2828∗28的，所以第一层的输入为28∗2828\*2828∗28。  
因为识别结果是0~9这10种，所以最后一层的输出就是10个。

我们需要定义神经网络结构，这部分在\_\_init\_\_(self)部分实现。  
且我们需要forward部分定义网络正向传播的方法。

```
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = NeuralNetwork().to(device)
print(model)
```

### 训练模型

首先，我们需要先定义损失函数和优化器（优化梯度下降算法）

```
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # lr为学习率
```

在一次循环中，神经网络通过forward进行预测（我们写的forward函数），然后再利用预测误差。通过反向传播来进行梯度下降（pytorch帮我们实现）。

```
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

```
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

### 开始训练！

```
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b54f877c691066629441aec9bfd484db.png#pic_center)