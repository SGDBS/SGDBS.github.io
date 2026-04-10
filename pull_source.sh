#!/bin/bash

# 确保在 source 分支
CURRENT=$(git branch --show-current)
if [ "$CURRENT" != "source" ]; then
    echo "==> 当前分支: $CURRENT，切换到 source 分支..."
    git checkout source || { echo "错误: 切换分支失败"; exit 1; }
fi

# 若有未提交改动，先暂存
STASHED=0
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "==> 检测到本地改动，暂存中..."
    git stash push -m "auto-stash before pull"
    STASHED=1
fi

# 拉取远程 source 分支
echo "==> 拉取 origin/source..."
git pull origin source

# 恢复暂存的改动
if [ "$STASHED" -eq 1 ]; then
    echo "==> 恢复暂存的本地改动..."
    git stash pop
fi

echo "==> 完成"
