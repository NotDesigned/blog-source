#!/bin/bash

# 提示用户输入 commit message
echo "请输入 commit message (直接回车使用默认值 'Update'):"
read -r commit_message

# 如果输入为空，使用默认值
if [ -z "$commit_message" ]; then
    commit_message="Update"
fi

echo "使用 commit message: $commit_message"
echo

# 执行部署流程
echo "正在执行 git pull..."
git pull

echo "正在清理 Hexo..."
hexo clean

echo "正在生成静态文件..."
hexo generate

echo "正在部署..."
hexo deploy

echo "正在添加文件到 git..."
git add .

echo "正在提交更改..."
git commit -m "$commit_message"

echo "正在推送到远程仓库..."
git push -u origin main

echo "部署完成！"