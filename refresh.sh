#!/bin/bash

# 当任何命令失败时，立即退出脚本
set -e

# --- 1. 获取用户输入的 Commit Message ---
echo "🚀 开始执行 Hexo 博客一键部署脚本..."
echo

echo "请输入本次提交的描述信息 (直接回车将使用默认值 'docs: Update content'):"
read -r commit_message

# 如果用户输入为空，则使用默认的 commit message
if [ -z "$commit_message" ]; then
    commit_message="docs: Update content"
fi

echo "✅ 使用 Commit Message: '$commit_message'"
echo "----------------------------------------"


# --- 2. 执行部署流程 ---

echo "➡️ 步骤 1/7: 正在从远程仓库拉取最新更改 (git pull)..."
# 如果 git pull 失败 (例如有合并冲突), 'set -e' 会让脚本在这里停止
git pull
echo "✅ Git Pull 成功。"
echo "----------------------------------------"

echo "➡️ 步骤 2/7: 正在清理 Hexo 缓存 (hexo clean)..."
hexo clean
echo "✅ 清理完成。"
echo "----------------------------------------"

echo "➡️ 步骤 3/7: 正在生成静态文件 (hexo generate)..."
hexo generate
echo "✅ 文件生成成功。"
echo "----------------------------------------"

echo "➡️ 步骤 4/7: 正在部署网站到服务器 (hexo deploy)..."
hexo deploy
echo "✅ 部署成功。"
echo "----------------------------------------"

echo "➡️ 步骤 5/7: 正在将所有更改添加到 Git (git add .)..."
git add .
echo "✅ 文件添加完成。"
echo "----------------------------------------"

echo "➡️ 步骤 6/7: 正在提交本地更改 (git commit)..."
git commit -m "$commit_message"
echo "✅ 提交成功。"
echo "----------------------------------------"

echo "➡️ 步骤 7/7: 正在推送到远程源文件仓库 (git push)..."
# 如果你的主分支不是 main，请修改下面的 "main"
git push origin main
echo "✅ 推送成功。"
echo "----------------------------------------"

echo "🎉 全部流程执行完毕，部署圆满完成！"