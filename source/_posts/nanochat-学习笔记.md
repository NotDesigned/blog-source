---
title: nanochat 学习笔记
date: 2026-01-10 16:20:15
tags:
- Code
- Nanochat
- Open Source
categories:
- Code Implementation
---

本文档记录我解读学习 Andrej Karpathy 开源项目 [nanochat](https://github.com/karpathy/nanochat) 的笔记。

学习的 git 代码版本是: `f5a0ea4`。

## 项目简介

nanochat 是一个用纯 Python 实现的全栈 GPT库，包括 pretrain、midtrain、finetune、inference 等。它的目标是最简化帮助理解大型语言模型（LLM）的工作原理。

计划从以下几个方面来学习这个项目：

1. **代码结构**：了解项目的整体架构和各个模块的功能。
2. **模型实现**：深入理解 Transformer 模型的实现细节。
3. **训练流程**：学习模型的训练过程，包括数据预处理、训练循环等。

对于里面涉及的算法，会结合相关论文进行学习和理解，但是应该不会去复现论文中的数学推导。

Wiki 参考：
- [nanochat Wiki](https://deepwiki.com/karpathy/nanochat)

## 项目安装

我使用的机器与环境如下：
- windows上的wsl2+Ubuntu 22.04 
- CUDA 13.1
- RAM: 48GB
- GPU: RTX 5080 16GB

```bash
git clone https://github.com/karpathy/nanochat.git
cd nanochat
uv venv
uv sync --extra gpu
source .venv/bin/activate
```

试试跑起来
```bash
python -m nanochat.report reset
python -m nanochat.dataset -n 240
python -m scripts.tok_train --max_chars=2000000000 --vocab_size=65536
python -m scripts.tok_eval
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20

```

## 代码结构

- `nanochat/`：核心代码目录，包含模型实现、数据处理等模块。
    - `adamw.py`：AdamW 优化器实现。
    - `checkpoint_manager.py`：检查点管理器，用于保存和加载模型,优化器状态等。
    - `common.py`：常用工具函数。有打印，logging，检测torch device等。
    - `core_eval.py`：评价模型性能的核心函数。
    - `dataloader.py`：数据加载器实现。
    - `dataset.py`：数据集处理模块。
    - `engine.py`：用于推理对话的引擎，发送和接收token
    - `execution.py`：一个沙盒执行环境
    - `gpt.py`：GPT 模型的核心实现。
    - `loss_eval.py`：损失评估模块。
    - `muon.py`：Muon 优化器实现。
    - `report.py`：报告生成模块。
    - `tokenizer.py`：分词器实现。


## 代码实现

从前到后，按照模型本身的逻辑顺序来学习代码实现，外加训练，评估等辅助功能。


### 数据集，分词器与词嵌入

数据集相关代码在 `dataset.py` 和 `dataloader.py` 中，分词器在 `tokenizer.py` 中。
