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

本文档记录解读学习 Andrej Karpathy 开源项目 [nanochat](https://github.com/karpathy/nanochat) 的笔记。

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


### 数据集与分词器

数据集相关代码在 `dataset.py` 和 `dataloader.py` 中，分词器在 `tokenizer.py` 中。

#### 分词器实现

首先，`tokenizer.py` 实现了两个分词器类：`HuggingFaceTokenizer` 和 `RustBPETokenizer`。按照描述，两者应该都是基于 BPE（Byte Pair Encoding）算法的分词器。只是 `HuggingFaceTokenizer` 使用了 Hugging Face 的 `tokenizers` 库，可能为了完整的兼容。而 `RustBPETokenizer` 是基于 `tiktoken` 库并用 RustBPE 训练。

```python
SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I haven't validated that this is actually a good idea, TODO.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

这标注了特殊的 token 列表 `SPECIAL_TOKENS`，以及用于预分词的正则表达式 `SPLIT_PATTERN`。

这里简单分析一下它的正则表达式逻辑：
- `'(?i:[sdmt]|ll|ve|re)`：匹配常见的缩写形式，如 's, 'd, 'm, 't, ll, ve, re（不区分大小写）。
- `[^\r\n\p{L}\p{N}]?+\p{L}+`：匹配以字母开头的单词，前面可以有一个非字母数字字符。
- `\p{N}{1,2}`：匹配1到2位的数字序列，标准的 GPT-4 使用的是1到3位数字。这里做了修改以节省 token 空间。
- ` ?[^\s\p{L}\p{N}]++[\r\n]*`：匹配非空白、非字母数字的字符，前面可以有一个空格，后面可以跟换行符。
- `\s*[\r\n]`：匹配换行符，前面可以有任意数量的空白字符。
- `\s+(?!\S)`：匹配空白字符，后面不跟非空白字符（即匹配行尾的空白）。
- `\s+`：匹配任意数量的空白字符。

先来看 `HuggingFaceTokenizer` 类：

[文档参考](https://huggingface.co/docs/tokenizers/v0.20.3/)

```python
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    # ...
    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 从一个文本迭代器，加上一个词汇表大小，来训练一个 Hugging Face BPE 分词器
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!
            unk_token=None,
            fuse_unk=False,
        ))
"""
当 byte_fallback 设为 True 时，分词器的处理逻辑会发生根本变化：

首选尝试：按照正常的词表（如 BPE 或 Unigram 算法生成的子词）进行分词。

触发降级（Fallback）：如果遇到一个完全不在词表里的字符，它不会直接输出 [UNK]。

字节化处理：分词器会将该未知字符转化为其 UTF-8 编码的字节（Bytes）。

字节 Token 映射：词表中预先保留了 256 个基础 Token，分别对应字节 0 到 255。分词器会将这个未知字符拆解为一系列字节 Token。

unk_token (str, optional) — 指定未知字符的替代 Token，这里 None，表示不使用特殊的未知 Token。

fuse_unk (bool, optional) — 是否将连续的未知 Token 融合为一个单独的 Token.

这里整体的意思是，把 byte-level 作为最后的兜底方案，确保任何输入字符都能被分词器处理，而不是直接归为未知 Token。
"""
        # Normalizer: None
        tokenizer.normalizer = None
"""
不进行任何正规化处理，保持原始文本不变。（比如不转换大小写，不去除标点，重音符号等）
"""
        # Pre-tokenizer: GPT-4 style
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(SPLIT_PATTERN)
        # huggingface 要求 PATTERN 必须用 Regex 包裹
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        """
        behavior="isolated"：表示匹配到的部分会被单独作为一个 token 处理，而不是与周围的文本合并。
        invert=False：表示按照正则表达式的匹配结果进行分割，而不是取反。
        pre_tokenizers.ByteLevel：将文本转换为字节级别的表示，确保每个字符都能被处理。
        add_prefix_space=False：不在文本前添加额外的空格。
        use_regex=False：不使用正则表达式进行字节级别的处理。

        这个预分词器的设计目的是先用 GPT-4 风格的正则表达式进行初步分割，然后再将每个部分转换为字节级别的表示，确保任何字符都能被处理。
        """
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        """
        vocab_size (int) — 词汇表的大小，决定了分词器可以识别的不同子词数量。
        show_progress (bool) — 是否在训练过程中显示进度条。
        min_frequency (int) — 一个子词被纳入词汇表的最低出现频率，这里设为 0，表示不设限制。
        initial_alphabet (List[str]) — 初始字母表，这里使用 ByteLevel 预分词器的字母表，确保所有字节都被包含在内。
        special_tokens (List[str]) — 特殊 Token 列表，如 <|bos|> 等，这些 Token 会被单独处理。
        """
        # 调用Huggingface的训练方法
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)
```

`RustBPETokenizer` 类的实现：

```python
import pickle
import rustbpe
import tiktoken

class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        """
        先训练一个 RustBPE 分词器，减去特殊 token 的数量，确保基础词汇表大小足够大（至少256个，满足所有byte）。
        """
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        # 此处将 rustbpe 输出的数据转换为 {字节序列: 优先级数值} 的字典格式。

        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        # 为特殊 token 分配唯一的 token id，确保它们不会与基础词汇表冲突。
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>") # 实例化并返回 RustBPETokenizer 对象
# ...
```

#### 数据集

