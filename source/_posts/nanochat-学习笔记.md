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

nanochat 是一个用（几乎，有一点 Rust 用来训练分词器）纯 Python 实现的全栈 GPT库，包括 pretrain、midtrain、finetune、inference 等。它的目标是最简化帮助理解大型语言模型（LLM）的工作原理。

计划从以下几个方面来学习这个项目：

1. **代码结构**：了解项目的整体架构和各个模块的功能。
2. **模型实现**：理解 Transformer 模型的实现细节。
3. **训练流程**：学习模型的训练过程，包括数据预处理、训练循环等。

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

数据是 fineweb_100b 数据集的一个子集，[link](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle)。下载到 `$BASE_DIR/base_data` 目录下。
数据被切片成多个 `.parquet` 文件。

```py
def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts
```

parquet 文件是一个列式存储格式，这里按行组（row group）来读取，避免一次性加载过多数据。

而 DataLoader 负责读取数据并生成训练所需的一个批次的 token 序列：

```py
def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches():
        parquet_paths = list_parquet_files()
        assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        first_pass = True
        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        while True: # iterate infinitely (multi-epoch)
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths): # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                # Start from resume point if resuming on same file, otherwise from DDP rank
                # I know this state resumption is a little bit tricky and a little bit hacky... sigh.
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    base_idx = resume_rg_idx // ddp_world_size # in units of ddp_world_size
                    base_idx += 1 # advance by 1 so that we definitely don't repeat data after resuming
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None # set to None as we only want to do this a single time
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist() # each batch is a parquet group, e.g. 1024 rows
                    # the tokenizer encode might want to go in even smaller batches, e.g. 128 rows
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size # advance to the next row group (in DDP)
                pq_idx += 1 # advance to the next parquet file
            first_pass = False
    batches = document_batches()

    # Now emit batches of tokens.
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) # in PyTorch, long=int64
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # we need this in case we wish to approximately resume training
        yield inputs, targets, state_dict
```

document_batches 函数流式 yields tokenized_batch_size 个文本列表+(parquet文件索引，行组索引)，然后外层循环不断累积 token 直到满足一个训练批次的需求，再切分成 inputs 和 targets 返回，并附带当前的 parquet 文件和行组索引状态，方便后续恢复训练时使用。

### 优化器

优化器代码在 `adamw.py` 和 `muon.py` 中。

#### AdamW 优化器

令参数为 $\theta_t$，学习率为 $\alpha$，一阶矩估计为 $m_t$，二阶矩估计为 $v_t$，偏置修正后的一阶矩为 $\hat{m}_t$，偏置修正后的二阶矩为 $\hat{v}_t$，权重衰减系数为 $\lambda$。
$$
\begin{align*}
\theta_{t+1} &= \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \lambda \theta_t \\
m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t\\
v_{t+1} &= \beta_2 v_t + (1 - \beta_2) g_t^2\\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\end{align*}
$$

这里采取分布式训练的方式实现 AdamW 优化器，关键是利用了 `torch.distributed.reduce_scatter_tensor`与 `torch.distributed.all_gather_into_tensor` 来实现梯度的分布式平均和切分，从而减少每个 GPU 的内存占用。

```python
    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                assert params[base_i].shape[0] % world_size == 0, f"First dim of parameter shape {params[base_i].shape} must be divisible by world size {world_size}"
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)
```
这里先对每个参数的梯度进行 `reduce_scatter` 操作，将梯度平均后切分成多个片段，每个 GPU 只保留自己负责的片段，减少内存占用。

目前有 grad_slices 列表，存储了每个参数在当前 GPU 上的梯度片段和对应的 future 对象。
接下来对每个参数组，读取AdamW 的超参数，再对每个参数进行更新：

状态存储在 self.state[p] 中，包括 step 计数器，一阶矩 exp_avg 和二阶矩 exp_avg_sq。

```python
        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = (exp_avg_sq / bias2).sqrt().add_(eps)
                step_size = lr / bias1
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_reduce_futures).wait()
        # 阻塞直到所有 all_gather 操作完成，确保所有 GPU 上的参数都同步更新完毕。
```

#### Muon 优化器

咕咕咕，可先看 [link](https://kexue.fm/archives/10592)。

### GPT 模型实现

架构流程：
1. 输入的Token进入输入嵌入层（Token Embeddings），输入类型：
```python
inputs: Tensor  # (B, T) long tensor of token indices
outputs: Tensor # (B, T, C) float tensor of logits over vocabulary
```
2. RoPE 位置嵌入层（Positional Embeddings）为每个位置添加位置信息。
```
pos_emb: Tensor # (T, C) float tensor of positional embeddings
```
加法广播到输入嵌入上

3. 多层 Transformer 块（Transformer Blocks）处理嵌入，捕捉上下文关系。

4. 输出层（Output Layer）将 Transformer 的输出映射到词汇表大小的 logits。

5. 从 logits 计算损失（Loss Computation），用于训练。或者按照multinomial分布采样生成下一个 token，用于推理。

我们需要的块包括：
- 输入嵌入层
- 预计算位置嵌入层
- Transformer 块 （多个堆叠+ 残差连接）
    - 层归一化
    - 多头自注意力机制 （推理时缓存 K,V）
    - 前馈神经网络
- 输出层 

表格如下

模块名称 | 输入维度 | 输出维度 | 核心功能
--- | --- | --- | ---
Token Embedding | (B,T) | (B,T,C) | 将离散索引转为连续向量
Pos Embedding | (T,C) | (T,C) | 提供序列位置的绝对/相对信息
Transformer Block | (B,T,C) | (B,T,C) | 包含 LayerNorm -> Self-Attention -> Residual -> LayerNorm -> FFN -> Residual
Output Head | (B,T,C) | (B,T,V) | 投影至词汇表空间

优化方面：

nanochat 使用了 KV Cache 来加速推理过程中的自注意力计算，并且使用了 GQA （Grouped Query Attention）来优化多头注意力的计算效率。

对于 KV Cache 的推理，要记录当前 Token 的位置索引，并索引正确的 RoPE 位置嵌入（按绝对位置）。

对于 GQA 的实现，即保持 Q 的头数不变，但将 K 和 V 的头数减少分组，每组共享 K 和 V。

先来看 `gpt.py` 中注意力模块的实现

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # 记录是第几层 Transformer，用于 KV Cache 索引
        self.layer_idx = layer_idx 
        # 有多少个 Q 头，多少个 KV 头
        self.n_head = config.n_head 
        self.n_kv_head = config.n_kv_head
        # 嵌入维度和每个头的维度
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # 定义线性层用于生成 Q、K、V 和输出投影
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        # 在预训练的时候，Tq == Tk，因为没有缓存
        # 在推理的时候，Tq 可以小于 Tk，因为有缓存，每次只处理一个 token，然后询问前面的缓存 + 当前 token
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
            # 训练时没有缓存，或者推理时当前查询数等于缓存的键值数，都可以直接使用因果注意力。
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
            # 推理时如果只有一个查询，可以直接让这个查询关注缓存中的所有键值对。
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
            # 推理时如果有多个查询，需要先让每个查询关注缓存中的所有键值对，然后在当前查询块内进行因果注意力。（下三角掩码）

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
```

简单的 MLP 和 Transformer Block 堆叠块

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x
```

最后是 GPT 模型的整体实现（去除部分辅助函数）：

```python
class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        # For DDP, we want vocab_size divisible by world_size. Also, there are potential performance benefits, see:
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} to be divisible by {pad_vocab_size_to}")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        # wte 是 word to embedding 的缩写，表示词嵌入层
        # h 是 transformer blocks 
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        # 这里的意思是预计算 RoPE 位置嵌入，长度是序列长度的 10 倍，以防止在推理时超出范围，比如模型上下文长度是 1024，则预计算长度是 10240。理论上 sin, cos 可以滚动计算，但为了简单起见，直接预计算一个较大的长度。
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast token embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95)):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)

        # 对2d矩阵参数使用 Muon 优化器，对1d的嵌入和输出层参数使用 AdamW 优化器。
        
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        # 这里利用 tanh 函数对 logits 进行平滑限制在 [-softcap, softcap]，防止极端值导致训练不稳定。

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
```

以下是 Gemini 的总结

权重初始化：

1. 嵌入层与输出层的极端标准差差异 `wte` (Token Embedding): `std=1.0`: Token 嵌入被初始化为标准正态分布。由于 $wte$ 的权重通常会随后被层归一化（LayerNorm）处理，较大的初始标准差可以为模型提供丰富的初始特征表示。`lm_head`: `std=0.001` 输出层使用了非常小的标准差。目的： 在训练开始时，使模型对所有词汇的预测概率趋于均匀分布。如果初始权重过大，模型会产生强烈的随机偏见，导致初始损失值（Loss）极高，增加收敛难度。

2. 均匀分布初始化与 $\sqrt{3}$ 的数学推导代码中使用了均匀分布 `uniform_(-s, s)` 而非正态分布，并定义了 $s = \sqrt{3} \times \frac{1}{\sqrt{n\_embd}}$。为什么用 Uniform： 注释提到是为了“避免离群值（outliers）”。正态分布理论上可能产生极大或极小的权重，而均匀分布的范围是严格受限的。$\sqrt{3}$ 的来源： 对于均匀分布 $U(-s, s)$，其方差为 $Var = \frac{(s - (-s))^2}{12} = \frac{4s^2}{12} = \frac{s^2}{3}$。为了让均匀分布的方差等于期望的方差 $\sigma^2$（即 $1/n\_embd$），则需要：
$$
\frac{s^2}{3} = \sigma^2 \implies s = \sigma \sqrt{3}
$$
这确保了无论使用哪种分布，权重的统计特性是一致的。

3. 投影层初始化为零 (`c_proj`: zeros)这是一个非常关键的技巧，常见于高性能模型实现（如 GPT-2 或部分版本的 Llama）：残差流的恒等映射： 在训练初始时刻，如果注意力投影 `c_proj` 为零，那么 Transformer 块的输出就等于输入（因为残差连接 $x + 0 = x$）。逻辑： 这种做法类似于“恒等函数”初始化，让模型先学习直通的数据流动，再逐渐通过训练学习如何修改残差流中的特征。这极大地提高了极深网络的训练稳定性。

4. 旋转位置嵌入（RoPE）的预计算操作： 调用内部函数生成 `cos` 和 `sin` 表。逻辑： 如前所述，这部分内容是确定性的数学值（频率分布），不需要学习，因此在初始化阶段一次性生成并缓存，以供推理和训练时快速索引。

5. 混合精度转换 (`bfloat16`)操作： 显式将 `wte.weight` 转为 `bfloat16`。优势： 显存节省： 词表往往很大，将其从 float32 转为 bf16 可以直接节省一半的词表显存。精度特性： `bf16` 具有与 `fp32` 相同的指数位范围，能够有效防止训练中的溢出风险。

RoPE 位置嵌入预计算公式：

设 $d$ 为每个注意力头的维度，$i$ 为通道索引，$t$ 为时间步索引，$\theta$ 为基频（通常取 10000）。则每个位置 $t$ 和通道 $i$ 的旋转频率和角度为：
$$
\text{freq}(t, i) = \frac{1}{\theta^{\frac{2i}{d}}}
$$
$$
\text{angle}(t, i) = t \times \text{freq}(t, i) = \frac{t}{\theta^{\frac{2i}{d}}}
$$

