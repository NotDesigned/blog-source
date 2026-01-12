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
