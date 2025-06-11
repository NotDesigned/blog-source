---
title: 子模函数以及Lovász拓展
date: 2025-06-11 17:04:05
tags:
- Optimization
- Submodular Functions
- Lovász Extension
categories:
- Mathematics
- Computer Science
---

参考文献: 

[Wikipedia - Submodular function](https://en.wikipedia.org/wiki/Submodular_function)

[Lecture 7: Submodular Functions, Lovász Extension and Minimization](https://www.cs.princeton.edu/~hy2/teaching/fall22-cos521/notes/SFM.pdf)

## 子模函数 (Submodular Function)

一个函数 $f: 2^N \to \mathbb{R}$ 被称为子模函数，如果对于任意的 $A \subseteq B \subseteq N$ 和 $x \in N \setminus B$，都有以下不等式成立：

$$
f(A \cup \{x\}) - f(A) \geq f(B \cup \{x\}) - f(B)
$$

一个等价的定义是，如果对于任意的 $A \subseteq B$ 和 $x \in N \setminus B$，都有：

$$
f(A \cup \{x\}) - f(A) \geq f(B \cup \{x\}) - f(B)
$$

第一个不等式可以被解释为“边际收益递减”，即添加一个元素到集合 $A$ 的收益大于或等于添加同样的元素到更大的集合 $B$ 的收益。

几个常见的子模函数示例包括：

- 令图 $G = (V, E)$，定义函数 $f(S) = |E(S, \bar S)|$，则$f$ 是子模的，其中 $E(S, \bar S)$ 表示集合 $S$ 和其补集 $\bar S$ 之间的边集。
- 令图 $G = (U, V, E)$ 为一个二分图，$|U| = n$。 对于所有 $S \subseteq U$，定义 $f(S) = |N(S)|$，其中 $N(S)$ 是 $S$ 的邻居节点集合，则 $f$ 是子模的。$f$也是单调的。
- 令 $x_1,x_2,\cdots, x_n$ 是 $n$ 个离散随机变量。对任意 $A \subseteq [n]$，定义 $f(A) = H(X_A)$，其中 $H(X_A)$ 是 $X_A$ 的熵，则 $f$ 是子模的。

## Lovász拓展 (Lovász Extension)

> In a word, the Lovasz extension is a weighted average of the values at the corners.

对于函数 $f: 2^N \to \mathbb{R}$，其Lovász拓展定义为：

$$
\begin{align}
\hat{f}(x) = \mathbb{E}_{\lambda \sim \text{Uniform}(0,1)}[f(\{i \in N : x_i \geq \lambda\})]
\end{align}
$$

其中 $x = (x_1, x_2, \ldots, x_n)\in \mathbb{R}^n$。

Lovász拓展被称为拓展，因为它保留了离散点集上的函数值。注意到，对于离散点$z\in \{0,1\}^n, \lambda \in [0, 1]$，都有：$\{ i | z_i\geq \lambda\} = \{ i | z_i = 1 \} = S$。所以 $\hat f$与 $f$ 在离散点集上是相同的。

### 等价的构造定义

将向量 $x$ 的分量按降序重新排列为 $x_{\sigma(1)} \geq x_{\sigma(2)} \geq \ldots \geq x_{\sigma(n)}$，然后定义集合序列 $S_i = \{ \sigma(1), \sigma(2), \ldots, \sigma(i) \} = \{i \in N : x_i \geq x_{\sigma(i)}\}$。

则：

$$
\begin{align}
\hat{f}(x) = \sum_{i=1}^n f(S_{i}) \cdot (x_{\sigma(i)} - x_{\sigma(i+1)})
\end{align}
$$

并约定 $x_{\sigma(n+1)} = 0$。

这个定义可以看作是对函数 $f$ 在每个集合 $S_i$ 上的值进行加权平均，其中权重是相邻元素之间的差值。

另一个等价的定义是，令 $B(f)$ 为 $f$ 的拟阵多面体（polymatrioid）

$$
B(f) = \{ w\in \mathbb{R}^n : w(N) = f(N), w(A) \leq f(A) \text{ for all } A \subseteq N \}
$$

则

$$
\begin{align}
\hat{f}(x) = \max_{y \in B(f)} \langle x, y \rangle 
\end{align}
$$

### 等价性证明

$(1)\iff (2)$

$$
\begin{align*}
\hat{f}(x) &= \mathbb{E}_{\lambda \sim \text{Uniform}(0,1)}[f(\{i \in N : x_i \geq \lambda\})] \\
&= \int_0^1 f(\{i \in N : x_i \geq \lambda\}) d\lambda \\
\end{align*}
$$
按 $\{x_{\sigma(i)}\}$ 分段积分：
$$
\begin{align*}
\hat{f}(x) &= \int_{\sigma(n)}^1 f(\{j \in N : x_j \geq \lambda\}) d\lambda + \sum_{i=1}^{n} \int_{x_{\sigma(i+1)}}^{x_{\sigma(i)}} f(\{j \in N : x_j \geq \lambda\}) \, d\lambda \\
&= 0 + \sum_{i=0}^{n} f(\{j \in N : x_j \geq x_{\sigma(i)}\}) \cdot (x_{\sigma(i)} - x_{\sigma(i+1)}) \\
&= \sum_{i=1}^{n} f(\{j \in N : x_j \geq x_{\sigma(i)}\}) \cdot (x_{\sigma(i)} - x_{\sigma(i+1)}) \\
&= \sum_{i=1}^{n} f(S_i) \cdot (x_{\sigma(i)} - x_{\sigma(i+1)})
\end{align*}
$$

$(2) \iff (3)$

**引理：**

对于子模函数 $f$，其拟阵多面体 $B(f)$ 的顶点一一对应于排列 $\pi$
$$
\begin{align*}
w_i^{\pi} = f(\{j \in N : \sigma^{-1}(j) \leq \sigma^{-1}(i)\}) - f(\{j \in N : \sigma^{-1}(j) < \sigma^{-1}(i)\})
\end{align*}
$$

证明：

$w^{\pi}(N)=f(N)$是平凡的

对于任意的 $A \subseteq N$，考虑一个排列 $\tau$，它首先列出 $A$ 中的元素（按某种顺序），然后列出 $N \setminus A$ 中的元素。

对于这个排列 $\tau$，我们有：
$$
\begin{align*}
w^{\tau}(A) &= f(A) - f(\emptyset) \\
&= f(A)
\end{align*}
$$

现在考虑任意的排列 $\pi$，我们可以证明如果某个 $a\notin A$ 出现在 $b\in A$ 的前面，则交换它们的位置不会减少 $w(A)$ 的值。

原来 $b$ 的贡献是 $f(S\cup \{a,b\}) - f(S\cup \{a\})$，
交换后变为 $f(S\cup {b}) - f(S)$。

而子模函数性质告诉我们，$f(S\cup \{a,b\}) - f(S\cup \{a\}) \geq f(S\cup \{b\}) - f(S)$。

因此，$w^{\pi}(A)$ 在所有排列 $\pi$ 中是最大的。

所以$w^{\pi}(A) \leq f(A), \forall \pi$

另外，$w(\{\sigma(1), \sigma(2), \ldots, \sigma(k)\}) = f(\{\sigma(1), \sigma(2), \ldots, \sigma(k)\}) $，$\forall k$。

满足 $n$ 个线性无关约束，故 $w^{\pi}$ 是 $B(f)$ 的顶点。

$\square$

现在我们可以证明 $(2) \iff (3)$。

不难知道最优解 $w*$ 一定是对应排列 $\sigma$ 的顶点。

注意到 $\{j| j \in N : \sigma^{-1}(j) \leq i\} = \{j| x_j \geq x_{\sigma(i)}\}$

我们有：

$$
\begin{align*}
\hat{f}(x) &= \max_{y \in B(f)} \langle x, y \rangle \\
&= \max_{\pi} \langle x, w^{\pi} \rangle \\
&= \sum_{i=1}^{n} x_{\sigma(i)} (f(\{j \in N : \sigma^{-1}(j) \leq i\}) - f(\{j \in N : \sigma^{-1}(j) < i\})) \\
&= \sum_{i=1}^{n} x_{\sigma(i)} (f(\{j \in N : x_j \geq x_{\sigma(i)}\}) - f(\{j \in N : x_j > x_{\sigma(i)}\})) \\
&= \sum_{i=1}^{n} f(\{j \in N : x_j \geq x_{\sigma(i)}\}) (x_{\sigma(i)} - x_{\sigma(i+1)}) \\
&= \sum_{i=1}^{n} f(S_i) (x_{\sigma(i)} - x_{\sigma(i+1)}) \\
\end{align*}
$$

### Lovász拓展在子模函数上的性质

我们有如下定理：

**定理**：对于函数 $f: 2^N \to \mathbb{R}$，$f$ 是子模函数当且仅当其 Lovász 拓展 $\hat{f}$ 是凸函数。

充分性：如果 $f$ 是子模函数，则对于任意的 $x, y \in \mathbb{R}^n$ 和 $\theta \in [0, 1]$，都有：
$$
\hat{f}(\theta x + (1 - \theta) y) \leq \theta \hat{f}(x) + (1 - \theta) \hat{f}(y)
$$
证明：由 Lovász 拓展的定义，我们有：
$$
\begin{align*}
\hat{f}(\theta x + (1 - \theta) y) &= \mathbb{E}_{\lambda \sim \text{Uniform}(0,1)}[f(\{i \in N : (\theta x + (1 - \theta) y)_i \geq \lambda\})] \\
&= \mathbb{E}_{\lambda \sim \text{Uniform}(0,1)}[f(\{i \in N : \theta x_i + (1 - \theta) y_i \geq \lambda\})] \\
&\leq \mathbb{E}_{\lambda \sim \text{Uniform}(0,1)}[\theta f(\{i \in N : x_i \geq \lambda\}) + (1 - \theta) f(\{i \in N : y_i \geq \lambda\})] \\
&= \theta \mathbb{E}_{\lambda \sim \text{Uniform}(0,1)}[f(\{i \in N : x_i \geq \lambda\})] + (1 - \theta) \mathbb{E}_{\lambda \sim \text{Uniform}(0,1)}[f(\{i \in N : y_i \geq \lambda\})] \\
&= \theta \hat{f}(x) + (1 - \theta) \hat{f}(y)
\end{align*}
$$
因此，$\hat{f}$ 是凸函数。

必要性：如果 $\hat{f}$ 是凸函数，则对于任意的 $A \subseteq B$ 和 $x \in N \setminus B$，（这里将$f$视作$\hat f$限制到$\{0,1\}^N$上）

都有：
$$
f(A \cup \{x\}) - f(A) \geq f(B \cup \{x\}) - f(B)
$$
证明：

令 $A' = A \cup \{x\}$ 和 $B' = B \cup \{x\}$

$$
\mathbb{1}_{A'}+\mathbb{1}_{B} = \mathbb{1}_{A} + \mathbb{1}_{B'} 
$$
因此，
$$
\begin{align*}
\hat{f}(\frac 12 \mathbb{1}_{A'} +\frac 12\mathbb{1}_{B}) &\leq \frac 12 \hat{f}(\mathbb{1}_{A'}) + \frac 12 \hat{f}(\mathbb{1}_{B}) \\
\iff \hat{f}(\frac 12 \mathbb{1}_{A} +\frac 12\mathbb{1}_{B'}) &\leq \frac 12 f(A') + \frac 12 f(B) \\
\end{align*}
$$

现在我们用Lovász拓展的定义来计算左侧，设 $z = \frac 12 \mathbb{1}_{A} +\frac 12\mathbb{1}_{B'}$，则：
- 对于 $i \in A$，$z_i = 1$
- 对于 $i \in (B\setminus A)\cup \{x\}$，$z_i = 1/2$
- 其他情况， $z_i = 0$

所以
$$
\begin{align*}
\hat{f}(\frac 12 \mathbb{1}_{A} +\frac 12\mathbb{1}_{B'}) &= \int_0^{\frac 12} f(\{i \in N : z_i \geq \lambda\}) d\lambda + \int_{\frac 12}^1 f(\{i \in N : z_i \geq \lambda\}) d\lambda \\
&= \int_0^{\frac 12} f(B') d\lambda + \int_{\frac 12}^1 f(A) d\lambda \\
&= \frac 12 f(B') + \frac 12 f(A) \\
\end{align*}
$$

因此，我们得到了：
$$
\begin{align*}
\frac 12 f(B') + \frac 12 f(A) &\leq \frac 12 f(A') + \frac 12 f(B)\\
\iff f(A \cup \{x\}) - f(A) &\geq f(B \cup \{x\}) - f(B)
\end{align*}
$$
这就证明了必要性。