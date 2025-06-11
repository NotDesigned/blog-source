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

对于函数 $f: 2^N \to \mathbb{R}$，其Lovász拓展定义为：

$$
\hat{f}(x) = \mathbb{E}_{\lambda \sim \text{Uniform}(0,1)}[f(\{i \in N : x_i \geq \lambda\})]
$$

其中 $x = (x_1, x_2, \ldots, x_n)\in \mathbb{R}^n$。

Lovász拓展被称为拓展，因为它保留了离散点集上的函数值。注意到，对于离散点$z\in \{0,1\}^n, \lambda \in [0, 1]$，都有：$\{ i | z_i\geq \lambda\} = \{ i | z_i = 1 \} = S$。所以 $\hat f$与 $f$ 在离散点集上是相同的。

