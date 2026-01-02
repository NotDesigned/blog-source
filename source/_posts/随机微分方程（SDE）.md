---
title: 随机微分方程（SDE）
date: 2026-01-01 21:01:50
tags: 
    - 随机微分方程
    - Diffusion
    - Ito Calculus
categories:
    - Mathematics
    - Finance
---

## 参考文献

基本上参考 Bernt Øksendal 的《Stochastic Differential Equations: An Introduction with Applications》。

加上一些个人的理解，和测度论上的一些补充。也许会简单过一些习题。

这里主要是速通，不会太过深入某些定理的证明细节。

前置：基本的概率论，实分析，最好有测度论知识。

## 1. 测度概率论和随机过程基础 (Chapter 2 of Øksendal)

在正式进入随机微分方程之前，先简单回顾一下测度论和概率论的基础知识。

一个代数 $\mathcal{A}$ 是定义在集合 $\Omega$ 上的子集族，满足：

1. $\Omega \in \mathcal{A}$；
2. 如果 $A \in \mathcal{A}$，则 $A^c \in \mathcal{A}$；
3. 如果 $A_1, A_2, \ldots, A_n \in \mathcal{A}$，则 $\bigcup_{i=1}^{n} A_i \in \mathcal{A}$。

一个 $\sigma$-代数 $\mathcal{F}$ 是定义在集合 $\Omega$ 上的子集族，满足：
1. $\Omega \in \mathcal{F}$；
2. 如果 $A \in \mathcal{F}$，则 $A^c \in \mathcal{F}$；
3. 如果 $A_1, A_2, \ldots \in \mathcal{F}$，则 $\bigcup_{i=1}^{\infty} A_i \in \mathcal{F}$。（补集一下自然也对可列交成立）

一个测度是定义在 $\sigma$-代数 $\mathcal{F}$ 上的函数 $\mu: \mathcal{F} \to [0, \infty]$，满足：
1. $\mu(\emptyset) = 0$；
2. 如果 $\{A_i\}_{i=1}^{\infty}$ 是 $\mathcal{F}$ 中的可列不交集族，则 $\mu\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} \mu(A_i)$。

我们可以证明测度满足以下几个有用的性质：
1. 单调性：如果 $A, B \in \mathcal{F}$ 且 $A \subseteq B$，则 $\mu(A) \leq \mu(B)$。
2. 次可加性：如果 $\{A_i\}_{i=1}^{\infty}$ 是 $\mathcal{F}$ 中的任意集合族，则 $\mu\left(\bigcup_{i=1}^{\infty} A_i\right) \leq \sum_{i=1}^{\infty} \mu(A_i)$。
3. 上连续性：如果 $\{A_n\}_{n=1}^{\infty}$ 是 $\mathcal{F}$ 中的递增集合族，即 $A_1 \subseteq A_2 \subseteq \ldots$，则 $\mu\left(\bigcup_{n=1}^{\infty} A_n\right) = \lim_{n \to \infty} \mu(A_n)$。
4. 下连续性：如果 $\{A_n\}_{n=1}^{\infty}$ 是 $\mathcal{F}$ 中的递减集合族，即 $A_1 \supseteq A_2 \supseteq \ldots$，且 $\mu(A_1) < \infty$，则 $\mu\left(\bigcap_{n=1}^{\infty} A_n\right) = \lim_{n \to \infty} \mu(A_n)$。

给定一族子集 $\mathcal{U} \subseteq \mathcal{P}(\Omega)$，存在唯一的最小 $\sigma$-代数 $\mathcal{H}_\mathcal{U}$，使得 $\mathcal{U} \subseteq \mathcal{H}_\mathcal{U}$，称为由 $\mathcal{U}$ 生成的 $\sigma$-代数。由此，我们可以定义 Borel $\sigma$-代数 $\mathcal{B}$ 为 拓扑空间 $\Omega$ 的所有开集生成的 $\sigma$-代数。

一个概率空间是一个三元组 $(\Omega, \mathcal{F}, P)$，其中 $\Omega$ 是样本空间，$\mathcal{F}$ 是定义在 $\Omega$ 上的 $\sigma$-代数，$P$ 是定义在 $\mathcal{F}$ 上的概率测度，满足 $P(\Omega) = 1$。如果 $\mathcal{F}$ 包含所有$P$-外测度零的子集 $G \subseteq \Omega$，即 $\inf{ P(A) : A \in \mathcal{F}, G \subseteq A } = 0$ ，则称概率空间 $(\Omega, \mathcal{F}, P)$ 是完备的。

给定一个概率空间 $(\Omega, \mathcal{F}, P)$，一个函数是$\mathcal{F}$-可测的，如果对于所有的 Borel 集合 $B \in \mathcal{B}(\mathbb{R})$，有 $X^{-1}(B) \in \mathcal{F}$。随机变量是定义在概率空间 $(\Omega, \mathcal{F}, P)$ 上的 $\mathcal{F}$-可测函数，一般取值在 $\mathbb{R}^n$ 上。

反过来，给定一个函数 $X: \Omega \to \mathbb{R}^n$，定义由 $\Omega$ 生成的 $\sigma$-代数为 $\sigma(X) = \{X^{-1}(B) : B \in \mathcal{B}\}$。 其中 $\mathcal{B}$ 是 $\mathbb{R}^n$ 上的 Borel $\sigma$-代数 （或写作 $\mathcal{H}_X$）。同时还引导一个 $\mathbb{R}^n$ 上的测度 $\mu_X$，定义为 $\mu_X(B) = P(X^{-1}(B))$，称为 $X$ 的分布（或诱导测度），又写作 $\mathrm{Law}(X)$。我们说随机变量 $X$ 服从分布 $\mu_X$，记作 $X \sim \mu_X$。

### Doob-Dynkin 引理

设 $X: \Omega \to \mathbb{R}^n$ 和 $Y: \Omega \to \mathbb{R}^n$ 是两个随机变量，则存在一个 Borel 可测函数 $f: \mathbb{R}^n \to \mathbb{R}^n$，使得 $Y = f(X)$ 当且仅当 $\sigma(Y) \subseteq \sigma(X)$（$Y$ 是 $\mathcal{\mathcal{H}}_X$-可测的）。

证明：

($\Rightarrow$) 如果存在 Borel 可测函数 $f$ 使得 $Y = f(X)$，则对于任意的 Borel 集合 $B \in \mathcal{B}$，$Y^{-1}(B) = X^{-1}(f^{-1}(B)).$ 由于 $f$ 是 Borel 可测的，$f^{-1}(B)$ 也是 Borel 集合，因此 $Y^{-1}(B) \in \sigma(X)$，即 $\sigma(Y) \subseteq \sigma(X)$。

($\Leftarrow$) 如果 $ Y = \mathbf{1}_A $ 是集合 $A \in \sigma(X)$ 的指示函数，则存在 Borel 集合 $B \in \mathcal{B}$，使得 $A = X^{-1}(B)$。定义函数 $f: \mathbb{R}^n \to \mathbb{R}$ 为 $f(x) = \mathbf{1}_B(x)$，则 $Y(\omega) = f(X(\omega))$ 对所有 $\omega \in \Omega$ 成立。然后可以将该结论推广到简单函数 $Y = \sum a_i \mathbf{1}_{A_i}$，其中 $A_i \in \sigma(X)$，再通过测度论里的套路构造逐点极限收敛序列$Y_k$，推广到非负的随机变量 $Y$，最后正负分解推广到任意随机变量 $Y$。后面此过程简要带过了。

这个引理的意义在于，它告诉 $Y$ 是否可以通过 $X$ 来表示，取决于 $Y$ 的信息（引导的测度）是否包含在 $X$ 的信息中。

### 期望

设 $(\Omega, \mathcal{F}, P)$ 是一个概率空间，$X: \Omega \to \mathbb{R}^n$ 是定义在该空间上的随机变量。$X$ 的期望值（或数学期望）定义为
$$
E[X] = \int_{\Omega} X(\omega) dP(\omega),
$$
前提是 $\int_{\Omega} \|X(\omega)\| dP(\omega) < \infty$，即绝对可积。

一般地，如果 $f: \mathbb{R}^n \to \mathbb{R}$ 是一个 Borel 可测函数，且 $E[|f(X)|] < \infty$，则 $f(X)$ 的期望值定义为
$$
E[f(X)] = \int_{\Omega} f(X(\omega)) dP(\omega) = \int_{\mathbb{R}^n} f(x) d\mu_X(x),
$$

特别地，随机变量 $X$ 的 $n$ 阶矩定义为
$$
E[X^n] = \int_{\Omega} X(\omega)^n dP(\omega),
$$

### $L^p$ 空间

设 $(\Omega, \mathcal{F}, P)$ 是一个概率空间，$p\in [1,\infty)$。我们定义随机变量 $X:\Omega \to \mathbb{R}^n$ 的 $L^p$ 范数为 
$$
\|X\|_p = \|X\|_{L^{p}(P)}= \left( E[\|X\|^p] \right)^{1/p} = \left( \int_{\Omega} \|X(\omega)\|^p dP(\omega) \right)^{1/p},
$$
如果 $p=\infty$，则定义为
$$
\|X\|_\infty = \|X\|_{L^{\infty}(P)}= \inf \{ N \geq 0 :  \|X(\omega)\| \leq N \text{ a.s.} \}.
$$
我们定义 $L^p$ 空间为
$$
L^p(P) = L^p(\Omega) = \{ X: \Omega \to \mathbb{R}^n ; X \text{ 是 } \mathcal{F}\text{-可测的且 } \|X\|_p < \infty \}.
$$

$L^p$ 空间配备范数 $\|\cdot\|_p$ 后是一个 Banach 空间。当 $p=2$ 时，$L^2$ 空间是一个 Hilbert 空间，内积定义为
$$
\langle X, Y \rangle = E[X \cdot Y] = \int_{\Omega} X(\omega) \cdot Y(\omega) dP(\omega).
$$

### 独立与条件期望

两个集合 $A, B \in \mathcal{F}$ 称为独立的，如果 $P(A \cap B) = P(A)P(B)$。一般的，一个集合族 $\{A_i\}_{i \in I} \subseteq \mathcal{F}$ 称为相互独立的，如果对于任意有限子集 $J \subseteq I$，有
$$
P\left( \bigcap_{j \in J} A_j \right) = \prod_{j \in J} P(A_j).
$$
随机变量 $X: \Omega \to \mathbb{R}^n$ 和 $Y: \Omega \to \mathbb{R}^m$ 称为独立的，如果它们引导的 $\sigma$-代数 $\sigma(X)$ 和 $\sigma(Y)$ 独立。我们也同时考虑一个随机变量 $X$ 和一个 $\sigma$-代数 $\mathcal{G} \subseteq \mathcal{F}$ 的独立性，定义为 $\sigma(X)$ 和 $\mathcal{G}$ 独立。

容易验证，若 $X$ 和 $Y$ 独立，则 $E[XY] = E[X]E[Y]$。证明是考虑简单函数（随机变量）。

设 $(\Omega, \mathcal{F}, P)$ 是一个概率空间，$\mathcal{G} \subseteq \mathcal{F}$ 是 $\sigma$-代数，$X: \Omega \to \mathbb{R}^n$ 是一个随机变量，且 $E[\|X\|] < \infty$。则条件期望 $E[X | \mathcal{G}]$ 定义为满足以下性质的 $\mathcal{G}$-可测随机变量：
1. $E[X | \mathcal{G}]$ 是 $\mathcal{G}$-可测的；
2. 对于所有 $A \in \mathcal{G}$，有 $E[X \mathbf{1}_A] = E[E[X | \mathcal{G}] \mathbf{1}_A]$。换言之，
$$
\int_{A} X(\omega) dP(\omega) = \int_{A} E[X | \mathcal{G}](\omega) dP(\omega).
$$
条件期望的存在性和唯一性（几乎处处相等意义下）可以通过Radon-Nikodym 定理来证明。

条件期望的一些重要性质包括：
1. 线性：对于任意的随机变量 $X, Y$ 和标量 $a, b \in \mathbb{R}$，有
$$
E[aX + bY | \mathcal{G}] = aE[X | \mathcal{G}] + bE[Y | \mathcal{G}].
$$
2. 全期望公式：$E[E[X | \mathcal{G}]] = E[X]$。
3. 如果 $X$ 是 $\mathcal{G}$-可测的，则 $E[X | \mathcal{G}] = X$ 几乎处处成立。
4. 如果 $X$ 独立于 $\mathcal{G}$，则 $E[X | \mathcal{G}] = E[X]$ 几乎处处成立。

### 随机过程

一个随机过程是定义在概率空间 $(\Omega, \mathcal{F}, P)$ 上的随机变量族 $\{X_t : t \in T\}$，其中 $T$ 是一个索引集，通常取为时间参数（如 $T = [0, \infty)$ 或 $T = \mathbb{N}$）。对于每个固定的 $\omega \in \Omega$，函数 $t \mapsto X_t(\omega)$ 称为该随机过程的一个样本路径。一个随机过程也可以被视为一个映射 $X: \Omega \times T \to \mathbb{R}^n$，满足对于每个固定的 $t \in T$，$X_t(\cdot)$ 是一个随机变量（$\mathcal{F}$-可测的）。这样考虑是因为我们经常需要 $X$ 是联合可测的，即 $X$ 作为 $\Omega \times T$ 上的函数是可测的。

> Q: 一个联合可测的随机过程，和一个只对 $t$ 可测的随机过程，有什么区别？和一个只有$X_t$ 可测的随机过程，有什么区别？
>
> A: 对 $t$ 可测的随机过程，意味着对于每个固定的 $\omega$，函数 $t \mapsto X_t(\omega)$ 是可测的。但是，对于联合可测的随机过程，我们可以应用 Fubini 定理，这说明 $I(\omega) = \int_{[0,r]} X_t(\omega) dt$ 也是一个随机变量（$\mathcal{F}$-可测的）。而仅仅对 $t$ 可测的随机过程，不能保证这一点。

我们可以将 $\omega$ 视同于路径 $t \mapsto X_t(\omega)$，从而将 $\Omega$ 视作所有从 $T$ 到 $\mathbb R^n$ 的函数空间 $\tilde \Omega = (\mathbb R^n)^T$ 上的一个子集。

定义在 $\tilde \Omega$ 上的自然 $\sigma$-代数为 $\tilde{\mathcal{F}}$，由所有形如 $\mathbb W=\{\tilde \omega \in \tilde \Omega : \tilde \omega(t_1) \in B_1, \ldots, \tilde \omega(t_k) \in B_k\}$ 的集合生成，其中 $t_i \in T$，$B_i \in \mathcal{B}(\mathbb{R}^n)$。称为柱集（cylinder sets）。这个代数记作 $\mathcal{B}(\mathbb W)$。（注意这个实际上包含了可数点）

> 对于连续函数空间，这个 $\sigma$-代数恰好等于由一致收敛拓扑诱导的 Borel $\sigma$-代数。（因为连续函数的上下极限可以只考虑稠密可数的有理数点）

对于随机过程 $X: \Omega \times T \to \mathbb{R}^n$，我们可以定义映射 $\pi: \Omega \to \tilde \Omega$，使得对于每个 $\omega \in \Omega$，$\pi(\omega)$ 是路径 $t \mapsto X_t(\omega)$。然后，我们可以通过 $\pi$ 将概率测度 $P$ 从 $\Omega$ 推到 $\tilde \Omega$ 上，得到测度 $\tilde P$，定义为对于所有 $A \in \tilde{\mathcal{F}}$，
$$
\tilde P(A) = P(\pi^{-1}(A)).
$$

这样，我们就得到了一个新的概率空间 $(\tilde \Omega, \tilde{\mathcal{F}}, \tilde P)$，其中 $\tilde{\mathcal{F}}$ 是 $\tilde \Omega$ 上的 $\sigma$-代数，$\tilde P$ 是定义在 $\tilde{\mathcal{F}}$ 上的概率测度。这个新的概率空间描述了随机过程 $X$ 的路径空间结构。

> 于是可以采用这样的观点：随机过程 $X$ 可以被视为定义在路径空间 $(\tilde \Omega = (\mathbb R^n)^{T}, \tilde{\mathcal{F}} = \mathcal{B}(\mathbb W), \tilde P)$ 上的路径测度 $\tilde P$。

