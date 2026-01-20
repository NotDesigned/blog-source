---
title: 随机微分方程（SDE）
date: 2026-01-01 21:01:50
tags:
- Stochastic Differential Equations
- Ito Calculus
categories:
- Mathematics
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

($\Leftarrow$) 如果 $Y = \mathbf{1}_A$ 是集合 $A \in \sigma(X)$ 的指示函数，则存在 Borel 集合 $B \in \mathcal{B}$，使得 $A = X^{-1}(B)$。定义函数 $f: \mathbb{R}^n \to \mathbb{R}$ 为 $f(x) = \mathbf{1}_B(x)$，则 $Y(\omega) = f(X(\omega))$ 对所有 $\omega \in \Omega$ 成立。然后可以将该结论推广到简单函数 $Y = \sum a_i \mathbf{1}_{A_i}$，其中 $A_i \in \sigma(X)$，再通过测度论里的套路构造逐点极限收敛序列$Y_k$，推广到非负的随机变量 $Y$，最后正负分解推广到任意随机变量 $Y$。

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
条件期望的存在性和唯一性（几乎处处相等意义下）可以通过 Radon-Nikodym 定理来证明。

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

定义在 $\tilde \Omega$ 上的自然 $\sigma$-代数为 $\tilde{\mathcal{F}}$，由所有形如 $\mathbb W=\{\tilde \omega \in \tilde \Omega : \tilde \omega(t_1) \in B_1, \ldots, \tilde \omega(t_k) \in B_k\}$ 的集合生成，其中 $t_i \in T$，$B_i \in \mathcal{B}(\mathbb{R}^n)$。称为柱集（cylinder sets）。这个代数记作 $\mathcal{B}(\mathbb W)$。（注意这个实际上包含了可数点，这里的有限 $k$ 可以加强为 $1$，但是实际应用中有限比较方便）

> 对于连续函数空间，这个 $\sigma$-代数恰好等于由一致收敛拓扑诱导的 Borel $\sigma$-代数。（因为连续函数的上下极限可以只考虑稠密可数的有理数点）

对于随机过程 $X: \Omega \times T \to \mathbb{R}^n$，我们可以定义映射 $\pi: \Omega \to \tilde \Omega$，使得对于每个 $\omega \in \Omega$，$\pi(\omega)$ 是路径 $t \mapsto X_t(\omega)$。然后，我们可以通过 $\pi$ 将概率测度 $P$ 从 $\Omega$ 推到 $\tilde \Omega$ 上，得到测度 $\tilde P$，定义为对于所有 $A \in \tilde{\mathcal{F}}$，
$$
\tilde P(A) = P(\pi^{-1}(A)).
$$

这样，我们就得到了一个新的概率空间 $(\tilde \Omega, \tilde{\mathcal{F}}, \tilde P)$，其中 $\tilde{\mathcal{F}}$ 是 $\tilde \Omega$ 上的 $\sigma$-代数，$\tilde P$ 是定义在 $\tilde{\mathcal{F}}$ 上的概率测度。

这样，随机过程 $X$ 可以被视为定义在路径空间 $(\tilde \Omega = (\mathbb R^n)^{T}, \tilde{\mathcal{F}} = \mathcal{B}(\mathbb W), \tilde P)$ 上的路径概率测度 $\tilde P$。

现在考虑从有限观测点出发，构造随机过程。

#### Kolmogorov 扩展定理（Kolmogorov Extension Theorem）

观测 $k\in \mathbb{N}$ 次，假定存在 $\nu_{t_1, t_2, \ldots, t_k}$ 是 $\mathbb{R}^{nk}$ 上的概率测度。

满足一致性条件：对于 $\forall k \in \mathbb{N},t_1, t_2, \ldots, t_k \in T$，以及任意的 Borel 集合 $B_1, B_2, \ldots, B_k \in \mathcal{B}(\mathbb{R}^n)$，有
$$
\nu_{t_1, t_2, \ldots, t_k}(B_1 \times B_2 \times \ldots \times B_k) = \nu_{t_1, t_2, \ldots, t_k, t_{k+1}}(B_1 \times B_2 \times \ldots \times B_k \times \mathbb{R}^n).
$$

那么存在 一个随机过程 $\{X_t\}$，定义在某个概率空间 $(\Omega, \mathcal{F}, P)$ 上，使得对于任意的 $k \in \mathbb{N}$ 和 $t_1, t_2, \ldots, t_k \in T$，随机变量 $(X_{t_1}, X_{t_2}, \ldots, X_{t_k})$ 的分布为 $\nu_{t_1, t_2, \ldots, t_k}$。换言之，对任意 Borel 集合 $B_1, B_2, \ldots, B_k \in \mathcal{B}(\mathbb{R}^n)$，有
$$
P(X_{t_1} \in B_1, X_{t_2} \in B_2, \ldots, X_{t_k} \in B_k) = \nu_{t_1, t_2, \ldots, t_k}(B_1 \times B_2 \times \ldots \times B_k).
$$

注意，这个定理并没有保证样本路径的正则性（如连续性或可测性）。它仅仅保证了存在一个随机过程，其有限维分布与给定的一致性条件相符。也就是说目前的 $\Omega$ 只能视作 $(\mathbb R^n)^T$。

为了得到具有更好路径性质的随机过程，我们需要以下定理。

#### Kolmogorov 连续性定理（Kolmogorov Continuity Theorem）

设 $\{X_t : t \in [0, T]\}$ 是定义在概率空间 $(\Omega, \mathcal{F}, P)$ 上的随机过程。假设存在常数 $\alpha, \beta, C > 0$，使得对于所有的 $s, t \in [0, T]$，有
$$
E[\|X_t - X_s\|^\alpha] \leq C |t - s|^{1 + \beta}.
$$
那么存在一个修改版本 $\{\tilde{X}_t : t \in [0, T]\}$，使得对于几乎所有的 $\omega \in \Omega$，路径 $t \mapsto \tilde{X}_t(\omega)$ 是 Hölder 连续的，指数为 $\gamma$，其中 $0 < \gamma < \frac{\beta}{\alpha}$。也就是说，存在随机变量 $K(\omega)$，使得对于所有的 $s, t \in [0, T]$，有
$$
\|\tilde{X}_t(\omega) - \tilde{X}_s(\omega)\| \leq K(\omega) |t - s|^{\gamma}.
$$

证明：

考虑所有二进位网格点 $\mathcal{D}_n = \{ \frac{k}{2^n} : k = 0, 1, \dots, 2^n \}$ 上的随机变量 $X_t$。利用假设的矩不等式与切比雪夫不等式
$$
P\left( |X_{\frac{k}{2^n}} - X_{\frac{k-1}{2^n}}| > \epsilon \right) \le \frac{E[|X_{\frac{k}{2^n}} - X_{\frac{k-1}{2^n}}|^\alpha]}{\epsilon^\alpha} \le \frac{C (2^{-n})^{1+\beta}}{\epsilon^\alpha}
$$
$M_n = \max_{1 \le k \le 2^n} |X_{\frac{k}{2^n}} - X_{\frac{k-1}{2^n}}|$, 则
$$
P(M_n > \epsilon) \le \sum_{k=1}^{2^n} P\left( |X_{\frac{k}{2^n}} - X_{\frac{k-1}{2^n}}| > \epsilon \right) \le \frac{C 2^{-n\beta}}{\epsilon^\alpha}
$$
取 $\epsilon = 2^{-n\gamma}$，其中 $0 < \gamma < \frac{\beta}{\alpha}$，则
$$
P(M_n > 2^{-n\gamma}) \le C 2^{n(\alpha \gamma - \beta)}
$$
由于 $\alpha \gamma - \beta < 0$，级数 $\sum_{n=1}^{\infty} P(M_n > 2^{-n\gamma})$ 收敛。由 Borel-Cantelli 引理，几乎处处存在随机变量 $N(\omega)$，使得对于所有的 $n \geq N(\omega)$， 有 $M_n(\omega) \leq 2^{-n\gamma}$。
因此，对于几乎所有的 $\omega \in \Omega$，存在 $N(\omega)$，使得对于所有的 $n \geq N(\omega)$ 和 $k = 1, 2, \ldots, 2^n$，有
$$
|X_{\frac{k}{2^n}}(\omega) - X_{\frac{k-1}{2^n}}(\omega)| \leq 2^{-n\gamma}.
$$
这表明对于几乎所有的 $\omega \in \Omega$，路径 $t \mapsto X_t(\omega)$ 在二进位网格点上是 Hölder 连续的。我们可以将这种 Hölder 连续性扩展到整个区间 $[0, T]$，从而得到修改版本 $\tilde{X}_t$，使得对于几乎所有的 $\omega \in \Omega$，路径 $t \mapsto \tilde{X}_t(\omega)$ 是 $\gamma$-Hölder 连续的。

所以对于矩被控的随机过程，我们总能找到一个修改版本，使得路径具有 Hölder 连续性。

### 布朗运动 （Brownian Motion）

考虑这样的一个过程，在任何时刻 $t_1$ 观察，在 $t_2$ 时候的概率分布只与经过的时间 $t_2 - t_1$ 与位移有关。 
$$
p(t,x,y) = (2\pi t)^{-n/2}\cdot \exp(-\frac{|x-y|^2}{2t}) \forall y\in \mathbb R^n, t > 0
$$
给定 $k$ 个观测点, $0\leq t_1 \leq t_2 \leq \cdots \leq t_k$，定义这样的概率度量为 $\nu_{t_1,\ldots, t_k}$ 在 $\mathbb R^{nk}$ 为
$$
v_{t_1,\ldots, t_k}(F_1\times\cdots \times F_k) = \int_{F_1\times \cdots \times F_k} p(t_1, x, x_1) p(t_2-t_1,x_1,x_2) \cdots p(t_k-t_{k-1}, x_{k-1}, x_k) d x_1 \cdots d x_k
$$
并且规定 $p(0,x,y) dy = \delta_x(y)$，其中 $dy=dy_1\cdots dy_k$ 是Lebesgue测度。
这些测度满足一致性条件，因此由 Kolmogorov 扩展定理，存在一个概率空间 $(\Omega, \mathcal{F}, P^{x})$ 和随机过程 $\{B_t : t \geq 0\}$，使得对于任意的 $k \in \mathbb{N}$ 和 $0 \leq t_1 \leq t_2 \leq \cdots \leq t_k$，随机变量 $(B_{t_1}, B_{t_2}, \ldots, B_{t_k})$ 的分布为 $\nu_{t_1, t_2, \ldots, t_k}$。
这个随机过程称为从点 $x$ 开始的布朗运动（Brownian Motion），记作 $B_t^x$ 或简称 $B_t$。

根据 Kolmogorov 连续性定理，布朗运动存在一个修改版本，使得对于几乎所有的 $\omega \in \Omega$，路径 $t \mapsto B_t(\omega)$ 是 Hölder 连续的，指数为 $\gamma$，其中 $0 < \gamma < \frac{1}{2}$。故可以视为 $C([0,\infty);\mathbb R^n)$ 上的概率测度。

布朗运动具有以下重要性质：
1. 是高斯过程：对于任意的 $k \in \mathbb{N}$ 和 $0 \leq t_1 < t_2 < \cdots < t_k$，随机变量 $Z=(B_{t_1}, B_{t_2}, \ldots, B_{t_k})$ 服从多元正态分布。
$$
\mathbb E^{x} \left[\exp\left(i \sum_j^{nk} u_j Z_j\right)\right] = \exp\left(i \sum_j^{nk} u_j M_j - \frac{1}{2} \sum_{j,l}^{nk} u_j c_{jl} u_l \right)
$$

其中 
$$
M = [x, x, \ldots, x]^T \in \mathbb R^{nk}
$$
$$
C = \begin{bmatrix}
t_1 I_n & t_1 I_n & \cdots & t_1 I_n \\
t_1 I_n & t_2 I_n & \cdots & t_2 I_n \\
\vdots & \vdots & \ddots & \vdots \\
t_1 I_n & t_2 I_n & \cdots & t_k I_n
\end{bmatrix} \in \mathbb R^{nk \times nk}
$$

我们有 $\mathbb E^{x}[(B_t - x)^2] = n t$。即布朗运动的方差与时间成正比； $\mathbb E^{x}[(B_t - x) (B_s - x)] = n \min(t,s)$。即布朗运动的协方差与时间的最小值成正比。

因此，$\mathbb E^{x}[ (B_t - B_s)^2] = n (t-s), \forall 0 \leq s < t$。

2. 具有独立增量：对于任意的 $0 \leq t_1 < t_2 < \cdots < t_k$，增量 $B_{t_2} - B_{t_1}, B_{t_3} - B_{t_2}, \ldots, B_{t_k} - B_{t_{k-1}}$ 是相互独立的随机变量。


3. 几乎处处连续路径：布朗运动存在一个修改版本，使得对于几乎所有的 $\omega \in \Omega$，路径 $t \mapsto B_t(\omega)$ 是连续的。

4. 几乎处处不可微：对于几乎所有的 $\omega \in \Omega$，路径 $t \mapsto B_t(\omega)$ 在任意时刻 $t$ 都不可微。

1, 2, 3 是直接由构造和 Kolmogorov 连续性定理得到的。

4 的详细证明这里略过，对于单点来说，考虑增量商 $\geq M$ 的概率，可以证明几乎处处存在一个无穷序列，使得增量商 $\geq M$，从而不可微。但是对于整个区间不存在可微点的概率，需要更复杂的论证。

## 2. Ito 积分，Ito 引理 和 Ito 公式 (Chapter 3, 4 of Øksendal)

### Ito 积分

首先我们要问，什么叫做随机积分？我们希望定义类似于 Riemann/Lebesgue 积分的东西，但是积分的变元是随机过程。这样的过程是因为现实中很多现象都具有随机性，比如股票价格的变化，粒子的无规则运动等。

形式化地说，假如我们有一个随机过程 $X_t$ 和布朗运动 $B_t$，我们希望定义积分
$$
\int_S^T X_t dB_t
$$
这里的 $dB_t$ “表示布朗运动的增量”。

回忆一下 Riemann 积分的定义，是将区间 $[S,T]$ 划分为小区间，然后在每个小区间上取一个点，计算函数值与区间长度的乘积之和，最后取极限。或者等价地，用上方控制的简单函数与下方控制的简单函数来逼近，然后两边取极限。

类似地，我们可以尝试定义随机积分为
$$
\sum_{j} f(t_j^*,\omega) (B_{t_{j+1}} - B_{t_j})(\omega)
$$
其中 $\{t_j\}$ 是 $[S,T]$ 的一个划分，$t_j^* \in [t_j, t_{j+1}]$ 是每个小区间内的某一个点。

我们现在考虑 $X_t = B_t$ 的情况，即积分
$$
\int_0^T B_t dB_t
$$
我们尝试用上面的和式来定义这个积分。取一个划分 $S=0 = t_0 < t_1 < \cdots < t_n = T$，则
$$
\begin{aligned}
\sum_{j=0}^{n-1} B_{t_j} (B_{t_{j+1}} - B_{t_j}) &= \frac{1}{2} \sum_{j=0}^{n-1} (B_{t_{j+1}}^2 - B_{t_j}^2) - \frac{1}{2} \sum_{j=0}^{n-1} (B_{t_{j+1}} - B_{t_j})^2 \\
= & \frac{1}{2} (B_T^2 - B_0^2) - \frac{1}{2} \sum_{j=0}^{n-1} (B_{t_{j+1}} - B_{t_j})^2
\end{aligned}
$$
而
$$
\mathbb E\left[\sum_{j=0}^{n-1} (B_{t_{j+1}} - B_{t_j})^2\right]  = \sum_{j=0}^{n-1} (t_{j+1} - t_j) = T
$$

而考虑求和
$$
\begin{aligned}
\sum_{j=0}^{n-1} B_{t_{j+1}} (B_{t_{j+1}} - B_{t_j}) &= \frac{1}{2} \sum_{j=0}^{n-1} (B_{t_{j+1}}^2 - B_{t_j}^2) + \frac{1}{2} \sum_{j=0}^{n-1} (B_{t_{j+1}} - B_{t_j})^2 \\
&= \frac{1}{2} (B_T^2 - B_0^2) + \frac{1}{2} \sum_{j=0}^{n-1} (B_{t_{j+1}} - B_{t_j})^2
\end{aligned}
$$
这说明，随机积分的结果与取点的位置有关。

这里的变化，正是因为布朗运动的增量 $(B_{t_{j+1}} - B_{t_j})$ 的平方的期望与区间长度成正比，而不是像确定性函数那样趋近于零。

对于 Ito 积分，我们选择在每个小区间的左端点取值，即定义为
$$
\int_S^T X_t dB_t := \lim_{|\Delta| \to 0} \sum_{j} X_{t_j} (B_{t_{j+1}} - B_{t_j})
$$
令最大区间长度 $|\Delta| = \max_j (t_{j+1} - t_j)$ 趋近于零。

顺便，我们也可以定义 Stratonovich 积分，其选择在每个小区间的中点取值，即定义为
$$
\int_S^T X_t \circ dB_t := \lim_{|\Delta| \to 0} \sum_{j} X_{\frac{t_j + t_{j+1}}{2}} (B_{t_{j+1}} - B_{t_j})
$$

那么现在我们就可以保证我们的积分在 $L^2$ 意义下收敛到一个随机变量了吗？
为了考察这个问题，我们先考虑简单函数的情况。

我们称随机过程 $X_t$ 是一个简单过程，如果它可以表示为
$$
X_t(\omega) = \sum_{j} e_j(\omega) \mathbf{1}_{[t_j, t_{j+1})}(t)
$$
其中 $\{t_j\}$ 是 $[S,T]$ 的一个划分。

显然，对于简单过程 $X_t$，我们可以定义 Ito 积分为
$$
\int_S^T X_t dB_t := \sum_{j} e_j (B_{t_{j+1}} - B_{t_j})
$$

然后我们考虑在 $L^2$ 意义下的收敛性。即计算
$$
\begin{aligned}
\mathbb E\left[\left(\int_S^T X_t dB_t\right)^2\right] &= \mathbb E\left[\left(\sum_{j} e_j (B_{t_{j+1}} - B_{t_j})\right)^2\right] \\
& = \sum_{i,j} \mathbb E[e_i e_j \Delta B_i \Delta B_j] \\
\end{aligned}
$$

为了进一步计算，我们假设 $e_j$ 是 $\mathcal{F}_{t_j}$-可测的，其中 $\mathcal{F}_t$ 是由 $\{B_s : s \leq t\}$ 生成的 $\sigma$-代数（则$\mathcal{F}_{s} \subseteq \mathcal{F}_{t}$ 对于 $s < t$）。

对于 $i < j$，有 $e_i, e_j, \Delta B_i$ 都是 $\mathcal{F}_{t_j}$-可测的，而 $\Delta B_j$ 独立于 $\mathcal{F}_{t_j}$，因此
$$
\begin{aligned}
\mathbb E[e_i e_j \Delta B_i \Delta B_j] &= \mathbb E\left[ \mathbb E[e_i e_j \Delta B_i \Delta B_j | \mathcal{F}_{t_j}] \right] \quad \text{（全期望公式）} \\
&= \mathbb E\left[ e_i e_j \Delta B_i \mathbb E[\Delta B_j | \mathcal{F}_{t_j}] \right] \quad \text{（条件期望的线性性质）} \\
&= 0 \quad \text{（} \mathbb E[\Delta B_j | \mathcal{F}_{t_j}] = \mathbb E[\Delta B_j] = 0 \text{）}
\end{aligned}
$$

类似地，对于 $i > j$，也有 $\mathbb E[e_i e_j \Delta B_i \Delta B_j] = 0$。
因此，只有当 $i = j$ 时，$\mathbb E[e_i e_j \Delta B_i \Delta B_j]$ 才可能非零。此时，
$$
\mathbb E[e_j^2 (\Delta B_j)^2] = \mathbb E[e_j^2] \mathbb E[(\Delta B_j)^2] = \mathbb E[e_j^2] (t_{j+1} - t_j)
$$
综上所述，我们得到了
$$
\mathbb E\left[\left(\int_S^T X_t dB_t\right)^2\right] = \sum_{j} \mathbb E[e_j^2] (t_{j+1} - t_j) = \mathbb E\left[\int_S^T X_t^2 dt\right]
$$

因此，我们得到了 Ito 等距公式 （简单过程版本）：
$$
\mathbb E\left[\left(\int_S^T X_t dB_t\right)^2\right] = \mathbb E\left[\int_S^T X_t^2 dt\right]
$$

左边是一个 Ito 积分作为随机变量的 $L^2$ 范数，右边是一个随机过程被视作 $\Omega \times [S,T]$ 上的概率测度/随机变量的 $L^2$ 范数。因此叫做等距公式。

这个实际上就在告诉我们，我们在时间这一维积分后，整体的长度是不变的。

现在我们将它推广到较为一般的过程并给出正式定义。

令 $\{\mathcal{N}_t\}$ 为一族递增的 $\sigma$-代数，称为过滤（filtration）。随机过程 $X_t$ 称为适应于过滤 $\{\mathcal{N}_t\}$ 的，如果对于每个 $t$，$X_t$ 是 $\mathcal{N}_t$-可测的。

令 $\mathcal{F}_t$ 为由布朗运动 $\{B_s : s \leq t\}$ 生成的自然过滤，即
$$
\mathcal{F}_t = \sigma(B_s : s \leq t) = \sigma(\omega; \omega(t_1)\in B_1, \ldots, \omega(t_k) \in B_k, \text{其中 } t_i \leq t, B_i \in \mathcal{B}(\mathbb{R}^n))
$$
这里我们假设 $\mathcal{F}_t$ 是完备化的，即包含所有的 $P$-零测集。

定义空间 $\mathcal{V}=\mathcal{V}(S,T)$ 为所有适应于过滤 $\{\mathcal{F}_t\}$ 的随机过程 $X_t$（或函数 $X: [S,T] \times \Omega \to \mathbb{R}$）且满足
1. $X$ 在 $[S,T] \times \Omega$ 上联合可测；
2. $\int_S^T E[X_t^2] dt < \infty$，即 $X_t \in L^2([S,T] \times \Omega)$。

首先，令 $g \in \mathcal{V}$ 是有界的，并且 $g$ 对于每个 $\omega$ 是连续的。则存在一列简单过程 $\{g_n\} \subset \mathcal{V}$，使得
$$
\lim_{n \to \infty} E\left[\int_S^T (g_n(t,\omega) - g(t,\omega))^2 dt\right] = 0.
$$

证明：
取 $\phi_n = \sum_j g(t_j, \omega) \mathbf{1}_{[t_j, t_{j+1})}(t)$，其中 $\{t_j\}$ 是 $[S,T]$ 的一个划分，且当 $n \to \infty$，划分的间距趋近于零。显然有 $\phi_n \in \mathcal{V}$，并且由于 $g$ 对于每个 $\omega$ 连续，因此 $\phi_n(t,\omega)$ 对于每个 $\omega$ 在 $[S,T]$ 上逐点收敛于 $g(t,\omega)$，且一致有界。

由有界收敛定理，我们有
$$
\lim_{n \to \infty} \mathbb E\left[\int_S^T (\phi_n(t,\omega) - g(t,\omega))^2 dt\right] = \mathbb E\left[\int_S^T \lim_{n \to \infty} (\phi_n(t,\omega) - g(t,\omega))^2 dt\right] = 0.
$$
因此，$\{\phi_n\}$ 是所需的简单过程列。