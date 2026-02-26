---
title: 'Unified Latents: 扩散模型的框架下，令生成与编码相互配合'
date: 2026-02-26 14:22:25
tags:
- Diffusion Models
categories:
- Machine Learning
---

本文介绍 google deepmind 团队提出的 Unified Latents 架构，在扩散模型的架构下统一了生成和表示压缩，并且给出了优雅的信息界。


## 生成与编码的权衡

在生成模型设定中（详细见[此文](https://notdesigned.github.io/2025/11/03/Reading-notes-of-The-Principle-of-Diffusion-Models/) ） ，首先我们有一个在高维空间中的分布

$$
x\sim \mu,\ p(x):=\mathrm{Law}(\mu) : \mathbb R^n \to [0,\infty)
$$

历史上， 变分自编码器（Variational Autoencoder, VAE）引入了一个潜变量 $z$，并且得出了一个关于 $p(x)$ 的下界。然后考虑真实的条件概率 $p(z|x)$ 与其变分近似 $q(z|x)$:
$$
\begin{aligned}
\log p(x) &= \mathbb E_{q(z|x)} \left[\log\frac{p(x,z)}{p(z|x)}\right] \\
&\geq \mathbb E_{q(z|x)}\left[\log \frac{p(x,z)}{q(z|x)}\right] \\
&= \underbrace{\mathbb E_{q(z|x)}\left[\log
p(x|z)\right]}_{\text{Reconstruction Term}} - \underbrace
{\mathrm{KL}(q(z|x) || p(z))}_{\text{Latent Regularization}}\\
\end{aligned}
$$

即：先从原始数据中压缩出潜变量 $z$，然后从潜变量 $z$ 中还原出 $x$。并且让编码器 $q(z|x)$ 靠近标准正态分布 $p(z)=\mathcal{N}(0,I)$ 以得到优良的潜空间性质。

在这里，编码器 $q(z|x)$ 从原始数据中提取出潜变量 $z$，而生成器 $p(x|z)$ 从潜变量 $z$ 中还原出原始数据 $x$。这个由信息瓶颈（Information Bottleneck）引入的潜变量 $z$ 连接了生成和编码两个过程。

### 信息瓶颈

信息瓶颈描述了在编码过程中，如何在保留有用信息的同时压缩数据。具体来说，信息瓶颈试图找到一个潜变量 $z$，使得它能够最大程度地保留关于输入数据 $x$ 的有用信息，同时尽可能地压缩 $z$ 的表示。

数学上，信息瓶颈可以通过以下优化问题来描述：
$$
\min_{p(z|x)} I(X;Z) - \beta I(Z;Y)
$$
其中，$I(X;Z) = \mathbb E_{p(x,z)}[\log\frac{p(x,z)}{p(x)p(z)}]$ 是输入数据 $X$ 和潜变量 $Z$ 之间的互信息，$I(Z;Y)$ 是潜变量 $Z$ 和目标变量 $Y$ 之间的互信息，$\beta$ 是一个权衡参数。

较大的 $\beta$ 会使得模型更倾向于最小化 $I(Z;Y)$，从而得到一个更保留输入数据 $X$ 的潜变量 $Z$，而较小的 $\beta$ 则会使得模型更倾向于最小化 $I(X;Z)$，从而得到一个更压缩的潜变量 $Z$。

变分自编码器通过调控 KL 散度项 $\mathrm{KL}(q(z|x) || p(z))$ 来实现信息瓶颈的效果。较大的 KL 散度会使得潜变量 $z$ 更接近于标准正态分布，从而令 $I(X;Z)$ 较小，得到一个更压缩的潜变量 $z$；反之亦然。

问题在于，变分自编码器中，我们无法显式地控制 $I(Z;X)$，或者说潜变量 $z$ 中保留了多少关于输入数据 $x$ 的信息，只能通过调节 KL 散度来间接控制。这就导致了复杂的调试过程，而且因模型而异。

## Unified Latents 架构

首先，Unified Latents 架构使用了一个确定性的编码器 $E: \mathbb R^n \to \mathbb R^d$，将输入数据 $x$ 映射到一个潜空间中的表示 $z_{\text {clean}}=E(x)$。

然后，对这个确定性的编码加噪得到带噪编码 $z_t =  z_{\text {clean}} + \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ 是一个高斯噪声。

接下来，使用两个独立的扩散模型，一个从噪声编码 $z_t$ 生成原始编码。另一个从干净编码 $z_{\text {clean}}$ 生成原始数据 $x$。

![Unified Latents 架构](Arch.png)

$$
z_t = \alpha_z(t) \cdot z_{\text {clean}} + \sigma_z(t) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$
信噪比为
$$
\mathrm{SNR}(t) = \frac{\alpha_z(t)^2}{\sigma_z(t)^2}
$$
同时定义 $\lambda(t) = \log \mathrm{SNR}(t)$ ，$\lambda$ 大意味着信号强、噪声弱（数据清晰）；$\lambda$ 小意味着噪声主导。

好处是可以方便地表达： $\alpha^2 = \text{sigmoid}(\lambda), \sigma^2 = \text{sigmoid}(-\lambda)$。

> 注：$\text{sigmoid}(t) = \frac{1}{1 + e^{-t}}$

![手画的演示](pic.png)

### 损失函数推导 与 ELBO 重加权

TODO.