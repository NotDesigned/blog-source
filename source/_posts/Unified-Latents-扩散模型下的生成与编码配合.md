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
{\mathrm{KL}(q(z|x) || p(z))}_{\text{Latent Encode Cost}}\\
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

问题在于，变分自编码器中，我们无法显式地控制 $I(Z;X)$，或者说潜变量 $z$ 中保留了多少关于输入数据 $x$ 的信息，只能通过调节损失函数里面的 KL 散度权重来间接控制。这就导致了复杂的调试过程，而且因模型而异。

## Unified Latents 架构

首先，Unified Latents 架构使用了一个确定性的编码器 $E: \mathbb R^n \to \mathbb R^d$，将输入数据 $x$ 映射到一个潜空间中的表示 $z_{\text {clean}}=E(x)$。

然后，对这个确定性的编码加噪得到带噪编码 $z_t = \alpha_z(t) z_{\text{clean}} + \sigma_z^2 \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$ 是标准高斯噪声。

接下来，使用两个独立的扩散模型，一个先验扩散模型（diffusion prior）负责从加噪编码 $z_t$ 生成原始编码。另一个从一个设定好的固定程度加噪（$\lambda(0)=5$，见下文）编码 $z_t$ 作为条件生成数据 $x$。

![Unified Latents 架构](Arch.png)

$$
z_t = \alpha_z(t) \cdot z_{\text {clean}} + \sigma_z(t) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$
信噪比为
$$
\mathrm{SNR}(t) = \frac{\alpha_z(t)^2}{\sigma_z(t)^2}
$$
同时定义 $\lambda(t) = \log \mathrm{SNR}(t)$ ，$\lambda$ 大意味着信号强、噪声弱（数据清晰）；$\lambda$ 小意味着噪声主导。

好处是可以方便地表达信号和噪声的比例： 
$$
\alpha^2 = \text{sigmoid}(\lambda), \sigma^2 = \text{sigmoid}(-\lambda).
$$

> 注：$\text{sigmoid}(t) = \frac{1}{1 + e^{-t}}$

![手画的演示](pic.jpg)

### 损失函数推导 与 ELBO 重加权

对于扩散模型，文中使用了以下的损失：
$$
\mathbb E_{t\in \text{Uniform}(0,1)}\left[-\frac{d\lambda(t)}{dt}\frac{\exp\lambda(t)}{2}\omega(\lambda(t))\|x-\hat x(x_t,\theta)\|^2\right] + \mathrm{KL}\left(p(x_1|x) | p(x_1)\right)
$$

逻辑是，首先最小化负对数似然展开成 ELBO。
$$
-\log p_\theta(x) \leq \mathbb E_{q(z|x)}[-\log p(x|z)] + \mathrm{KL}(q(z|x) || p(z))
$$

在扩散模型的设定里，$z$ 是标准高斯噪声 $\mathcal{N}(0,I)$。

但是不同于 VAE，扩散模型中的 $z$ 代表的是一整个时间步的噪声过程，也即是路径空间 $\Omega \times [0,1]$ 上的作为随机变量。然后这个路径由前向和后向过程 Itô SDE 定义。

根据 Girsanov 定理，路径空间上的 KL 散度可以被重写为一个时间积分：
$$
\text{KL}[\mathbb{Q}^{\leftarrow} \| \mathbb{P}_\theta] = \frac{1}{2} \int_0^1 g(t)^2 \, \mathbb{E}_{q(x_t)} \left[\left\| \nabla_{x_t} \log q_t(x_t) - s_\theta(x_t, t) \right\|^2 \right] dt
$$
其中 $s_\theta(x_t, t)$ 是模型的得分函数，$\nabla_{x_t} \log q_t(x_t)$ 是真实的得分函数。

而 $x$ 和 $s$ 的关系由扩散模型的定义给出：
$$
s_\theta(x_t, t) = \frac{1}{\sigma(t)} (x_t - \alpha(t) \hat{x}(x_t, t))
$$

然后由 SDE 的对应关系给出，$g(t)^2 = -\dot{\lambda} \sigma_t^2$。

带入化简一下就可以得到文中使用的损失函数。

而这里只降噪到 $\lambda(0)=5$ 之类的 $z_0$，所以实际上 time integral 的下界不是 0。然后还需要加上 $z_1$ 端点的 KL 散度：
$$
\mathrm{KL}\left(p(z_1|x) | \mathcal{N}(0,I)\right)
$$

而 reweight 就是在路径空间上对不同时间步的损失进行加权，根据 Jensen's inequality 可以证明，重加权后的损失函数仍然是 ELBO 的上界。

> TODO: 有时间写一个详细的推导过程。

## 双阶段训练

TODO.