---
title: Reading notes of The Principle of Diffusion Models
date: 2025-11-03 21:08:49
tags:
- Diffusion Models
- Generative Models
categories:
- Machine Learning
---

This note is to summarize key points, note down key formulae and gives some discussion for the book "The Principle of Diffusion Models" by Song et al. 

[arxiv](https://www.arxiv.org/pdf/2510.21890)

## 1. Deep Generative Modeling

TODO.

## 2. Variational Perspective: From VAEs to DDPMs

### 2.1 Variational Autoencoder

**Model structure**

$$
x\xrightarrow{\mathrm{encoder},q_{\theta}(z|x)} z\sim \mathcal{N}(0,1) \xrightarrow{\mathrm{decoder}, p_{\phi}(x|z)} \hat{x}
$$

**Key formulae:**

The learned data distribution:

$$
p_{\phi}(x) = \mathbb{E}_{p(z)}[p_{\phi}(x|z)] = \int p_{\phi}(x|z)p(z)dz
$$

Intractable to compute due to integral over latent variable $z$.

Maximizing log-likelihood $\iff$ minimizing the KL divergence between true data distribution $q(x)$ and learned data distribution $p_{\phi}(x)$:

$$
\begin{aligned}
\arg\max_{\phi} \mathbb{E}_{q(x)}[\log p_{\phi}(x)] &\iff \arg\min_{\phi} \mathbb{E}_{q(x)}[\log q(x) - \log p_{\phi}(x)] \\
&\iff \arg\min_{\phi} \mathrm{KL}(q(x) || p_{\phi}(x))
\end{aligned}
$$

**Evidence Lower Bound (ELBO):**

Idea: 

We try to estimate $p_{\phi}(x)$ by Bayes' theorem

$$
p_{\phi}(x) = \frac{p_{\phi}(x,z)}{p_{\phi}(z|x)}
$$

Now the posterior $p_{\phi}(z|x)$ is also intractable, so we introduce a variational distribution $q_{\theta}(z|x)$ to approximate it.

Lower bound of the log-likelihood $\log p_{\phi}(x)$:

$$
\begin{aligned}
\log p_{\phi}(x) &= \mathbb{E}_{q_{\theta}(z|x)}\left[\log \frac{p_{\phi}(x,z)}{p_{\phi}(z|x)}\right] \\
&= \mathbb{E}_{q_{\theta}(z|x)}\left[\\log \frac{p_{\phi}(x,z)}{q_{\theta}(z|x)} \cdot \frac{q_{\theta}(z|x)}{p_{\phi}(z|x)}\right] \\
&= \mathbb{E}_{q_{\theta}(z|x)}\left[\log \frac{p_{\phi}(x,z)}{q_{\theta}(z|x)}\right] + \mathbb{E}_{q_{\theta}(z|x)}\left[\log \frac{q_{\theta}(z|x)}{p_{\phi}(z|x)}\right] \\
&= \mathbb{E}_{q_{\theta}(z|x)}\left[\log \frac{p_{\phi}(x,z)}{q_{\theta}(z|x)}\right] + \mathrm{KL}(q_{\theta}(z|x) || p_{\phi}(z|x)) \\
&\geq \mathbb{E}_{q_{\theta}(z|x)}\left[\log \frac{p_{\phi}(x,z)}{q_{\theta}(z|x)}\right] \\
&= \mathcal{L}(\theta, \phi; x)
\end{aligned}
$$

Quick summary:

By introducing a latent variable $z$ and an inaccurate posterior approximation $q_{\theta}(z|x)$, we derive a lower bound of the log-likelihood $\log p_{\phi}(x)$, which is called Evidence Lower Bound (ELBO). It is the expectation of the **Bayes-like ratio** $\frac{p_{\phi}(x,z)}{q_{\theta}(z|x)}$ under the $q_{\theta}(z|x)$.
