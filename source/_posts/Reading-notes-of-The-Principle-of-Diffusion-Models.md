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

**Core Idea**: 

We try to estimate $p_{\phi}(x)$ by Bayes' theorem

$$
p_{\phi}(x) = \frac{p_{\phi}(x,z)}{p_{\phi}(z|x)}
$$

Now the posterior $p_{\phi}(z|x)$ is also intractable, so we introduce a variational distribution $q_{\theta}(z|x)$ to approximate it.

Proof of being lower bound of the log-likelihood $\log p_{\phi}(x)$:

$$
\begin{aligned}
\log p_{\phi}(x) &= \mathbb{E}_{q_{\theta}(z|x)}\left[\log \frac{p_{\phi}(x,z)}{p_{\phi}(z|x)}\right] \\
&= \mathbb{E}_{q_{\theta}(z|x)}\left[\log \frac{p_{\phi}(x,z)}{q_{\theta}(z|x)} \cdot \frac{q_{\theta}(z|x)}{p_{\phi}(z|x)}\right] \\
&= \mathbb{E}_{q_{\theta}(z|x)}\left[\log \frac{p_{\phi}(x,z)}{q_{\theta}(z|x)}\right] + \mathbb{E}_{q_{\theta}(z|x)}\left[\log \frac{q_{\theta}(z|x)}{p_{\phi}(z|x)}\right] \\
&= \mathbb{E}_{q_{\theta}(z|x)}\left[\log \frac{p_{\phi}(x,z)}{q_{\theta}(z|x)}\right] + \mathrm{KL}(q_{\theta}(z|x) || p_{\phi}(z|x)) \\
&\geq \mathbb{E}_{q_{\theta}(z|x)}\left[\log \frac{p_{\phi}(x,z)}{q_{\theta}(z|x)}\right] \\
&= \underbrace{\mathbb E_{q_{\theta}(z|x)}\left[\log p_{\phi}(x|z)\right]}_{\text{Reconstruction Term}} - \underbrace {\mathrm{KL}(q_{\theta}(z|x) || p(z))}_{\text{Latent Regularization}}\\
&= \mathcal{L}(\theta, \phi; x)
\end{aligned}
$$

**Quick summary**:

By introducing a latent variable $z$ and an inaccurate posterior approximation $q_{\theta}(z|x)$, we derive a lower bound of the log-likelihood $\log p_{\phi}(x)$, which is called Evidence Lower Bound (ELBO). 

Interpretation 1.

It is the expectation of the **Bayes-like ratio** $\frac{p_{\phi}(x,z)}{q_{\theta}(z|x)}$ by replacing the intractable true posterior $p_{\phi}(z|x)$ with the variational distribution $q_{\theta}(z|x)$.

Interpretation 2.

It consists of two terms: a reconstruction term that encourages the decoder to reconstruct $x$ from $z$, and a latent regularization term that encourages the encoder distribution $q_{\theta}(z|x)$ to be close to the prior $p(z)$.

**Reason of blurriness of Standard VAE**:

The standard VAE employs Gaussian distribution for both encoder and decoder. 

$$
\begin{aligned}
q_{\theta}(z|x) &= \mathcal{N}(z; \mu_{\theta}(x), \sigma_{\theta}^2(x)I)\\
p_{\phi}(x|z) &= \mathcal{N}(x; \mu_{\phi}(z), \sigma^2 I)
\end{aligned}
$$

The optimal decoder mean $\mu_{\phi}(z)$ is the conditional expectation $\mathbb{E}[x|z]$, which is the average of all possible $x$ that can be mapped to the same $z$. This averaging effect leads to blurriness in generated samples.

Derivation of optimal decoder mean:

First note that
$$
\begin{aligned}
\mathbb{E}_{q_{\theta}(z|x)}[\log p_{\phi}(x|z)]
&= \mathbb{E}_{q_{\theta}(z|x)}\left[-\frac{1}{2\sigma^2}\|x - \mu_{\phi}(z)\|^2 + \text{const}\right] \\
&= -\frac{1}{2\sigma^2} \mathbb{E}_{q_{\theta}(z|x)}[\|x - \mu_{\phi}(z)\|^2] + \text{const} \\
\end{aligned}
$$

And take expectation over $p(x)$:

$$
\begin{aligned}
\mathbb{E}_{p(x)}[\mathbb{E}_{q_{\theta}(z|x)}[\log p_{\phi}(x|z)]]
&= -\frac{1}{2\sigma^2} \mathbb{E}_{p(x)}[\mathbb{E}_{q_{\theta}(z|x)}[\|x - \mu_{\phi}(z)\|^2]] + \text{const} \\
&= -\frac{1}{2\sigma^2} \mathbb{E}_{q_{\theta}(z)}[\mathbb{E}_{q_{\theta}(x|z)}[\|x - \mu_{\phi}(z)\|^2]] + \text{const} \\
\end{aligned}
$$

where $q_{\theta}(x|z)= \frac{p(x) q_{\theta}(z|x)}{q_{\theta}(z)}$ is the posterior distribution of $x$ given $z$ under the encoder.

For the inner expectation $\mathbb{E}_{q_{\theta}(x|z)}[\|x - \mu_{\phi}(z)\|^2]$, it is minimized when $\mu_{\phi}(z) = \mathbb{E}_{q_{\theta}(x|z)}[x]$.


**Think**: 
Which part of the structure of VAE leads to blurriness most?

The encoder or the decoder?

Answer: 

It depends on your perspective.

The encoder mixes different $x$ into the same $z$, create ambiguity. The Gaussian assumption of $z$ enforces this mixing, otherwise the aggregate posterior $q_{\theta}(z)=\int p(x) q_{\theta}(z|x) dx$ cannot match the simple prior $p(z)$.

The Gaussian decoder reconstructs the average of these ambiguous $x$, leading to blurriness.

