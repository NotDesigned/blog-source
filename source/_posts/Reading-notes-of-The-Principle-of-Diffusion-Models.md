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

**Evidence Lower Bound (ELBO):**

Lower bound of the log-likelihood $\log p_{\phi}(x)$:
