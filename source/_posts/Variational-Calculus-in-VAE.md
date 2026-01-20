---
title: Variational Calculus in VAE
date: 2025-08-25 14:28:57
tags:
- Generative Models
- VAE
- Calculus of Variations
categories:
- Machine Learning
---

Given a probability distribution $p(x)$, we want to find an encoder $q_{\theta}(z|x)$ that approximates the true posterior $p(z|x)$. And we resort to minimize the Kullback-Leibler divergence between them:

$$
\min D_{KL}\left(q_{\theta}(z|x) \| p(z|x)\right)
$$

And $D_{KL}\left(q_{\theta}(z|x) \| p(z|x)\right)$ can be computed as:

$$
\begin{align*}
D_{KL}\left(q_{\theta}(z|x) \| p(z|x)\right) &= \mathbb{E}_{q_{\theta}(z|x)}\left[\log\frac{q_{\theta}(z|x)}{p(z|x)}\right]\\
&=\int q_{\theta}(z|x) \log\frac{q_{\theta}(z|x)}{p(z|x)} dz\\
&= J[q]\\
s.t. \int q_{\theta}(z|x) dz = 1
\end{align*}
$$
We introduce a Lagrange multipler $\lambda$, and we can write the constrained optimization problem as:

$$
\begin{align*}
S[q, \lambda] &= J[q] + \lambda\left(\int q_{\theta}(z|x) dz-1\right)\\
&= \int \left[ q(z) \log q(z) -q(z) \log p(z|x)\right] dz+ \lambda\left(\int q(z) dz- 1\right)\\
&= \int \left[ q(z) \log q(z) -q(z) \log p(z|x) + \lambda q(z) \right] dz - \lambda

\end{align*}
$$

Use the Euler-Lagrange Equation w.r.t. $q$
$$
\frac{\partial L}{\partial q} -\frac{d}{dz} \frac{\partial L}{\partial \dot q} = 0
$$

Reduce to 
$$
\frac{\partial L}{\partial q} = 0
$$
since $L$ is not explicitly dependent on $\dot q$.

Therefore
$$
\log q(z) + 1 - \log p(z|x) + \lambda  =0 
$$

We can solve the above equations and get
$$
q(z) = \exp\left(\log p(z|x) - 1 - \lambda\right) = C \cdot p(z|x)
$$
And with normalizing constraint, 
$$
\int q(z) dz = 1
$$
we know that $q(z) = p(z|x)$.

And the next step is to decompose 
$$
\begin{align*}
D_{KL}\left(q_{\theta}(z|x) \| p(z|x)\right) &= \mathbb{E}_{q_{\theta(z|x)}}\left[\log \frac{q_{\theta}(z|x)}{p(z)}\right] + \mathbb{E}_{q_{\theta(z|x)}}\left[\log \frac{p(z)}{p(z|x)}\right] \\
&= D_{KL}\left(q_{\theta}(z|x) \| p(z)\right) + \mathbb{E}_{q_\theta(z|x)}\left[\log p(x)\right]\\
&= D_{KL}\left(q_{\theta}(z|x) \| p(z)\right) + \log p(x)
\end{align*}
$$

which is equivalent to maximizing the ELBO:

$$
\mathcal{L}(q) = - D_{KL}\left(q_{\theta}(z|x) \| p(z)\right) =\mathbb {E}_{z\sim q(z|x)}\left [ \log\frac{p(x,z)}{q_{\theta}(z|x)}\right]
$$

And ELBO can be expressed as:

$$
\mathcal{L}(q) = \mathbb{E}_{z\sim q(z|x)} [\log p_{\theta}(x|z)] - D_{KL}(q_{\theta}(z|x) \| p(z))
$$
And $p_{\theta}(x|z)$ represent the decoder.