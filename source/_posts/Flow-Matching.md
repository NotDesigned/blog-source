---
title: Flow Matching
date: 2025-08-09 13:49:15
tags:
- Flow Matching
categories:
- Machine Learning
- Deep Learning
- Generative Models
math: true
---

## Reference

[Flow Matching For Generative Modeling](https://arxiv.org/pdf/2210.02747)

## Preliminaries

The objective of generative modeling is to learn the underlying distribution $p_{data}(x)$ of the training data $x$. This is typically achieved by training a generative model $p_{model}(x)$ to approximate $p_{data}(x)$, allowing for the generation of new samples from the learned distribution.

Let $X$ be a complete and separable metric space such as $\Omega \subseteq \mathbb R^n$. Then, the space of probability measures $\mathcal{P}(X)$ is defined as the set of all Borel probability measures on $X$. 

Generative models usually aims to learn a mapping from a simple distribution (e.g., Gaussian) to the complex data distribution $p_{data}(x)$ by transforming samples from the simple distribution into samples from the data distribution.

### Topology on $\mathcal P(X)$

#### Weak Topology 

the **weak topology** on $P(X)$ is defined as the coarsest topology making all functionals of the form 
$$
\mu \mapsto \int_{X}f\,d\mu
$$
continuous, where $f\in C_{b}(X)$ ranges over all the bounded continuous functions on $X$.

**Convergence characterization**: A sequence $(\mu_n)$
converges to $\mu$ in the weak topology if and only if
$$
\int_X f \, d\mu_n \to \int_X f \, d\mu
$$
for all bounded continuous functions $f: X \to \mathbb{R}$.

Actually, let $\mathcal M(X)$ be the space of all signed, finite measure on $X$. $C_b(X)$ can be seen as the dual space induced by the pairing:
$$
\langle f,\mu \rangle = \int_{X}f\,d\mu
$$

The relation between a measure on $X$ and a linear functional (dual space) on $C_b(X)$:
$$
\begin{align*}
\Phi:\mathcal{M}(X) &\to C_b(X)^*\\
\mu&\mapsto [f\mapsto \int_X f\, d\mu] = \langle f, \mu \rangle
\end{align*}
$$

$\Phi$ is linear and injective.

When $X$ is compact, $\Phi$ is bijective by the Riesz representation theorem. $\mathcal{M}(X)\cong C_b(X)^*$.

When $X$ is locally compact and separable, $\mathcal{M}(X)\cong C_0(X)^*$, where $C_0(X)$ is the compactly support function on $X$, which vanishing on infinity.

And we have the **weak*** topology on the dual space of $C_b(X)$
$$
L_n\overset{w^*}{\to} L \iff L_n(f)\to L(f),\quad \forall f\in C_b(X)
$$

So $\Phi$ induces the weak* topology on $\mathcal M(X)$ and its subset $P(X)$.

#### Wasserstein Metric Topology

When $X$ is a compact Riemann manifold $(M,g)$, the $p$-Wasserstein distance is a metric on $P(X)$, and induced a topology on $M$.

**Kantorovich-Rubinstein** duality states that for any $p$-Wasserstein distance, we have:

$$
W_1(\mu, \nu) = \sup_{f \in \text{Lip}_1(M)} \left( \int_M f \, d\mu - \int_M f \, d\nu \right)
$$
where $W_1(\mu, \nu)$ is the $1$-Wasserstein distance.

This plays an important role in WGAN.

So $1$-Wasserstein convergence is equivalent to convergence over all Lipschitz functions $f$, which is stronger than weak convergence (all $C_b$ functions).

Generally, we have 
$$
W_n(\mu, \nu)\leq W_m(\mu, \nu),\forall n\leq m
$$

So $2$-Wasserstein topology is weaker than $1$-Wasserstein metric topology, etc.

But if we assume all the probability measure has compact support, the convergence is equivalent.

---

Let $\mu \in \mathcal{P}(X)$ be the simple distribution, e.g. $\mu=\mathcal{N}(0, I)$, and let $\nu \in \mathcal{P}(X)$ be the data distribution. The goal is to learn a mapping $f: X \to X$ such that $f_{*}\mu = \nu$, where $f_{*}\mu$ is the pushforward measure of $\mu$ under the mapping $f$ defined by $f_{*}\mu(A) := \mu(f^{-1}(A))$, where $A\in \mathcal{B}_X$ is a Borel set.
    
$$
\begin{align*}
f_{*}: \mathcal{P}(X) &\to \mathcal{P}(X) \\
\mu &\mapsto f_{*}\mu = \nu
\end{align*}
$$

But in reality, we only have the samples from the data distribution $\nu$. So the best we can do is to approximate $\nu$ by 
$$
\nu \approx \frac{1}{N}\sum_{i=1}^{N} \delta_{x_i}
$$
where $x_i\in X$ is the sample data and $\delta_{x_i}$ is the Dirac measure centered at $x_i$.

This can be viewed as a semi-discrete optimal transport problem, where we aim to find a mapping $f: X \to X$ that pushes forward the simple distribution $\mu$ to the empirical distribution $\nu$.

### Data Manifold Hypothesis
We view the real distribution as a distribution over $\mathcal{M}$, where $\mathcal{M}$ is a manifold embedded in $\mathbb{R}^n$ and $n$ is the dimensionality of the data. More precisely, we assume that the data distribution $\nu$ has support on a lower-dimensional manifold $\mathcal{M} \subset \mathbb{R}^n$, typically with $\dim(\mathcal{M}) \ll n$. This is the data manifold hypothesis, which posits that high-dimensional data often lies on or near a lower-dimensional manifold.

### Theoretical Framework vs. Practical Implementation

**Theoretical assumption**: 

The true data distribution $p_{data}(x)$ is absolutely continuous with respect to the volume measure on the data manifold $\mathcal{M}$, i.e., there exists a smooth density function $\rho_{data}: \mathcal{M} \to \mathbb{R}_+$ such that:

$$
d\nu = \rho_{data}(x) \, d\text{vol}_\mathcal{M}(x)
$$

**Practical reality**: 

We only observe finite samples $\{x_i\}_{i=1}^N$ from the true distribution, leading to the empirical distribution:

$$
\hat{\nu}_N = \frac{1}{N}\sum_{i=1}^{N} \delta_{x_i}
$$
Note that each Dirac measure $\delta_{x_i}$ is not absolutely continuous with respect to the volume measure, since $\{x_i\}$ has zero volume but $\delta_{x_i}(\{x_i\}) = 1$.

Regularization assumption: We assume that the empirical distribution $\hat{\nu}_N$ converges weakly to the true continuous distribution $\nu$ as $N \to \infty$:

$$
\hat{\nu}_N \rightharpoonup \nu \quad \text{as } N \to \infty
$$
This weak convergence justifies treating the discrete empirical distribution as an approximation to the underlying continuous distribution.

But we can instead use a Gaussian kernel to smooth the empirical distribution:

$$
\tilde{\nu}_N = \frac{1}{N}\sum_{i=1}^{N} K_\sigma(x - x_i) \delta_{x_i}
$$
where $K_\sigma(x)$ is a Gaussian kernel with bandwidth $\sigma$.

We shall see this is the choice of the flow matching.

### Normalizing Flow

We learn a bijective mapping $f: \mathbb{R}^n \to \mathbb{R}^n$ such that the pushforward measure $f_{*}\mu$ matches the target distribution $\nu$.

The Normalizing Flow method assumes that $f$ can be expressed as a sequence of invertible transformations, typically using neural networks, such that the Jacobian determinant can be efficiently computed. 

$$ 
y \sim \nu, y = f(x), x \sim \mu= \mathcal{N}(0, I)
$$

$$
\begin{align*}
f &= f_n \cdots \circ f_1\\

x_1 &= f_1(x) \\
x_2 &= f_2(x_1) \\
\vdots \\
y = x_n &= f_n(x_{n-1})
\end{align*}
$$

And each $f_i$ is a neural network parameterized by $\theta_i$, i.e., $f_i = f_i(x; \theta_i)$.

#### Loss and Training

The objective is maximum likelihood estimation or KL divergence, equivalently.

By the change of variables formula, we have:
$$
p_Y(y) = p_X(f^{-1}(y)) \left| \det \frac{\partial f^{-1}}{\partial y} \right|
$$

$$
\begin{align*}
\mathcal L(\theta) &= \frac{1}{N}\sum_{i=1}^N \log p_Y(y_i) \\
&= \frac{1}{N}\sum_{i=1}^N \log p_X(f^{-1}(y_i)) \left| \det \frac{\partial f^{-1}}{\partial y_i} \right|\\
\end{align*}
$$

## Continuous Normalizing Flows, Flow Matching

Let $n\to \infty$, it is equivalent to construct a vector field (and corresponding ODE) and take the transform process as particle flows.

Given a time dependent transformation
$$
\begin{align*}
g: [0,1] \times \mathbb \Omega &\to \mathbb \Omega\\
(t,x) &\mapsto g(t,x)
\end{align*}
$$

The flow $g$ induces a time-dependent one parameter group of diffeomorphisms:
$$
g_t(x) = g(t,x)
$$
And the trajectory of a particle $x$ is denoted as $\gamma_x(t)$:
$$
\gamma_x(t) = g(t,x) = g_t(x)
$$
Note that here particle $x$ means it was initially at position $x$ at time $t=0$.

We want a gradual transformation of the input distribution to the target distribution, i.e.
$$
g_0 = \text{id}, g_1 = T
$$
And $T$ satisfies $T_* \mu = \nu$.

Fix a position $y$ and describe the speed of the particles at $y$, we can define the velocity field $v_t(y)$ as:
$$
v_t(y) = \frac{\partial}{\partial t} g(t, x)\bigg|_{g(t,x)=y} = \frac{\partial g}{\partial t}(t, g_t^{-1}(y))
$$

And the Eulerian description and Lagrangian description are related as follows:
Eulerian:
$$
\begin{cases}
\frac{d}{dt}g_t(x) = v_t(g_t(x)),\\
g_0(x)=x.
\end{cases}
$$
Lagrangian:
$$
\begin{cases}
\frac{d}{dt}\gamma_x(t) = v_t(\gamma_x(t)),\\
\gamma_x(0)=x.
\end{cases}
$$

We shall now consider the conservation of mass, i.e., we want to ensure that the flow preserves the total mass $1$ of the distribution.

We thus introduce a density function $\rho$ induced by the flow
$$
\begin{align*}
\rho(t,x): [0,1] \times \Omega &\to \mathbb{R}\\
(t,x) &\mapsto \rho(t,x)
\end{align*}
$$
And denote $\rho_t(x) = \rho(t,x)$.

We require
$$
\rho_t = (g_t)_*\rho_0
$$
Particularly, $\rho_0 = \mu, \rho_1 = \nu$.

## Rectified Flow Matching

TODO

