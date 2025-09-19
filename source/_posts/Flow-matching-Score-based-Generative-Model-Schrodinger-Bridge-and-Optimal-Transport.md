---
title: >-
  Flow matching, Score-based Generative Model, Schrödinger Bridge and Optimal
  Transport
date: 2025-09-18 20:18:45
tags:
- Stochastic Differential Equations
- Optiaml Transport
categories:
- Study Notes
- Machine Learning
- Mathematics
- Physics for AI
---

## Reference

[Denoising Diffusion Probabilistic Models](http://arxiv.org/abs/2006.11239)

[Diffusion Schrödinger Bridge Matching](https://arxiv.org/abs/2303.16852)

[Simplified Diffusion Schrödinger Bridge](https://arxiv.org/abs/2403.14623)

[Speed-accuracy relations for diffusion models: Wisdom from nonequilibrium thermodynamics and optimal transport](https://arxiv.org/abs/2407.04495)


## Introduction

Generative models resonate with two deep principles: the thermodynamics of entropy and the mathematics of optimal transport.

## Symbols and Preliminaries

### Probability Space

- $(\Omega,\mathcal F,\mathbb P)$ is a probability space.
- $(E,\mathcal E) = (\mathbb R^d, \mathcal B(\mathbb R^d))$ is the measurable space.
- A random variable is a measurable map $X:(\Omega,\mathcal F)\to(E,\mathcal E)$, i.e.
  $$
  X^{-1}(B) \in \mathcal F, \quad \forall B\in \mathcal B(\mathbb R^d).
  $$
- The distribution (pushforward measure) of $X$ is
  $$
  \mu = \mathrm{Law}(X) = \mathbb P \circ X^{-1}, \quad \mu \in \mathcal P(\mathbb R^d).
  $$
- If $\mu$ is absolutely continuous w.r.t. the Lebesgue measure $\lambda$, then there exists a density function $p(x): E\to [0,+\infty)$ such that
  $$
  d\mu(x) = p(x)\, dx.
  $$
- Notation: we write $X\sim \mu$; if $\mu$ admits a density, we often abbreviate $X\sim p(x)$.

### Stochastic Process

- A **stochastic process** is a time-parametrized family of random variables 
  $$
  \{X_t: t\in [0,T]\},\quad X_t:(\Omega, \mathcal{F}) \to (E,\mathcal{E}).
  $$
- The marginal distribution at time $t$ is
  $$
  \mu_t = \mathrm{Law}(X_t),\quad X_t\sim \mu_t.
  $$

#### Standard Brownian Motion (Wiener Process)

A $d$-dimensional **standard Brownian motion** $(W_t)_{t\ge0}$ with respect to a filtration $(\mathcal F_t)_{t\ge0}$ satisfies:
1. $W_0 = 0$ almost surely.
2. For $0\le s<t$, the increment $W_t-W_s \sim \mathcal N(0,(t-s)I_d)$.
3. The increments $W_t-W_s$ are independent of $\mathcal F_s$ (independent increments).
4. The paths $t\mapsto W_t(\omega)$ are almost surely continuous.

#### Itô Integral and Itô's Lemma

**Quadratic Variation.**  
For a continuous semimartingale $X=(X_t)_{t\ge0}$, the quadratic variation is defined as
$$
[X]_t = \lim_{\|\Pi\|\to 0} \sum_{k} (X_{t_{k+1}}-X_{t_k})^2,
$$
where $\Pi=\{0=t_0<\cdots<t_n=t\}$ is a partition of $[0,t]$ and the limit is in probability.  

For one-dimensional Brownian motion $(W_t)$,
$$
[W]_t = t, \quad \text{a.s.}
$$
In higher dimensions, for $W=(W^{(1)},\dots,W^{(d)})$,
$$
[W^{(i)},W^{(j)}]_t = 
\begin{cases}
t & i=j,\\
0 & i\ne j,
\end{cases}
$$
where $[X,Y]_t$ denotes the quadratic covariation.

---

**Itô Integral.**  
Let $(W_t)$ be a Brownian motion and let $\phi_t$ be an adapted process with 
$$
\mathbb E\int_0^T \|\phi_t\|^2 dt < \infty.
$$
Then the Itô integral is defined as
$$
\int_0^T \phi_t\, dW_t = L^2\text{-}\lim_{\|\Pi\|\to0}\sum_k \phi_{t_k}(W_{t_{k+1}}-W_{t_k}).
$$
It satisfies the **Itô isometry**:
$$
\mathbb E\left[\left(\int_0^T \phi_t\, dW_t\right)^2\right] 
= \mathbb E\int_0^T \phi_t^2\, dt.
$$

---

**Itô's Lemma (Itô formula, one-dimensional).**  
Suppose $X_t$ satisfies the SDE
$$
dX_t = f(X_t,t)\,dt + g(X_t,t)\,dW_t,
$$
and let $F:\mathbb R\times[0,T]\to\mathbb R$ be $C^{2,1}$ (twice continuously differentiable in $x$ and once in $t$). Then
$$
dF(X_t,t) = \Big(\partial_t F + f\,\partial_x F + \tfrac{1}{2} g^2\,\partial_{xx}F\Big)\,dt
+ g\,\partial_x F\, dW_t.
$$

---

**Multidimensional Itô's Lemma.**  
If $X_t\in\mathbb R^d$ satisfies
$$
dX_t = f(X_t,t)\,dt + G(X_t,t)\,dW_t, \quad G\in\mathbb R^{d\times m},
$$
and $F:\mathbb R^d\times[0,T]\to\mathbb R$ is $C^{2,1}$, then
$$
dF(X_t,t) = \Big(\partial_t F 
+ f^\top \nabla_x F 
+ \tfrac12 \mathrm{Tr}\!\big(GG^\top \nabla_x^2 F\big)\Big)dt
+ (\nabla_x F)^\top G\, dW_t.
$$

---

**Remark.**  
- The term $\tfrac12 g^2 \partial_{xx}F$ (or $\tfrac12 \mathrm{Tr}(GG^\top\nabla^2F)$ in higher dimensions) arises from the quadratic variation of Brownian motion, i.e. $[W]_t=t$.  
- This correction term is what distinguishes Itô calculus from classical calculus and is fundamental in stochastic analysis.


### Other

The Kullback–Leibler (KL) divergence between two probability measures are:
$$
D_{\mathrm{KL}}(\mu \| \nu)=\int_{E}\log\left(\frac{d\mu}{d\nu}(x)\right) d\mu(x),
$$
where the $\frac{d\mu}{d\nu}$ is the Radon-Nikodym derivative.

For KL divergence between Gaussian distribution, we have

$$
\begin{align*}
\mathrm{KL}(\mathcal{N}({\mu}_x,{\Sigma}_x)\|\mathcal{N}({\mu}_y,{\Sigma}_y))
&=\frac{1}{2}\left[\log\frac{|{\Sigma}_y|}{|{\Sigma}_x|} - d + \text{tr}({\Sigma}_y^{-1}{\Sigma}_x)
+ ({\mu}_y-{\mu}_x)^T {\Sigma}_y^{-1} ({\mu}_y-{\mu}_x)\right]
\end{align*}
$$

## Diffusion Models

### DDPM

$$
x\sim q(x)
$$

How to sample from $x$?

We define a markovian stochastic process $x_t$ as the **forward process**, such that:
$$
x_0=x\\
q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1}),
$$
where $q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t I) = \frac{1}{\sqrt{(2\pi\beta_t)^d}}\exp(-\frac{\|x_t-\sqrt{1-\beta_t}x_{t-1}\|^2}{2\beta_t})$.

We call $\{\beta_t\}_{t=1}^T$ as a variance schedule.

By induction, we know $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\quad \epsilon \sim\mathcal{N}(0,I),\quad \bar\alpha_t =\prod_{s=1}^{t}(1-\beta_s)$,

$$
x_t \sim \mathcal{N}(\sqrt{\bar \alpha_t}x_0, (1-\bar\alpha_t)I).
$$

So we can sample $x_t$ directly.

Assume $\bar\alpha_t \to 0$, we can regard
$$
x_T \sim N(0,I).
$$

Now given $x_T$, we want sample $x_0$ from $x_T$.

Calculate the posterior probability density:
$$
q(x_t|x_{t+1}) = \int q(x_{t-1}|x_{t},x_{0}) q(x_0|x_t) d x_0
$$
Which is impractical to calculate since we has to integrate over $x_0$.

Instead we consider
$$
q(x_{t-1}|x_t,x_0) = \frac{q(x_t|x_{t-1})q(x_{t-1}|x_{0})}{q(x_t|x_0)}
$$
So 
$$
q(x_{t-1}|x_t,x_0)\propto q(x_t|x_{t-1})q(x_{t-1}|x_{0})
$$
is still a Gaussian distribution.
$$
q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1};\mu,\Sigma)
$$
Recall $\Sigma^{-1}=\Sigma_{1}^{-1}+\Sigma_{2}^{-1}, \Sigma^{-1}\mu = \Sigma_1^{-1}\mu_1+\Sigma_1^{-1}\mu_2$

Note that
$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I)=\mathcal{N}(x_{t-1};\frac{1}{\sqrt{1-\beta_t}}x_{t}, \frac{\beta_t}{1-\beta_t}I)
$$
$\Sigma_1=\frac{\beta_t}{1-\beta_t} I,\mu_1=\frac{1}{\sqrt{1-\beta_t}}x_{t}$.

$\Sigma_2=(1-\bar\alpha_{t-1})I,\mu_2=\sqrt{\bar \alpha_{t-1}}x_0$.

$\Sigma_t=(\frac{1-\beta_{t}}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}})^{-1}I=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_tI=\tilde \beta_t I$

$\mu_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t(\frac{\sqrt{1-\beta_t}}{\beta_t}x_t+\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0) = \frac{\beta_t\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_t}x_0 + \frac{(1-\bar\alpha_{t-1})\sqrt{1-\beta_t}}{1-\bar\alpha_t}x_t$.

Now, to train the $p_{\theta}(x_{t-1}|x_t)$, fix the variance $\tilde \beta_t$, let model predict the $\mu_t$.

we minimize the KL divergence:
$$
L_t(\theta)=\mathrm{KL}(q(x_{t-1}|x_0,x_t)\|p_{\theta}(x_{t-1}|x_t)) \propto \mathbb{E}[\|\mu_t(x_t,x_0)-\mu_{\theta}(x_t,t)\|^2]
$$
Or let the model predict the noise $\epsilon=\frac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}$, $x_0=\frac{1}{\sqrt{\bar\alpha_t}}(x_t-\sqrt{1-\bar\alpha_t}\epsilon)$.

And minimize
$$
\mathbb E_{x_0,\epsilon,t}\left[\|\epsilon-\epsilon_{\theta}(x_t,t)\|^2\right].
$$
This reparametrization setting generally yields better performance.

### Score-based Generative Model (SGM)

We view noise injection as an SDE and learn the **score** $s_t(x) := \nabla_x \log p_t(x)$ of the noisy marginal $(p_t)_{t\in [0,1]}$. Sampling is done by integrating a reverse-time SDE (or its ODE counterpart), where the score guides the dynamics back to data.

#### Forward SDE
Let $X_t\in \mathbb R^d$ solve:
$$
dX_t = f(X_t,t) dt + g(t) dW_t, \quad t\in [0,1],
$$
with $f:\mathbb R^d\times [0,1] \to \mathbb R^d$, diffusion scale $g:[0,1]\to \mathbb R_+$. 
Denote $p_t = \mathrm{Law}(X_t)$.

---

#### DDPM as VP-SDE

Set the Variance-Preserving (VP) SDE
$$
dX_t = -\frac{1}{2}\beta(t)X_t dt + \sqrt{\beta(t)} d W_t,
$$
with $\beta(t)\geq 0$ integratable and $X_0 \sim p_{data}$.

This linear SDE has the explicit solution
$$
X_t = \sqrt{\bar \alpha_t} X_0 + \underbrace{\int_0^t \exp\left(-\frac{1}{2}\int_s^t \beta(u) du\right)\sqrt{\beta(s)} dW_s}_{\text{zero-mean Gaussian}},
$$
where $\bar\alpha_t:=\exp\left(-\int_0^t\beta(s)ds\right)$.

Hence the marginal conditional matches DDPM.

$$
X_t|X_0\sim \mathcal{N}(\sqrt{\bar\alpha_t}X_0, (1-\bar\alpha_t)I).
$$

**Claim.** The DDPM forward chain with $\beta_1,\ldots,\beta_T$ is an Euler-Maruyama discretization of the VP-SDE with piecewise-constant $\beta(t)$ and share the same marginals $q(x_t|x_0)$.

#### Variance-Exploding SDE (VE-SDE)

Alternatively, set
$$
dX_t = g(t)\,dW_t, \qquad f\equiv 0,
$$
with $g(t)=\sqrt{d[\sigma^2(t)]/dt}$ and $\sigma(0)=0$. Then
$$
X_t = X_0 + \sigma(t)Z,\quad Z\sim\mathcal N(0,I).
$$
So $p_t = p_0 * \mathcal N(0,\sigma(t)^2I)$, i.e. Gaussian smoothing.


#### Reverse-time SDE and Probability Flow ODE

The reverse SDE (Anderson, 1982) is
$$
dX_t = \big(f(X_t,t)-g(t)^2\nabla_x\log p_t(X_t)\big)dt + g(t)\,d\bar W_t,
$$
where $\bar W_t$ is a backward Wiener process. Replacing $\nabla_x\log p_t$ with a learned $s_\theta$ gives a generative sampler.

The equivalent deterministic probability flow ODE is
$$
\frac{dX_t}{dt} = f(X_t,t) - \tfrac12 g(t)^2 \nabla_x\log p_t(X_t).
$$
This is the continuous analogue of DDIM.

#### Noise Prediction vs. Score Prediction

For VP,
$$
\nabla_{x_t}\log q(x_t|x_0) = -\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t} = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
$$
Thus noise prediction and score prediction are equivalent:
$$
s_\theta(x_t,t) = -\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}.
$$

An alternative data predictor:
$$
\hat x_0(x_t) = \frac{1}{\sqrt{\bar\alpha_t}}\big(x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)\big).
$$

#### Denoising Score Matching (DSM)

Training objective:
$$
\min_\theta\; \mathbb E_{x_0,\sigma,z}\;\Big\|s_\theta(x_0+\sigma z,\sigma) + \tfrac{z}{\sigma}\Big\|^2.
$$

For VP, choosing $\sigma(t)=\sqrt{1-\bar\alpha_t}$ reduces DSM to the DDPM noise-prediction MSE.

**Derivation**

TODO: Weak form.

## Flow Matching Models

See [Flow Matching](https://notdesigned.github.io/2025/08/09/Flow-Matching/)

## Schrödinger Bridge Models

