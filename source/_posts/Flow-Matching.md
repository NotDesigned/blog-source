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

[Continuous Normalizing Flows](https://voletiv.github.io/docs/presentations/20200901_Mila_CNF_Vikram_Voleti.pdf)

[Flow Straight and Fast](https://arxiv.org/abs/2209.03003)

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
\mu&\mapsto [f\mapsto \int_X f\, d\mu = \langle f, \mu \rangle] 
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

When $X$ is a compact Riemann manifold $(M,g)$, the $p$-Wasserstein distance is a metric on $\mathcal{P}(X)$, and induced a topology on $M$. More precisely, for probability distribution with finite $p$-th moment, $\mathcal{P}_p(X)$.

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

### (Discrete) Normalizing Flow

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

## Continuous Normalizing Flows

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
And denote $\rho_t(x):\Omega \to \mathbb {R} $ as $\rho(t,x)$.

We require
$$
\rho_t = (g_t)_*\rho_0
$$
Particularly, $\rho_0 = \mu, \rho_1 = \nu$.

And we assume $v$ is tangent to the boundary $\partial \Omega$, and the flow remains inside of $\Omega$, thus $\rho_t \in \mathcal{P}(\Omega)$.

It must satisfy the continuity equation to make the probability mass conserve
$$
\partial_t \rho_t + \nabla \cdot (v_t \rho_t) = 0
$$

### Proof of continuity equation

---

Let $\phi\in C_b(\Omega)$, 
$$
\begin{align*}
\int_{\Omega} \partial_t \phi\rho_t\, d\Omega &=\int_{\Omega} \partial_t \phi\, d\rho_t\\
&= \frac{d}{dt} \int_{\Omega} \phi(x) \, d((g_t)_* \rho_0)  \\
&= \frac{d}{dt} \int_{\Omega} \phi(g_t^{-1})\, d\rho_0 \\
&= \int_{\Omega} \nabla \phi(g_t^{-1}) \cdot v_t(g_t^{-1}) \, d\rho_0 \\
&= \int_{\Omega} \nabla \phi \cdot v_t \, d\rho_t\\
&= -\int_{\Omega} \phi \left(\nabla\cdot v_t\rho_t \right) d\Omega
\end{align*} 
$$
By the arbitrariness of $\phi$, we get the continuity equation:
$$
\partial_t \rho_t + \nabla \cdot (v_t \rho_t) = 0
$$

We say the $(\rho_t,v_t)$ solve the continuity in distributional sense, which is called a weak solution or distributional solution.

From the Lagrangian description, we know the formula:
$$
J_t = \left |\det \frac{\partial g_t}{\partial x} \right | = \left | F_t \right|
$$
$$
\rho_t(g_t)\cdot J_t = \rho_0
$$
Take differential on both side,
$$
J_td \rho_t(g_t) + \rho_t(g_t) dJ_t = 0
$$
And 
$$
\frac{d J_t}{dt}=J_t\cdot \operatorname{tr}\left(F_t^{-1} \frac{dF_t}{dt}\right)
$$

$$
\frac{d F_t}{d t} = \frac{\partial}{\partial t}\frac{\partial g_t}{\partial x} = \frac{d}{d x} \frac{\partial g_t}{\partial t} = \frac{d}{dx} v_t(g_t) = \nabla v_t(g_t) \cdot F_t
$$
Hence
$$
\frac{d J_t}{d t}= J_t \cdot \operatorname{tr}\left(\nabla v_t(g_t)\right) = J_t \cdot( \nabla \cdot v_t(g_t))
$$

Take it back, and eliminate $J_t$
$$
d\rho_t(g_t) + \rho_t(g_t) (\nabla \cdot v_t(g_t)) = 0
$$
And $d\rho_t(g_t) = \frac{\partial \rho_t}{\partial t}(g_t) + \nabla \rho_t(g_t) \cdot v_t(g_t)$.
$$
\frac{\partial \rho_t}{\partial t}(g_t) + \nabla \rho_t(g_t) \cdot v_t(g_t) + \rho_t(g_t) (\nabla \cdot v_t(g_t)) = 0
$$
$$
\frac{\partial \rho_t}{\partial t}(g_t) + \nabla \cdot (v_t(g_t) \rho_t(g_t)) = 0
$$
since $g_t$ is a diffeomorphism, we have:
$$
\frac{\partial \rho_t}{\partial t} + \nabla \cdot (v_t \rho_t) = 0
$$

Note: If we further assume the incompressibility condition, i.e.
$$
\frac{\partial \rho_t}{\partial t} = 0
$$
then we have
$$
\nabla \cdot v_t = 0
$$

---

So we know 
$$
\frac{\partial \rho_t}{\partial t} (x)= -\nabla \cdot (v_t(x) \cdot \rho_t(x))
$$

Now we focus on a particle is initially at $x$, moving on trajectory $x_t=\gamma_x(t)$

$$
\begin{align*}
\frac{d}{dt} \rho_t(\gamma_x(t)) &= \frac{\partial \rho_t}{\partial t}(\gamma_x(t)) + \nabla \rho_t(\gamma_x(t)) \cdot \frac{d}{dt} \gamma_x(t) \\
&= \frac{\partial \rho_t}{\partial t}(\gamma_x(t)) + \nabla \rho_t(\gamma_x(t)) \cdot v_t(\gamma_x(t)) \\
&= -\nabla \cdot (v_t\rho_t)(\gamma_x(t)) + \nabla \rho_t(\gamma_x(t)) \cdot v_t(\gamma_x(t))\\
&= -\rho_t(\gamma_x(t))\nabla \cdot v_t(\gamma_x(t))
\end{align*}
$$

So 
$$
\frac{d}{dt}\log \rho_t(\gamma_x(t)) = \frac{1}{\rho_t(\gamma_x(t))} \frac{d}{dt} \rho_t(\gamma_x(t)) = - \nabla \cdot v_t(\gamma_x(t))
$$
Thus
$$
\log \rho_{1}(x_1) - \log \rho_{0}(x_0)= -\int_0^1 \nabla \cdot v_t(x_t) dt
$$

### Loss and Training

We simulate the flow by using a neural network $g_\theta$ with parameter $\theta$.

Similar to the discrete normalizing flow, we use MLE:

$$
\begin{align*}
\mathcal L(\theta)&= -\mathbb{E}_{x_1\sim \nu}\left[\log \rho_{1,\theta}(x_1)\right] \\
&= -\mathbb{E}_{x_1\sim\nu}\left[\log \mu(x_0)-\int_0^1 \nabla \cdot v_{t,\theta}(x_t)\right]
\end{align*}
$$

Or minimize the KL divergence between $\rho_{1,\theta}$ and $\nu$.

## Flow Matching

Instead of learning $g$, flow matching directly learns the velocity field $v_{t,\theta}(x)$.

The theoretical objective is:
$$
\mathcal{L}_{FM}=\mathbb E_{t\sim\mathcal{U}[0,1],x_t\sim \rho_t} \|v_{t,\theta}(x_t)-v_t(x_t)\|^2
$$
But this cannot be trained directly since we don't know $v_t$.

### Conditional Flow Matching

conditioning on $x_1$, CFM use the conditional distribution $\rho_t(x_t|x_1)$

Key theorem:

Given conditional probability path $\rho_t(x_t|x_1)$ satisfying the continuity equation with conditional vector field $v_t(x_t| x_1)$.

Then the marginal vector field $v_t$:
$$
v_t(x)=\int v_t(x|x_1) \frac{\rho_t(x|x_1)\nu(x_1)}{\rho_t(x)} d{x_1}
$$
satisfy the continuity equation and generate the marginal probability path $\rho_t(x_t)$.

And one can choose any conditional probability path as long as
$$
\rho_0(x|x_1) = \mu, \rho_1(x|x_1) = \delta(x-x_1)
$$
And the paper use Gaussian kernal as an approximation.

If the above condition is satisfied, the paper proves that optimizing the objective of CFM is equivalent to optimizing the objective of the original flow model.

$$
\mathcal L_{CFM} = \mathbb{E}_{t\sim\mathcal{U}[0,1],x_1\sim\nu,x_t\sim\rho_t(x_t|x_1)} \left[\|v_{t,\theta}(x_t|x_1) - v_t(x_t|x_1)\|^2\right]
$$

And $\nabla_{\theta} \mathcal L_{CFM}= \nabla_{\theta} \mathcal L_{FM}$.

And you can also condition on $x_0$ or both $x_0$ and $x_1$.

## Rectified Flow

In the paper, **rectified flow** optimized the velocity field $v_t$ by minimizing the following objective:

$$
\mathcal{L}_{RF}=\int_0^1 \mathbb{E} \left[\|(X_1-X_0)-v(X_t,t)\|^2 \right]\, \text{d}t,\, \text{with}\,\, X_t=tX_1+(1-t)X_0
$$
where $X_0\sim \mu, X_1\sim \nu$.

First, lets translate this into the CFM language.

Conditioning on $x_0$ and $x_1$, rectified flow defined a deterministic path as the interpolation between $x_0$ and $x_1$:
$$
\begin{align*}
g_t(x|x_0,x_1) &= t x_1 + (1-t)x_0\\
v_t(x|x_0,x_1) &= \partial_t g_t(x|x_0,x_1) = x_1 - x_0\\
\rho_t(x|x_0,x_1) &= (g_t(x|x_0,x_1))_* \rho_0(x|x_0,x_1)\\
&= (g_t(x|x_0,x_1))_* \delta(x - x_0) \\
&= \delta(g_t^{-1}(x|x_0,x_1) - x_0) \left| \det \frac{\partial g_t^{-1}}{\partial x} \right|\\
&= \delta(\frac{x-tx_1}{1-t} - x_0) \frac{1}{(1-t)^n}\\
&= \delta\left(\frac{x-tx_1-(1-t)x_0}{1-t}\right) \frac{1}{(1-t)^n}\\
&= \delta\left(x-tx_1-(1-t)x_0\right) (1-t)^{n} \frac{1}{(1-t)^n}\\
&= \delta\left(x-tx_1-(1-t)x_0\right)\\
\end{align*}
$$

And apply the CFM objective:
$$
\begin{align*}
\mathcal{L}_{RF} &= \mathbb{E}_{t\sim\mathcal{U}[0,1],x_0\sim\mu,x_1\sim\nu,x_t\sim\rho_t(x|x_0,x_1)} \left[\|v_{t,\theta}(x_t|x_0,x_1) - v_t(x_t|x_0,x_1)\|^2\right]\\
&= \int_0^1 \mathbb{E}_{x_0\sim\mu,x_1\sim\nu,x_t\sim\rho_t(x|x_0,x_1)} \left[\|v_{t,\theta}(x_t|x_0,x_1) - (x_1-x_0)\|^2\right]\, \text{d}t\\
&= \int_0^1 \mathbb{E}_{x_0\sim\mu,x_1\sim\nu} \left[ v_{t,\theta}(x_t|x_0,x_1) - (x_1-x_0)\right]^2 \, \text{d}t\\
\end{align*}
$$
And this is exactly the objective of rectified flow.

Actually, the author of [Flow Matching For Generative Modeling](https://arxiv.org/pdf/2210.02747) also proposed an identical objective, but they use the name "OT-Flow" instead of "rectified flow". We shall see the reason in the next section.

### Relation to Optimal Transport

What does the rectified flow and conditional flow matching have to do with optimal transport? Let's recall the method of conditional probability path. 

The claim is that to get the vector field $v_t$, we can use the conditional vector field $\rho_t(x_t|x_0,x_1)$ and do summation over all $x_0$ and $x_1$. Then we shall get the vector field $v_t$ that satisfies the continuity equation and generates the marginal probability path $\rho_t(x_t)$.

It is not difficult to see that summing over the conditional vector field will yield a valid vector field that satisfies the continuity equation and transport the distribution from $\mu$ to $\nu$. But what is the property of such vector field?

Let's think about a principle in traditional physics, the **principle of least action**. It states that the path taken by a system between two states is the one for which the action is stationary (usually minimized).

In the context of traditional physics, the shortest path between two points is a straight line. But in quantum mechanics, the particle is no longer deterministic as a point, but rather a wave function that can be spread out over a region of space assigning a probability amplitude to each point. But the principle of least action still holds, and the path taken by the particle is the one that minimizes the action.

### Benamou-Brenier Theory

The Benamou-Brenier theory states that the optimal transport problem can be reformulated as a dynamic problem, where the optimal transport plan is the one that minimizes the action functional.

In every time $t$, we have a distribution $\rho_t$ and a velocity field $v_t$. The kinetic energy is given by the integral of the squared velocity field:
$$
E(t) = \int_{\Omega} \frac{1}{2}\|v_t(x)\|^2 \rho_t(x) \, dx
$$

The action functional is then defined as:
$$
A(\rho, v) = \int_0^1 E(t) \, dt
= \int_0^1 \int_{\Omega}\frac{1}{2}\|v_t(x)\|^2 \rho_t(x) \, dx \, dt
$$

And we see the action functional is exactly the total energy of using the velocity field $v_t$ to transport the distribution $\rho_t$ from $\mu$ to $\nu$.

**Problem** (Benamou-Brenier): Given two probability measures $\mu$ and $\nu$, find the optimal transport plan that minimizes the action functional $A(\rho, v)$ subject to some conditions.
$$
\begin{align*}
\text{min} \quad & A(\rho, v) \\
\text{s.t.} \quad &(\rho,v)\in V(\mu,\nu)
\end{align*}
$$
**Admissible pairs** $V(\mu,\nu)$ are pairs $(\rho, v)$ such that:
- $\rho_0 = \mu$, $\rho_1 = \nu$,
- $\partial_t \rho_t + \nabla \cdot (v_t \rho_t) = 0$ in distributional sense,
- $\bigcup_{t\in[0,1]} \text{supp}(\rho_t) \subseteq \Omega$,
- $v \in L^2(d\rho_t(x) dt)$, i.e., the velocity field is square integrable with respect to the measure $\rho_t$.
- $\rho \in C([0,1]; w^*\text{-}P_{ac}(\mathbb \Omega))$, i.e., the probability path is absolutely continuous with respect to the weak* topology on the space of probability measures.

**Theorem**: (Benamou-Brenier)

The $2$-Wasserstein distance between two probability measures $\mu$ and $\nu$ can be expressed as the infimum of the action functional over all admissible pairs $(\rho, v)$
$$
\mathcal{T}_2 (\mu, \nu) = \inf_{(\rho, v) \in V(\mu, \nu)} A(\rho, v)
$$

This means that the optimal transport plan is the one that minimizes the action functional, which is equivalent to minimizing the total energy of the velocity field.

#### Variational Calculus

$$
\begin{align*}
\min A(\rho, v) &= \int_0^1 \int_{\Omega} \frac{1}{2}\|v_t(x)\|^2 \rho_t(x) \, dx \, dt \\
\text{s.t.} \quad & \partial_t \rho_t + \nabla \cdot (v_t \rho_t) = 0 \\
& \rho_0 = \mu, \rho_1 = \nu
\end{align*}
$$
We introduce a Lagrange multiplier $\phi_t(x)$ to enforce the continuity equation constraint:
$$
\begin{align*}
\mathcal{J}(\rho, v, \phi) &=\int_0^1 \int_{\Omega} \left[\frac{1}{2}\|v_t(x)\|^2 \rho_t(x)+\phi_t(x) \left( \partial_t \rho_t(x) + \nabla \cdot (v_t(x) \rho_t(x)) \right) \right] \,dx \, dt
\end{align*}
$$

**Variation with respect to $v$**:
$$
\frac{\delta \mathcal{J}}{\delta v_t} = \int_0^1 \int_{\Omega} \left[ v_t\cdot \delta v_t \rho_t + \phi_t \nabla \cdot (\delta v_t \rho_t) \right] \, dx \, dt
$$
Using integration by parts on the second term
$$
\begin{align*}
\int_{\Omega} \phi_t \nabla \cdot (\delta v_t \rho_t) \, dx &= \int_{\Omega} \nabla \cdot (\phi_t \rho_t \delta v_t) \, dx - \int_{\Omega} \nabla \phi_t \cdot (\delta v_t \rho_t) \, dx\\
&=\int_{\partial \Omega} \phi_t \rho_t \delta v_t \cdot n \, dS - \int_{\Omega} \nabla \phi_t \cdot (\delta v_t \rho_t) \, dx\\
&= - \int_{\Omega} \nabla \phi_t \cdot (\delta v_t \rho_t) \, dx
\end{align*}
$$
where the boundary term vanishes due to the assumption that $v$ is tangent to the boundary $\partial \Omega$.

we get
$$
\frac{\delta \mathcal{J}}{\delta v_t} = \int_0^1 \int_{\Omega} \left[ v_t\rho_t - \rho_t\nabla \phi_t \right] \cdot \delta v_t  \, dx \,dt
$$

Which forces
$$
v_t =  \nabla \phi_t
$$

**Variation with respect to $\rho$**:
$$
\frac{\delta \mathcal{J}}{\delta \rho_t} = \int_0^1 \int_{\Omega} \left[ \frac{1}{2}\|v_t(x)\|^2 + \phi_t(x) \partial_t \delta \rho_t(x) + \phi_t(x) \nabla \cdot (v_t(x) \delta \rho_t(x)) \right] \, dx \, dt
$$
Similarly, using integration by parts on the second term and the third term
$$
\begin{align*}
\int_{\Omega} \phi_t\partial_t\delta\rho_t \, d\Omega &= -\int_{\Omega} \partial_t\phi_t\delta\rho_t\, d\Omega\\
\int_{\Omega} \phi_t \nabla \cdot (v_t \delta \rho_t)\, d\Omega &= - \int_{\Omega} \nabla \phi_t \cdot (v_t \delta \rho_t) \, d\Omega
\end{align*}
$$
Note $\rho_0 = \mu$ and $\rho_1 = \nu$ are fixed, so variations $\delta \rho_t$ are zero at the endpoints $(\delta \rho_0 = \delta \rho_1 = 0)$. 

we get
$$
\begin{align*}
\frac{\delta \mathcal{J}}{\delta \rho_t} &= \int_0^1 \int_{\Omega} \left[ \frac{1}{2}\|v_t(x)\|^2 - \partial_t\phi_t(x) - \nabla \phi_t(x) \cdot v_t(x) \right] \delta\rho_t(x) \, dx \, dt
\end{align*}
$$
Setting this to zero for all $\delta \rho_t(x)$, we have
$$
\frac{1}{2}\|v_t(x)\|^2 - \partial_t\phi_t(x) - \nabla \phi_t(x) \cdot v_t(x) = 0
$$

Substituting $v_t = \nabla \phi_t$, we have
$$
\partial_t \phi_t(x) + \frac{1}{2}\|v_t(x)\|^2 = 0
$$

To summarize, we have three equations for the action functional, which gives us the necessary conditions for optimality:
$$
\begin{cases}
v_t = \nabla \phi_t, \quad \text{Velocity field as gradient of potential} \\ 
\partial_t \phi_t(x) + \frac{1}{2}\|v_t(x)\|^2 = 0, \quad \text{Hamilton-Jacobi equation} \\
\partial_t \rho_t + \nabla \cdot (v_t \rho_t) = 0. \quad \text{Continuity equation for mass conservation}
\end{cases}
$$