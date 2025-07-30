---
title: Wasserstein Distance and Optimal Transport
date: 2025-07-16 20:47:31
tags: 
- Wasserstein Distance
- Optimal Transport
categories:
- Mathematics
- Study Notes
---

# Optimal Transport Problem 

## Definition

Given two probability measures $\mu$ and $\nu$ on measurable spaces $(X, \mathcal{A})$ and $(Y, \mathcal{B})$, respectively, the **Optimal Transport Problem** seeks to find a transport plan $\pi$ that minimizes the cost of transporting mass from $\mu$ to $\nu$. The cost function is typically defined as a function $c: X \times Y \to \mathbb{R}$, which quantifies the cost of moving a unit mass from point $x \in X$ to point $y \in Y$.

The problem can be formulated as follows (Monge's formulation):
$$
\begin{aligned}
\inf_{T} \int_{X} c(x, T(x)) \, d\mu(x)
\end{aligned}
$$
where $T: X \to Y$ is a transport map that pushes $\mu$ forward to $\nu$, i.e., $T_* \mu = \nu$.

Such a transport map $T$ is called an **optimal transport map** if it minimizes the cost function, although it is not always guaranteed to exist since the pushforward is not necessarily possible. 

More generally, we can express the problem as (Kantorovich's formulation):
$$
\begin{aligned}
\inf_{\pi \in \Pi(\mu, \nu)} \int_{X \times Y} c(x, y) \, d\pi(x, y)
\end{aligned}
$$
where $\Pi(\mu, \nu)$ is the set of all joint distributions of $\mu$ and $\nu$.

## Dual Formulation

The primal problem is a convex optimization problem since it is linear w.r.t. $\pi$ and the feasible set is convex. 

The constraint is that $\pi$ must be a joint distribution of $\mu$ and $\nu$, which can be expressed as:
$$
\begin{aligned}
\int_{X \times Y} f(x) \, d\gamma(x, y) &= \int_X f(x) \, d\mu(x), \quad \forall f \in C(X)\\
\int_{X \times Y} g(y) \, d\gamma(x, y) &= \int_Y g(y) \, d\nu(y), \quad \forall g \in C(Y)
\end{aligned}
$$
where $C(X)$ and $C(Y)$ are the spaces of continuous functions on $X$ and $Y$, respectively.

This form allows us to derive the dual formulation of the optimal transport problem, whose Lagrangian is given by:
$$
\begin{aligned}
L(\pi, \lambda, \mu) &= \int_{X \times Y} c(x,y) d\pi + \int_{X} \lambda(x) ( d\mu -\operatorname{Proj}_X(d\pi)) + \int_{Y} \mu(y) (d\nu - \operatorname{Proj}_Y(d\pi)) \\
&= \int_{X \times Y} \left ( c(x,y) - \lambda(x) - \mu(y) \right ) d\pi + \int_{X} \lambda(x) d\mu + \int_{Y} \mu(y) d\nu
\end{aligned}
$$

Thus the dual problem can be expressed as:
$$
\begin{aligned}
\sup_{\lambda, \mu} \inf_{\pi \in \Pi(\mu, \nu)} L(\pi, \lambda, \mu) 
\end{aligned}
$$

For any point $(x,y)$, if $c(x,y) - \lambda(x) - \mu(y) < 0$, then the value of $L(\pi, \lambda, \mu)$ can be made arbitrarily to $-\infty$ by increasing $\pi$.

Therefore, the optimal solution must satisfy:
$$
c(x,y) \geq \lambda(x) + \mu(y), \quad \forall (x,y) \in X \times Y
$$

And the dual problem now becomes:
$$
\begin{aligned}
\sup_{\lambda, \mu} \left \{ \int_{X} \lambda(x) d\mu + \int_{Y} \mu(y) d\nu : c(x,y) \geq \lambda(x) + \mu(y), \forall (x,y) \in X \times Y \right \}
\end{aligned}
$$

Or more compactly:
$$
\begin{aligned}
\sup_{\lambda, \mu} \left \{ \int_{X} \lambda(x) d\mu + \int_{Y} \mu(y) d\nu : \lambda \oplus \mu \leq c \right \}
\end{aligned}
$$

The strong duality holds under mild conditions, such $X$ and $Y$ are compact and $c$ is continuous. More generally, $c$ can be a lower semicontinuous function and exist a feasible $\pi$ with finite cost.

## Wasserstein Distance

When $X = Y, c(x,y) = d(x,y)$ is a metric, the optimal transport problem induces the **Wasserstein distance**:
$$
\begin{aligned}
W_p(\mu, \nu) = \inf_{\pi \in \Pi(\mu, \nu)} \left( \int_{X \times Y} d(x,y)^p \, d\pi(x,y) \right)^{1/p}
\end{aligned}
$$
This distance is a metric on the space of probability measures with finite $p$-th moment, and it satisfies the properties of a metric:
1. **Non-negativity**: $W_p(\mu, \nu) \geq 0$ for all $\mu, \nu$.
2. **Identity of indiscernibles**: $W_p(\mu, \nu) = 0$ if and only if $\mu = \nu$.
3. **Symmetry**: $W_p(\mu, \nu) = W_p(\nu, \mu)$.
4. **Triangle inequality**: $W_p(\mu, \nu) + W_p(\nu, \sigma) \geq W_p(\mu, \sigma)$ for all $\mu, \nu, \sigma$.
