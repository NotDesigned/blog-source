---
title: Random Coordinate Descent Methods
date: 2025-06-14 13:18:06
tags:
- Coordinate Descent
categories:
- Mathematics
- Optimization
---

This is a reading note on the paper "Random Coordinate Descent Methods for Minimizing Decomposable Submodular Functions" by A. ENE, Huy L. NGUYEN

[arxiv](https://arxiv.org/abs/1502.02643)

Knowledge prerequisite on submodular function (in Chinese):

[子模函数以及Lovasz拓展](https://notdesigned.github.io/2025/06/11/%E5%AD%90%E6%A8%A1%E5%87%BD%E6%95%B0%E4%BB%A5%E5%8F%8ALovasz%E6%8B%93%E5%B1%95/)

## Preliminaries

Let $V = \{1,2,\ldots, n\}$. 

Regarding $w\in \mathbb{R}^n$ as a modular function $w(A) = \sum_{i\in A} w_i$.

Let $F: 2^V \to \mathbb{R}$ be a submodular function of the form:
$$
F = \sum_{i=1}^r F_i\\
F(\emptyset) = 0
$$
where each $f_i$ is a simple submodular function.

Additionally, we assume minimizing $F_i+w$ is easy, i.e., there exists efficient oracles to find the minimizer of $F_i + w$ for each $i$ and $w\in \mathbb R^n$.

We want to minimize $F$.

We pivot to the minimization of the Lovász extension $f$.

Let $f$ be the Lovász extension of $F$ written as the support function of the base polytope $B(F)$:
$$
f(x) = \max_{w\in B(F)} \langle w, x \rangle
$$
where $B(F) = \{w\in \mathbb{R}^n: w(A) \leq F(A), \forall A\subseteq V, w(V)=F(V)\}$.

And let $f_i$ be the Lovász extension of $F_i$, $B_i$ be the base polytope of $F_i$.

The paper considers a proximal version of the problem:

$$
\min_{x\in \mathbb{R}^n} f(x) + \frac{1}{2}\|x\|^2 \equiv \min_{x\in \mathbb{R}^n} \sum_{i=1}^r f_i(x) + \frac{1}{2r}\|x\|^2
$$

Given an optimal solution $x$ to the proximal problem, one can obtain an optimal solution to the discrete problem by thresholding $x$ at $0$; more precisely, the optimal solution to the discrete problem is given by $A = \{i: x_i \geq 0\}$.

### Lemma 1

The dual of the proximal problem is given by:

$$
\max_{y^(1)\in B_1, \ldots, y^(r)\in B_r} -\frac{1}{2}\|\sum_{i=1}^r y^{(i)}\|^2.
$$

The primal and dual variables are related by:
$$
x = -\sum_{i=1}^r y^{(i)}.
$$

## Application on Densest Subgraph Problem

To see the definition, please refer to the [Densest Subgraph Problem](https://notdesigned.github.io/2025/06/16/Densest-Subgraph/) post.

The quadratic formulation of DDS problem is defined as follows:

$$
\begin{align*}
\mathsf{QP}(c):&&\min&&\sqrt c \cdot \sum_{u\in V}w_{\alpha}(u)^2+\frac 1{\sqrt c} \cdot & \sum_{v\in V}w_{\beta}(v)^2 \\
&& \text{s.t.}&&\alpha_{u,v}+\beta_{v,u} &=1,&& \forall (u,v) \in E\\
&& &&\sum_{(u,v) \in E}\alpha_{u,v}&=w_{\alpha}(u),&& \forall u \in V \\
&& &&\sum_{(u,v) \in E}\beta_{v,u}&=w_{\beta}(v),&& \forall v \in V \\
&& &&\alpha_{u,v},\beta_{v,u} &\geq 0 && \forall (u,v) \in E
\end{align*}
$$

The dual of such quadratic program is formulated as follows:

### Lagrangian relaxation
$$
\begin{align*}
L = & \sqrt{c} \sum_{u} w_\alpha(u)^2 + \frac{1}{\sqrt{c}} \sum_{v} w_\beta(v)^2 \\
& + \sum_{(u,v) \in E} \lambda_{u,v}(1 - \alpha_{u,v} - \beta_{v,u}) \\
& + \sum_{u \in V} \mu_u\left(w_\alpha(u) - \sum_{(u,v) \in E} \alpha_{u,v}\right) \\
& + \sum_{v \in V} \nu_v\left(w_\beta(v) - \sum_{(u,v) \in E} \beta_{v,u}\right) \\
& - \sum_{(u,v) \in E} \rho_{u,v} \alpha_{u,v} - \sum_{(u,v) \in E} \sigma_{v,u} \beta_{v,u}
\end{align*}
$$

Now consider the KKT conditions:

### KKT Conditions

#### Stationarity Conditions
$$
\begin{align*}
\frac{\partial L}{\partial \alpha_{u,v}} &= -\lambda_{u,v} - \mu_{u} - \rho_{u,v} = 0 \\
\frac{\partial L}{\partial \beta_{v,u}} &= -\lambda_{u,v} - \nu_{v} - \sigma_{v,u} = 0 \\
\frac{\partial L}{\partial w_\alpha(u)} &= 2\sqrt{c} w_\alpha(u) - \mu_u = 0 \\
\frac{\partial L}{\partial w_\beta(v)} &= \frac{2}{\sqrt{c}} w_\beta(v) - \nu_v = 0\\
\end{align*}
$$
$$
\begin{align*}
\Rightarrow 
\begin{cases}
w_{\alpha}(u) &= -\frac{\mu_u}{2\sqrt{c}} \\
w_{\beta}(v) &= -\frac{\nu_v\sqrt{c}}{2}\\
\lambda_{u,v} &= -\mu_u - \rho_{u,v} = -\nu_v - \sigma_{v,u}\\
\end{cases}
\end{align*}
$$

#### Primal Feasibility Conditions
$$
\begin{align*}
\alpha_{u,v} + \beta_{v,u} &= 1, && \forall (u,v) \in E \\
\sum_{(u,v) \in E} \alpha_{u,v} &= w_{\alpha}(u), && \forall u \in V \\
\sum_{(u,v) \in E} \beta_{v,u} &= w_{\beta}(v), && \forall v \in V \\
\alpha_{u,v}, \beta_{v,u} &\geq 0, && \forall (u,v) \in E
\end{align*}
$$

#### Dual Feasibility Conditions

$$
\begin{align*}
\rho_{u,v} &\geq 0, && \forall (u,v) \in E \\
\sigma_{v,u} &\geq 0, && \forall (u,v) \in E
\end{align*}
$$

#### Complementary Slackness Conditions
$$
\begin{align*}
\rho_{u,v} \cdot \alpha_{u,v} &= 0, && \forall (u,v) \in E \\
\sigma_{v,u} \cdot \beta_{v,u} &= 0, && \forall (u,v) \in E
\end{align*}
$$

### Analysis & Simplification

Substituting stationarity conditions back to Lagrangian:
$$
\begin{align*}
\inf_{\alpha,\beta,w}L(\alpha,\beta,w,\lambda,\mu,\nu,\rho,\sigma) = & \sqrt{c} \sum_{u} \frac{\mu_u^2}{4c} + \frac{1}{\sqrt{c}} \sum_{v} \frac{\nu_v^2c}{4} \\
& + \sum_{(u,v) \in E} \lambda_{u,v}(1 - \alpha_{u,v} - \beta_{v,u}) \\
& + \sum_{u \in V} \mu_u\left(-\frac{\mu_u}{2\sqrt{c}} - \sum_{(u,v) \in E} \alpha_{u,v}\right) \\
& + \sum_{v \in V} \nu_v\left(-\frac{\nu_v\sqrt{c}}{2} - \sum_{(u,v) \in E} \beta_{v,u}\right) \\
& - \sum_{(u,v) \in E} \rho_{u,v} \alpha_{u,v} - \sum_{(u,v) \in E} \sigma_{v,u} \beta_{v,u} \\

=& -\frac {1}{4}\sum_{u} \frac{\mu_u^2}{\sqrt{c} } - \frac{1}{4} \sum_{v} \nu_v^2\sqrt{c} + \sum_{(u,v) \in E} \lambda_{u,v}(1 - \alpha_{u,v} - \beta_{v,u}) \\
& - \sum_{u \in V} \mu_u \sum_{(u,v) \in E} \alpha_{u,v} \\
& - \sum_{v \in V} \nu_v \sum_{(u,v) \in E} \beta_{v,u} \\
& - \sum_{(u,v) \in E} \rho_{u,v} \alpha_{u,v} - \sum_{(u,v) \in E} \sigma_{v,u} \beta_{v,u} \\

=& -\frac {1}{4}\sum_{u} \frac{\mu_u^2}{\sqrt{c} } - \frac{1}{4} \sum_{v} \nu_v^2\sqrt{c} + \sum_{(u,v) \in E} \lambda_{u,v} \\
& - \sum_{(u,v) \in E} (\mu_u + \nu_v + \lambda_{u,v}) \alpha_{u,v} \\
& - \sum_{(u,v) \in E} (\mu_u + \nu_v + \lambda_{u,v}) \beta_{v,u} \\

=& -\frac {1}{4}\sum_{u} \frac{\mu_u^2}{\sqrt{c} } - \frac{1}{4} \sum_{v} \nu_v^2\sqrt{c} + \sum_{(u,v) \in E} \lambda_{u,v} \\
\end{align*}
$$

To maximize $\lambda_{u,v}$, WLOG

When $\mu_u\geq \nu_v$, we can set $\rho_{u,v} = 0$ and $\sigma_{v,u} = \mu_u-\nu_v$. Then $\lambda_{u,v} = - \mu_u - \rho_{u,v} = -\mu_u$.

When $\nu_v\geq \mu_u$, we can set $\rho_{u,v} = \nu_v - \mu_u$ and $\sigma_{v,u} = 0$. Then $\lambda_{u,v} = -\nu_v - \sigma_{v,u} = -\nu_v$.

Thus, the analysis shows that we can set $\lambda_{u,v} = -\max(\mu_u, \nu_v)$.

So the dual objective function can be simplified to:
$$
\begin{align*}
\mathsf{DQP}(c) &= \max_{\mu,\nu,\lambda} -\frac{1}{4\sqrt{c}} \sum_{u\in V} \mu_u^2  -\frac{\sqrt{c}}{4} \sum_{v\in V} \nu_v^2 - \sum_{(u,v) \in E} \max(\mu_u , \nu_v)\\
\end{align*}
$$

### Dual Quadratic Program

Set $x=(-\mu, -\nu)$, we can rewrite the dual quadratic program as follows:

$$
\begin{align*}
\mathsf{DQP}(c):&\min_{x\in \mathbb R^{2n}}  \frac{1}{4}x^TWx + \sum_{(u,v)\in E}\max(x_u,x_{n+v})\\

&\text{where } W=\begin{bmatrix}
\frac{1}{\sqrt c}I_n & 0 \\
0 & \sqrt{c}I_n
\end{bmatrix}
\end{align*}
$$

#### Submodular Function Representation

Let define the submodular function $F:2^{V\coprod V}\to \mathbb R$ as follows:

$$
F(S,T) = \sum_{(u,v)\in E} F_{u,v}(S,T) = |E(S,T)|
$$

$F_{u,v}(S,T)$ is defined as follows:

$$
F_{u,v}(S,T) = \begin{cases}
0 & \text{if } u\notin S \text{ and } v\notin T \\
0 & \text{if } u\in S \text{ and } v\notin T \\
0 & \text{if } u\notin S \text{ and } v\in T\\
-1 & \text{if } u\in S \text{ and } v\in T 
\end{cases}
$$

The base polytope $B(F_{u,v})$ is defined as follows:

$$
B(F_{u,v}) = \{w\in \mathbb{R}^{V\coprod V}: w(S,T) \leq F(S,T), \forall S\subseteq V, T\subseteq V, w(V,V)=F(V,V)\}
$$

Obviously we have $w_{j}=0$ for all $j\neq u \text{ or } {v+n}$. (Here use $v+n$ to denote the vertex corresponding to $v$ in the second copy of $V$.)

And $w_{u}+w_{v+n} = -1$.