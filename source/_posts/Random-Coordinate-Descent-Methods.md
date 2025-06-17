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
$$
\begin{align*}
\mathsf{DQP}(c):&&\min_{x\in \mathbb R^{2n}} && \frac{1}{2}x^TWx + \sum_{(u,v)\in E}\max(x_u,x_{n+v})\\

&& \text{s.t.} && x_u\geq 0, && \forall u\in V\\
&& && x_{n+v}\geq 0, && \forall v\in V\\
&& && x_u + x_{n+v} = 1, && \forall (u,v)\in E\\
\text{where } W=\begin{bmatrix}
\frac{1}{2\sqrt c}I_n & 0 \\
0 & \frac{\sqrt{c}}{2}I_n
\end{bmatrix}
\end{align*}
$$

Let define the submodular function $F:2^{V\coprod V}\to \mathbb R$ as follows:

$$
F(S,T) = \sum_{i=1}^{|E|} F_{u,v}(S,T) 
$$

$F_{u,v}(S,T)$ is defined as follows:

$$
F_{u,v}(S,T) = \begin{cases}
0 & \text{if } u\notin S \text{ and } v\notin T \\
1 & \text{if } u\in S \text{ and } v\notin T \\
1 & \text{if } u\notin S \text{ and } v\in T\\
1 & \text{if } u\in S \text{ and } v\in T 
\end{cases}
$$

The base polytope $B(F_{u,v})$ is defined as follows:

$$
B(F_{u,v}) = \{w\in \mathbb{R}^{V\coprod V}: w(S,T) \leq F(S,T), \forall S\subseteq V, T\subseteq V, w(V,V)=F(V,V)\}
$$

Obviously we have $w_{j}=0$ for all $j\neq u \text{ or } {v+n}$. (Here use $v+n$ to denote the vertex corresponding to $v$ in the second copy of $V$.)

And $w_{u}+w_{v+n} = 1$.

