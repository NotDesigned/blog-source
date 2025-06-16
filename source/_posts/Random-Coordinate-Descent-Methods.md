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

