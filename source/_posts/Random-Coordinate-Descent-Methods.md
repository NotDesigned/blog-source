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
\mathsf{QP}(c):&&\min&&\sqrt c \cdot \sum_{u\in V}\left(\sum_{(u,v) \in E}\alpha_{u,v}\right)^2+\frac 1{\sqrt c} \cdot & \sum_{v\in V}\left(\sum_{(u,v) \in E}\beta_{v,u}\right)^2 \\
&& \text{s.t.}&&\alpha_{u,v}+\beta_{v,u} &=1,&& \forall (u,v) \in E\\
&& &&\alpha_{u,v},\beta_{v,u} &\geq 0 && \forall (u,v) \in E
\end{align*}
$$

The dual of such quadratic program is formulated as follows:

### Lagrangian relaxation
$$
\begin{align*}
L = & \sqrt{c} \sum_{u\in V}\left(\sum_{(u,v) \in E}\alpha_{u,v}\right)^2 + \frac{1}{\sqrt{c}} \sum_{v\in V}\left(\sum_{(u,v) \in E}\beta_{v,u}\right)^2 \\
& + \sum_{(u,v) \in E} \lambda_{u,v}(1 - \alpha_{u,v} - \beta_{v,u}) \\
& - \sum_{(u,v) \in E} \rho_{u,v} \alpha_{u,v} - \sum_{(u,v) \in E} \sigma_{v,u} \beta_{v,u}
\end{align*}
$$

Now consider the KKT conditions:

### KKT Conditions

#### Stationarity Conditions
$$
\begin{align*}
\frac{\partial L}{\partial \alpha_{u,v}} &= 2\sqrt{c} \sum_{(u,s) \in E}\alpha_{u,s} - \lambda_{u,v} - \rho_{u,v} = 0 \\
\frac{\partial L}{\partial \beta_{v,u}} &= \frac{2}{\sqrt{c}} \sum_{(t,v) \in E}\beta_{v,t} - \lambda_{u,v} - \sigma_{v,u} = 0
\end{align*}
$$

Define:
$$
w_{\alpha}(u) = \sum_{(u,v) \in E}\alpha_{u,v}, \quad w_{\beta}(v) = \sum_{(u,v) \in E}\beta_{v,u}
$$

Then:
$$
\begin{align*}
2\sqrt{c} w_{\alpha}(u) - \lambda_{u,v} - \rho_{u,v} &= 0\\
\frac{2}{\sqrt{c}} w_{\beta}(v) - \lambda_{u,v} - \sigma_{v,u} &= 0
\end{align*}
$$
$$
\begin{align*}
\Rightarrow 
\begin{cases}
\lambda_{u,v} + \rho_{u,v} &= 2\sqrt{c} \sum_{(u,s) \in E}\alpha_{u,s}\\
\lambda_{u,v} + \sigma_{v,u} &= \frac{2}{\sqrt{c}} \sum_{(t,v) \in E}\beta_{v,t}
\end{cases}
\end{align*}
$$

#### Primal Feasibility Conditions
$$
\begin{align*}
\alpha_{u,v} + \beta_{v,u} &= 1, && \forall (u,v) \in E \\
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

From the stationarity conditions, we notice that for any vertex $u$, the quantity $2\sqrt{c} \sum_{(u,s) \in E}\alpha_{u,s}$ must be equal to $\lambda_{u,v} + \rho_{u,v}$ for **all edges** $(u,v)$ incident to $u$. This gives us:

$$\lambda_{u,v_1} + \rho_{u,v_1} = \lambda_{u,v_2} + \rho_{u,v_2} = \cdots$$ 

for all edges $(u,v_1), (u,v_2), \ldots$ incident to vertex $u$.

Similarly, for any vertex $v$:
$$\lambda_{u_1,v} + \sigma_{v,u_1} = \lambda_{u_2,v} + \sigma_{v,u_2} = \cdots$$

Let us define:
$$x_u := 2\sqrt{c} \sum_{(u,s) \in E}\alpha_{u,s}, \quad y_v := \frac{2}{\sqrt{c}} \sum_{(t,v) \in E}\beta_{v,t}$$

Then we have:
$$\lambda_{u,v} + \rho_{u,v} = x_u, \quad \lambda_{u,v} + \sigma_{v,u} = y_v$$

Substituting stationarity conditions back to Lagrangian:
$$
\begin{align*}
\inf_{\alpha,\beta,w}L(\alpha,\beta,w,\lambda,\rho,\sigma) = & \sqrt{c} \sum_{u\in V}\left(\sum_{(u,v) \in E}\alpha_{u,v}\right)^2 + \frac{1}{\sqrt{c}} \sum_{v\in V}\left(\sum_{(u,v) \in E}\beta_{v,u}\right)^2 \\
& + \sum_{(u,v) \in E} \lambda_{u,v}(1 - \alpha_{u,v} - \beta_{v,u}) \\
& - \sum_{(u,v) \in E} \rho_{u,v} \alpha_{u,v} - \sum_{(u,v) \in E} \sigma_{v,u} \beta_{v,u}\\
= & \sqrt{c} \sum_{u\in V}\left(\sum_{(u,v) \in E}\alpha_{u,v}\right)^2 + \frac{1}{\sqrt{c}} \sum_{v\in V}\left(\sum_{(u,v) \in E}\beta_{v,u}\right)^2 \\
& + \sum_{(u,v) \in E} \lambda_{u,v} \\
& - \sum_{(u,v) \in E} (\lambda_{u,v} + \rho_{u,v}) \alpha_{u,v} - \sum_{(u,v) \in E} (\lambda_{u,v} + \sigma_{v,u}) \beta_{v,u}\\
= & \sqrt{c} \sum_{u\in V}\left(\sum_{(u,v) \in E}\alpha_{u,v}\right)^2 + \frac{1}{\sqrt{c}} \sum_{v\in V}\left(\sum_{(u,v) \in E}\beta_{v,u}\right)^2 \\
& + \sum_{(u,v) \in E} \lambda_{u,v} - \sum_{(u,v) \in E} \left(2\sqrt{c} \sum_{(u,s) \in E}\alpha_{u,s}\right) \alpha_{u,v} \\
& - \sum_{(u,v) \in E} \left(\frac{2}{\sqrt{c}} \sum_{(t,v) \in E}\beta_{v,t}\right) \beta_{v,u} \\
= & \sqrt{c} \sum_{u\in V}\left(\sum_{(u,v) \in E}\alpha_{u,v}\right)^2 + \frac{1}{\sqrt{c}} \sum_{v\in V}\left(\sum_{(u,v) \in E}\beta_{v,u}\right)^2 \\
& + \sum_{(u,v) \in E} \lambda_{u,v} - 2\sqrt{c} \sum_{u \in V} \left(\sum_{(u,v) \in E}\alpha_{u,v}\right)^2 \\
& - \frac{2}{\sqrt{c}} \sum_{v \in V} \left(\sum_{(u,v) \in E}\beta_{v,u}\right)^2 \\
= & -\sqrt{c} \sum_{u\in V}\left(\sum_{(u,v) \in E}\alpha_{u,v}\right)^2 - \frac{1}{\sqrt{c}} \sum_{v\in V}\left(\sum_{(u,v) \in E}\beta_{v,u}\right)^2 \\
& + \sum_{(u,v) \in E} \lambda_{u,v} \\
= & -\frac{1}{4\sqrt{c}} \sum_{u\in V} x_u^2 - \frac{\sqrt{c}}{4} \sum_{v\in V} y_v^2 + \sum_{(u,v) \in E} \lambda_{u,v}
\end{align*}
$$

To maximize the dual objective, we want to maximize $\lambda_{u,v}$ which means minimizing $\rho_{u,v} + \sigma_{v,u}$ subject to:

- $\rho_{u,v}, \sigma_{v,u} \geq 0$
- $\lambda_{u,v} = x_u - \rho_{u,v} = y_v - \sigma_{v,u}$

This gives us $\rho_{u,v} - \sigma_{v,u} = x_u - y_v$.

**Case 1:** If $x_u \geq y_v$, set $\sigma_{v,u} = 0$ and $\rho_{u,v} = x_u - y_v \geq 0$. Then $\lambda_{u,v} = y_v$.

**Case 2:** If $x_u \leq y_v$, set $\rho_{u,v} = 0$ and $\sigma_{v,u} = y_v - x_u \geq 0$. Then $\lambda_{u,v} = x_u$.

Therefore: $\lambda_{u,v} = \min(x_u, y_v)$.

So the dual objective function can be simplified to:
$$
\begin{align*}
\mathsf{DQP}(c) &= \max_{x,y} -\frac{1}{4\sqrt{c}} \sum_{u\in V} x_u^2 - \frac{\sqrt{c}}{4} \sum_{v\in V} y_v^2 + \sum_{(u,v) \in E} \min(x_u, y_v)
\end{align*}
$$

And the constraints are given by the primal feasibility conditions, which is equivalent to:

$$
\begin{align*}
\frac{1}{2\sqrt{c}}\sum_{u \in V} x_u + \frac{\sqrt{c}}{2}\sum_{v \in V} y_v = |E| \\
x_u, y_v \geq 0, \quad \forall u \in V, v \in V
\end{align*}
$$

Because a valid solution of primal can be constructed by using flow theory.

### Dual Quadratic Program

Set new $x=(\frac{x}{2}, \frac{y}{2})$, we can rewrite the dual quadratic program as follows:

$$
\begin{align*}
\mathsf{DQP}(c):&\min_{x\in \mathbb R^{2n}}  \frac{1}{2}x^TWx - \sum_{(u,v)\in E}\min(x_u,x_{n+v})\\
&\text{s.t. } a^Tx = |E| \\
&\quad\quad\quad x_u, y_v \geq 0, \quad \forall u \in V, v \in V \\
&\text{where } W=\begin{bmatrix}
\frac{1}{\sqrt c}I_n & 0 \\
0 & \sqrt{c}I_n
\end{bmatrix}
, \quad \vec{a} = [\frac{1}{\sqrt{c}}, \frac{1}{\sqrt{c}}, \ldots, \frac{1}{\sqrt{c}},\sqrt{c}, \sqrt{c}, \ldots, \sqrt{c}]^T
\end{align*}
$$

#### Submodular Function Representation

Let define the submodular function $F:2^{V\coprod V}\to \mathbb R$ as follows:

$$
F(S,T) = \sum_{(u,v)\in E} F_{u,v}(S,T) = -|E(S,T)|
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

the Lovász extension of $F_{u,v}$ is given by:

$$
f_{u,v}(x) = \max_{w\in B(F_{u,v})} \langle w, x \rangle
$$

which just equals to $-\min(x_u, x_{n+v})$.

Thus, the dual quadratic program can be rewritten as:
$$
\begin{align*}
\mathsf{DQP}(c):&\min_{x\in \mathbb R^{2n}}  \frac{1}{2}x^TWx + \sum_{(u,v)\in E} f_{u,v}(x)\\
&\text{s.t. } a^Tx = |E| \\
&\quad\quad\quad x_u, y_v \geq 0, \quad \forall u \in V, v \in V \\
&\text{where } W=\begin{bmatrix}
\frac{1}{\sqrt c}I_n & 0 \\
0 & \sqrt{c}I_n
\end{bmatrix}, \quad \vec{a} = [\frac{1}{\sqrt{c}}, \frac{1}{\sqrt{c}}, \ldots, \frac{1}{\sqrt{c}},\sqrt{c}, \sqrt{c}, \ldots, \sqrt{c}]^T
\end{align*}
$$

### Another Derivation 

Just ignore the quadratic term, we can rewrite the dual quadratic program as a ratio linear program (RLP):
$$
\begin{align*}
\mathsf{RLP}(c):&\max_{x\in \mathbb R^{2n}}  \sum_{(u,v)\in E}\min(x_u,x_{n+v})\\
&\text{s.t. } a^Tx = |E| \\
&\quad\quad\quad x_u, y_v \geq 0, \quad \forall u \in V, v \in V \\
&\text{where } \vec{a} = [\frac{1}{\sqrt{c}}, \frac{1}{\sqrt{c}}, \ldots, \frac{1}{\sqrt{c}},\sqrt{c}, \sqrt{c}, \ldots, \sqrt{c}]^T
\end{align*}
$$

Let's introduce the definition of c-biased density:
$$
\rho_c(S,T) = \frac{2 \sqrt c \sqrt {c'}}{c+c'}  \frac{|E(S,T)|}{\sqrt{|S||T|}}=\frac{2|E(S,T)|}{\frac{1}{\sqrt c} |S|+ \sqrt c |T|}\\
\text{where } c' = \frac{|S|}{|T|}
$$

And the c-biased DDS is defined as the subgraph G(S,T) such that $\rho_c(S,T)$ is maximized.

We have the following theorem: 
#### Theorem 1

The optimal value of $\mathsf{RLP}(c)$, $\text{OPT}({\mathsf{RLP}(c)})=\frac{|E|}{2}\rho^*_c(S,T)$ and the optimal solution can be recovered by thresholding.

##### Proof

First, we show that $\text{OPT}({\mathsf{RLP}(c)})\geq \frac{|E|}{2}\rho^*_c(S,T)$.

For any vertex sets $S \subseteq V, T \subseteq V$, consider the feasible solution to RLP:

$x_u = \alpha$ for $u \in S$, $x_u = 0$ for $u \notin S$

$x_{n+v} = \beta$ for $v \in T$, $x_{n+v} = 0$ for $v \notin T$

The constraint gives us:

$$
\frac{1}{\sqrt{c}}|S| \cdot \alpha + \sqrt{c}|T| \cdot \beta = |E|
$$

The objective value is:

$$
\sum_{(u,v) \in E} \min(x_u, x_{n+v}) = |E(S,T)| \cdot \min(\alpha, \beta)
$$

To maximize this, we want to maximize $\min(\alpha, \beta)$ subject to the constraint.

If $\alpha = \beta$, then: $\alpha = \frac{|E|}{\frac{1}{\sqrt{c}}|S| + \sqrt{c}|T|}$

Objective = $|E(S,T)| \cdot \frac{|E|}{\frac{1}{\sqrt{c}}|S| + \sqrt{c}|T|}$

This can be rewritten as:
$$
\frac{2|E(S,T)|}{\frac{1}{\sqrt{c}}|S| + \sqrt{c}|T|} \cdot \frac{|E|}{2} = \rho_c(S,T) \cdot \frac{|E|}{2}
$$

Therefore: $\text{OPT}({\mathsf{RLP}(c)}) \geq \frac{|E|}{2} \max_{S,T} \rho_c(S,T)$

Now we show that $\text{OPT}({\mathsf{RLP}(c)}) \leq \frac{|E|}{2} \max_{S,T} \rho_c(S,T)$.

The LP objective can be rewritten as:
$$
\sum_{(u,v) \in E} \min(x^*_u, y^*_v) = \sum_{(u,v) \in E} \int_0^{\infty} \mathbf{1}[x^*_u \geq t \text{ and } y^*_v \geq t] \, dt
$$

For each threshold $t \geq 0$, define:

- $S_t = \{u \in V : x^*_u \geq t\}$ (vertices in first part with value ≥ t)
- $T_t = \{v \in V : y^*_v \geq t\}$ (vertices in second part with value ≥ t)

Then the number of edges between $S_t$ and $T_t$ is given by:
$$\sum_{(u,v) \in E} \mathbf{1}[x^*_u \geq t \text{ and } y^*_v \geq t] = |E(S_t, T_t)|$$

Consider the constraint:
$$
\begin{align*}
\frac{1}{\sqrt{c}}\sum_{u} x^*_u + \sqrt{c}\sum_{v} y^*_v &= \frac{1}{\sqrt{c}}\sum_{u} \int_0^{\infty} \mathbf{1}[x^*_u \geq t] \, dt + \sqrt{c}\sum_{v} \int_0^{\infty} \mathbf{1}[y^*_v \geq t] \, dt\\
&= \int_0^{\infty} \left( \frac{1}{\sqrt{c}}|S_t| + \sqrt{c}|T_t| \right) dt
\end{align*}
$$

So 
$$
\frac{\text{LP objective}}{|E|} = \frac{\int_0^{\infty} |E(S_t, T_t)| \, dt}{\int_0^{\infty} \left( \frac{1}{\sqrt{c}}|S_t| + \sqrt{c}|T_t| \right) dt}
$$

By averaging principle:

$$
\frac{\int g(t) dt}{\int h(t) dt} \leq \max_t \frac{g(t)}{h(t)}
$$

We get:
$$
\frac{\text{LP objective}}{|E|} \leq \max_{t \geq 0} \frac{|E(S_t, T_t)|}{\frac{1}{\sqrt{c}}|S_t| + \sqrt{c}|T_t|}
$$

Thus:
$$
\text{OPT}({\mathsf{RLP}(c)}) \leq \frac{|E|}{2} \max_{S,T} \rho_c(S,T)
$$

This completes the proof of Theorem 1.

And $\text{DQP}$ can be considered as a proximal version of $\text{RLP}$.