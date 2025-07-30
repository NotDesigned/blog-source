---
title: Random Coordinate Descent Methods
date: 2025-06-14 13:18:06
tags:
- Coordinate Descent
- Optimization
categories:
- Mathematics
- Study Notes
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
\begin{align*}
F &= \sum_{i=1}^r F_i\\
F(\emptyset) &= 0
\end{align*}
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
&\quad\quad\quad x_u \geq 0, \quad \forall u \in V\\
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

Thus, with $a^Tx$ normalized from $|E|$ to $2$,

the dual quadratic program can be rewritten as:
$$
\begin{align*}
\mathsf{DQP}(c):&\min_{x\in \mathbb R^{2n}}  \frac{1}{2}x^TWx + \sum_{(u,v)\in E} f_{u,v}(x)\\
&\text{s.t. } a^Tx = 2 \\
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
&\text{s.t. } a^Tx = 2 \\
&\quad\quad\quad x_u, y_v \geq 0, \quad \forall u \in V, v \in V \\
&\text{where } \vec{a} = [\frac{1}{\sqrt{c}}, \frac{1}{\sqrt{c}}, \ldots, \frac{1}{\sqrt{c}},\sqrt{c}, \sqrt{c}, \ldots, \sqrt{c}]^T
\end{align*}
$$

Let's introduce the definition of c-biased density:
$$
\begin{gather*}
\rho_c(S,T) = \frac{2 \sqrt c \sqrt {c'}}{c+c'}  \frac{|E(S,T)|}{\sqrt{|S||T|}}=\frac{2|E(S,T)|}{\frac{1}{\sqrt c} |S|+ \sqrt c |T|}\\
\text{where } c' = \frac{|S|}{|T|}
\end{gather*}
$$

And the c-biased DDS is defined as the subgraph G(S,T) such that $\rho_c(S,T)$ is maximized.

We have the following theorem: 
#### Theorem 1

The optimal value of $\mathsf{RLP}(c)$, $\text{OPT}({\mathsf{RLP}(c)})=\rho^*_c(S,T)$ and the optimal solution can be recovered by thresholding.

##### Proof

First, we show that $\text{OPT}({\mathsf{RLP}(c)})\geq \rho^*_c(S,T)$.

For any vertex sets $S \subseteq V, T \subseteq V$, consider the feasible solution to RLP:

$x_u = \alpha$ for $u \in S$, $x_u = 0$ for $u \notin S$

$x_{n+v} = \beta$ for $v \in T$, $x_{n+v} = 0$ for $v \notin T$

The constraint gives us:

$$
\frac{1}{\sqrt{c}}|S| \cdot \alpha + \sqrt{c}|T| \cdot \beta = 2
$$

The objective value is:

$$
\sum_{(u,v) \in E} \min(x_u, x_{n+v}) = |E(S,T)| \cdot \min(\alpha, \beta)
$$

To maximize this, we want to maximize $\min(\alpha, \beta)$ subject to the constraint.

If $\alpha = \beta$, then: $\alpha = \frac{2}{\frac{1}{\sqrt{c}}|S| + \sqrt{c}|T|}$

Objective = $|E(S,T)| \cdot \frac{2}{\frac{1}{\sqrt{c}}|S| + \sqrt{c}|T|}$

This can be rewritten as:
$$
\frac{2|E(S,T)|}{\frac{1}{\sqrt{c}}|S| + \sqrt{c}|T|}  = \rho_c(S,T) 
$$

Therefore: $\text{OPT}({\mathsf{RLP}(c)}) \geq \max_{S,T} \rho_c(S,T)$

Now we show that $\text{OPT}({\mathsf{RLP}(c)}) \leq \max_{S,T} \rho_c(S,T)$.

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
\frac{\text{LP objective}}{2} = \frac{\int_0^{\infty} |E(S_t, T_t)| \, dt}{\int_0^{\infty} \left( \frac{1}{\sqrt{c}}|S_t| + \sqrt{c}|T_t| \right) dt}
$$

By averaging principle:

$$
\frac{\int g(t) dt}{\int h(t) dt} \leq \max_t \frac{g(t)}{h(t)}
$$

We get:
$$
\frac{\text{LP objective}}{2} \leq \max_{t \geq 0} \frac{|E(S_t, T_t)|}{\frac{1}{\sqrt{c}}|S_t| + \sqrt{c}|T_t|}
$$

Thus:
$$
\text{OPT}({\mathsf{RLP}(c)}) \leq \max_{S,T} \rho_c(S,T)
$$

This completes the proof of Theorem 1.

And $\text{DQP}$ can be considered as a proximal version of $\text{RLP}$.

### Complexity Analysis

Let $\kappa = \max\{\sqrt c, \frac{1}{\sqrt c}\}$,

$$
\begin{gather*}
r=m\\
y\in \R^{2n*r}=(y^{(1)},...,y^{(r)})\\
W=\begin{bmatrix}
c^{1/4}I_n & 0 \\
0 & c^{-1/4}I_n
\end{bmatrix},
S=\frac{1}{\sqrt r}[W,W,...,W]\in{\R^{2n\times{2nr}}}\\
%f=\sum_{(u,v)\in E}f_{u,v}(x)\\
g=r||Sy||^2\\
\nabla g(y)=2rS^T Sy\\
||\nabla_ig(x)-\nabla_ig(y)||\leq L_i||x^{(i)}-y^{(i)}||,\text{for all vectors }x,y\in \R^{2nr} \text{that differ only in block } i\\
L_i=2\max\{\sqrt c,\frac{1}{\sqrt c}\}=2\kappa\\
S^T S = \frac{1}{r} \begin{bmatrix}
W^T W & W^T W & \cdots & W^T W \\
W^T W & W^T W & \cdots & W^T W \\
\vdots & \vdots & \ddots & \vdots \\
W^T W & W^T W & \cdots & W^T W
\end{bmatrix}
\end{gather*}
$$

### Key Sets and Definitions

**Feasible Set:**
$Y = \prod_{i=1}^r B(F_i) = B(F_1) \times B(F_2) \times \cdots \times B(F_r)$

**Null Space:**
$Q = \{y \in \mathbb{R}^{2nr} : Sy = 0\} = \left\{y \in \mathbb{R}^{2nr} : \sum_{i=1}^r W y^{(i)} = 0\right\}$

**Optimal Solution Set:**
$E = \{y \in Y : g(y) = \min_{z \in Y} g(z)\}$

**Alternative Characterization of $E$:**
$E = \{y \in Y : d(y, Q) = d(Y, Q)\}$

where $d(y, Q) = \min_{z \in Q} \|y - z\|$ and $d(Y, Q) = \min_{y \in Y} d(y, Q)$.

**Geometric Interpretation:**

- $E$ represents the set of points in the feasible region $Y$ that are closest to satisfying the constraint $Sy = 0$
- Points in $E$ are optimal solutions to our proximal optimization problem
- The set $E$ connects to optimal solutions of the original discrete problem via thresholding

#### Analogue of theorem 2:

$$
||S(y-y^*)|| \geq \frac{1}{2nr}||y-y^*||
$$

Proof:

The result is identical to the one in the original paper.

[Proof of gemini](https://g.co/gemini/share/7a4f1299ca26)

#### Analogue of lemma 7:
Let $R \subseteq \{1, 2, \ldots, r\}$ be a random subset where each $i \in \{1, 2, \ldots, r\}$ is included independently with probability $1/r$. For vectors $x, h \in \mathbb{R}^{2nr}$, let $h_R$ be defined by $(h_R)^{(i)} = h^{(i)}$ if $i \in R$ and $(h_R)^{(i)} = 0$ otherwise. Then:
$$
E[g(x + h_R)] \leq g(x) + \frac{1}{r}\langle\nabla g(x), h\rangle + \frac{2\kappa}{r}\|h\|^2
$$

**Proof:**

We have:
$$E[g(x + h_R)] = E[r\|S(x + h_R)\|^2]$$
$$= E[r\|Sx\|^2 + r\|Sh_R\|^2 + 2r\langle Sx, Sh_R\rangle]$$
$$= g(x) + rE[\|Sh_R\|^2] + \frac{1}{r}\langle\nabla g(x), h\rangle$$

The key term to bound is:
$$E[\|Sh_R\|^2] = E\left[\left\|\frac{1}{\sqrt{r}} \sum_{i \in R} W h^{(i)}\right\|^2\right] = \frac{1}{r} E\left[\left\|W \sum_{i \in R} h^{(i)}\right\|^2\right]$$

Using the matrix norm bound:
$$E\left[\left\|W \sum_{i \in R} h^{(i)}\right\|^2\right] \leq \|W\|^2 E\left[\left\|\sum_{i \in R} h^{(i)}\right\|^2\right]$$

From standard coordinate descent analysis:
$$E\left[\left\|\sum_{i \in R} h^{(i)}\right\|^2\right] \leq \frac{2}{r} \sum_{i=1}^r \|h^{(i)}\|^2 = \frac{2}{r} \|h\|^2$$

Since $\|W\|^2 = \kappa$:
$$E[\|Sh_R\|^2] \leq \frac{1}{r} \cdot \kappa \cdot \frac{2}{r} \|h\|^2 = \frac{2\kappa}{r^2} \|h\|^2$$

Therefore:
$$E[g(x + h_R)] \leq g(x) + \frac{1}{r}\langle\nabla g(x), h\rangle + \frac{2\kappa}{r}\|h\|^2$$

#### Analogue of theorem 8:
Consider iteration $k$ of the APPROX algorithm. Let $y_k=\theta_k^2u_{k+1}+z_{k+1}$, Let $y^*=\arg \min_{y\in E} ||y-y_k||$ is the optimal solution that is closest to $y_k$.
Then we have: 
$$
E_{\xi_\ell}[g(y_{\ell+1}) - g(y^*)] \leq \frac{8r^2 \kappa}{(k - 1 + 2r)^2} \left(\left(1 - \frac{1}{r}\right)(g(y_\ell) - g(y^*)) + 2\|y_\ell - y^*\|^2\right)
$$
**Proof:**

This follows directly from applying Fercoq-Richtárik's Theorem 3 with:

- $\tau = 1$ (sampling parameter)
- $\nu_i = 4\kappa$ for each $i \in \{1, 2, \ldots, r\}$

#### Geometric Bound from Theorem 2 Analogue

We need the relationship between function values and distances. 

**Key Geometric Relationship:** For our specific objective function $g(y) = r\|Sy\|^2$, we have:
$$g(y_\ell) = g(y^*) + \langle\nabla g(y^*), y_\ell - y^*\rangle + \int_0^1 \langle\nabla g(y^* + t(y_\ell - y^*)) - \nabla g(y^*), y_\ell - y^*\rangle dt$$

This uses the fundamental theorem of calculus. For $\phi(t) = g(y^* + t(y_\ell - y^*))$:
$$g(y_\ell) - g(y^*) = \int_0^1 \phi'(t) dt = \int_0^1 \langle\nabla g(y^* + t(y_\ell - y^*)), y_\ell - y^*\rangle dt$$

Since $\nabla g(z) = 2rS^T Sz$:
$$\nabla g(y^* + t(y_\ell - y^*)) - \nabla g(y^*) = 2rS^T S \cdot t(y_\ell - y^*)$$

Therefore:
$$\int_0^1 \langle\nabla g(y^* + t(y_\ell - y^*)) - \nabla g(y^*), y_\ell - y^*\rangle dt = \int_0^1 2rt\|S(y_\ell - y^*)\|^2 dt = r\|S(y_\ell - y^*)\|^2$$

Since $y^*$ is optimal, $\langle\nabla g(y^*), y_\ell - y^*\rangle \leq 0$ (first-order optimality condition).

Thus:
$$g(y_\ell) - g(y^*) \geq r\|S(y_\ell - y^*)\|^2$$

#### Applying Theorem 2 Analog

From our Theorem 2 analog:
$$\|S(y_\ell - y^*)\| \geq \frac{1}{2nr} \|y_\ell - y^*\|$$

Therefore:
$$g(y_\ell) - g(y^*) \geq r\|S(y_\ell - y^*)\|^2 \geq \frac{r}{(2nr)^2} \|y_\ell - y^*\|^2$$
$$= \frac{1}{4n^2r} \|y_\ell - y^*\|^2$$

**Geometric Bound:**
$$\|y_\ell - y^*\|^2 \leq 4n^2r (g(y_\ell) - g(y^*))$$

#### Single Epoch Analysis

Consider epoch $\ell$. Let $y_{\ell+1}$ be the solution constructed by the APPROX algorithm after running for $T$ iterations starting with $y_\ell$ (so $z_0 = y_\ell$).

Let $y^* = \arg\min_{y \in E} \|y - y_{\ell+1}\|$ be the optimal solution closest to $y_{\ell+1}$.

By Theorem 8 Analog with $k = T$:
$$E_{\xi_\ell}[g(y_{\ell+1}) - g(y^*)] \leq \frac{8r^2 \kappa}{(T - 1 + 2r)^2} \left(\left(1 - \frac{1}{r}\right)(g(y_\ell) - g(y^*)) + 2\|y_\ell - y^*\|^2\right)$$

From Step 3:
$$\|y_\ell - y^*\|^2 \leq 4n^2r (g(y_\ell) - g(y^*))$$

Therefore:
$$E_{\xi_\ell}[g(y_{\ell+1}) - g(y^*)] \leq \frac{8r^2 \kappa}{(T - 1 + 2r)^2} \left(\left(1 - \frac{1}{r}\right) + 2 \cdot 4n^2r\right) (g(y_\ell) - g(y^*))$$

$$= \frac{8r^2 \kappa}{(T - 1 + 2r)^2} \left(1 - \frac{1}{r} + 8n^2r\right) (g(y_\ell) - g(y^*))$$

For large $n, r$, the dominant term is $8n^2r$, so:
$$E_{\xi_\ell}[g(y_{\ell+1}) - g(y^*)] \leq \frac{8r^2 \kappa \cdot 8n^2r}{(T - 1 + 2r)^2} (g(y_\ell) - g(y^*))$$

$$= \frac{64n^2r^3 \kappa}{(T - 1 + 2r)^2} (g(y_\ell) - g(y^*))$$

#### Epoch Length Calculation

To achieve a factor of $\frac{1}{2}$ improvement per epoch, we need:
$$\frac{64n^2r^3 \kappa}{(T - 1 + 2r)^2} \leq \frac{1}{2}$$

This gives us:
$$(T - 1 + 2r)^2 \geq 128n^2r^3 \kappa$$

$$T - 1 + 2r \geq 8\sqrt{2} nr^{3/2} \kappa^{1/2}$$

For large $r$, we can approximate:
$$T \geq 8\sqrt{2} nr^{3/2} \kappa^{1/2} \approx 11.31 nr^{3/2} \kappa^{1/2}$$

To ensure the bound holds robustly, we choose:
$$T = 12nr^{3/2} \kappa^{1/2}$$

This gives us:
$$E_{\xi_\ell}[g(y_{\ell+1}) - g(y^*)] \leq \frac{1}{2} (g(y_\ell) - g(y^*))$$

### Conclusion

From now on, here an iteration is defined as one pass through the entire set of edges $|E| = r = m$ for the convenience of comparison.

After $\ell$ epochs of the ACDM algorithm (equivalently, $12n\sqrt{r\kappa}\ell$ iterations), we have:
$E[g(y_{\ell+1}) - g(y^*)] \leq \frac{1}{2^{\ell+1}}(g(y_0) - g(y^*))$

where $y^* = \arg\min_{y \in E} \|y - y_{\ell+1}\|$ is the optimal solution in $E$ that is closest to $y_{\ell+1}$.

#### Theorem 5.6

Suppose $\|g(y)-g(y^*)\| \leq \epsilon$, then $\|y\|_{\infty}-\|y^*\|_{\infty} \leq 2 \sqrt {\kappa \epsilon}$

Now we can use the above result to give a guarantee for the $\epsilon$-approximation c-DDS subproblem.

To solve the $\epsilon$-approximation c-DDS subproblem, we need 
$$
\|y\|_{\infty}-\|y^*\|_{\infty} \leq \epsilon
$$

which requires:
$$
\|g(y)-g(y^*)\| \leq \frac{\epsilon^2}{4\kappa}
$$
Thus, in expectation, we need to run the algorithm for $\ell$ epochs, where:
$$
\ell = \log_2\left(\frac{g(y_0) - g(y^*)}{\epsilon^2/(4\kappa)}\right) 
$$

We upper bound $\|g(y_0) - g(y^*)\|$ by the maximum value of the objective function $g$.

Denote $\psi(G) = \max\{ \Delta^+(G),  \Delta^-(G)\}$, where $\Delta^+(G)$ and $\Delta^-(G)$ denote the maximum outdegree and indegree of directed graph $G$, respectively

We can show that $g(y) = \sqrt c \sum_{u\in V}\left(\sum_{(u,v) \in E}\alpha_{u,v}\right)^2 + \frac{1}{\sqrt{c}} \sum_{v\in V}\left(\sum_{(u,v) \in E}\beta_{v,u}\right)^2$ is upper bounded by $\psi(G)\cdot |E| \cdot \kappa$

Thus, we have:
$$\ell = \log_2\left(\frac{\psi(G)\cdot |E|}{\epsilon^2}\right)$$

So we need to run the algorithm in iterations of 
$$
T \cdot \ell = 12n\sqrt{m\kappa} \log_2\left(\frac{\psi(G)\cdot |E|}{\epsilon^2}\right)
$$

In terms of asymptotic complexity, this gives us:
$$
\mathcal{O}\left(n\sqrt {m\kappa} \log\left(\frac{\psi(G)\cdot m}{\epsilon^2}\right)\right)
$$
to solve a c-DDS subproblem.

The set of C is defined as 
$$
C=\{\frac ab | 1\leq a,b \leq n, a,b \in \mathbb{Z}^+\}
$$

For the DDS problem, define $\Phi = \sum_{c\in C} \max\{c^{\frac 14},c^{-\frac 14}\}$

Then the total complexity of solving all c-DDS subproblems is:
$$
\mathcal{O}\left(\Phi n\sqrt {m} \log\left(\frac{\psi(G)\cdot m}{\epsilon^2}\right)\right)
$$