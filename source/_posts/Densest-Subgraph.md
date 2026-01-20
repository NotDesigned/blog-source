---
title: Densest Subgraph
date: 2025-06-16 14:53:23
tags:
- Graph Theory
- Optimization
categories:
- Mathematics
---

Reference:

1. [A Convex-Programming Approach for Efficient Directed Densest Subgraph Discovery](https://dl.acm.org/doi/pdf/10.1145/3514221.3517837)

2. [Efficient and Scalable Directed Densest Subgraph Discovery]()


## UDS problem
### Definition
The Undirected Densest Subgraph (DS) problem is defined as follows:

$$
\max_{S\subseteq V} \frac{E(S)}{|S|}
$$

where $E(S)$ is the number of edges in the subgraph induced by $S$.
### Linear Program

TBD.

## DDS problem

### Definition

The Directed Densest Subgraph (DDS) problem is defined as follows:
$$
\max_{S,T\subseteq V} \frac{E(S,T)}{\sqrt{|S||T|}}
$$
where $E(S,T)$ is the number of edges from $S$ to $T$.

### Linear Program
The DDS problem can be formulated as the following linear program:

$$
\begin{align*}
\mathsf{LP}(c):\quad \text{max } &  \sum_{(u,v)\in E}x_{u,v} \\
\text{s.t. } & \sum_{v\in V} x_{uv} \leq s_u, \forall u\in S \\
& \sum_{u\in V} x_{uv} \leq t_v, \forall v\in T \\
& x_{uv} \geq 0, \forall (u,v)\in E\\
& \sum_{u\in V} s_u = a\sqrt c\\
& \sum_{v\in V} t_v = \frac{b}{\sqrt c}\\
& a+b = 2
\end{align*}
$$
And we have two theorems:

#### Theorem 1

For a fixed $c$, consider two arbitrary sets $P$,$Q\subseteq V$, and let $c'=\frac{|P|}{|Q|}$. Then $\text{OPT(LP(c))}\geq \frac{2\sqrt{cc'}}{c+c'}\rho(P,Q)$.

#### Theorem 2

Given a feasible solution $(x,s,t,a,b)$ to $\text{LP(c)}$, we can construct an $(S,T)$-induced subgraph $G[S,T]$ such that $\sqrt{ab}\rho(S,T)\geq \text{OPT(LP(c))}=\sum_{(u,v)\in E}x_{u,v}$.

By setting $c=\frac{|S^*|}{|T^*|}$ in  theorem 1, we have:
$$
\max_c{\text{OPT(LP(c))}}\geq \rho(S^*,T^*)
$$
 
From theorem 2, we have $\max_{c} \text{OPT(LP(c))}\leq\sqrt{ab}\rho(S,T)\leq \rho(S,T)\leq \rho(S^*,T^*)$.

So $\rho^*=\rho(S^*,T^*)=\max_{c} \text{OPT(LP(c))}$

### Dual Program

The dual program of the DDS problem can be formulated as follows:

$$
\begin{align*}
\mathsf{DP}(c):&&\min && \max_{u \in V} \{r_{\alpha}(u), r_{\beta}(u)\}\\
&& \text{s.t.} && \alpha_{u,v}+\beta_{v,u} &=1, && \forall (u,v) \in E\\
&& && 2\sqrt{c} \sum_{(u,v) \in E}\alpha_{u,v} &= r_{\alpha}(u), && \forall u \in V \\
&& && \frac{2}{\sqrt{c}} \sum_{(u,v) \in E}\beta_{v,u} &= r_{\beta}(v), && \forall v \in V \\
&& && \alpha_{u,v}, \beta_{v,u} &\geq 0 && \forall (u,v) \in E
\end{align*}
$$

### Quadratic Program

The dual program can be transformed into a quadratic program (QP) as follows:

$$
\begin{align*}
\mathsf{QP}(c):&&\min&&\sqrt c \cdot \sum_{u\in V}w_{\alpha}(u)^2+\frac 1{\sqrt c} \cdot & \sum_{v\in V}w_{\beta}(v)^2 \\
&& \text{s.t.}&&\alpha_{u,v}+\beta_{v,u} &=1,&& \forall (u,v) \in E\\
&& &&\sum_{(u,v) \in E}\alpha_{u,v}&=w_{\alpha}(u),&& \forall u \in V \\
&& &&\sum_{(u,v) \in E}\beta_{v,u}&=w_{\beta}(v),&& \forall v \in V \\
&& &&\alpha_{u,v},\beta_{v,u} &\geq 0 && \forall (u,v) \in E
\end{align*}
$$

