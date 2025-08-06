---
title: Differential Manifold
date: 2025-08-04 10:28:20
tags:
- Mathematics
categories:
- Mathematics
- Study Notes
math: true
---

## References

- John Lee, Introduction to Smooth Manifolds

- Loring Tu, An Introduction to Manifolds

- Victor Guillemin and Alan Pollack, Differential Topology

For Chinese reader, you can refer to the website http://staff.ustc.edu.cn/~wangzuoq/Courses/23F-Manifolds/#notes for Chinese course note in USTC.

## Manifolds and Smooth maps

### Topological Manifolds

Let $M$ be a **Hausdorff** and **second countable** topological space that is locally homeomorphic to Euclidean space $\mathbb{R}^n$. We say that $M$ is a **manifold of dimension $n$** if $\forall p \in M, \exists U \subseteq M,\exists \phi: U \to \mathbb{R}^n$ such that $\phi$ is a homeomorphism onto its image and $\phi(U)$ is an open subset of $\mathbb{R}^n$. Sometime we denote it as $M^n$.

We denote $(U,\phi)$ as a coordinate chart, $\phi$ as a coordinate map, and $\phi^{-1}$ as a parameterization of $U$.

### Atlas and $C^{k}$-structure

An **atlas** on a topological manifold $M$ is a collection of coordinate charts $\mathcal{A} = \{(U_\alpha, \phi_\alpha)\}_{\alpha \in I}$ where $\bigcup_{\alpha \in I} U_\alpha = M$.

Let $(U,\phi)$ and $(V, \psi)$ be two coordinate charts in the atlas $\mathcal{A}$. The **transition map** from $(U, \phi)$ to $(V, \psi)$ is defined as $\psi \circ \phi^{-1}: \phi(U \cap V) \to \psi(U \cap V)$.

We call the transition map **$C^{k}$ compatible** if it is a $C^{k}$ function. And we say that the atlas $\mathcal{A}$ is a **$C^{k}$-atlas** if $\forall (U, \phi), (V, \psi) \in \mathcal{A}, U\cap V \neq \emptyset$, the transition map $\psi \circ \phi^{-1}\in C^{k}(\phi(U \cap V)\cap \psi(U \cap V))$.

We say $(U, \phi)$ are **$C^{k}$-compatible with $\mathcal{A}$** if it is $C^{k}$ compatible with every chart in $\mathcal{A}$.

Let $\mathcal{A}$ be a $C^{k}$-atlas on a topological manifold $M$. Maximal extension of $\mathcal{A}$: $\overline{\mathcal{A}} = \{(U, \phi) : (U, \phi) \text{ is } C^{k} \text{-compatible with } \mathcal{A}\}$. It is unique by definition and is a $C^{k}$-atlas on $M$.

And it is not hard to prove that if $\overline{\mathcal{A}} = \overline{\mathcal{B}}$ for another $C^{k}$-atlas $\mathcal{B}$, then $\mathcal{A} \cup \mathcal{B}$ is also a $C^{k}$-atlas on $M$.

Actually, Let $\mathcal{F}$ be the collection of all $C^{k}$-compatible coordinate charts on $M$. Then $(\mathcal{F}, \subseteq)$ is a directed set. And $\overline{\mathcal{A}}$ is the maximal element of $\mathcal{F}$.

We call $\overline{\mathcal{A}}$ a maximal $C^{k}$-atlas or a **$C^{k}$-structure** on $M$. A manifold with a $C^{k}$-structure is called a **$C^{k}$-manifold**.

### Smooth Manifolds

A **smooth manifold** is a $C^{\infty}$-manifold.

### Coordinate Functions

Let $(U, \phi)$ be a coordinate chart on a $n$-manifold $M$. The **coordinate functions** are the components of the coordinate map $\phi: U \to \mathbb{R}^n$, denoted as $\phi = (\phi^1, \phi^2, \ldots, \phi^n)$. We call the coordinate functions $\phi^i: U \to \mathbb{R}$ the **$i$-th coordinate function**.

For $p\in U$, we write $\phi(p) = (x^1, x^2, \ldots, x^n)$, where $x^i = \phi^i(p)$.

And we call $(x^1, x^2, \ldots, x^n)$ the local coordinates of $p$ in the chart $(U, \phi)$.

### Tangent Vectors and Tangent Spaces

#### Germs of functions

A **germ of a function** at a point $p \in M$ is an equivalence class of functions defined on a neighborhood of $p$. Two functions $f$ and $g$ are equivalent if they agree on some neighborhood of $p$. 

Formally, let $(f,U)$ and $(g,V)$ be two functions defined on neighborhoods $U$ and $V$ of $p$. If there exists a neighborhood $W \subseteq U \cap V$ such that $f|_W = g|_W$, then we say that the germ of $f$ at $p$ is equal to the germ of $g$ at $p$, denoted as $(f,U) \sim (g,V)$.

A germ of a function at $p$ is denoted as $[f]_p$, $[f]$ or $f$ if the context is clear.

We denote $C^{\infty}_p(M)$ or $C^{\infty}_p$ as the set of germs of smooth functions at $p \in M$. Formally, $C^{\infty}_p(M) = \{ [f]_p : f \in C^{\infty}(U), U \text{ is a neighborhood of } p \}$. 

If we write $f \in C^{\infty}_p(M)$, it means that $[f]_p\in C^{\infty}_p(M)$.

#### Definition 

There is three common definitions of tangent vectors on a manifold $M$: 
1. **Tangent Vector as Derivation**: A tangent vector at a point $p \in M$ is a linear map $v: C^{\infty}_p(M) \to \mathbb{R}$ that satisfies the Leibniz rule: $v(fg) = v(f)g(p) + f(p)v(g)$ for all $f, g \in C^{\infty}_p(M)$.
2. **Tangent Vector as Equivalence Class of Curves**: A tangent vector at a point $p \in M$ is an equivalence class of curves $\gamma: (-\epsilon, \epsilon) \to M$ such that $\gamma(0) = p$.
3. **Tangent Vector as Derivative of Coordinate Functions**: A tangent vector at a point $p \in M$ is a vector in the tangent space $T_pM$ defined as the set of all derivations at $p$ or equivalence classes of curves through $p$.

The **tangent space** at a point $p \in M$, denoted as $T_pM$, is the set of all tangent vectors at $p$. It is a vector space over $\mathbb{R}$ of dimension $n$, same as the dimension of the manifold $M$.

We shall mainly focus on the first definition of tangent vectors as derivations, which is the most general and abstract definition. 

The tangent space is equipped with a natural $\mathbb{R}$-vector space structure.

##### Coordinate Expression

Let $(U,x)=(U,x^1,\ldots,x^n)$ be a coordinate chart containing $p$. 

Consider $\left.\frac{\partial}{\partial x^i}\right|_p: C^{\infty}_p(M)\to \mathbb{R}$ act on $f:U\to \mathbb R$ defined as 
$$\left.\frac{\partial}{\partial x^i}\right|_p([f])=\left.\frac{\partial(f\circ x^{-1})}{\partial x^i}\right |_{x(p)}$$

We know that $\left \{\left.\frac{\partial}{\partial x^{1}}\right|_p,\ldots,\left.\frac{\partial}{\partial x^n}\right|_p\right \}$ forms a basis of $T_p M$, $\dim T_pM = \dim M=n $.

By definition, we can prove the Kronecker delta property:
$$ \left.\frac{\partial}{\partial x^i}\right|_p(x^j) = \frac{\partial x^j\circ x^{-1}}{\partial x^i} \bigg|_{x(p)} = \delta^j_i $$
where $\delta^i_j$ is the Kronecker delta, which is $1$ if $i=j$ and $0$ otherwise.

So every $X_p \in T_pM$ can be expressed as a linear combination of the basis vectors: $X_p = \sum_{i=1}^n X_p^i \left.\frac{\partial}{\partial x^i}\right|_p$, where $X_p^i =X_p(x^i)\in \mathbb{R}$.

The coordinate expression of a tangent vector $X_p$ at $p$ in the chart $(U,x)$ is given by the tuple $(a_1, a_2, \ldots, a_n)$, where $a_i$ are the coefficients in the linear combination.

### Fiber Bundles

#### Definition of Fiber Bundle

A **fiber bundle** is a structure $(E, M, \pi, F)$ consisting of:

- **Total space**: $E$ (a manifold)
- **Base space**: $M$ (a manifold) 
- **Bundle projection**: $\pi: E \to M$ (a smooth surjective map)
- **Typical fiber**: $F$ (a manifold)

such that the **local triviality condition** holds: for each point $p \in M$, there exists an open neighborhood $U$ of $p$ and a diffeomorphism $\phi: \pi^{-1}(U) \to U \times F$ satisfying:
$$ \text{pr}_1 \circ \phi = \pi|_{\pi^{-1}(U)} $$
where $\text{pr}_1: U \times F \to U$ is the projection onto the first factor.

The pair $(U, \phi)$ is called a **local trivialization** or **bundle chart**.

For each $p \in M$, the **fiber** over $p$ is defined as $F_p = \pi^{-1}(p)$. The local triviality ensures that each fiber $F_p$ is diffeomorphic to the typical fiber $F$.

Intuitively, a fiber bundle describes a space that 'locally looks like a product' but may have global twisting, like the Möbius strip over a circle.

#### Vector Bundles

A **vector bundle** is a fiber bundle $(E, M, \pi, \mathbb{R}^k)$ where:

1. Each fiber $F_p = \pi^{-1}(p)$ has the structure of a $k$-dimensional real vector space.
2. The local trivializations $\phi: \pi^{-1}(U) \to U \times \mathbb{R}^k$ are **linear** on each fiber, meaning that for each $p \in U$, the restriction $\phi|_{F_p}: F_p \to \{p\} \times \mathbb{R}^k \cong \mathbb{R}^k$ is a vector space isomorphism.

The integer $k$ is called the **rank** of the vector bundle.

### Tangent Bundle

The **tangent bundle** of a manifold $M$, denoted as $TM$, is the vector bundle whose total space is the disjoint union of all tangent spaces:
$$ TM = \bigcup_{p \in M} T_pM = \{(p, X_p) : p \in M, X_p \in T_pM\}$$

The **bundle projection** $\pi: TM \to M$ is defined by $\pi(p, X_p) = p$.

**Local Trivialization**: Let $(U, x = (x^1, \ldots, x^n))$ be a coordinate chart on $M$. The tangent bundle can be locally trivialized over $U$ by the map:
$$ \Phi: \pi^{-1}(U) \to U \times \mathbb{R}^n $$
$$ \Phi(p, X_p) = \left(p, (X_p^1, \ldots, X_p^n)\right) $$
where $X_p = \sum_{i=1}^n X_p^i \left.\frac{\partial}{\partial x^i}\right|_p$ is the coordinate representation of the tangent vector $X_p$.

**Manifold Structure**: The tangent bundle $TM$ is a smooth manifold of dimension $2n$, where $n$ is the dimension of $M$. The coordinate charts on $TM$ are given by $(U \times \mathbb{R}^n, \Phi^{-1})$ where $(U, x)$ ranges over all coordinate charts on $M$.

**Transition Maps**: If $(U, x)$ and $(V, y)$ are two overlapping coordinate charts on $M$, the transition map between the corresponding bundle charts is:
$$ \Phi_V \circ \Phi_U^{-1}: (U \cap V) \times \mathbb{R}^n \to (U \cap V) \times \mathbb{R}^n $$
$$ (p, (v^1, \ldots, v^n)) \mapsto \left(p, \left(\sum_{i=1}^n v^i \frac{\partial y^j}{\partial x^i}\bigg|_p\right)_{j=1}^n\right) $$

This map is smooth, confirming that $TM$ has a smooth manifold structure.

### Sections of Fiber Bundles

#### Definition of Sections

Let $\pi: E \to M$ be a fiber bundle projection. A **section** of $E$ is a map $s: M \to E$ such that $\pi \circ s = \text{id}_M$. In other words, for each point $p \in M$, we have $s(p) \in E_p$ (the fiber over $p$).

A section $s$ is called **smooth** if $s: M \to E$ is a smooth map between manifolds.

#### Local Expression of Smoothness

Let $(U, \phi)$ be a local trivialization of $E$ over an open set $U \subseteq M$, where $\phi: \pi^{-1}(U) \to U \times F$ for some typical fiber $F$. Any section $s$ over $U$ can be written as:
$$ \phi \circ s|_U: U \to U \times F $$
which has the form $p \mapsto (p, f(p))$ for some function $f: U \to F$.

Sometimes we just say $f$ is a section of $E$ over $U$.

#### Space of Sections

We denote by $\Gamma(E)$ or $\Gamma(M, E)$ the space of all smooth sections of the fiber bundle $E \to M$.

**For General Fiber Bundles**: The space $\Gamma(E)$ has the structure of a set with pointwise operations (when they make sense on the typical fiber).

**For Vector Bundles**: When $E \to M$ is a vector bundle with typical fiber $\mathbb{R}^k$, the space $\Gamma(E)$ has additional algebraic structures:

1. **Vector space structure**: For sections $s_1, s_2 \in \Gamma(E)$ and scalars $a, b \in \mathbb{R}$:
   $$ (as_1 + bs_2)(p) = as_1(p) + bs_2(p) \in E_p $$

2. **$C^\infty(M)$-module structure**: For a smooth function $f \in C^\infty(M)$ and a section $s \in \Gamma(E)$:
   $$ (fs)(p) = f(p) \cdot s(p) \in E_p $$

These structures exist because each fiber $E_p$ is a vector space, allowing us to perform linear operations.

### Differential

Let $F: M^m \to N^n$ be a smooth map between manifolds $M$ and $N$. The **differential** of $F$ at a point $p \in M$, denoted as $dF_p: T_pM \to T_{F(p)}N$, is a linear map between tangent spaces induced by the pushforward of $F$.

It is defined as follows: for any tangent vector $X_p \in T_pM$ and any smooth function $f \in C^{\infty}_{F(p)}(N)$,
$$ dF_p(X_p)f = X_p(f \circ F) $$

**Local Coordinate Expression**: Let $(U, x)$ and $(V, y)$ be coordinate charts around $p$ and $F(p)$ respectively, and let $\tilde F = y \circ F \circ x^{-1}$ be the local representation of $F$. Then:
$$
\begin{align*}
dF_p(X_p) &= \sum_{j=1}^n X_p(y^j\circ F) \frac{\partial}{\partial y^j} \bigg|_{F(p)}\\
&= \sum_{j=1}^n \sum_{i=1}^m X_p^i \frac{\partial \tilde F^j}{\partial x^i} \bigg|_{x(p)} \frac{\partial}{\partial y^j} \bigg|_{F(p)}\\
&= \sum_{j=1}^n \left(\sum_{i=1}^m X_p^i \frac{\partial \tilde F^j}{\partial x^i} \bigg|_{x(p)}\right) \frac{\partial}{\partial y^j} \bigg|_{F(p)}
\end{align*}
$$

In matrix form, if $J_F(p) = \left(\frac{\partial \tilde F^j}{\partial x^i} \bigg|_{x(p)}\right)$ is the Jacobian matrix of $F$ at $p$, then:
$$ dF_p(X_p) = J_F(p) X_p $$
where $X_p$ is considered as a column vector in local coordinates.

### Vector Fields

A **vector field** on a manifold $M$ is a smooth section of the tangent bundle $TM$. That is, a vector field $X$ is a smooth map $X: M \to TM$ such that $\pi \circ X = \text{id}_M$, where $\pi: TM \to M$ is the bundle projection.

Equivalently, a vector field assigns to each point $p \in M$ a tangent vector $X_p \in T_pM$ in a smooth manner.

**Local Coordinate Expression**: 

The partial derivative operator $\frac{\partial}{\partial x^i}$ can be viewed as a vector field on $M$ in the coordinate chart $(U, x)$, where it acts on smooth functions $f \in C^\infty(U)$ by:
$$ \frac{\partial f}{\partial x^i} = \left.\frac{\partial(f\circ x^{-1})}{\partial x^i}\right|_{x(p)} $$

where $x^i$ is the $i$-th coordinate function in the chart $(U, x)$.

In a coordinate chart $(U, x = (x^1, \ldots, x^n))$, a vector field $X$ can be expressed as:
$$ X = \sum_{i=1}^n X^i \frac{\partial}{\partial x^i} $$
where $X^i: U \to \mathbb{R}$ are smooth functions called the **components** of the vector field $X$ with respect to the coordinate chart $(U, x)$.

The smoothness of the vector field $X$ is equivalent to the smoothness of all its component functions $X^i$.

**Space of Vector Fields**: We denote by $\mathfrak{X}(M)$ or $\Gamma^{\infty}(TM)$ the space of all smooth vector fields on $M$. This space is both a vector space over $\mathbb{R}$ and a module over the ring $C^\infty(M)$ of smooth functions on $M$.

### Partition of Unity

### Local Behavior of Smooth Maps

### Homotopy 

### Sard's theorem

### Submanifold

### Whitney Embedding Theorem 

### Tubular Neighborhood Theorem

### Manifold with Boundary

First, we denote $\mathbb R^n_+ = \{(x^1, \ldots, x^n) \in \mathbb R^n : x^n \geq 0\}$ as the half-space in $\mathbb R^n$.

A **manifold with boundary** is a topological space $M$ that is **Hausdorff**, **second countable** and $\forall p\in M$, there exists a neighborhood $U$ of $p$ that is homeomorphic to an open subset of $\mathbb R^n_+$.

or equivalently, a **Hausdorff**, **second countable**  topological space $M$ is a manifold with boundary if it is locally homeomorphic to $\mathbb R^n_+$ or $\mathbb R^n$.

Let $M$ be a manifold with boundary. The **boundary** of $M$, denoted as $\partial M$, is defined as the set of points in $M$ that is not locally homeomorphic to $\mathbb R^n$ i.e.,
$$ \partial M = \{ p \in M : \text{there is no neighborhood } U \cong \mathbb R^n \text{ around } p \} $$
and $\text{int}(M) = M \setminus \partial M$ is the interior of $M$.

In other words, $p\in \partial M$ if and only if there exists a local coordinate chart $(U,\phi)$ around $p$ such that $\phi(U) \subseteq \mathbb R^n_+$ and $\phi(p) \in \partial \mathbb R^n_+= \{(x^1, \ldots, x^{n-1}, 0) : x^1, \ldots, x^{n-1} \in \mathbb R\}$.

We may denote $(M,\partial M)$ for a manifold with boundary and $M$ can be called as manifold without boundary. Sometime $M$ can be either situation depending on the context.

Properties:
- $\partial M$ is a manifold without boundary with dimension $n-1$.

- The result of atlas on manifold with boundary is similar. 

- For the tangent space of $(M,\partial M)$ remains a full space of $\mathbb R^n$ at any point $p\in M$.

## Transversality and Intersection Theory

