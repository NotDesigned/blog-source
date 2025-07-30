---
title: Differential Geometry
date: 2025-07-30 10:47:26
tags:
- Mathematics
categories:
- Mathematics
- Study Notes
---

## Manifolds

### Topological Manifolds

Let $M$ be a Hausdorff and second countable topological space that is locally homeomorphic to Euclidean space $\mathbb{R}^n$. We say that $M$ is a **manifold of dimension $n$** if $\forall p \in M, \exists U \subseteq M,\exists \phi: U \to \mathbb{R}^n$ such that $\phi$ is a homeomorphism onto its image and $\phi(U)$ is an open subset of $\mathbb{R}^n$. Sometime we denote it as $M^n$.

We denote $(U,\phi)$ as a coordinate chart, $\phi$ as a coordinate map, and $\phi^{-1}$ as a parameterization of $U$.

### Atlas and $C^{k}$-structure

An **atlas** on a topological manifold $M$ is a collection of coordinate charts $\mathcal{A} = {(U_\alpha, \phi_\alpha)}_{\alpha \in I}$ where $\bigcup_{\alpha \in I} U_\alpha = M$.

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

Consider $\frac{\partial}{\partial x^i}|_p: C^{\infty}_p(M)\to \mathbb{R}$ act on $f:U\to R$ defined as $\frac{\partial}{\partial x^i}|_p([f])=\frac{\partial(f\circ x^{-1})}{\partial x^i}$

We know that $\left. \{\frac{\partial}{\partial x^{1}}|_p,\ldots,\frac{\partial}{\partial x^n}|_p\right \}$ forms a basis of $T_p M$, $\dim T_pM = \dim M=n $.

So every $X_p \in T_pM$ can be expressed as a linear combination of the basis vectors: $X_p = \sum_{i=1}^n a_i \frac{\partial}{\partial x^i}|_p$, where $a_i \in \mathbb{R}$.

The coordinate expression of a tangent vector $X_p$ at $p$ in the chart $(U,x)$ is given by the tuple $(a_1, a_2, \ldots, a_n)$, where $a_i$ are the coefficients in the linear combination.

### Tangent Bundle

The **tangent bundle** of a manifold $M$, denoted as $TM$, is the disjoint union of all tangent spaces at each point in $M$:
$$ TM = \bigcup_{p \in M} T_pM = \{(p, X_p) : p \in M, X_p \in T_pM\}$$

The tangent bundle $TM$ is a manifold of dimension $2n$, where $n$ is the dimension of the manifold $M$.

Proof Sketch: The tangent bundle $TM$ can be covered by coordinate charts of the form $(U, \phi) \times \mathbb{R}^n$, where $U$ is an open subset of $M$ and $\phi: U \to \mathbb{R}^n$ is a coordinate chart. The transition maps between these charts are smooth, making $TM$ a smooth manifold.

### Vector Fields

A **vector field** on a manifold $M$ is a smooth section of the tangent bundle $TM$. It assigns to each point $p \in M$ a tangent vector $X_p \in T_pM$ in a smooth manner.

Formally, a vector field $X$ is a smooth map $X: M \to TM$ such that for each $p \in M$, the projection $\pi: TM \to M$ satisfies $\pi(X(p)) = p$.

We know that $X_p$ can be expressed in local coordinates as $X_p = \sum_{i=1}^n a^i(p) \frac{\partial}{\partial x^i}|_p$, where $a^i(p)\in \mathbb{R}$

So we can write a vector field $X$ in local coordinates as:
$$ X = \sum_{i=1}^n a^i \frac{\partial}{\partial x^i} $$

where $a^i: M \to \mathbb{R}$ are smooth functions on $M$.


