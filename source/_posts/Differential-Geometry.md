---
title: Differential Geometry
date: 2025-07-30 10:47:26
tags:
- Mathematics
categories:
- Mathematics
- Study Notes
---

## Manifolds and its Structures

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

Consider $\left.\frac{\partial}{\partial x^i}\right|_p: C^{\infty}_p(M)\to \mathbb{R}$ act on $f:U\to \mathbb R$ defined as $$\left.\frac{\partial}{\partial x^i}\right|_p([f])=\left.\frac{\partial(f\circ x^{-1})}{\partial x^i}\right |_{x(p)}$$

We know that $\left \{\left.\frac{\partial}{\partial x^{1}}\right|_p,\ldots,\left.\frac{\partial}{\partial x^n}\right|_p\right \}$ forms a basis of $T_p M$, $\dim T_pM = \dim M=n $.

By definition, we can prove the Kronecker delta property:
$$ \left.\frac{\partial}{\partial x^i}\right|_p(x^j) = \frac{\partial x^j\circ x^{-1}}{\partial x^i} \bigg|_{x(p)} = \delta^j_i $$
where $\delta^i_j$ is the Kronecker delta, which is $1$ if $i=j$ and $0$ otherwise.

So every $X_p \in T_pM$ can be expressed as a linear combination of the basis vectors: $X_p = \sum_{i=1}^n X_p^i \left.\frac{\partial}{\partial x^i}\right|_p$, where $X_p^i =X_p(x^i)\in \mathbb{R}$.

The coordinate expression of a tangent vector $X_p$ at $p$ in the chart $(U,x)$ is given by the tuple $(a_1, a_2, \ldots, a_n)$, where $a_i$ are the coefficients in the linear combination.

### Tangent Bundle

The **tangent bundle** of a manifold $M$, denoted as $TM$, is the disjoint union of all tangent spaces at each point in $M$:
$$ TM = \bigcup_{p \in M} T_pM = \{(p, X_p) : p \in M, X_p \in T_pM\}$$

The tangent bundle $TM$ is a manifold of dimension $2n$, where $n$ is the dimension of the manifold $M$.

Proof Sketch: The tangent bundle $TM$ can be covered by coordinate charts of the form $(U, \phi) \times \mathbb{R}^n$, where $U$ is an open subset of $M$ and $\phi: U \to \mathbb{R}^n$ is a coordinate chart. The transition maps between these charts are smooth, making $TM$ a smooth manifold.

### Differential 

Let $F: M^m \to N^n $ be a smooth map between manifolds $M$ and $N$. The **differential** of $F$ at a point $p \in M$, denoted as $dF_p: T_pM \to T_{F(p)}N$, is a linear map of tangent spaces induced by the pushforward of $F$.
It is defined as follows: for any tangent vector $X_p \in T_pM$,
$$ dF_p(X_p)f = X_p(f \circ F) $$
where $f \in C^{\infty}_{F(p)}(N)$ is a smooth function on $N$.

Let $\tilde F: U \to V$ be a local representation of $F$ 
$$ \tilde F = \psi \circ F \circ \phi^{-1} $$
where $(U, \phi)$ is a coordinate chart on $M$ and $(V, \psi)$ is a coordinate chart on $N$. Then the differential can be expressed in local coordinates as:
$$
\begin{align*}
dF_p(X_p) &= \sum_{j=1}^n X_p(y^j\circ F) \frac{\partial}{\partial y^j} \bigg|_{F(p)}\\
&= \sum_{j=1}^n X_p(\tilde F^j\circ \phi) \frac{\partial}{\partial y^j} \bigg|_{F(p)}\\
&= \sum_{j=1}^n \sum_{i=1}^m X_p^i \frac{\partial \tilde F^j}{\partial x^i} \bigg|_{\phi(p)} \frac{\partial}{\partial y^j} \bigg|_{F(p)}\\
&= \sum_{j=1}^n \left(\sum_{i=1}^m X_p^i \frac{\partial \tilde F^j}{\partial x^i} \bigg|_{\phi(p)}\right) \frac{\partial}{\partial y^j} \bigg|_{F(p)}\\
\end{align*}
$$
In a matrix form, if we denote the Jacobian matrix of $F$ at $p$ as $J_F(p) = \left(\frac{\partial \tilde F^j}{\partial x^i} \bigg|_{\phi(p)}\right)$, then the differential can be expressed as:
$$ dF_p(X_p) = J_F(p) X_p $$

where $X_p$ is considered as a column vector in the local coordinates.

### Vector Fields

A **vector field** on a manifold $M$ is a smooth section of the tangent bundle $TM$. It assigns to each point $p \in M$ a tangent vector $X_p \in T_pM$ in a smooth manner.

Formally, a vector field $X$ is a smooth map $X: M \to TM$ such that for each $p \in M$, the projection $\pi: TM \to M$ satisfies $\pi(X(p)) = p$.

We know that $X_p$ can be expressed in local coordinates as $X_p = \sum_{i=1}^n X^i(p) \left.\frac{\partial}{\partial x^i}\right|_p$, where $X^i(p)\in \mathbb{R}$

So we can write a vector field $X$ in local coordinates as:
$$ X = \sum_{i=1}^n X^i \frac{\partial}{\partial x^i} $$

where $X^i: M \to \mathbb{R}$ are smooth functions on $M$.


## Exterior Algebra, Differential Forms

**Note**: we discuss multilinear functions over $\mathbb{R}$-vector spaces, but the theory can be generalized to $\mathbb{C}$-vector spaces or other vector spaces over a field.

### Dual Space

Let $V,W$ be a $\mathbb{R}$-vector space, we denote all of the mapping from $V$ to $W$ as $\text{Hom}(V,W)$.

The **dual space** of $V$, denoted as $V^{\vee}$ or $V^*$, is the vector space of all linear maps from $V$ to $\mathbb{R}$, i.e., $V^* = \text{Hom}(V, \mathbb{R})$.

The elements of the dual space $V^*$ are called **covectors** or **dual vectors**.

The basis of the dual space $V^*$ is defined as the set of linear functionals that map each basis vector of $V$ to $\mathbb{R}$, while mapping all other basis vectors to zero.

To be specific, let $V$ be $n$-dimensional vector space, if $\{e_1, e_2, \ldots, e_n\}$ is a basis of $V$, then the dual basis $\{\alpha^1, \alpha^2, \ldots, \alpha^n\}$ of $V^*$ is defined by:
$$ \alpha^i(e_j) = \delta^i_j $$
where $\delta^i_j$ is the Kronecker delta.

To show that $\{\alpha^1, \alpha^2, \ldots, \alpha^n\}$ is a basis of $V^*$, we need to show that it is linearly independent and spans $V^*$.

Let $f \in V^*$ be a linear functional. We can express $f$ in terms of the dual basis as:
$$ f = \sum_{i=1}^n f(e_i) \alpha^i $$

where $f(e_i)$ are the coefficients of the linear functional $f$ in the dual basis. This shows that $\{\alpha^1, \alpha^2, \ldots, \alpha^n\}$ spans $V^*$.

If $\sum_{i=1}^n c_i \alpha^i = 0$ for some coefficients $c_i$, then for any basis vector $e_j \in V$, we have:
$$ \sum_{i=1}^n c_i \alpha^i(e_j) = \sum_{i=1}^n c_i \delta^i_j = c_j = 0 $$
for all $j = 1, 2, \ldots, n$. This implies that all coefficients $c_i = 0$, showing that the dual basis is linearly independent.

### Multilinear function

Let $V^k$ be the $k$-fold Cartesian product of a vector space $V$, i.e., $V^k = \underbrace{V \times V \times \ldots \times V}_{k\text{ times}}$ . A **multilinear function** or **$k$-linear function** is a function $f: V^k \to \mathbb{R}$ that is linear in each argument separately. It is a type $(0,k)$ tensor.

One can prove that all multilinear functions on $V$ forms a vector space, denoted as $L_k(V)$ or $\text{Mult}(V^k; \mathbb{R})$.

#### Examples

1. **Dot Product**: For a $\mathbb{R}$-vector space $V$ with a standard basis $\{e_1, e_2, \ldots, e_n\}$, the dot product is a bilinear function $f: V^2 \to \mathbb{R}$ defined as:
   $$ f(x,y) = \sum_{i=1}^n x^i y^i $$
   where $x = \sum_{i=1}^n x^i e_i$ and $y = \sum_{i=1}^n y^i e_i$ are vectors in $V$.

2. **Determinant**: The determinant is a multilinear function $f: V^n \to \mathbb{R}$ defined on the space of $n \times n$ matrices. It is linear in each row (or column) of the matrix.

Let $\sigma\in S_k$ be a permutation of the indices $\{1, 2, \ldots, k\}$. The left action of $\sigma$ on a multilinear function $f: V^k \to \mathbb{R}$ is defined as:
$$ \sigma f(v_1, v_2, \ldots, v_k) = f(v_{\sigma(1)}, v_{\sigma(2)}, \ldots, v_{\sigma(k)}) $$

A **symmetric multilinear function** is a multilinear function that is invariant under any permutation of its arguments. 

Formally, $\sigma f = f $ for any permutation $\sigma\in S_k$.

An **alternating multilinear function** is a multilinear function that changes sign when two of its arguments are swapped, or equivalently, changes by a factor of $sgn(\sigma)$ when permuted by $\sigma$.

Formally, $\sigma f = \text{sgn}(\sigma) f$ for any permutation $\sigma\in S_k$, where $\text{sgn}(\sigma)$ is the sign of the permutation.

All symmetric multilinear functions on $V$ forms a vector space, denoted as $S_k(V)$ or $\text{Sym}(V^k; \mathbb{R})$.
All alternating multilinear functions on $V$ forms a vector space, denoted as $A_k(V)$ or $\text{Alt}(V^k; \mathbb{R})$.


#### Symmetrizing and Alternating Operators

Let $V$ be a vector space and $f: V^k \to \mathbb{R}$ be a multilinear function. We can define the **symmetrizing operator** and **alternating operator** on $f$.

The **symmetrizing operator** $S: L_k(V) \to S_k(V)$ is defined as:
$$ S(f)(v_1, v_2, \ldots, v_k) = \frac{1}{k!} \sum_{\sigma \in S_k} \sigma f $$

The symmetrizing operator takes a multilinear function and produces a symmetric multilinear function by averaging over all permutations of its arguments.

The **alternating operator** $A: L_k(V) \to A_k(V)$ is defined as:
$$ A(f)(v_1, v_2, \ldots, v_k) = \frac{1}{k!} \sum_{\sigma \in S_k} \text{sgn}(\sigma) \sigma f $$   

The alternating operator takes a multilinear function and produces an alternating multilinear function by summing over all permutations of its arguments, weighted by the sign of the permutation.

We shall only prove the alternating operator does produce an alternating multilinear function. The proof for the symmetrizing operator is trivial.

**Proof**: Let $f$ be a multilinear function and $A(f)$ be the result of the alternating operator. For any permutation $\tau \in S_k$, we have:

$$
\begin{align*}
\tau A(f) &= \frac{1}{k!} \sum_{\sigma \in S_k} \text{sgn}(\sigma) \tau \circ\sigma f \\
&= \frac{1}{k!} \sum_{\sigma \in S_k} \text{sgn}(\tau) \text{sgn}(\tau \circ \sigma) \tau \circ\sigma f \\
&= \frac{1}{k!} \sum_{\sigma' \in S_k} \text{sgn}(\sigma') \sigma' f \\
&= \text{sgn}(\tau) A(f)
\end{align*} 
$$

The alternating operator $A$ satisfies the following properties:
1. **Linearity**: For $f_1, f_2 \in L_k(V)$ and $r,s\in \mathbb{R}$, we have:
   $$ A(r f_1 + s f_2) = r A(f_1) + s A(f_2) $$
2. $A(f\otimes A(g)) = A(A(f)\otimes g) = A(f\otimes g)$
3. $A\circ A = A$ 

### Tensor Product

Let $f\in L_k(V)$ and $g\in L_l(V)$ be two multilinear functions on $V$. The **tensor product** of $f$ and $g$, denoted as $f \otimes g$, is a multilinear function on $V^{k+l}$ defined as:
$$ (f \otimes g)(v_1, v_2, \ldots, v_{k+l}) = f(v_1, v_2, \ldots, v_k) g(v_{k+1}, v_{k+2}, \ldots, v_{k+l}) $$

The tensor product satisfies associativity and bilinearity.
Formally, we have:
1. **Associativity**: $(f \otimes g) \otimes h = f \otimes (g \otimes h)$ for any multilinear function $h \in L_m(V)$.
2. **Bilinearity**: For $f,f_1,f_2 \in L_k(V)$ and $g,g_1,g_2 \in L_l(V)$; $r,s\in \mathbb R$, we have:
    $$ (r f_1 + s f_2) \otimes g = r (f_1 \otimes g) + s (f_2 \otimes g) $$
    $$ f \otimes (r g_1 + s g_2) = r (f \otimes g_1) + s (f \otimes g_2) $$

**Example**: Let $(e_1,\ldots,e_n)$ be a basis of $V$, $(\alpha^1,\ldots,\alpha^n)$ be the dual basis of $V^*$. Then the inner product is a bilinear function $f: V\times V \to \mathbb{R}$ defined as:
$$ \langle x,y\rangle = \sum_{i=1}^n x^i y^i $$
where $x = \sum_{i=1}^n x^i e_i$ and $y = \sum_{i=1}^n y^i e_i$ are vectors in $V$.

Now, let $g_{ij}=\langle e_i,e_j\rangle$, $\langle x,y\rangle = \sum x^i y^j g_{ij} = \sum \alpha^i(x) \alpha^j(y) g_{ij} = \sum_{i,j} g_{ij} (\alpha^i \otimes \alpha^j)(x,y)$.

So we can write the inner product as a tensor product of the dual basis:
$$ \langle x,y\rangle = \sum_{i,j} g_{ij} (\alpha^i \otimes \alpha^j)(x,y) $$

### Wedge Product

The **wedge product** or **exterior product** is an operation on alternating multilinear functions that produces a new alternating multilinear function. It is denoted by $\wedge$ and is defined as follows:

For all $f \in A_k(V)$ and $g \in A_l(V)$, the wedge product $f \wedge g \in A_{k+l}(V)$ is defined by:
$$ (f \wedge g) = \binom{k+l}{k} A(f\otimes g)$$ 
or equivalently,
$$(f \wedge g)(v_1, v_2, \ldots, v_{k+l}) = \frac{1}{k!l!} \sum_{\sigma \in S_{k+l}} \text{sgn}(\sigma) f(v_{\sigma(1)}, v_{\sigma(2)}, \ldots, v_{\sigma(k)}) g(v_{\sigma(k+1)}, v_{\sigma(k+2)}, \ldots, v_{\sigma(k+l)}) $$

To avoid division by factorials, we can define the wedge product as sum over sorted permutations:
$$ (f \wedge g)(v_1, v_2, \ldots, v_{k+l}) =  \sum_{\substack{\sigma \in S_{k+l} \\ \sigma(1) < \sigma(2) < \ldots < \sigma(k) \\ \sigma(k+1) < \sigma(k+2) < \ldots < \sigma(k+l)}} \text{sgn}(\sigma) f(v_{\sigma(1)}, v_{\sigma(2)}, \ldots, v_{\sigma(k)}) g(v_{\sigma(k+1)}, v_{\sigma(k+2)}, \ldots, v_{\sigma(k+l)}) $$

Let $f \in A_k(V)$ and $g \in A_l(V)$ be alternating multilinear functions. The wedge product satisfies the following properties:
1. **Bilinearity**: It is bilinear on $f$ and $g$.
2. **Associativity**: It is associative, i.e., $(f \wedge g) \wedge h = f \wedge (g \wedge h)$ for any alternating multilinear function $h$.
3. **Anticommutativity**: It is anticommutative, i.e., $f \wedge g = (-1)^{kl} g \wedge f$. Particularly, $f \wedge f = 0$ when $k$ is odd.
4. $c\wedge f=cf$, $\forall c\in \mathbb R$.

Let $r\in N$, $f_i\in A_{k_i}(V)$ for $i=1,2,\ldots,r$, the wedge product $f_1 \wedge f_2 \wedge \ldots \wedge f_r$ is $\binom{k_1+\ldots+k_r}{k_1,\ldots,k_r} A(f_1\otimes f_2\otimes \ldots \otimes f_r)$, where $\binom{k_1+\ldots+k_r}{k_1,\ldots,k_r}$ is the multinomial coefficient. 

or equivalently,
$$ (f_1 \wedge f_2 \wedge \ldots \wedge f_r)(v_1, v_2, \ldots, v_{k_1+\ldots+k_r}) = \frac{1}{k_1!k_2!\ldots k_r!} \sum_{\sigma \in S_{k_1+\ldots+k_r}} \text{sgn}(\sigma) f_1(v_{\sigma(1)}, v_{\sigma(2)}, \ldots, v_{\sigma(k_1)}) f_2(v_{\sigma(k_1+1)}, v_{\sigma(k_1+2)}, \ldots, v_{\sigma(k_1+k_2)}) \ldots f_r(v_{\sigma(k_1+\ldots+k_{r-1}+1)}, v_{\sigma(k_1+\ldots+k_{r-1}+2)}, \ldots, v_{\sigma(k_1+\ldots+k_r)}) $$

This shows the associativity directly.

#### Relation to determinant

Let $\alpha^1,\ldots,\alpha^n\in V^*$, $v_1,\ldots,v_n\in V$, then:
$$ (\alpha^1 \wedge \alpha^2 \wedge \ldots \wedge \alpha^n)(v_1, v_2, \ldots, v_n) = \det[(\alpha^i(v_j))_{i,j}] $$

This is the determinant of the matrix formed by evaluating the dual basis vectors on the vectors $v_1, v_2, \ldots, v_n$.

### Graded Algebra and Exterior Algebra

#### Graded Algebra

Let $\mathbb K$ be a field, the **graded algebra** $(A,\times)$ is a vector space $A$ over $\mathbb K$ equipped with a direct sum decomposition $A = \bigoplus_{k=0}^{\infty} A_k$, where each $A_k$ is a vector space of degree $k$. The multiplication operation $\times: A \times A \to A$ satisfies:
$$ \times: A_k \times A_l \to A_{k+l} $$

$(A,\times)$ is called **anticommutative** if for all $a \in A_k$ and $b \in A_l$, $ab = (-1)^{kl} ba$.

A homomorphism of graded algebra is an algebra homomorphism that preserves the degree.

#### Exterior Algebra

The **exterior algebra** or **Grassmann algebra** $(\bigwedge V, \wedge)$ is a **anticommutative graded algebra** defined on a vector space $V$ with the wedge product $\wedge$, where $\bigwedge V = \bigoplus_{k=0}^{\infty} A_k(V)$. When $V$ is finite-dimensional, we can write $\bigwedge V = \bigoplus_{k=0}^{\dim V} A_k(V)$. 

This is because $A_k(V) = 0$ for $k > \dim V$, which we shall prove later.

#### Basis of $L_k$

Let $I=(i_1,i_2,\ldots,i_k)$ be a $k$-indices, $e_1,\ldots,e_n$ and $\alpha^1,\ldots,\alpha^n$ are the basis of the vector space $V$ and its dual $V^*$ respectively. Denote $(e_{i_1}, e_{i_2}, \ldots, e_{i_k})$ as $e_I$, and $\alpha^I = \alpha^{i_1} \wedge \alpha^{i_2} \wedge \ldots \wedge \alpha^{i_k}$. $\alpha^I(e_J) = \delta^I_J$, where $\delta^I_J$ is the generalized Kronecker delta, which is $1$ if $I=J$ and $0$ otherwise.

Let $I=(1 \leq i_1 < \cdots < i_k \leq n)$, we claim that $\{\alpha^I\}_{I}$ is a basis of $A_k(V)$.

The proof is similar to the proof in the dual space section.

Corollary 1: The dimension of $A_k(V)$ is $\binom{n}{k}$, where $n = \dim V$.

Corollary 2: The dimension of $\bigwedge V$ is $\sum_{k=0}^{\dim V} \binom{n}{k} = 2^n$, where $n = \dim V$.

Corollary 3: The dimension of $A_k(V)$ is $0$ for $k > \dim V$.

Proof: For $k > \dim V$, the set of indices $I=(i_1,i_2,\ldots,i_k)$ cannot be chosen such that $1 \leq i_1 < i_2 < \ldots < i_k \leq n$, hence $A_k(V) = 0$.

