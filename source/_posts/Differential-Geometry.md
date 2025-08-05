---
title: Differential Geometry
date: 2025-07-30 10:47:26
tags:
- Mathematics
categories:
- Mathematics
- Study Notes
---

## References

- John Lee, Introduction to Smooth Manifolds

- Loring Tu, An Introduction to Manifolds

- Victor Guillemin and Alan Pollack, Differential Topology

For Chinese reader, you can refer to the website http://staff.ustc.edu.cn/~wangzuoq/Courses/23F-Manifolds/#notes for Chinese course note in USTC.

## Review on Manifolds and its Structures

Please refer to the "Differential Manifold" article.

For the sake of this articles, you at least need to understand the following concept in that article.

- Topological Manifold
- Smooth Manifold
- Manifold with Boundary
- Partition of Unity
- Submersion, Immersion and Embedding
- etc.

## Lie Algebra and Lie Groups

TODO

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

Formally, $\sigma f = f$ for any permutation $\sigma\in S_k$.

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
$$
\begin{equation*} 
(f \wedge g)(v_1, v_2, \ldots, v_{k+l}) =  \sum_{\substack{\sigma \in S_{k+l} \\ \sigma(1) < \sigma(2) < \ldots < \sigma(k) \\ \sigma(k+1) < \sigma(k+2) < \ldots < \sigma(k+l)}} \text{sgn}(\sigma) f(v_{\sigma(1)}, v_{\sigma(2)}, \ldots, v_{\sigma(k)}) g(v_{\sigma(k+1)}, v_{\sigma(k+2)}, \ldots, v_{\sigma(k+l)})
\end{equation*}
$$

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

The **exterior algebra** or **Grassmann algebra** $(\bigwedge V, \wedge)$ is a **anticommutative graded algebra** defined on a vector space $V$ with the wedge product $\wedge$, where $\bigwedge V = \bigoplus_{k=0}^{\infty} A_k(V)$. When $V$ is finite-dimensional, we can write $\bigwedge V = \bigoplus_{k=0}^{\dim V} A_k(V)$ or $\bigwedge V = \bigoplus_{k=0}^{n} \bigwedge ^k V$, where $n = \dim V$. 

This is because $A_k(V) = 0$ for $k > \dim V$, which we shall prove later.

$\bigwedge^k V$ is called the **$k$-th exterior power** of $V$.

#### Basis of $L_k$

Let $I=(i_1,i_2,\ldots,i_k)$ be a $k$-indices, $e_1,\ldots,e_n$ and $\alpha^1,\ldots,\alpha^n$ are the basis of the vector space $V$ and its dual $V^*$ respectively. Denote $(e_{i_1}, e_{i_2}, \ldots, e_{i_k})$ as $e_I$, and $\alpha^I = \alpha^{i_1} \wedge \alpha^{i_2} \wedge \ldots \wedge \alpha^{i_k}$. $\alpha^I(e_J) = \delta^I_J$, where $\delta^I_J$ is the generalized Kronecker delta, which is $1$ if $I=J$ and $0$ otherwise.

Let $I=(1 \leq i_1 < \cdots < i_k \leq n)$, we claim that $\{\alpha^I\}_{I}$ is a basis of $A_k(V)$.

The proof is similar to the proof in the dual space section.

Corollary 1: The dimension of $A_k(V)$ is $\binom{n}{k}$, where $n = \dim V$.

Corollary 2: The dimension of $\bigwedge V$ is $\sum_{k=0}^{n} \binom{n}{k} = 2^n$, where $n = \dim V$.

Corollary 3: The dimension of $A_k(V)$ is $0$ for $k > \dim V$.

Proof: For $k > \dim V$, the set of indices $I=(i_1,i_2,\ldots,i_k)$ cannot be chosen such that $1 \leq i_1 < i_2 < \ldots < i_k \leq n$, hence $A_k(V) = 0$.

### Inner Product

Let $V$ be a vector space, $v\in V$, and $\omega \in \bigwedge^k V$. The **inner product** of $v$ and $\omega$, denoted as $\iota_v \omega$, is a linear map $\iota_v: \bigwedge^k V \to \bigwedge^{k-1} V$ defined by:
$$ \iota_v(\omega) = \sum_{i=1}^k (-1)^{i-1} \alpha^i(v) \alpha^{1} \wedge \ldots \wedge \widehat{\alpha^i} \wedge \ldots \wedge \alpha^{k} $$

where $\widehat{\alpha^i}$ means that the $i$-th term is omitted from the wedge product.

The inner product $\iota_v$ is called the **interior product** or **contraction** with respect to the vector $v$. It reduces the degree of the form by 1.

Let $\alpha \in \bigwedge^k V, \beta \in \bigwedge^l V$, the inner product satisfies the following properties:
1. **Nilpotency**: $\iota_v \iota_v \alpha = 0$ for any $v \in V$.
2.  $\iota_v(\alpha \wedge \beta) = \iota_v(\alpha) \wedge \beta + (-1)^k \alpha \wedge \iota_v(\beta)$.
3. **Linearity**: $\iota_{av + bw}(\alpha) = a \iota_v(\alpha) + b \iota_w(\alpha)$ for any $a,b \in \mathbb{R}$ and $v,w \in V$.

### Pullback 

Let $L: W \to V$ be a linear map between vector spaces $W$ and $V$. The **pullback** of exterior forms under $L$, denoted as $L^*: \bigwedge^k V \to \bigwedge^k W$, is defined by:
$$ L^*(\omega)(w_1, w_2, \ldots, w_k) = \omega(L(w_1), L(w_2), \ldots, L(w_k)) $$
for all $w_1, w_2, \ldots, w_k \in W^*$.

The pullback satisfies the following properties:
1. $L^*(\alpha \wedge \beta) = L^*(\alpha) \wedge L^*(\beta)$ 
2. $\iota_w(L^*(\alpha)) = L^*(\iota_{L(w)}(\alpha))$.

### Differential Forms 

We have defined the exterior algebra $\bigwedge V$ and the wedge product $\wedge$ on an abstract vector space $V$. Now we can define **differential forms** on a manifold $M$, where $V$ is the tangent space $T_pM$ at a point $p \in M$.

#### Cotangent Space and Cotangent Bundle

The **cotangent space** $T^*_pM$ at a point $p \in M$ is the dual space of the tangent space $T_pM$. It consists of all linear functionals on $T_pM$. Elements of the cotangent space are called **covectors** or **differential 1-forms**.

The cotangent space $T^*_pM$ is a vector space of dimension $n$, where $n$ is the dimension of the manifold $M$.

The **cotangent bundle** $T^*M$ is the disjoint union of all cotangent spaces at each point in $M$:
$$ T^*M = \bigcup_{p \in M} T^*_pM = \{(p, \omega_p) : p \in M, \omega_p \in T^*_pM\} $$

The cotangent bundle $T^*M$ is a manifold of dimension $2n$.

Local coordinates on the cotangent bundle can be defined as $(x^1, x^2, \ldots, x^n, \omega_1, \omega_2, \ldots, \omega_n)$, where $(x^1, x^2, \ldots, x^n)$ are local coordinates on $M$ and $(\omega_1, \omega_2, \ldots, \omega_n)$ are the components of a covector in the cotangent space with respect to the dual basis $\{dx^1, dx^2, \ldots, dx^n\}$.

#### Differential $k$-Forms

Let $M$ be a smooth manifold of dimension $n$. A **differential $k$-form** on $M$ is a smooth section of the $k$-th exterior power of the cotangent bundle, denoted as $\bigwedge^k T^*M$.

In other words, a differential $k$-form on a open subset $U \subseteq M$ is a mapping $$ \omega: U \to \bigwedge^k T^*M $$ that assigns to each point $p \in U$ an alternating multilinear function $\omega(p): T_pM^k \to \mathbb{R}$ and $\omega \in A_k(T_pM)$ for all $p \in U$.

Let $\{e_1 = \left.\frac{\partial}{\partial x^1}\right|_p, e_2 = \left.\frac{\partial}{\partial x^2}\right|_p, \ldots, e_n = \left.\frac{\partial}{\partial x^n}\right|_p\}$ be a basis of $T_pM$. The corresponding dual basis of $T^*_pM$ is $\{dx^1, dx^2, \ldots, dx^n\}$, where $dx^i(e_j) = \delta^i_j$. 

A basis of $A_k(T_pM)$ is then given by $\{dx^I\}_{I}$, where $I = (i_1, i_2, \ldots, i_k)$ is a $k$-tuple of indices with $1 \leq i_1 < i_2 < \ldots < i_k \leq n$, and $dx^I$ denotes the wedge product $dx^{i_1} \wedge dx^{i_2} \wedge \ldots \wedge dx^{i_k}$. 

Therefore, a differential $k$-form $\omega$ can be expressed in local coordinates as:
$$ \omega = \sum_{I} f_I dx^I $$
where $f_I: U \to \mathbb{R}$ are functions on $U$.

A differential $k$-form $\omega$ is called **smooth** if all the functions $f_I$ are smooth functions on $U$.

Let $\Omega^k(M)=\Gamma^{\infty}(\bigwedge^k T^*M)$ be the space of all smooth differential $k$-forms on a manifold $M$. It is a vector space over $\mathbb{R}$.

The wedge product of two differential forms $\omega_1 \in \Omega^k(M)$ and $\omega_2 \in \Omega^l(M)$ is pointwise defined as:
$$ (\omega_1 \wedge \omega_2)(p) = \omega_1(p) \wedge \omega_2(p) = \sum_{I,J} f_I(p) g_J(p) dx^I \wedge dx^J $$
where $f_I$ and $g_J$ are the coefficients of $\omega_1$ and $\omega_2$ in local coordinates, respectively. Note that if $I$ and $J$ are such that $I \cap J \neq \emptyset$, then $dx^I \wedge dx^J = 0$ due to the anticommutativity of the wedge product.
The wedge product of differential forms is bilinear and associative, and it satisfies the anticommutativity property:
$$ \omega_1 \wedge \omega_2 = (-1)^{kl} \omega_2 \wedge \omega_1 $$

Let $\Omega^*(M) = \bigoplus_{k=0}^n \Omega^k(M)$ be the space of all smooth differential forms on $M$. It is a graded algebra with respect to the wedge product $\wedge$.

#### Coordinate Functions and Their Differentials

Let $(U, x) = (U, x^1, \ldots, x^n)$ be a coordinate chart on a smooth manifold $M$ of dimension $n$. The coordinate functions $x^i: U \to \mathbb{R}$ are smooth functions that assign to each point $p \in U$ its $i$-th coordinate $x^i(p)$. 

The **coordinate 1-forms** $dx^i$ are smooth sections of the cotangent bundle, each $dx^i: U \to T^*M$ satisfies $$\pi \circ dx^i = \text{id}_U$$
where $\pi: T^*M \to M$ is the bundle projection. 

We have $dx^i|_p: T_pM \to \mathbb{R}$ defined by $dx^i|_p(v) = v(x^i(p))$ for any tangent vector $v \in T_pM$. The coordinate 1-forms satisfy the dual basis property: $$\left.dx^i\right|_p\left(\left.\frac{\partial}{\partial x^j}\right|_p\right) = \delta^i_j$$
making $\{dx^1|_p, \ldots, dx^n|_p\}$ the dual basis of $T_p^*M$ corresponding to the coordinate basis $\{\frac{\partial}{\partial x^1}|_p, \ldots, \frac{\partial}{\partial x^n}|_p\}$ of $T_pM$. 

Any differential $k$-form on $U$ can be uniquely expressed as $\omega = \sum_{|I|=k} f_I \, dx^I$, where $I = (i_1, \ldots, i_k)$ with $1 \leq i_1 < \cdots < i_k \leq n$, $dx^I = dx^{i_1} \wedge \cdots \wedge dx^{i_k}$, and $f_I: U \to \mathbb{R}$ are smooth functions.

### Pullback of Differential Forms

Let $F: M \to N$ be a smooth map between manifolds, and let $\omega \in \Omega^k(N)$ be a differential $k$-form on $N$. The **pullback** of $\omega$ by $F$, denoted as $F^*\omega$, is a differential $k$-form on $M$ defined by:
$$ (F^*\omega)(p)(v_1, v_2, \ldots, v_k) = \omega(F(p))(dF_p(v_1), dF_p(v_2), \ldots, dF_p(v_k)) $$
where $dF_p: T_pM \to T_{F(p)}N$ is the differential of $F$ at the point $p$.

#### Local Coordinate Expression

Let $(U, x^1, \ldots, x^m)$ and $(V, y^1, \ldots, y^n)$ be coordinate charts on $M$ and $N$ respectively, with $F(U) \subseteq V$. 

Let $F: M \to N$ have local coordinate representation $F^i = y^i \circ F \circ x^{-1}$, where $(x^1, \ldots, x^m)$ are coordinates on $M$ and $(y^1, \ldots, y^n)$ are coordinates on $N$.

If $\omega$ has the local expression:
$$ \omega = \sum_{|I|=k} f_I \, dy^I $$
where $f_I: V \to \mathbb{R}$ are smooth functions.

Then the pullback $F^*\omega$ has the expression:
$$ 
\begin{align*}
F^*\omega &= \sum_{|I|=k} (f_I \circ F) \, F^*(dy^I) \\
&= \sum_{|I|=k} (f_I \circ F) \, d(y^{i_1} \circ F) \wedge \cdots \wedge d(y^{i_k} \circ F) 
\end{align*}
$$

Note:
$$ d(y^{i_j} \circ F) = d(F^{i_j} \circ x) = \sum_{l=1}^m \frac{\partial F^{i_j}}{\partial x^l} dx^l $$

Therefore:
$$ F^*\omega = \sum_{|I|=k} (f_I \circ F) \, \left(\sum_{l_1=1}^m \frac{\partial F^{i_1}}{\partial x^{l_1}} dx^{l_1}\right) \wedge \cdots \wedge \left(\sum_{l_k=1}^m \frac{\partial F^{i_k}}{\partial x^{l_k}} dx^{l_k}\right) $$

#### Properties of Pullback

The pullback operation satisfies the following important properties:

1. **Linearity**: $F^*(a\omega_1 + b\omega_2) = aF^*\omega_1 + bF^*\omega_2$ for $a,b \in \mathbb{R}$

2. **Preservation of wedge product**: 
   $$ F^*(\omega_1 \wedge \omega_2) = F^*\omega_1 \wedge F^*\omega_2 $$

3. **Functoriality**: If $G: L \to M$ and $F: M \to N$ are smooth maps, then:
   $$ (F \circ G)^* = G^* \circ F^* $$

4. **Identity**: $(\text{id}_M)^* = \text{id}_{\Omega^*(M)}$

### Exterior Derivative

The **Exterior Derivative** is an operator $d: \Omega^k(M) \to \Omega^{k+1}(M)$ that generalizes the concept of differentiation to differential forms. It is defined as follows:
$$
d\omega = \sum_{|I|=k} \left( \sum_{j=1}^n \frac{\partial f_I}{\partial x^j} \right) dx^j \wedge dx^I
$$
where $\omega = \sum_{|I|=k} f_I \, dx^I$ is a differential $k$-form on $M$, and $dx^I = dx^{i_1} \wedge \cdots \wedge dx^{i_k}$ for $I = (i_1, \ldots, i_k)$.

It can be naturally extended to the exterior algebra $\bigwedge^*(M)$ as follows:
$$ d(\omega) = \sum_{k=0}^{n} d\omega_k $$

where $\omega = \sum_{k=0}^{n} \omega_k$ and $\omega_k \in \Omega^k(M)$.

The exterior derivative satisfies the following properties:
1. **Linearity**: $d(a\omega_1 + b\omega_2) = a d\omega_1 + b d\omega_2$ for $a,b \in \mathbb{R}$.
2. **Nilpotency**: $d^2 = 0$, i.e., $d(d\omega) = 0$ for any differential form $\omega$.
3. **Antiderivation Property**: $d(\omega_1 \wedge \omega_2) = d\omega_1 \wedge \omega_2 + (-1)^k \omega_1 \wedge d\omega_2$ for $\omega_1 \in \Omega^k(M)$ and $\omega_2 \in \Omega^l(M)$.
4. **Pullback Compatibility**: For any smooth map $F: M \to N$, we have:
   $$ F^*(d\omega) = d(F^*\omega) $$
5. **Vanishing of Top Forms**: $d\omega = 0$ if $\omega\in \Omega^n(M)$, where $n = \dim M$.

We only prove the nilpotency here:

Proof: 

Only to prove $\omega\in \Omega^k(M)$, $\omega = \sum_{|I|=k} \alpha^{I} dx^{I}$
$$
\begin{align*}

d \omega &= \sum_{|I|=k}\sum_{i\in[n]}\frac{\partial \alpha^I}{\partial x^i}d x^i \wedge dx^I\\

d(d\omega) &= \sum_{i,j\in [n]}\sum_{|I|=k} \frac{\partial^2 \alpha^I}{\partial x^i\partial x^j} d x^i \wedge d x^j \wedge d x^I \\
&= \sum_{1\leq i<j\leq n} \sum_{|I|=k}\left ( \frac{\partial^2 \alpha^I}{\partial x^i\partial x^j} - \frac{\partial^2 \alpha^I}{\partial x^j\partial x^i} \right ) d x^i \wedge d x^j \wedge d x^I \\
&= 0
\end{align*}
$$
since mixed partial derivatives commute for smooth functions.

### Integration of Differential Forms

TODO.

### Stokes' Theorem

TODO.

### de Rham Theory

Let $M$ be a smooth manifold, $\omega \in \Omega^k(M)$ be a smooth differential $k$-form. We call $\omega$ a **closed form** if $d\omega = 0$, and we call $\omega$ an **exact form** if there exists a differential $(k-1)$-form $\eta \in \Omega^{k-1}(M)$ such that $\omega = d\eta$.

We denote the space of closed $k$-forms on $M$ as $Z^k(M) = \{\omega \in \Omega^k(M) : d\omega = 0\}$ and the space of exact $k$-forms as $B^k(M) = \{d\eta : \eta \in \Omega^{k-1}(M)\}$.

Let $\dim M = n$, we have the following result:

- $B^k(M) = Z^k(M) = \{0\}$ if $k > n$.
- $B^0(M) = \{0\}$, $Z^0(M) = \{f\in C^{\infty}(M) | df=0\}\simeq \mathbb{R}^{K}$, where $K$ is the number of connected components of $M$.
- $Z^n(M) = \Omega^n(M)$ 

By definition, 
$$
Z^k(M) = \ker (d: \Omega^k(M) \to \Omega^{k+1}(M))
$$
$$
B^k(M) = \text{Im} (d: \Omega^{k-1}(M) \to \Omega^k(M))
$$

Since $d^2 = 0$, we have the following property:
$$
B^k(M) \subseteq Z^k(M) \subseteq \Omega^k(M)
$$

We define the **de Rham cohomology** of $M$ as:
$$ H^k_{dR}(M) = Z^k(M) / B^k(M) $$
and $[\omega]$ is the equivalence class of $\omega \in Z^k(M)$ in $H^k_{dR}(M)$, called the **de Rham cohomology class** of $\omega$.

Example:

For $M = \mathbb{R}$, $Z^0(\mathbb{R}) = \Omega^0(\mathbb{R}) = \mathbb R$, $B^0(\mathbb{R}) = \{0\}$, hence $H^0_{dR}(\mathbb{R}) \simeq \mathbb{R}$, and $H^k_{dR}(\mathbb{R}) \simeq \{0\}$ for $k > 0$.

Let $\dim M = n$, we have the following results:
- $H^k_{dR}(M) = \{0\}$ for $k > n$.
- $H^0_{dR}(M) \simeq \mathbb{R}^{K}$, where $K$ is the number of connected components of $M$.

#### Betti Numbers and Euler Characteristic

If $\dim H^k_{dR}(M) < \infty$ for any $k$, we define the **Betti number** $b_k(M) = \dim H^k_{dR}(M)$, which is called the $k$-th Betti number of the manifold $M$.

Euler characteristic of $M$ is defined as:
$$ \chi(M) = \sum_{k=0}^{\dim M} (-1)^k b_k(M) $$

#### Chain Complex

The **chain complex** is a sequence of abelian groups (or modules) $\cdots A_0, A_1, A_2\cdots$ connected by homomorphisms $d_n: A_n\to A_{n-1}$, such that the image of one homomorphism is contained in the kernel of the next. The composition of any two consecutive maps shall be the zero maps, $ d_n \circ d_{n+1} = 0$ or $d^2=0$ for short. 
$$ \cdots \xleftarrow{d_0} A_0 \xleftarrow{d_1} A_1 \xleftarrow{d_2} A_2 \xleftarrow{d_3} \cdots$$

The **cochain complex** is the dual notion to the chain complex.

$$ \cdots \xrightarrow{d^0} A^0 \xrightarrow{d^1} A^1 \xrightarrow{d^2} A^2 \xrightarrow{d^3} \cdots$$
where $d^{n}\circ d^{n+1} = 0 $.

The elements in the kernel of $d$ are called **(co)cycles** (or **closed** elements), and the elements in the image of $d$ are called **(co)boundaries** (or **exact** elements). Right from the definition of the differential, all boundaries are cycles. The $n$-th **(co)homology** group $H_n(H^n)$ is the group of (co)cycles modulo (co)boundaries in degree $n$, that is,

$$ H_n = \ker d_n/\text{im} d_{n+1} \left ( H^n = \ker d^n/\text{im} d^{n-1}\right)$$

A **exact sequence** is a (co)chain complex whose (co)homology groups are all zero, which means all closed elements are exact. A short exact sequence is a bounded exact sequence in which only the groups $A_k$, $A_{k+1}$, $A_{k+2}$ may be nonzero. For example, the following chain complex is a short exact sequence.

$$ \cdots \rightarrow 0 \rightarrow Z \xrightarrow {\times p} Z \twoheadrightarrow Z/p \rightarrow 0 \rightarrow \cdots $$


#### de Rham Complex

The **de Rham complex** is the sequence of differential forms:
$$ 0 \to \Omega^0(M) \xrightarrow{d} \Omega^1(M) \xrightarrow{d} \Omega^2(M) \xrightarrow{d} \ldots \xrightarrow{d} \Omega^n(M) \to 0 $$

### Poincaré Lemma

TODO.