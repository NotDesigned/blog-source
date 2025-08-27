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

## Review on Manifolds

Please refer to the "[Differential Manifold](https://notdesigned.github.io/2025/08/04/Differential-Manifold/)" article.

For the sake of this articles, one at least need to understand the following concept in that article.

- Topological Manifold
- Smooth Manifold
- Manifold with Boundary
- Partition of Unity
- Submersion, Immersion and Embedding
- etc.

## Lie Algebra and Lie Groups

### Smooth Differential Operators 

A **smooth differential operator** of order $n$ on $M$ is a linear map $P: C^{\infty}(M) \to C^{\infty}(M)$ that can be expressed locally as a finite sum of the form on each coordinate chart $(U,x)$:
$$
P=\sum_{|j|\leq n} a_j(p) \left(\frac{\partial}{\partial x^1}\right)^{j_1}\left(\frac{\partial}{\partial x^2}\right)^{j_2} \cdots \left(\frac{\partial}{\partial x^m}\right)^{j_m}
$$
where $a_j(p)$ are smooth functions on $M$, $m=\dim M$, and $j=(j_1,j_2,\ldots,j_m)$ is a multi-index with $|j|=j_1+j_2+\ldots+j_m$.

And by definition, a smooth vector field $X$ on $M$ is a smooth differential operator of order $1$.

Using partition of unity, we can prove that any smooth differential operator can be expressed as composition of at most $n$ smooth vector fields.

And $\operatorname{supp}(Pf) \subseteq \operatorname{supp}(f)$ for any smooth function $f\in C^{\infty}(M)$.

Peetre proves that any linear differential operator that satisfies the above condition is a smooth differential operator.

### Lie Bracket on Smooth Vector Fields

Given two smooth vector fields $X,Y$ on a smooth manifold $M$, the **Lie bracket** of $X$ and $Y$, denoted as $[X,Y]$, on a local coordinate $(U,\phi)$ it is defined as:
$$
[X,Y]= X \circ Y - Y \circ X = \sum_{i=1}^m \left( X^i \frac{\partial Y^j}{\partial \phi^i} - Y^i \frac{\partial X^j}{\partial \phi^i} \right) \frac{\partial}{\partial \phi^j}
$$
note the second order derivative is diminished, it is a the smooth vector field on $M$.

Generally speaking, if $X$ is a differential operator of order $p$, $Y$ is of order $q$. $X\circ Y-Y\circ X$ shall be a differential operator of order $p+q-1$.

The Lie bracket $[\cdot,\cdot]: \mathfrak{X}(M) \times \mathfrak{X}(M) \to \mathfrak{X}(M)$ satisfy the following property for any $X,Y,Z\in\mathfrak{X}(M)$

1. Antisymmetry: $[X,Y]=-[Y,X]$
2. Jacobi Identity: $[X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0$.
3. $\mathbb R$-Linearity: $[aX+bY,Z]=a[X,Z]+b[Y,Z],\forall a,b\in \mathbb R$

Note that it is not linear with respect to the second argument.

The Lie bracket gives a **Lie Algebra** structure on $\mathfrak{X}(M)$ $(\text{or } \Gamma^{\infty}(TM))$.

### Integral Curves

Let $X\in \mathfrak{X}(M)$, if a smooth curve $\gamma: I \to M$ is such that 

$$
\dot \gamma(t) = X_{\gamma(t)}
$$
for all $t \in I$, then $\gamma$ is called an **integral curve** of the vector field $X$.

if $0\in I$, then $\gamma(0)$ is called the **initial point** of the integral curve.

Example:

---

For the vector field $\widetilde{X} = \sum_{i=1}^n a^i \frac{\partial}{\partial x^i}$, the integral curve is:
$$
\widetilde{\gamma}(t)=(c_1 + a^1 t, c_2 + a^2 t, \ldots, c_n + a^n t)
$$
where $c_i = \gamma(0)^i$ are the initial point of the integral curve.

---

Assume $\gamma$ is in a coordinate chart $(U,x)$, we write $\phi(\gamma(t))=(x^1(\gamma(t)),\cdots,x^n(\gamma(t)))$.

We denote $\gamma^i:I\to \mathbb{R}=x^i\circ \gamma$ as the $i$-th coordinate of $\gamma$ on $U$.

$$
\dot \gamma(t)=(d\gamma)_t(\frac{d}{dt})|_{t}=\sum_{i=1}^n (d\gamma)_t(\frac{d}{dt})|_{t}(x^i)\left.\frac{\partial}{\partial x^i}\right|_{\gamma(t)}=\sum_{i=1}^n\frac{d (x^i\circ \gamma)}{dt}\left.\frac{\partial}{\partial x^i}\right|_{\gamma(t)}=\sum_{i=1}^n \dot{\gamma}^i \left.\frac{\partial}{\partial x^i}\right|_{\gamma(t)}
$$

So the equation $\dot \gamma(t) = X_{\gamma(t)}$ can be written as:

$$
\sum_{i=1}^n \dot{\gamma}^i(t) \left.\frac{\partial}{\partial x^i}\right|_{\gamma(t)} = \sum_{i=1}^n X^i(\gamma(t)) \left.\frac{\partial}{\partial x^i}\right|_{\gamma(t)}
$$

So we have the system of ordinary differential equations:
$$\dot{\gamma}^i(t) = X^i\circ x^{-1}(\gamma^1(t),\ldots,\gamma^n(t))$$
for $i=1,2,\ldots,n$.

Conversely, every system of ordinary differential equations of the form above defines a unique integral curve $\gamma$ on $U$ of the vector field $X$.

By the existence and uniqueness theorem of ordinary differential equations on $\mathbb R^n$, we have the following theorem:

Assume $X\in \mathfrak{X}(M)$ is a smooth vector field on a smooth manifold $M$, then for every $p\in M$, there exists a neighborhood $U$ of $p$, $\epsilon>0$ and a smooth mapping $\Gamma: (-\epsilon,\epsilon) \times U \to M$ such that:
1. $\Gamma(0,p)=p$ for all $p\in U$.
2. For fixed $q\in U$, $\gamma_q(t)=\Gamma(t,q)$ is the integral curve of $X$ with initial point $\gamma_q(0)=q$ for all $t\in (-\epsilon,\epsilon)$.
3. The integral curve in $[2]$ is unique in the sense that if $\sigma: I \to U$ is another smooth curve with $\sigma(0)=q$, then $\gamma_q(t)=\sigma(t)$ for all $t\in (-\epsilon,\epsilon)\cap I$.

### Reparametrization

Generally, the reparametrization of an integral curve $\gamma$ is not an integral curve of the vector field $X$. But if the reparametrization is linear, the reparametrization is still an integral curve of the vector field $X$.

If $\gamma:I\to M$ is an integral curve of $X$, then:
1. Let $I_{a}={t|t+a\in I}$, $\gamma_a:I_a\to M$ be the curve defined by $\gamma_a(t)=\gamma(t+a)$ for all $t\in I_a$. Then $\gamma_a$ is also an integral curve of $X$.
2. Let $I^{a}={t|at\in I} (a\neq 0)$, then $\gamma^a:I^a\to M, \gamma^a(t)=\gamma(at)$ is an integral curve of $X^a=aX$

For any $p\in M$, the integral curve starting at $p$ has a maximal interval of existence $J_p$. It is easy to prove that $J_p$ must be open.

Denote the maximal curve as 
$$
\gamma_p:J_p\to M
$$

We have the following property:

> For $X\in \mathfrak{X}(M)$, $\gamma_{\gamma_p(s)}(t)=\gamma_p(t+s)$ if $t,s,t+s\in J_p$.

And $X\in \mathfrak{X}(M)$ is called a **complete vector field** if $J_p=\mathbb R,\forall p\in M$

### Flow

The **flow** is a mapping on set $X$
$$ 
\Phi: X\times \mathbb{R} \to X
$$
such that $\Phi(x,0)=x,\Phi(\Phi(x,s),t) = \Phi(x,s+t)$.

In certain situations one might also consider local flows, which are defined only in some subset $\operatorname{dom}\Phi=\mathcal{X}$ is an open subset of $X\times\mathbb R$.

We can see that a vector field $X$ on $M$ induced a flow on $M$.

We have the following theorem stating the smoothness of such flow:
> For any $X\in \Gamma^{\infty}(TM)$, the flow $\Phi:M\times R\to M$ is smooth on its domain $\operatorname{dom}\Phi=\{(p,t)|p\in M, t\in J_p\}$.

**Proof**

---

By the property of flow, it suffices to prove that $\Phi$ is smooth around any point $(p,0)$. And then we can transfer the smoothness to any point $(q,t)$ by the property of flow.

By the fundamental theorem above, there exists a neighborhood $U$ of $p$ and $\epsilon>0$ such that $\Phi(q,t): U \times (-\epsilon,\epsilon)$ is smooth.
So $\Phi$ is smooth around $(p,0)$.

---

If the vector field $X$ is complete, then the flow $\Phi$ is defined on the whole $M\times \mathbb R$.

We call the flow $\Phi$ of a complete vector field $X$ the **global flow** of $X$. Otherwise, we call it a **local flow**.

### Completeness and the induced diffeomorphism

We shall naturally ask, when is a vector field complete?

A sufficient condition is that the vector field has compact support.

We define the support of a vector field $X\in \mathfrak{X}(M)$ as:
$$
\operatorname{supp}(X) = \overline{\{p\in M|X_p\neq 0\}}
$$

We state the proof idea here:

- For $X_p = 0$, the integral curve is constant, so it is defined on $\mathbb R$.
- For $X_p \neq 0$, then the integral curve $\gamma_p(t)$ shall be always in $C=\operatorname{supp}(X)$. And there exists a $\epsilon_p$ such that $\Gamma(q,t)$ is defined on $U_p\times (-\epsilon_p,\epsilon_p)$. Now since $C$ is compact, we can cover $C$ with finite many $U_{p_i}$, and let $\epsilon = \min \epsilon_{p_i}$. Then for any $q\in C$, $\gamma_q(t)$ is defined on $(-\epsilon,\epsilon)$. And by proper reparametrization, we can extend the integral curve to the whole $\mathbb R$.

As a corollary, a vector field on a compact manifold is always complete.

#### Induced Diffeomorphism

Let $X\in \mathfrak{X}(M)$ be a complete vector field on $M$, then the flow $\Phi:M\times \mathbb R\to M$ induces a one-parameter group of diffeomorphisms $\{\phi_t:M\to M|t\in \mathbb R\}$ defined as:
$$
\phi_t(p) = \Phi(p,t)
$$
And $\phi_t$ satisfies the following properties:
- $\phi_0 = \text{id}_M$
- $\phi_{s+t} = \phi_s \circ \phi_t$ for all $s,t\in \mathbb R$.
- Each $\phi_t$ is a diffeomorphism and $\phi_t^{-1} = \phi_{-t}$.

That is to say, a complete vector field on $M$ induces a family of diffeomorphisms from $M$ to itself.

### Dynamical Systems Induced by Vector Fields

Mathematically, a **dynamical system** is a triple $(T,X,\Phi)$, where $X$ is a set called the **state space**, $T$ is a semi-group of transformation parameters called the **time set**, and $\Phi$ is a mapping from $X \times T$ to $X$ that satisfies the properties of flow. 

So a complete vector field $X$ on a smooth manifold $M$ induces a dynamical system $(\mathbb R, M, \Phi)$, where $\Phi$ is the flow of $X$.

#### Application in Morse Theory

The following theorem is a fundamental result in Morse theory indicating the topological structure of the manifold $M$ is determined by the properties of the vector field $X$ around the critical points.

Let $M$ be a smooth manifold, and $f: M \to \mathbb R$ be a smooth function on $M$. For any $a\in \mathbb R$, we define the **sublevel set** $M_a = \{p\in M | f(p) \leq a\}=f^{-1}((-\infty,a))$

> For $a<b$, assuming that $f^{-1}([a,b])$ is compact and $\forall c\in [a,b]$ is a regular value of $f$, then there exists a diffeomorphism $\varphi:M\to M$ such that $\varphi(M^a)=M^b$ (a deformation retract, more precisely).


**Proof**

First we embed $M$ into a Eulidean space, so a inner product is given in every tangent space $T_pM$.

We define a **gradient vector field** $\nabla f$ as following:
$$
\langle \nabla f|_p,X_p\rangle = df_p(X_p)= X_p(f), \quad \forall X_p\in T_pM
$$
We take a compact support bump function $h$ such that $f^{-1}([a,b])\subseteq \operatorname{supp}(h)\subseteq U$ and $U$ is a open set not containing any critical point.

And 
$$
X:=\frac{h}{\langle \nabla f, \nabla f \rangle} \nabla f
$$
is a compact support vector field on $M$.

Let $\varphi: M \times \mathbb{R} \to M$ be the global flow generated by $X$, then 
$$
\frac{d}{dt}\varphi_t^*f(p) = \frac{d}{dt} f(\varphi_t(p)) = \langle \nabla f, X \rangle = \frac{h}{\langle \nabla f, \nabla f \rangle} \langle \nabla f, \nabla f \rangle = h
$$

So we have $f(\varphi_t(p)) = t + f(p)$ for any $\varphi_t(p)\in f^{-1}([a,b])$.

Thus, the diffeomorphism $\varphi_{b-a}$ maps $M^a$ to $M^b$.

### Lie Derivative

The **Lie derivative** of a function $f\in C^{\infty}(M)$ with respect to a vector field $X\in \mathfrak{X}(M)$ is defined as:
$$
\mathcal{L}_X f = X(f) = \lim_{t\to 0} \frac{f\circ \varphi_t - f}{t}
$$
where $\varphi_t$ is the flow of $X$.

The **Lie derivative** of a vector field $Y\in \mathfrak{X}(M)$ with respect to a vector field $X\in \mathfrak{X}(M)$ is defined as:
$$
\mathcal{L}_X Y = \frac{d}{dt}\bigg|_{t=0} (d\varphi_{-t})_{\varphi_t(p)}(Y_{\varphi_t(p)}) = \lim_{t\to 0} \frac{(d\varphi_{-t})_{\varphi_t(p)}(Y_{\varphi_t(p)}) - Y_p}{t} = [X,Y]
$$
where $\varphi_t$ is the flow of $X$.

The second equality is because for any $f\in C^{\infty}(U)$ defined around $p$:
$$
\begin{align*}
(d\varphi_{-t})_{\varphi_t(p)}(Y_{\varphi_t(p)})(f) &= Y_{\varphi_t(p)}(f\circ \varphi_{-t}) \\ 
&= Y(f\circ \varphi_{-t})\circ \varphi_t(p) \\
&= (\varphi_t^*(Y(f\circ \varphi_{-t})))(p) \\
&= (\varphi_t^*Y\varphi_{-t}^*f)(p)
\end{align*}
$$

And 
$$
\begin{align*}
\frac{d}{dt}\bigg|_{t=0} (d\varphi_{-t}) _{\varphi_t(p)}(Y_{\varphi_t(p)}) f &= \frac{d}{dt}\bigg|_{t=0} (\varphi_t^*Y\varphi_{-t}^*f) \\
&= \frac{d}{dt}\bigg|_{t=0} \varphi_t^*Yf + Y\frac{d}{dt}\bigg|_{t=0} \varphi_{-t}^*f\\
&= XYf-YXf = [X,Y]f
\end{align*}
$$

The Lie derivative is a derivation on the Lie algebra $\mathfrak{X}(M)\to \mathfrak{X}(M)$. That is to say, for any $X,Y,Z\in \mathfrak{X}(M)$ and $a,b\in \mathbb R$, we have the Leibniz rule:
$$
L_X([Y,Z]) = [L_X Y, Z] + [Y, L_X Z]
$$

**Proof**

By the Jacobi identity, we have:
$$
\begin{align*}
L_X([Y,Z]) &= [X,[Y,Z]] \\
&= - [Y,[Z,X]] - [Z,[X,Y]] \\
&= [ [X,Y], Z] + [Y, [X,Z]] \\ 
&= [L_X Y, Z] + [Y, L_X Z]
\end{align*}
$$

So we can see the Jacobi identity as the Leibniz rule of the Lie derivative.

### Lie Group

Let $G$ be a group, we say $G$ is a **Lie group** if it is a smooth manifold and the group operation $\mu:G\times G\to G, (g_1,g_2)\mapsto g_1\cdot g_2$ is smooth mappings.

Examples of Lie groups include:
- $\mathbb{R}^n$ with addition as the group operation.
- $\mathbb{R}^n\setminus\{0\}$ with multiplication as the group operation.
- The general linear group $\operatorname{GL}(n,\mathbb{R})$, which consists of all invertible $n \times n$ matrices with real entries, with matrix multiplication as the group operation.
- The special orthogonal group $\operatorname{SO}(n)$, which consists of all $n \times n$ orthogonal matrices with determinant 1, with matrix multiplication as the group operation.
- $\mathbb S^1$, the circle group, which can be identified with the unit complex numbers under multiplication.
- $\mathbb S^3$, the 3-sphere, which can be identified with unit quaternions under multiplication.

A Lie group must satisfy the following properties:
- The base space $G$ (i.e. the underlying topological space) is orientable.
- The fundamental group $\pi_1(G)$ is a abelian group for a connected Lie group $G$.
- Every Lie group is parallelizable, i.e., its tangent bundle is trivial $TM \cong G \times \mathbb{R}^n$, where $n = \dim G$.


#### Left and Right Multiplication

Assume $G$ is a Lie group, for any $g\in G$, the left multiplication and right multiplication induce two mappings on $G$:
$$
L_g: G \to G, \quad L_g(h) = g\cdot h
$$
$$
R_g : G \to G, \quad R_g(h) = h\cdot g
$$

Note we can write them as the composition of the group operation and the inclusion map:
$$
L_g = \mu \circ l_g=(i_g, \text{id}_G), \quad R_g = \mu \circ r_g=(\text{id}_G, i_g)
$$
where $i_g: \{e\} \to G$ is the inclusion map from the identity element $e$ to $g$.

And $L_g^{-1} = L_{g^{-1}}, R_g^{-1} = R_{g^{-1}}$.

Thus, $L_g$ and $R_g$ are diffeomorphisms on $G$.

To show the usefulness of left and right multiplication, we leveraged them to prove that every Lie group is parallelizable.

**Theorem**
Every Lie group $G$ is parallelizable, i.e., its tangent bundle is trivial $TG \cong G \times \mathbb{R}^n$, where $n = \dim G$.

**Proof** 
Consider 
$$
\begin{align*}
\Phi: G \times T_eG &\to TG \\
(g,v) &\mapsto (g, (dL_g)_e(v))
\end{align*}
$$
where $e$ is the identity element of $G$.
It is easy to see that $\Phi$ is a bijection. And since $L_g$ is a diffeomorphism, $(dL_g)_e$ is a linear isomorphism from $T_eG$ to $T_gG$. So $\Phi$ is a diffeomorphism.

Use the same technique, we can also show that the inverse mapping $g\mapsto g^{-1}$ is a smooth diffeomorphism on $G$.

And the differential of the inverse mapping is given by:
$$
(d\iota)_a X_a = -(dL_{a^{-1}})_e (dR_{a^{-1}})_a X_a
$$

And $(d\mu)_{e,e}(X_e,Y_e) = X_e+Y_e, (d\iota)_e(X_e) = -X_e$.

### Lie Algebra

From the proof above, we see that the tangent space $T_eG$ determines the structure around any point $g\in G$.

So we focus on the structure around the identity element $e$ of the Lie group $G$.

**Definition**

Let $V$ be a vector space over $\mathbb R$ (or any field $\mathbb K$) with a map $[\cdot,\cdot]: V \times V \to V$ satisfying the following properties:
- Antisymmetry: $[X,Y] = -[Y,X]$ for all $X,Y\in V$.
- Linearity: $[aX+bY,Z]=a[X,Z]+b[Y,Z],\forall a,b\in \mathbb R$.
- Jacobi Identity: $[X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0$ for all $X,Y,Z\in V$.

Then $(V,[\cdot,\cdot])$ is called a **Lie algebra** and the map $[\cdot,\cdot]$ is called the **Lie bracket**.

#### Left Invariant Vector Fields

Assume $G$ is a Lie group, $X_e\in T_eG$ is a vector in the tangent space at the identity element $e$. We can define a vector field $X\in \mathfrak{X}(G)$ as follows:
$$
X(g) = (dL_g)_e(X_e)
$$
This vector field is called a **left invariant vector field** because it is invariant under left multiplication, i.e., $dL_g X(h) = X(gh)$ for all $g,h\in G$.

It is obvious that any left invariant vector field is uniquely determined by its value at the identity element $e$.

Denote the set of all left invariant vector fields on $G$ as $\mathfrak{g} = \{X\in \mathfrak{X}(G)| X(g) = (dL_g)_e(X_e), \forall g\in G\}$.

The set $\mathfrak{g}$ is a vector space over $\mathbb R$ and $\mathfrak{g} \simeq T_eG$, $\dim \mathfrak{g} = \dim G$.

For any $X,Y\in \mathfrak{g}, g\in G$, we can verify that:
$$
\begin{align*}
( (dL_g)_e([X,Y]_e) )f &= [X,Y]_e (f\circ L_g) \\
&= (X(Y(f\circ L_g)) - Y(X(f\circ L_g)))(e) \\
&= (XYf-YXf)(g)\\
&= [X,Y]_g f
\end{align*}
$$
Thus $[X,Y]$ is also a left invariant vector field.

So $\mathfrak{g}$ is a Lie algebra with the Lie bracket induced by the Lie bracket of $\mathfrak{X}(G)$, which is called **the Lie Algebra on Lie group $G$**.

**Example**

$\text{GL}(n,\mathbb{R})$ is a open subset of $M(n,\mathbb{R})\cong \mathbb R^{n^2}$

So $\mathfrak{gl}(n,\mathbb{R})$ as the tangent space of $\text{GL}(n,\mathbb{R})$ on $e=I_n$ is isomorphic to $M(n,\mathbb R)$.

And $[X_A,X_B]= X_{[A,B]}$ for all $A,B\in T_{I_n} \text{GL}(n,\mathbb R)$.

### Homomorphism of Lie Groups and Lie Algebras

There are two types of homomorphism, one is the **Lie group homomorphism** and the other is the **Lie algebra homomorphism**, which are morphism between Lie group categories and Lie algebra categories respectively.

Let $G,H$ be Lie groups, if smooth mapping $\phi: G\to H$ is a group homomorphism, we call $\phi$ a Lie group homomorphism. If $\phi$ is a diffeomorphism, we call $\phi$ a Lie group isomorphism.

Let $\mathfrak{g,h}$ be Lie algebras, if a linear mapping $L:\mathfrak{g}\to \mathfrak{h}$ satisfies $L([X,Y]) = [L(X),L(Y)]$ for all $X,Y\in \mathfrak{g}$, we call $L$ a Lie algebra homomorphism. If $L$ is invertible, we call $L$ a Lie algebra isomorphism. Note that if $L$ is invertible, then $L^{-1}$ is also a Lie algebra homomorphism.

**Examples**

For any Lie group $G$, the conjugation mapping $C_g: G\to G$ defined as:
$$
C_g(h) = g h g^{-1} = L_g \circ R_{g^{-1}}(h)
$$
is a Lie group isomorphism for any $g\in G$. The inverse mapping is given by $C_{g^{-1}}$.

For any $X\in GL(n,\mathbb R)$, the adjoint mapping $\operatorname{Ad}_X: \mathfrak{gl}(n,\mathbb R) \to \mathfrak{gl}(n,\mathbb R)$ defined as:
$$
\operatorname{Ad}_X(Y) = XYX^{-1}
$$
is a Lie group isomorphism. The inverse mapping is given by $\operatorname{Ad}_{X^{-1}}$.

#### Induced Lie Algebra Homomorphism

Let $\phi: G\to H$ be a Lie group homomorphism, then the differential of $\phi$ at the identity element $e\in G$, induced a mapping $d\phi: \mathfrak{g} \simeq T_eG \to T_{\phi(e)}H\simeq \mathfrak{h}$, is a Lie algebra homomorphism from the Lie algebra $\mathfrak{g}$ of $G$ to the Lie algebra $\mathfrak{h}$ of $H$.

To write it more explicitly, for any $X\in \mathfrak{g}$, $X_e\in T_eG$ is the value of $X$ at $e$.
$$
\begin{align*}
d\phi_e: T_eG &\to T_{\phi(e)}H \\
X_e &\mapsto (d\phi)_e(X_e) 
\end{align*}
$$
$$
\begin{align*}
d\phi: \mathfrak{g} &\to \mathfrak{h} \\
X &\mapsto [h \mapsto dL_h(d\phi_e(X_e))]
\end{align*}
$$

To see that $d\phi$ is a Lie algebra homomorphism, we need to show that $d\phi([X,Y]) = [d\phi(X), d\phi(Y)]$ for any $X,Y\in \mathfrak{g}$.

For any $f\in C^{\infty}(H)$, we have:
$$
\begin{align*}
(d\phi([X,Y]))_{e_{h}} f &= dL_{e_{h}}(d\phi_e([X,Y]_e)) f \\
&= (d\phi_e([X,Y]_e))(f) \\
&= [X,Y]_e (f\circ \phi) \\
\end{align*}
$$

$$
\begin{align*}
[d\phi(X), d\phi(Y)]_{e_{h}} f &= (d\phi(X) d\phi(Y) f - d\phi(Y) d\phi(X) f)(e_{h}) \\
&= (d\phi(X)(d\phi(Y) f))(e_{h}) - (d\phi(Y)(d\phi(X) f))(e_{h}) \\
&= (d\phi_e(X_e))(d\phi(Y) f) - (d\phi_e(Y_e))(d\phi(X) f) \\
&= X_e (d\phi(Y) f \circ \phi) - Y_e (d\phi(X) f \circ \phi) \\
\end{align*}
$$

To move on, we need:

**Lemma**: 

$$
(d\phi(X)f) \phi = X(f\circ \phi)
$$

**Proof**:
For any $g\in G$, we have:
$$
\begin{align*}
(d\phi(X)f) \phi(g) &= d\phi(X)|_{\phi(g)} f \\
&= (dL_{\phi(g)})_{e_H}(d\phi_e(X_e)) f \\
&= d\phi_g (dL_g)_e (X_e) f \\
&= d\phi_g X_g f \\
&= X_g (f \circ \phi) \\
&= X(f \circ \phi) (g)
\end{align*}
$$

> For the third step, $\phi$ is a group homomorphism, we know that $\phi(g h) = \phi(g)\phi(h)$, so $L_{\phi(g)} \circ \phi = \phi \circ L_g$, differentiating both sides at $e$ gives us $d(L_{\phi(g)})_{e_H} \circ d\phi_e = d\phi_g \circ (dL_g)_e$.

That is to say, the definition of the push forward $d\phi(X)$ by the differential of the diffeomorphism $\phi$ is equivalent to the corresponding vector field induced by the left action $dL_{\phi(\cdot)}$ on $d\phi_e(X_e)$.

So we have:
$$
\begin{align*}
[d\phi(X), d\phi(Y)]_{e_{h}} f &= X_e (d\phi(Y) f \circ \phi) - Y_e (d\phi(X) f \circ \phi) \\
&= X_e (Y (f\circ \phi)) - Y_e (X (f\circ \phi)) \\
&= [X,Y]_e (f\circ \phi) \\
&= (d\phi([X,Y]))_{e_{h}} f
\end{align*}
$$

The above proof can be summerized as the following:

Step 1: $d\phi(X)$ is $\phi$-related to $X$.

For $X \in \mathfrak{g}$, $d\phi(X) \in \mathfrak{h}$ is left-invariant, and for $f \in C^\infty(H)$, since $\phi(g h) = \phi(g) \phi(h)$, we have $L_{\phi(g)} \circ \phi = \phi \circ L_g$. 

Differentiating at $e$: $d(L_{\phi(g)}){e_H} \circ d\phi_e = d\phi_g \circ (dL_g)e.$ Thus: $d\phi(X)|_{\phi(g)} f = (dL{\phi(g)})_{e_H} (d\phi_e(X_e)) f = d\phi_g ((dL_g)_e X_e) f = X_g (f \circ \phi).$ 

So, $d\phi(X) f \circ \phi = X (f \circ \phi)$, meaning $X$ is $\phi$-related to $d\phi(X)$.

Step 2: Lie Bracket Preservation

If $X, Y \in \mathfrak{g}$ are $\phi$-related to $d\phi(X), d\phi(Y) \in \mathfrak{h}$, then $[X, Y]$ is $\phi$-related to $[d\phi(X), d\phi(Y)]$. 

At $g = e$: $[d\phi(X), d\phi(Y)]_{e_H} = d\phi_e ([X, Y]_e).$ Since both are left-invariant, $[d\phi(X), d\phi(Y)] = d\phi([X, Y])$. Linearity of $d\phi$ follows from $d\phi_e$. 

Thus, $d\phi$ is a Lie algebra homomorphism.

#### Adjoint Map

Since every element $g\in G$ gives a isomorphism on Lie group:
$$
\begin{align*}
C_g: G&\to G \\
x&\mapsto gxg^{-1}
\end{align*}
$$
Such map induces a isomorphism on Lie algebra:
$$
\text{Ad}_g = d C_g\big|_e: \mathfrak{g}\to\mathfrak{g}
$$

So this yields a new mapping
$$
\begin{align*}
\text{Ad}: G&\to \text{GL}(\mathfrak g)\\
g&\mapsto \text{Ad}_g
\end{align*}
$$
as a group homomorphism between $G$ and $\text{GL}(\mathfrak g)$, which is called the **adjoint representation** of $G$.

Again, take the induced mapping of $\text{Ad}$, we get
$$
\begin{align*}
\text{ad}: \mathfrak{g} &\to \text{End}(\mathfrak{g}) \\
X &\mapsto d \text{Ad} \big|_e (X)
\end{align*}
$$

We shall see that $\text{ad}(X)(Y) = [X,Y]$.

### Exponential Map

Recall that Lie algebra can be viewed as the tangent space of $T_eG$ of some Lie group $G$. For now, we put whether such $G$ can always be found aside. And a tangent vector $X\in T_eG$ induced a left-invariant vector field $X$ by the left multiplication $L_g$ on $G$. Such vector fields could induce a flow $\Phi^X$ on $G$, i.e. we can build a connection between the Lie algebra and the induced flow on the group. 

We shall ask the following question:

- Is $\Phi^X$ complete?
- How does $\Phi^X$ change regarding different choices of $X$?

The answer to the first question is yes.

Although $G$ is not necessarily compact, but if $U\in e$ we locally have $\Gamma: (-\epsilon, \epsilon) \times U \to G$ as the fundamental theorem states, then we can apply $L_g$ to transform the local maps to anywhere $g$, thus the integral curve can be extended to the whole manifold.

Note 
$$
\phi^X_t(g) = g \phi_t^X(e)
$$
where $\phi^X_t$ are the diffeomorphism induced by the flow $\Phi^X$ at time $t$.

**Exercise**: Prove it.

> Hint: First assume $g\in U$, and show that any $g'\in G$ can be written as finite product of $g_i\in U$

So 
$$
\phi_t^X = R_{\phi_{t}^X(e)}
$$
And 
$$
g\phi_{s+t}^X(e) = \phi_{t}^X(\phi_{s}^X(g)) = \phi_{t}^X(g\phi_{s}^X(e)) = g\phi_{s}^X(e)\phi_{t}^X(e)
$$
for all $g\in G$

Thus
$$
\phi_{s+t}^X(e) = \phi_{s}^X(e)\phi_{t}^X(e)
$$

So the mapping 
$$
\begin{align*}
\rho^X: \mathbb{R} &\to G \\
t &\mapsto \phi^X_t(e)
\end{align*}
$$
is a smooth group homomorphism. 

**Exercise**:
> Prove it is smooth.

So $\rho^X$ is called the **one-parameter subgroup** of $G$.

And every $X\in\mathfrak{g}$ gives rise to a one-parameter subgroup $\rho^X$ of $G$.

Conversely, let $\rho$ be a one-parameter subgroup of $G$. Then 
$$
X_g := \frac{d}{dt}\bigg|_{t=0} R_{\rho(t)}g
$$
is a left-invariant vector field on $G$.

**Proof**:
$$
\begin{align*}
dL_h(X_g) &= dL_h (\frac{d}{dt}\bigg|_{t=0} R_{\rho(t)}g)\\
&= \frac{d}{dt}\bigg|_{t=0}(L_hR_{\rho(t)}g) \\
&= \frac{d}{dt}\bigg|_{t=0}(R_{\rho(t)}hg) \\
&= X_{hg}
\end{align*}
$$

So $\rho^X$ has a bijection with $X\in \mathfrak{g}$, which is the third interpretation of the Lie algebra---the infinitesimal generator of all the one-parameter subgroup of $G$.

**Exercise**
> Now, prove $\text{ad}(X)(Y)=[X,Y]$.

To study the second problem, we define the exponential map
$$
\begin{align*}
\exp: \mathfrak{g} &\to G \\
X &\mapsto \phi^X_1(e)
\end{align*}
$$
Note $\exp(tX) = \phi_{1}^{tX}(e) = \phi_{t}^{X}(e)$.

The exponential map has the following properties:
1. **Differentiable**: The exponential map is a smooth (infinitely differentiable) map.

2. **Local Diffeomorphism**: The exponential map is a local diffeomorphism around the zero vector in the Lie algebra $\mathfrak{g}$. This means that for small enough $X \in \mathfrak{g}$, the map $\exp$ is invertible in a neighborhood of $0 \in \mathfrak{g}$.

3. $\exp(tX)\cdot \exp(sX) = \exp((t+s)X)$

**Examples**

(1). $G=\mathbb R^*$, $\mathfrak{g}=T_1 G= \{c\cdot \frac{d}{dx}\big|_{1}\}\cong \mathbb R$.

To calculate $\exp(X)=\phi_{1}^X(1)$, write $X= x\frac{d}{dx} \in \mathfrak{g}$.

And 
$$
\dot \gamma(t) = X_{\gamma(t)}
$$
Note $X_{\gamma(t)} = \gamma(t) x \frac{d}{d x^0}$
$$
\frac{d}{dt}\gamma(t) = \gamma(t) \cdot x
$$
yields $\gamma(t) = \exp(tx)$.

(2). For $G=(S^1,\cdot)$,
$$
\begin{align*}
\exp: i\mathbb R=T_e S^1&\to S^1 \\
\exp(ix) &\mapsto e^{ix}
\end{align*}
$$
(3). For $G=\text{GL}(n,\mathbb R)$
$$
\begin{align*}
\exp: \mathfrak{gl}(n,\mathbb R)&\to \text{GL}(n,\mathbb R) \\
A &\mapsto e^{A}=\sum_{i=0}^{\infty} \frac{1}{i!}A^i
\end{align*}
$$
where $A^0 = I$.

$\exp:\mathfrak{g} \to G$ is a smooth mapping, and its differential at $e$ is the identity map
$$
(d \exp)_0 = Id_{\mathfrak g}: T_0\mathfrak{g}\simeq \mathfrak{g} \to \mathfrak{g} \simeq T_e G
$$ 

**Proof**
Consider a vector field $\tilde X$ manifold $G\times \mathfrak{g}$, $\tilde X_{(g,X)}=(X_g,0)$

The integral curve just be $\gamma(t) = (g \cdot \exp(tX), X)$

The flow is smooth, thus
$$
\begin{align*}
\mathfrak{g}&\to \mathbb{R}\times G\times \mathfrak g &&\to G\times \mathfrak g &&\to G\\
X &\mapsto (1,e,X) &&\mapsto (\exp(X),X) &&\mapsto \exp(X)
\end{align*}
$$
is smooth.

The identity property is given by the following:
$$
d(exp)_0 X = \frac{d}{dt}\bigg|_{t=0} \exp(tX) = \frac{d}{dt}\bigg|_{t=0} \phi_t^X(e) = X_e
$$

Particularly, $\exp$ is a diffeomorphism around $0$. That is to say, locally around $e$, the Lie group looks like the corresponding Lie algebra.

### Naturality

Given any Lie group homomorphism $\varphi: G\to H$, the diagram

$$
\begin{array}{ccc}
\mathfrak{g} & \xrightarrow{\quad\quad d\varphi\quad\quad} & \mathfrak{h} \\[2em]
\Big\downarrow{\exp_G} & & \Big\downarrow{\exp_H} \\[2em]
G & \xrightarrow{\quad\quad\varphi\quad\quad} & H
\end{array}
$$

commute.

**Exercise**
Proof it
> Hint: For any $X\in \mathfrak g$, $\varphi \circ \exp_{G}(tX)$ represents a one-parameter groups in $H$, what is the infinitesimal generator?

Particularly, for mapping $C_g:G\to G$ and $\text{Ad}: G\to \text{GL}(\mathfrak g)$, we have:
1. $g(\exp(tX))g^{-1} = \exp(t\text{Ad}_g(X))$
2. $\text{Ad}(\exp(tX)) = \exp(t\text{ad}(X))$

### Baker-Campbell-Hausdorff Formula

TODO.


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
$$
A(f)(v_1, v_2, \ldots, v_k) = \frac{1}{k!} \sum_{\sigma \in S_k} \text{sgn}(\sigma) \sigma f
$$   

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
$$ 
(f \wedge g) = \binom{k+l}{k} A(f\otimes g)
$$ 
or equivalently,
$$
(f \wedge g)(v_1, v_2, \ldots, v_{k+l}) = \frac{1}{k!l!} \sum_{\sigma \in S_{k+l}} \text{sgn}(\sigma) f(v_{\sigma(1)}, v_{\sigma(2)}, \ldots, v_{\sigma(k)}) g(v_{\sigma(k+1)}, v_{\sigma(k+2)}, \ldots, v_{\sigma(k+l)}) 
$$

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

Let $\alpha \in \bigwedge^k V, \beta \in \bigwedge^l V$, the inner product satisfies the following properties:
1. **Nilpotency**: $\iota_v \iota_v \alpha = 0$ for any $v \in V$.
2.  $\iota_v(\alpha \wedge \beta) = \iota_v(\alpha) \wedge \beta + (-1)^k \alpha \wedge \iota_v(\beta)$.
3. **Linearity**: $\iota_{av + bw}(\alpha) = a \iota_v(\alpha) + b \iota_w(\alpha)$ for any $a,b \in \mathbb{R}$ and $v,w \in V$.

The inner product $\iota_v$ is called the **interior product** or **contraction** with respect to the vector $v$. It reduces the degree of the form by 1.

Observe the definition, we see that it just the evaluation of the first argument of the alternating multilinear function $\omega$ at the vector $v$, and then take the wedge product of the remaining arguments.

$$ \iota_v(\omega)(v_1, v_2, \ldots, v_{k-1}) = \omega(v, v_1, v_2, \ldots, v_{k-1}) $$

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

In other words, a differential $k$-form on a open subset $U \subseteq M$ is a mapping 
$$
\omega: U \to \bigwedge^k T^*M
$$ 
that assigns to each point $p \in U$ an alternating multilinear function $\omega(p): T_pM^k \to \mathbb{R}$ and $\omega \in A_k(T_pM)$ for all $p \in U$.

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

The **coordinate 1-forms** $dx^i$ are smooth sections of the cotangent bundle, each $dx^i: U \to T^*M$ satisfies 
$$
\pi \circ dx^i = \text{id}_U
$$
where $\pi: T^*M \to M$ is the bundle projection. 

We have $dx^i|_p: T_pM \to \mathbb{R}$ defined by $dx^i|_p(v) = v(x^i(p))$ for any tangent vector $v \in T_pM$. The coordinate 1-forms satisfy the dual basis property:
$$
\left.dx^i\right|_p\left(\left.\frac{\partial}{\partial x^j}\right|_p\right) = \delta^i_j
$$
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
4. **Pullback Compatibility**: For any smooth map $F: M \to N$, we have $F^*(d\omega) = d(F^*\omega)$. One can verify this property using local coordinates and chain rules.
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

### Lie Derivative on Differential Forms
The **Lie derivative** of a differential form $\omega \in \Omega^k(M)$ with respect to a vector field $X$ is defined as:
$$ \mathcal{L}_X \omega = \frac{d}{dt} \bigg|_{t=0} \phi_t^* \omega $$
where $\phi_t$ is the flow of the vector field $X$ at time $t$.

The Lie derivative satisfy the following property:
1. $d\mathcal L_X \omega=\mathcal L_X d\omega$
2. $\mathcal L_X(\omega \wedge \eta)=\mathcal L_X \omega \wedge \eta + \omega \wedge \mathcal L_X\eta$
3. $\mathcal L_{[X_1,X_2]} \omega = \mathcal L_{X_1}\mathcal L_{X_2}\omega - \mathcal L_{X_2}\mathcal L_{X_1}\omega$
4. $(\mathcal L_X{\omega})(X_1,\cdots,X_k)=\mathcal L_X(\omega(X_1,\cdots,X_k))-\sum_i\omega(X_1,\cdots,\mathcal L_X X_i,\cdots,X_k)$
5. **Cartan's Magic Formula**
$$
L_{X}\omega=d\iota_X\omega + \iota_Xd\omega
$$


**Proof**

1. $d\mathcal L_X \omega = d\left(\frac{d}{dt} \bigg|_{t=0} \phi_t^* \omega \right)= \frac{d}{dt} \bigg|_{t=0} d(\phi_t^* \omega) = \frac{d}{dt} \bigg|_{t=0} \phi_t^* (d\omega) = \mathcal L_X d\omega$
2. $\mathcal L_X(\omega \wedge \eta) = \frac{d}{dt} \bigg|_{t=0} \phi_t^* (\omega \wedge \eta) = \frac{d}{dt} \bigg|_{t=0} (\phi_t^* \omega \wedge \phi_t^* \eta) = \frac{d}{dt}\bigg|_{t=0} (\phi_t^*\omega)\wedge \eta+\omega\wedge\frac{d}{dt}\bigg|_{t=0}(\phi_t^*\eta) = \mathcal L_X \omega \wedge \eta + \omega \wedge \mathcal L_X\eta$
3. Use (5).
4. Do induction as (5).
5. For the $0$-form and $1$-form, it is trivial proved to be true. Then inductively prove by decomposing it on local coordinates.
Let $\omega$ be a $k$-form. Locally, we have:
$$
\omega=f dx^1\wedge \cdots \wedge dx^n=dx^1\wedge \omega_1
$$

$$
\begin{align*}
\mathcal L_X \omega &= \mathcal L_X (dx^1) \wedge \omega_1 + dx^1 \wedge \mathcal L_X\omega_1\\
&=[\iota_X d(dx^1)+d(\iota_X dx^1)]\wedge \omega_1+dx^1\wedge [\iota_X d\omega_1+ d(\iota_X \omega_1)]
\end{align*}
$$
$$
\begin{align*}
d\iota_X \omega + \iota_X d\omega &= d\iota_X(dx^1\wedge \omega_1) + \iota_X (-dx^1\wedge d\omega_1)\\
&=d((\iota_Xdx^1)\wedge \omega_1-dx^1\wedge \iota_X \omega_1) -(\iota_X dx^1)\wedge d\omega_1 + dx^1\wedge\iota_X d\omega_1\\
&=d(\iota_X dx^1)\wedge \omega_1+(\iota_Xdx^1)\wedge d\omega_1+dx^1\wedge d(\iota_X\omega_1) -(\iota_X dx^1)\wedge d\omega_1 + dx^1\wedge\iota_X d\omega_1\\
&=d(\iota_X dx^1)\wedge \omega_1+dx^1\wedge d(\iota_X\omega_1) + dx^1\wedge\iota_X d\omega_1\\
&=d(\iota_X dx^1)\wedge \omega_1+dx^1\wedge [\iota_X d\omega_1+d(\iota_X \omega_1)]\\
&=\mathcal L_X \omega
\end{align*}
$$

### Integration of Differential Forms

#### Highest Degree Forms
Let $M$ be a smooth manifold of dimension $n$. The **highest degree differential form** on $M$ is a differential $n$-form, denoted as $\omega \in \Omega^n(M)$. It can be expressed in local coordinates $(U,x)$ as:
$$ \omega = f \, dx^1 \wedge dx^2 \wedge \ldots \wedge dx^n $$
where $f: M \to \mathbb{R}$ is a smooth function and $\{dx^1, dx^2, \ldots, dx^n\}$ is the basis of the cotangent space at each point in $M$.

If we change the local coordinates to $(y^1, y^2, \ldots, y^n)$ by the pullback map $\phi: V \to U$, $y=\phi^*x = x\circ \phi$, then the highest degree form transforms as:
$$ \phi^*(dx^1 \wedge dx^2 \wedge \ldots \wedge dx^n) = \det\left(\frac{\partial y^i}{\partial x^j}\right) dy^1 \wedge dy^2 \wedge \ldots \wedge dy^n $$

We can define the integral of a highest degree form over $U$ as:
$$ \int_U \omega = \int_U f \, dx^1 \wedge dx^2 \wedge \ldots \wedge dx^n= \int_U f dx^1 dx^2 \cdots dx^n $$
where the integral is taken over the open set $U \subseteq M$.

To make integral is well-defined and does not depend on the choice of local coordinates, we need to ensure that the integral is invariant under coordinate changes. 

But according to the change of variables formula on integral over $\mathbb{R}^n$, we only have:
$$
\int_U f dx^1dx^2 \cdots dx^n = \int_V f \left|\det\left(\frac{\partial y^i}{\partial x^j}\right)\right| dy^1 dy^2 \cdots dy^n
$$

So to make the integral well-defined, we need to restrict the transformation to be orientation-preserving, i.e., $\det\left(\frac{\partial y^i}{\partial x^j}\right) > 0$. 

We call the two charts $(U,x)$ and $(V,y)$ **orientation-compactible** if the Jacobian determinant $\det\left(\frac{\partial y^i}{\partial x^j}\right)$ is positive for all points in $U \cap V$.

For a manifold, if all of the transition maps between charts are orientation-preserving, we say that the manifold is **oriented**. And if such a choice of orientation can be made, we call the manifold **orientable**. And we call such atlas $\mathcal{A}$ an **oriented atlas** or **orientation** of the manifold $M$.

#### Integration on Manifolds

Let $M$ be a smooth oriented manifold of dimension $n$. And let $(\rho, U)$ be a smooth partition of unity subordinate to an open cover $\{U_i\}_{i \in I}$ of $M$, where $\rho_i: M \to [0,1]$ are smooth functions such that $\sum_{i \in I} \rho_i = 1$ on $M$ and $\text{supp}(\rho_i) \subseteq U_i$.
The integral of a differential $n$-form $\omega \in \Omega^n(M)$ over the manifold $M$ is defined as:
$$
\int_M \omega = \sum_{i \in I} \int_{U_i} \rho_i \omega
$$
where $\rho_i \omega$ is a highest degree form on $U_i$ and the integral is taken over the open set $U_i$. 

Note $\rho_i \omega$ is compactly supported in $U_i$ and is a smooth differential $n$-form on $U_i$. So the integral $\int_{U_i} \rho_i \omega$ is well-defined and finite.
The integral $\int_M \omega$ is independent of the choice of partition of unity and the open cover $\{U_i\}_{i \in I}$.

#### Change of Variables

Let $\phi: M \to N$ be a smooth map between oriented manifolds $M$ and $N$ of dimension $n$, with orientation $\mathcal{A,B}$ respectively. We say that $\phi$ is **orientation-preserving** if for every chart $(\psi, V)\in \mathcal{B}$, the chart $(\psi \circ \phi, \phi^{-1}(V))$ is orientation-compactible with $\mathcal{A}$. We say that $\phi$ is **orientation-reversing** if for every chart $(\psi, V)\in \mathcal{B}$, the chart $(\psi \circ \phi, \phi^{-1}(V))$ is not orientation-compactible with $\mathcal{A}$.

Actually, mapping $\phi$ is either orientation-preserving or orientation-reversing if $M$ and $N$ are connected. Otherwise, it either preserves or reverses the orientation on each connected component of $M$ and $N$.

The **change of variables formula** for the integral of a differential $n$-form $\omega \in \Omega^n(M)$ under the map $\phi$ is given by:

if $\phi$ is an orientation-preserving map, then:
$$
\int_N \phi^* \omega = \int_M \omega
$$
if $\phi$ is an orientation-reversing map, then:
$$
\int_N \phi^* \omega = -\int_M \omega
$$

The relation of the orientability and the differential forms is given by the following theorem:

**Theorem**: A smooth manifold $M$ is orientable if and only if there exists a nowhere vanishing differential $n$-form $\omega \in \Omega^n(M)$.

And we call such $n$-form $\omega$ a **volume form** on $M$.

### Stokes Theorem

#### Induced Orientation on the Boundary

Let $M^n$ be an oriented smooth manifold with boundary $\partial M$. The boundary $\partial M$ is a smooth manifold of dimension $n-1$. And $\mu$ is a volume form on $M$ that defines the orientation of $M$.

We can define an orientation on $\partial M$ induced by the orientation of $M$ as follows:
At each point $p \in \partial M$, we can choose a local chart $(U_{\alpha}, x^1, x^2, \ldots, x^n)$ around $p$ such that $x^n \geq 0$ on $U_{\alpha}$ and $x^n = 0$ on $U \cap \partial M$. The orientation on $\partial M$ is defined by the volume form:
$$ \eta_{\alpha}=\iota_{e_n} \mu = \iota_{-\frac{\partial}{\partial x^n}} \mu $$
where $e_n = -\frac{\partial}{\partial x^n}$ is the outward-pointing normal vector field on $\partial M$.

Or explicitly, 
$$
\eta_{\alpha} = (-1)^{n} f_{\alpha} dx^1 \wedge dx^2 \wedge \ldots \wedge dx^{n-1}
$$
if $\mu = f_{\alpha} dx^1 \wedge dx^2 \wedge \ldots \wedge dx^n$ on $U_{\alpha}$.

And $\eta$ gives a nowhere vanishing $n-1$ form on $\partial M$.

To integrate a $n-1$-form $\omega$ on $\partial M$, we locally write
$$
\omega =  (-1)^n f\ dx^1\wedge \cdots \wedge dx^{n-1}
$$
Then 
$$
\int_U \omega=\int_V f\ dx^1\cdots dx^{n-1}
$$

The **Stokes Theorem** state that, for any $\omega \in \Omega^{n-1}(M)$
$$
\int_{\partial M} \iota_{\partial M}^* \omega = \int_{M} d\omega
$$

**Proof**

If $\omega$ is compactly support in coordinate chart $(U,x)$

$$
\omega = \sum_{i}(-1)^{i-1}f_i dx^1\wedge\cdots\wedge\widehat{dx^i}\wedge\cdots\wedge dx^n\\
$$
If $U\simeq \mathbb R^{n}$, $\omega$ is $0$ on $\partial M$, then:
$$
\begin{align*}
\int_U d\omega &= \int_{\mathbb R^n}\sum_i \frac{\partial f_i}{\partial x^i} dx^1 \cdots dx^n\\
&= \sum_{i=1}^n \int_{\mathbb R^{n-1}}\left(\int_{-\infty}^{\infty}\frac{\partial f_i}{\partial x^i}dx^i\right) dx^1\cdots\widehat{dx^i}\cdots dx^n\\
&=0=\int_M \iota_{\partial M}^*\omega 
\end{align*}
$$

If $U\simeq \mathbb R^n_+$, then:
$$
\begin{align*}
\int_U d\omega &= \int_{\mathbb R^n_+}\sum_i \frac{\partial f_i}{\partial x^i} dx^1\cdots  dx^n\\
&= \sum_{i=1}^{n-1} \int_{\mathbb R^{n-1}}\left(\int_{-\infty}^{\infty}\frac{\partial f_i}{\partial x^i}dx^i\right) dx^1\cdots\widehat{dx^i}\cdots dx^n + \int_{\mathbb R^{n-1}}\left(\int_{0}^{\infty}\frac{\partial f_n}{\partial x^n}dx^n\right) dx^1\cdots dx^{n-1}\\
&= \int_{\mathbb R^{n-1}} f_n( x^1,\cdots,x^{n-1},0) dx^1\cdots dx^{n-1}\\
&=\int_{\partial M} \iota_{\partial M}^*\omega 
\end{align*}
$$
The last step is because $\iota_{\partial M}^*(dx^i)=dx^i$ for $i=1,\cdots,n-1$ and $\iota_{\partial M}^*(dx^n)=0$.

For general $\omega$, we can use partition of unity to prove it.

## de Rham Theory

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
B^k(M) = \text{im} (d: \Omega^{k-1}(M) \to \Omega^k(M))
$$

Since $d^2 = 0$, we have the following property:
$$
B^k(M) \subseteq Z^k(M) \subseteq \Omega^k(M)
$$

We define the **de Rham cohomology** of $M$ as:
$$ H^k_{dR}(M) = Z^k(M) / B^k(M) $$
and $[\omega]$ is the equivalence class of $\omega \in Z^k(M)$ in $H^k_{dR}(M)$, called the **de Rham cohomology class** of $\omega$.

$H^k_{dR}(M)$ is a vector space over $\mathbb{R}$ and we shall see that it is linear with respect to the cup product.

Example:

For $M = \mathbb{R}$, $Z^0(\mathbb{R}) = \Omega^0(\mathbb{R}) = \mathbb R$, $B^0(\mathbb{R}) = \{0\}$, hence $H^0_{dR}(\mathbb{R}) \simeq \mathbb{R}$, and $H^k_{dR}(\mathbb{R}) \simeq \{0\}$ for $k > 0$.

For $M = \mathbb{S^1}$, we know $H^0_{dR}(\mathbb{S^1}) \simeq \mathbb{R}$ and $H^k_{dR}(\mathbb{S^1}) \simeq \{0\}$ for $k > 1$. So we only need to compute $H^1_{dR}(\mathbb{S^1})$.

$$
\begin{align*}
Z^1(\mathbb{S^1}) = \Omega^1(\mathbb{S^1}) &= \{fd\theta|f\in C^{\infty}(\mathbb{S^1})\}\\
& = \{fd\theta | f\in C^{\infty}(\mathbb R), f \text{ periodic with period } 2\pi\}\\
\end{align*}
$$

Now, consider $\omega \in B^1(\mathbb{S^1})$, we have $\omega = d\eta$ for some $\eta \in \Omega^0(\mathbb{S^1})$. Since $\eta$ is a smooth function on $\mathbb{S^1}$, it can be expressed as $f(\theta)$ for some periodic function $f$ with period $2\pi$. Thus, we have:
$$
\omega = d\eta = df(\theta) = g(\theta)d\theta
$$
where $g$ is a periodic function with period $2\pi$ and $\int_0^{2\pi} g(\theta) d\theta = 0$.
Thus, $H^1_{dR}(\mathbb{S^1}) \simeq \mathbb{R}$, which is the first de Rham cohomology group of the circle since the cohomology class is represented by the integral value of the 1-form over the circle.

Let $\dim M = n$, we have the following results:
- $H^k_{dR}(M) = \{0\}$ for $k > n$.
- $H^0_{dR}(M) \simeq \mathbb{R}^{K}$, where $K$ is the number of connected components of $M$.

### Betti Numbers and Euler Characteristic

If $\dim H^k_{dR}(M) < \infty$ for any $k$, we define the **Betti number** 
$$b_k(M) = \dim H^k_{dR}(M)$$
which is called the $k$-th Betti number of the manifold $M$.

The **Euler characteristic** of $M$ is defined as:
$$ \chi(M) = \sum_{k=0}^{\dim M} (-1)^k b_k(M) $$

From the above result, we know $\chi(\mathbb R) = 1 - 0 = 1$, and $\chi(\mathbb{S^1}) = 1 - 1 = 0$.

### Pullback 

Let $F: M \to N$ be a smooth map between manifolds, the pullback of the de Rham cohomology is defined as:
$$ F^*: H^k_{dR}(N) \to H^k_{dR}(M) $$
by $F^*([\omega]) = [F^*\omega]$ for $\omega \in Z^k(N)$.

Since $dF^*= F^*d$, $F^*$ maps closed forms to closed forms and exact forms to exact forms. 
$$ F^*(B^k(N)) \subseteq B^k(M) \quad \text{and} \quad F^*(Z^k(N)) \subseteq Z^k(M) $$

Hence $F^*$ induces a well-defined map on cohomology.

Note that the pullback of de Rham cohomology satisfies the contravariant functoriality:
1. $(F \circ G)^* = G^* \circ F^*$ for smooth maps $G: L \to M$ and $F: M \to N$.
2. $(\text{id}_M)^* = \text{id}_{H^*_{dR}(M)}$.

Thus, the de Rham cohomology is a contravariant functor from the category of smooth manifolds to the category of graded algebras over a field. So a diffeomorphism $F: M \to N$ induces an isomorphism $F^*: H^*_{dR}(N) \to H^*_{dR}(M)$.

Particularly, $b_k(M)$ and $\chi(M)$ are diffeomorphism invariants.

### Cup Product

Note that the wedge product $\wedge$ on differential forms induces a operation on de Rham cohomology, called the **cup product**.
$$ \cup: H^k_{dR}(M) \times H^l_{dR}(M) \to H^{k+l}_{dR}(M) $$
defined by
$$ [\omega_1] \cup [\omega_2] = [\omega_1 \wedge \omega_2] $$
for $\omega_1 \in Z^k(M)$ and $\omega_2 \in Z^l(M)$.


Assume $\omega_1 = d\eta_1$ and $\omega_2 = d\eta_2$, $\omega,\zeta$ are closed forms, to prove cup product is well-defined, we calculate 
$$
\begin{align*}
(\omega + d\eta_1) \wedge (\zeta + d\eta_2) &= \omega \wedge \zeta + \omega \wedge d\eta_2 + d\eta_1 \wedge \zeta + d\eta_1 \wedge d\eta_2\\
&= \omega \wedge \zeta + \omega \wedge d\eta_2 + (-1)^{k} d\omega \wedge \eta_2 - (-1)^{k} d\omega \wedge d\eta_2 \\
&+ d\eta_1 \wedge \zeta + d\eta_1 \wedge d\eta_2\\
&= \omega \wedge \zeta + d((-1)^{k} \omega \wedge \eta_2) + d\eta_1 \wedge \zeta + d\eta_1 \wedge d\eta_2 \\ 
&= \omega \wedge \zeta + d((-1)^{k} \omega \wedge \eta_2) + d(\eta_1 \wedge \zeta) + d(\eta_1 \wedge d\eta_2)\\
&= \omega \wedge \zeta + d\left ((-1)^{k} \omega \wedge \eta_2 + \eta_1 \wedge \zeta + \eta_1 \wedge d\eta_2\right)
\end{align*}
$$
Therefore, we have
$$ 
[\omega + d\eta_1] \cup [\zeta + d\eta_2]  = [\omega \wedge \zeta]
$$
so the cup product is well-defined.

The cup product is bilinear, associative, and commutative up to a sign:
$$ [\omega_1] \cup [\omega_2] = [\omega_1 \wedge \omega_2] = [(-1)^{kl} \omega_2 \wedge \omega_1] = (-1)^{kl} [\omega_2] \cup [\omega_1] $$

The cup product induces a graded algebra structure on the de Rham cohomology:
$$ H^*_{dR}(M) = \bigoplus_{k=0}^{\dim M} H^k_{dR}(M) $$
with the cup product $\cup$ as the multiplication operation.

And $(H^*_{dR}(M), +, \cup)$ becomes a graded commutative ring with identity, called the **de Rham cohomology ring** of $M$.

### Homotopy Invariance

We say $M$ and $N$ are homotopy equivalent if there exist continuous maps $f: M \to N$ and $g: N \to M$ such that $g \circ f$ is homotopic to the identity map on $M$ and $f \circ g$ is homotopic to the identity map on $N$.

If $M$ and $N$ are homotopy equivalent, then $H^k_{dR}(M) \simeq H^k_{dR}(N)$ for all $k$. This is stronger than the fact that $H^k_{dR}(M)$ is a diffeomorphism invariant, as homotopy equivalence does not even require the manifolds to have the same dimension, e.g. $\mathbb R^n\setminus \{0\}$ and $\mathbb S^{n-1}$.

Note: This shows that the de Rham cohomology is determined by the topological structure of the manifold, not relevant to the smooth structure.

#### Poincaré Lemma

The **Poincaré Lemma** states that if $U$ is a star-shape area in $\mathbb R^n$, then $H^k_{dR}(U)=0$ for $k>1$. Particularly, $H^k_{DR}(\mathbb R^n) = 0$.

This is trivial since the star shape area can always contract to a point.

Moreover, since for any point in a manifold has a neighborhood that is diffeomorphic to a star shape area in $\mathbb R^n$, we have the following collary:

For any $k$-th closed form $\omega \in Z^k(M)$ and any $p\in M$, there is a neighborhood $U\ni p$ and $(k-1)$-form $\eta\in \Omega^k(U)$ such that $\omega = d \eta$ on $U$.

#### Proof of Homotopy Invariance

It suffices to prove that functor $f\rightsquigarrow  f^*$ is homotopic invariant:

> If two smooth maps $f,g: M \to N$ are homotopic, then $f^* = g^*: H^k_{dR}(N) \to H^k_{dR}(M)$. 

Because assume this holds, if $M$ and $N$ are homotopy equivalent, we can use smooth approximations of the continuous maps to get smooth maps $f: M \to N$ and $g: N \to M$ such that $g \circ f$ and $f \circ g$ are homotopic to the identity maps on $M$ and $N$, respectively. Then we have
$$ (g \circ f)^* = f^* \circ g^* = \text{id}_{H^*_{dR}(M)} $$
$$ (f \circ g)^* = g^* \circ f^* = \text{id}_{H^*_{dR}(N)} $$
which implies that $f^*$ and $g^*$ are isomorphisms.

Now we prove the claim that if $f,g: M \to N$ are homotopic, then $f^* = g^*$.

We define the **cochain homotopy** as follows:
> if $f,g\in C^{\infty}(M,N)$ are homotopic. If there is a sequence of mapping $h_k:\Omega^k(M) \to \Omega^{k-1}(M)$ satisfying:
$$ g^*-f^* = d_{M}h_k+h_{k+1}d_{N}.$$
we say the sequence $h=(h_k)$ is a cochain homotopy between $f^*$ and $g^*$.

If there exists such a cochain homotopy $h$, for any $\omega \in Z^k_{dR}(N)$

$$
g^*\omega - f^* \omega = (dh_{k}+h_{k+1}d)\omega = d(h_k\omega) +h_{k+1}(d\omega) = d(h_k\omega) \in B^k(M)
$$

So $f^*([\omega])=[f^*\omega] = [g^* \omega] = g^*([\omega])$.

Now we prove the existence of the cochain homotopy:

First, we prove a lemma

---
Let $X$ be a complete vector field on $M$, $\phi_t$ be the flow generated by $X$. Then there exists a linear operator $Q_k:\Omega^k(M)\to\Omega^{k-1}(M)$ s.t. :
$$
\phi_1^*\omega-\omega= d Q_k(\omega)+ Q_{k+1}(d\omega)
$$
**Proof**
$$
\frac{d}{dt}\phi^*_t\omega=\left.\frac{d}{ds}\right|_{s=0}\phi^*_{s+t}=\left.\frac{d}{ds}\right|_{s=0}\phi^*_{s}\phi^*_{t}\omega=\mathcal{L}_X(\phi_t^*\omega)=d\iota_X(\phi^*_t\omega)+\iota_X d(\phi^*_t\omega)
$$
Therefore, denote $Q_k(\omega)=\int_{0}^{1}\iota_X(\phi_t^*\omega)dt$
$$
\phi_1^*\omega-\omega = \int_{0}^{1}\left (\frac{d}{dt}\phi^*_t\omega\right ) = dQ_k(\omega) + Q_{k+1}(d\omega)
$$

---

Now we can construct the cochain homotopy between $f^*$ and $g^*$ as follows:
Let $W=M\times R$, then $X=\frac{\partial}{\partial t}$ is a complete vector field on $W$. Let $\phi_t: W \to W$ be the flow generated by $X$, then $\phi_t(x,s)=(x,s+t)$ is a smooth map.

By the Lemma, we have a linear operator $Q_k: \Omega^k(W) \to \Omega^{k-1}(W)$ such that:
$$
\phi_1^*\omega - \omega = d Q_k(\omega) + Q_{k+1}(d\omega) 
$$

By the Whitney Approximation theorem, we can find a smooth homotopy $H: M \times [0,1] \to N$ such that $H(x,0) = f(x)$ and $H(x,1) = g(x)$ for all $x \in M$.

Let $\iota:M \hookrightarrow W$ be the inclusion map, $\iota(x) = (x,0)$, then 
$$
f = H \circ \iota, \quad g = H \circ \phi_1 \circ \iota
$$
Then we have:
$$
g^*\omega - f^*\omega = \iota^* \phi_1^* H^* \omega - \iota^* H^* \omega = \iota^* (dQ_k+Q_{k+1}d)H^* \omega = (d \iota^* Q_k H^* + \iota^* Q_{k+1} H^* d)\omega
$$

So $h_k = \iota^* Q_k H^*$ and $h_{k+1} = \iota^* Q_{k+1} H^*$ is a cochain homotopy between $f^*$ and $g^*$.

### de Rham Theorem

The famous **de Rham theorem** states that the de Rham cohomology is isomorphic to the singular cohomology with real coefficients, i.e., for any smooth manifold $M$,
$$ H^k_{dR}(M) \simeq H^k_{sing}(M; \mathbb{R}) $$

which we shall not prove here. 

This theorem reveals the duality between the topological structure and the algebraic structure (differential forms) on a manifold.

### Chain Complex

The **chain complex** is a sequence of abelian groups (or modules) $\cdots A_0, A_1, A_2\cdots$ connected by homomorphisms $d_n: A_n\to A_{n-1}$, such that the image of one homomorphism is contained in the kernel of the next. The composition of any two consecutive maps shall be the zero maps, $d_n \circ d_{n+1} = 0$ or $d^2=0$ for short. 
$$ \cdots \xleftarrow{d_0} A_0 \xleftarrow{d_1} A_1 \xleftarrow{d_2} A_2 \xleftarrow{d_3} \cdots$$

The **cochain complex** is the dual notion to the chain complex.

$$ \cdots \xrightarrow{d^0} A^0 \xrightarrow{d^1} A^1 \xrightarrow{d^2} A^2 \xrightarrow{d^3} \cdots$$
where $d^{n}\circ d^{n+1} = 0$.

The elements in the kernel of $d$ are called **(co)cycles** (or **closed** elements), and the elements in the image of $d$ are called **(co)boundaries** (or **exact** elements). Right from the definition of the differential, all boundaries are cycles. The $n$-th **(co)homology** group $H_n(H^n)$ is the group of (co)cycles modulo (co)boundaries in degree $n$, that is,

$$ H_n = \ker d_n/\text{im } d_{n+1} \left ( H^n = \ker d^n/\text{im } d^{n-1}\right)$$

A **exact sequence** is a (co)chain complex whose (co)homology groups are all zero, which means all closed elements are exact. A short exact sequence is a bounded exact sequence in which only the groups $A_k$, $A_{k+1}$, $A_{k+2}$ may be nonzero. For example, the following chain complex is a short exact sequence.

$$ \cdots \rightarrow 0 \rightarrow Z \xrightarrow {\times p} Z \twoheadrightarrow Z/p \rightarrow 0 \rightarrow \cdots $$

### de Rham Complex

The **de Rham complex** is the sequence of differential forms:
$$ 0 \to \Omega^0(M) \xrightarrow{d} \Omega^1(M) \xrightarrow{d} \Omega^2(M) \xrightarrow{d} \ldots \xrightarrow{d} \Omega^n(M) \to 0 $$

#### The Zig-Zag Lemma

TODO.

### Mayer-Vietoris Sequence

TODO.

## Poincaré Duality

### Singular Homology

TODO.

### Poincaré Duality Theorem

TODO.

## Hodge Theory

TODO.

## Sheaf Theory 

TODO.