---
title: Renormalization Group Flow as Optimal Transport
date: 2025-08-21 20:07:27
tags:
- Renormalization Group
- Optimal Transport
categories:
- Physics
---

## Reference 

[Renormalization Group Flow as Optimal Transport](https://arxiv.org/pdf/2202.11737)

## Introduction

I have very little knowledge in physics, so there might be many errors in this post. If you see any errors, please point it out in the comment!

Renormalization theory, to put it simply, is a theory describing that every theory is the optimal theory under our capability of understanding, thus might be only effective to some extent.

## Ising Model

### The Lattice

A $d$-dimensional lattice $\mathbb Z^d$ is the set of all integer points in $d$-dimensional Euclidean space $\mathbb R^d$. To make it easier to deal with, we usually take a subset $L\subseteq \mathbb Z^d$ and put **periodic boundary conditions** on $L$ (like the Snake game).

Every point $i=(i_1,\ldots,i_d)\in \mathbb Z^d$ in the lattice is called a **site**.

For every two neighbors $i,j$, if $\|i-j\|_{1}=1$, we call them **nearest neighbors**, denoted as $\langle i,j \rangle$.

### Configuration Space

For every site $i\in L$, we have a state $s_i\in V$, where $V=\{-1,1\}$ is the set of all possible states of the system.

A configuration $\sigma$ is a function $\sigma:L\to V$, which means every site $i\in L$ has a state $\sigma(i)\in V$.

The configuration space $\Lambda$ is the set of all possible configurations, i.e., $\Lambda = \{ \sigma : L \to V \}\cong V^{N}$, where $N=|L|$ is the number of sites in $L$.

### Operators

An operator $O$ is a functional $O:\Lambda\to \mathbb R$, which means it takes a configuration $\sigma$ and returns a real number.

The most important operator is the **Hamiltonian** $H$, which describes the energy of a configuration $\sigma$.

#### Symmetry

We may constrain the possible configurations by some symmetry. 

We say a physical system $(\mathcal S, S, \mathcal O)$ with state space $\mathcal S$, law of dynamics $S$ and observables $\mathcal O$ has a symmetry $T:\mathcal{S} \to \mathcal{S}$ if for any observable $O \in \mathcal{O}$, we have $O(T(\sigma)) = O(\sigma)$ for all $\sigma \in \mathcal{S}$.

So we can use the symmetry to reduce the number of operators we need to consider. 

For the Ising model, we mainly focus on two operators:

#### Magnetization Operator
$$
O_M(\sigma) = \sum_{i\in L} h_i \sigma(i)
$$

#### Nearest-Neighbor Interaction Operator
$$
O_{NN}(\sigma) = \sum_{\langle i,j\rangle} J_{i,j} \sigma(i)\sigma(j)
$$

The Hamiltonian is defined as a linear combination over a basis $\{O\}$ of the linear vector space $\Lambda$.

For the Ising model, 
$$
H(J,h;\sigma) = O_{NN}(\sigma) + h O_M(\sigma) = \sum_{\langle i,j\rangle} J_{i,j} \sigma(i)\sigma(j) + \sum_{i\in L}h_i\sigma(i)
$$

where $J,h$ are coupling constants describing the strength of the interaction and the external field.

When the external field is zero everywhere, $h = 0$, the Ising model is symmetric under switching the value of the spin in all the lattice sites; a nonzero field breaks this symmetry.

### The Partition Function

The partition function $Z$ is a function of the Hamiltonian $H$ and the temperature $T$, defined as:
$$
Z(J,h,\beta) = \sum_{\sigma \in \Lambda} e^{-\beta H(J,h;\sigma)}
$$

The configuration probability is given by the Boltzmann distribution
$$
P(\sigma) = \frac{1}{Z(J,h,\beta)} e^{-\beta H(J,h;\sigma)}
$$
where $\beta = 1/k_{B} T$, and $k_{B}$ is the Boltzmann constant.

## Continuous Generalization

### From Lattice to Continuum

Let the distance between two points approach $0$, then $\sigma\in \Lambda$ becomes a real scalar field $\phi(x)$ on $\Omega \subset \mathbb R^d$.

The Hamiltonian $H$ becomes the action functional $S$ as the integral of the Lagrangian density
$$
S[\phi] = \int_{\Omega} \mathcal{L}(\phi,\partial_\mu \phi) d^d x
$$

The partition function becomes the functional integral:
$$
Z[\phi] = \int \mathcal{D}\phi \, e^{-S[\phi]} 
$$

### Fourier Transform Convention and Momentum Space

**Convention**: We use the Fourier transform convention:
$$
\phi(x) = \int \frac{d^d p}{(2\pi)^d} e^{ip \cdot x} \tilde{\phi}(p)
$$
$$
\tilde{\phi}(p) = \int d^d x \, e^{-ip \cdot x} \phi(x)
$$

where $p \cdot x = p_\mu x^\mu$ (Einstein summation convention), and we set $\hbar = c = 1$.

**Key Identity**: The crucial relationship between position and momentum space derivatives is:
$$
\frac{\partial}{\partial x^\mu} \phi(x) \leftrightarrow i p_\mu \tilde{\phi}(p)
$$

**Derivation of the duality**:
Starting from the Fourier transform:
$$
\frac{\partial \phi(x)}{\partial x^\mu} = \frac{\partial}{\partial x^\mu} \int \frac{d^d p}{(2\pi)^d} e^{ip \cdot x} \tilde{\phi}(p)
$$

Since $\frac{\partial}{\partial x^\mu} e^{ip \cdot x} = \frac{\partial}{\partial x^\mu} e^{ip_\nu x^\nu} = i p_\mu e^{ip \cdot x}$, we get:
$$
\frac{\partial \phi(x)}{\partial x^\mu} = \int \frac{d^d p}{(2\pi)^d} (i p_\mu) e^{ip \cdot x} \tilde{\phi}(p)
$$

This shows that **differentiation in position space becomes multiplication by $ip_\mu$ in momentum space**.

### Momentum Space Integrals

**Parseval's theorem** for scalar fields gives us:
$$
\int d^d x \, \phi(x)^2 = \int \frac{d^d p}{(2\pi)^d} \tilde{\phi}(p) \tilde{\phi}(-p)
$$

**For derivatives**: Using the duality $\partial_\mu \leftrightarrow i p_\mu$:
$$
\int d^d x \, (\partial_\mu \phi(x))^2 = \int \frac{d^d p}{(2\pi)^d} (i p_\mu)^2 \tilde{\phi}(p) \tilde{\phi}(-p) = \int \frac{d^d p}{(2\pi)^d} p_\mu^2 \tilde{\phi}(p) \tilde{\phi}(-p)
$$

More generally:
$$
\int d^d x \, (\partial_\mu \phi)(\partial^\mu \phi) = \int \frac{d^d p}{(2\pi)^d} p^2 \tilde{\phi}(p) \tilde{\phi}(-p)
$$
where $p^2 = p_\mu p^\mu = |\vec{p}|^2$ in Euclidean space.

### Transformation of Operators

Now we can transform our operators to momentum space:

**Magnetization operator** (with external source $j(x)$):
$$
O_M[\phi] = \int d^d x \, j(x) \phi(x) \to \int \frac{d^d p}{(2\pi)^d} \tilde{j}(p) \tilde{\phi}(-p)
$$

**Kinetic term**:
$$
\int d^d x \, \frac{1}{2}(\partial_\mu\phi)^2 \to \int \frac{d^d p}{(2\pi)^d} \frac{1}{2} p^2 \tilde{\phi}(p) \tilde{\phi}(-p)
$$

**Mass term**:
$$
\int d^d x \, \frac{1}{2} m^2 \phi^2 \to \int \frac{d^d p}{(2\pi)^d} \frac{1}{2} m^2 \tilde{\phi}(p) \tilde{\phi}(-p)
$$

### The Action in Momentum Space

If we assume the interaction coupling constants are uniform, we can write the action as:
$$
S[\phi] = \int d^d x \left[\frac{1}{2}(\partial_{\mu}\phi)^2+\frac{1}{2}m^2\phi^2+\frac{u_0}{4!}\phi^4+j(x)\phi(x)\right]
$$

**In momentum space**, this becomes:
$$
S[\tilde{\phi}] = \int \frac{d^d p}{(2\pi)^d} \left[\frac{1}{2}(p^2+m^2)\tilde{\phi}(p)\tilde{\phi}(-p) + \tilde{j}(p)\tilde{\phi}(-p)\right ] + S_{int}[\tilde{\phi}]
$$

where $\tilde{j}(p)$ is the Fourier transform of $j(x)$ and $S_{int}[\tilde{\phi}]$ contains the interaction terms.

## Propagator

### Definition

After defining the behavior of the field $\phi$ $(S[\phi])$ and its statistical distribution $P[\phi]= \exp(-S[\phi])$, we shall ask the fundamental question---What is the relation between the different point in a field.

In other words, if I know the value $\phi(x)$ at point $x$, what can I say about the value $\phi(y)$ at another point $y$?

We introduce the **propagator** (also Green's function, two-point correlation function)

$$
G(x,y) \equiv \langle \phi(x) \phi(y) \rangle = \frac{1}{Z} \int \mathcal{D}\phi \, \phi(x) \phi(y) e^{-S[\phi]}
$$
as the expectation value of the product of the fields at two different points.



In the momentum space, the propagator is defined as:
$$
D(p) \equiv \langle \tilde{\phi}(p) \tilde{\phi}(-p) \rangle = \frac{1}{Z} \int \mathcal{D}\tilde{\phi} \, \tilde{\phi}(p) \tilde{\phi}(-p) e^{-S[\tilde{\phi}]}
$$

The relation is given by the Fourier transform:

$$
\begin{align*} G(x, y) &= \langle \phi(x)\phi(y) \rangle \\ 
&= \left\langle \left( \int \frac{d^d p}{(2\pi)^d} e^{ip \cdot x} \tilde{\phi}(p) \right) \left( \int \frac{d^d q}{(2\pi)^d} e^{iq \cdot y} \tilde{\phi}(q) \right) \right\rangle \\
&= \int \frac{d^d p}{(2\pi)^d} \int \frac{d^d q}{(2\pi)^d} e^{ip \cdot x} e^{iq \cdot y} \langle \tilde{\phi}(p)\tilde{\phi}(q) \rangle \\ 
&= \int \frac{d^d p}{(2\pi)^d} \int \frac{d^d q}{(2\pi)^d} e^{ip \cdot x} e^{iq \cdot y} \left( (2\pi)^d \delta^{(d)}(p+q) D(p) \right) \\
\end{align*} 
$$
Let $q = -p$：
$$ 
\begin{align*} G(x, y) &= \int \frac{d^d p}{(2\pi)^d} e^{ip \cdot x} e^{-ip \cdot y} D(p) \\ 
&= \int \frac{d^d p}{(2\pi)^d} e^{ip \cdot (x-y)} D(p) 
\end{align*}
$$

### The Source Field Method (as Partitional Generating Function)

Calculate the propagator directly from the definition is very complicated due to the presence of interactions. 

For the **free field** (no interactions, $u_0 = 0$), the action is quadratic:
$$
S_0[\tilde{\phi}] = \int \frac{d^d p}{(2\pi)^d} \frac{1}{2}(p^2+m^2)\tilde{\phi}(p)\tilde{\phi}(-p)
$$

If we calculate the propagator directly, we get:

$$
\begin{align*}
D(p) = \frac{1}{Z} \int \mathcal{D}\tilde{\phi} \, \tilde{\phi}(p) \tilde{\phi}(-p) e^{-S_0[\tilde{\phi}]} \\
= \frac{\int \mathcal{D}\tilde{\phi} \, \tilde{\phi}(p) \tilde{\phi}(-p) e^{-S_0[\tilde{\phi}]}}{\int \mathcal{D}\tilde{\phi} \, e^{-S_0[\tilde{\phi}]}}\\
\end{align*}
$$

To calculate this propagator, we need to evaluate the Gaussian integrals involved.

But actually, we can use the technique of generating function.

Recall that, for a random variable $X$, we can define its generating function as:

$$
G_X(t) = \mathbb{E}[e^{tX}] 
$$
And its moment $\mathbb{E}[X^n]$ can be obtained by differentiating $G_X(t)$ $n$ times and evaluating at $t=0$:

Now we are dealing with a field $\phi$ and functional partition function $Z[\phi] = \mathbb{E}[\exp(-S[\phi])]$.

Similarly, we introduce a source field $J$ and define the partition function with source $J$ as:
$$
Z[J] = \mathbb{E}[\exp(-S[\phi] + \int d^d x J(x) \phi(x))]
$$
or, in momentum space 
$$
Z[\tilde J] = \mathbb{E}[\exp(-S[\tilde{\phi}] + \int d^d p \tilde J(p) \tilde{\phi}(-p))]
$$
Take the functional derivative of $Z[J]$, we obtain:
$$
\frac{\delta}{\delta \tilde J(p)} Z[\tilde J] = \mathbb{E}[\tilde{\phi}(-p) \exp(-S[\tilde{\phi}] + \int d^d \tilde J(p) \tilde \phi(-p))]
$$
$$
\frac{\delta}{\delta \tilde J(p)}\frac{\delta}{\delta \tilde J(-p)} Z[\tilde J] = \mathbb{E}[\tilde{\phi}(-p) \tilde{\phi}(p) \exp(-S[\tilde{\phi}] + \int d^d \tilde J(p) \tilde \phi(-p))]
$$
And evaluate it at $\tilde J = 0$, this is exactly the definition of propagator:
$$
D(p) = \frac{1}{Z} \left. \frac{\delta}{\delta \tilde J(p)}\frac{\delta}{\delta \tilde J(-p)} Z[\tilde J] \right|_{\tilde J = 0} = \left. \frac{\delta}{\delta \tilde J(p)}\frac{\delta}{\delta \tilde J(-p)} W[\tilde J] \right|_{\tilde J = 0}
$$
where $W[\tilde J] = \ln Z[\tilde J]$.

For the free field,
$$
\begin{align*}
Z_0[\tilde J] &= \int \mathcal{D}\tilde{\phi} \, \exp \left[ -\frac{1}{2} \int \frac{d^d p}{(2\pi)^d} (p^2 + m^2) \tilde{\phi}(p) \tilde{\phi}(-p) + \int d^d p \tilde J(p) \tilde{\phi}(-p) \right]\\
&=\int \mathcal{D} \tilde \phi \exp\left(-\int\frac{d^d p}{(2\pi)^d}\left[\frac{1}{2}\tilde\phi'(-p)(p^2+m^2)\tilde\phi'(p)+(p^2+m^2)^{-1}\tilde J(p)\tilde J(-p)\right]\right)\\
&= Z_0[0] \exp \left[ \frac{1}{2} \int \frac{d^d p}{(2\pi)^d} (p^2 + m^2)^{-1} \tilde J(p) \tilde J(-p) \right]
\end{align*}
$$
where $\tilde\phi(p) = \tilde \phi'(p) + D(p)\tilde J(p)$. 

This substitution is called the completing the square technique.

The propagator is:
$$
D(p) = \frac{1}{p^2 + m^2}
$$

This gives the familiar result that in position space, the two-point correlation function is:
$$
\langle \phi(x) \phi(y) \rangle = \int \frac{d^d p}{(2\pi)^d} \frac{e^{ip \cdot (x-y)}}{p^2 + m^2}
$$

### Connection to Exact RG

This momentum space formulation naturally leads to the **exact renormalization group** approach, where we systematically integrate out high-momentum modes while keeping track of how the effective action changes. The cutoff function $K_\Lambda(p^2)$ mentioned in the paper provides a smooth way to suppress high-momentum modes above the scale $\Lambda$.

## Renormalization

If we directly calculate the correlation, we will encounter infinity when $d>4$. 

### Introduction

TODO.

### Polchinski Equation

Here is the detailed version of section $2.1.1$ in the paper.

$$
Z_{\Lambda}[J] =\int \mathcal D\phi \exp (-S_{\Lambda}[\phi,J]) =\int \mathcal D\phi \, \exp \left[ -\frac{1}{2} \int \frac{d^d p}{(2\pi)^d} \left(\phi(p)\phi(-p) (p^2 + m^2) K_\Lambda(p^2)^{-1} + J(p) \phi(-p)\right) - S_{int,\Lambda} [\phi] \right]
$$

where $S_{int,\Lambda}[\phi]$ is the interaction term that depends on the cutoff scale $\Lambda$, $K_\Lambda(p^2)$ is a soft cutoff function, i.e. it is $1$ for $p^2 < \Lambda^2$ and smoothly approach $0$ for $p^2 > \Lambda^2$.

Then we consider a small scale $\Lambda_{R}$, which is the scale at which we want to integrate out the high-momentum modes.

We want the probability functional shall not change under the perturbation of our scale $\Lambda$, so the partition function can only change by multiple amount inrelevent to $\phi$ and only relevent to $\Lambda$.
$$
- \Lambda \frac{d}{d \Lambda} Z_{\Lambda}[J] = C_{\Lambda} Z_{\Lambda}[J]
$$

Calculate the LHS directly
$$
\begin{align*}
-\Lambda \frac{d}{d \Lambda} Z_{\Lambda}[J] &= -\Lambda \int \mathcal D\phi \, \left[ \exp (-S_{\Lambda}[\phi,J]) \cdot \left(-\frac{dS_{\Lambda}[\phi,J]}{d\Lambda}\right) \right] \\
&= \int \mathcal D\phi \, \left( \Lambda \frac{dS_{\Lambda}[\phi,J]}{d\Lambda} \right) \exp (-S_{\Lambda}[\phi,J]) \\
&=\int \mathcal D\phi \, \left( \frac{1}{2}\int \frac{d^d p}{(2\pi)^d}  \phi(p) \phi(-p) (p^2 + m^2) \Lambda \frac{\partial K_\Lambda^{-1}(p^2)}{\partial \Lambda} + \Lambda \frac{\partial S_{int,\Lambda}[\phi]}{\partial \Lambda} \right) \exp (-S_{\Lambda}[\phi,J])\\
&=Z_{\Lambda}[J]\cdot\left\langle \frac{1}{2}\int \frac{d^d p}{(2\pi)^d} \phi(p) \phi(-p) (p^2 + m^2) \Lambda \frac{\partial K_\Lambda^{-1}(p^2)}{\partial \Lambda} + \Lambda \frac{\partial S_{int,\Lambda}[\phi]}{\partial \Lambda}  \right\rangle \bigg|_{J}
\end{align*}
$$

So
$$
- \Lambda \frac{1}{Z_{\Lambda}[J]}\frac{d}{d \Lambda} Z_{\Lambda}[J]=-\Lambda \frac{d \ln Z_{\Lambda}[J]}{d\Lambda}=C_{\Lambda} = \left\langle \frac{1}{2}\int \frac{d^d p}{(2\pi)^d}\phi(p) \phi(-p) (p^2 + m^2) \Lambda \frac{\partial K_\Lambda^{-1}(p^2)}{\partial \Lambda} + \Lambda \frac{\partial S_{int,\Lambda}[\phi]}{\partial \Lambda}  \right\rangle \bigg|_{J}
$$

By the arbitrarity of $J$,
$$
\frac{1}{2}\int \frac{d^d p}{(2\pi)^d} \left( \phi(p) \phi(-p) (p^2 + m^2) \Lambda \frac{\partial K_\Lambda^{-1}(p^2)}{\partial \Lambda}\right) + \Lambda \frac{\partial S_{int,\Lambda}[\phi]}{\partial \Lambda}  = C_{\Lambda}
$$
And the standard form is:
$$
\Lambda \frac{\partial S_{int,\Lambda}}{\partial\Lambda} = -\frac{dS_{int}}{d\Lambda/\Lambda} = \frac{1}{2} \int \frac{d^dp}{(2\pi)^d} \dot{C}_{\Lambda}(p^2) \left[ \frac{\delta^2 S_{int}}{\delta\phi(p)\delta\phi(-p)} - \frac{\delta S_{int}}{\delta \phi(p)} \frac{\delta S_{int}}{\delta \phi(-p)} \right]
$$

To make it match the form of the RHS, we should adjust the coupling constants in the interaction term $S_{int,\Lambda}[\phi]$ accordingly.

