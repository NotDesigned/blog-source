---
title: Renormalization Group Flow as Optimal Transport
date: 2025-08-21 20:07:27
tags:
- Renormalization Group
- Optimal Transport
categories:
- Physics
- Paper Study
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

### Free Field Propagator

For the **free field** (no interactions, $u_0 = 0$), the action is quadratic:
$$
S_0[\tilde{\phi}] = \int \frac{d^d p}{(2\pi)^d} \frac{1}{2}(p^2+m^2)\tilde{\phi}(p)\tilde{\phi}(-p)
$$

The **inverse propagator** in momentum space is:
$$
D^{-1}(p) = p^2 + m^2
$$

The **propagator** (Green's function) is:
$$
D(p) = \frac{1}{p^2 + m^2}
$$

This gives the familiar result that in position space, the two-point correlation function is:
$$
\langle \phi(x) \phi(y) \rangle = \int \frac{d^d p}{(2\pi)^d} \frac{e^{ip \cdot (x-y)}}{p^2 + m^2}
$$

### Physical Interpretation

1. **Mass scale**: The parameter $m^2$ sets the mass scale of the theory. When $m^2 > 0$, we have a massive theory with exponential decay of correlations at distances $\sim 1/m$.

2. **Momentum cutoff**: In practical calculations, we often introduce a UV cutoff $\Lambda$ such that $|p| < \Lambda$, regularizing high-momentum divergences.

3. **Renormalization**: The parameters $m^2, u_0$ typically depend on the cutoff scale $\Lambda$ and must be renormalized to obtain finite physical predictions.

### Connection to Exact RG

This momentum space formulation naturally leads to the **exact renormalization group** approach, where we systematically integrate out high-momentum modes while keeping track of how the effective action changes. The cutoff function $K_\Lambda(p^2)$ mentioned in the paper provides a smooth way to suppress high-momentum modes above the scale $\Lambda$.

## Exact RG 

TODO.