---
title: QAOA
date: 2025-12-15 15:42:28
tags:
- Quantum Algorithms
- Machine Learning
- Optimization
---

# Quantum Approximate Optimization Algorithm (QAOA)

## Problem Setting

Let $z=z_1 z_2 \ldots z_n$ be a string of $n$ bits, where each bit $z_i \in \{0, 1\}$. We define a cost function $C(z)$ that assigns a real value to each bit string. The goal is to find the bit string $z^*$ that maximizes the cost function:
$$
z^* = \arg\max_{z \in \{0,1\}^n} C(z)=\arg\max_{z \in \{0,1\}^n} \sum_{\alpha} C_{\alpha}(z)
$$
where $C_{\alpha}(z)$ are indicator functions $\mathbf{1}_{\text{clause } \alpha \text{ is satisfied by } z}$.

## Hamiltonian Representation

We can represent the cost function as a Hamiltonian operator acting on a quantum state. The Hamiltonian $H_C$ corresponding to the cost function $C(z)$ is given by:
$$
H_C = \sum_{\alpha} H_{C_{\alpha}}
$$
where each $H_{C_{\alpha}}$ is a diagonal operator in the computational basis defined as:
$$
H_{C_{\alpha}} \ket{z}= C_{\alpha}(z) \ket{z}
$$
Each $H_{C_{\alpha}}$ is Hermitian, so is $H_C$ and $H_C \ket{z} = C(z) \ket{z}$.

We now seek to find the maximum eigenvalue $\lambda^*$ of $H_C$ and the corresponding eigenstate $\ket{z^*}$.

## What we can learn from Score Based Model (SBM)

In EBM, we define a probability distribution via the energy function (Hamiltonian):

$$
p(z) = \frac{\exp(-E(z))}{Z}, Z = \int \exp(-E(z)) dz
$$

consider a parameterized model $p(z; \theta)$, we can learn the parameters $\theta$ by learning the score function $\nabla_z \log p(z; \theta)$.

$$
\boxed{
\mathcal{L}_{\mathrm{SM}}(\theta) = \mathbb{E}_{p_{data}(x)} \left[\mathrm{Tr}(\nabla_x s_{\theta}(x)) + \frac{1}{2} \| s_{\theta}(x) \|_2^2\right]
}
$$
where $s_{\theta}(x)= \nabla_x \log p(x; \theta)$ is the score function parameterized by $\theta$.

---

In QAOA's scenario, the energy function is fixed and we want to find the state $\ket{z^*}$ that maximizes the energy.

Let's define a quantum state $\ket{\psi(\theta)} = U(\theta) \ket{0}$ parameterized by $\theta$. 

We can measure:
$$
\langle H_C \rangle_{\theta} = \bra{\psi(\theta)} H_C \ket{\psi(\theta)}
$$
We want to maximize:
$$
J(\theta) = \langle \psi(\theta) | H_C | \psi(\theta) \rangle = \sum_{z \in \{0,1\}^n} P(z|\theta) C(z)
$$
where $P(z|\theta) = |\langle z | \psi(\theta) \rangle|^2$ is the probability of measuring the state $\ket{z}$ given the parameter $\theta$.

In the first glance, we try to estimiate the log gradient:
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \sum_{z} P(z|\theta) C(z) \\
&= \sum_{z} C(z) \nabla_\theta P(z|\theta) \\
&= \sum_{z} C(z) P(z|\theta) \nabla_\theta \log P(z|\theta) \\
&= \mathbb{E}_{z \sim P(z|\theta)} \left[ C(z) \nabla_\theta \log P(z|\theta) \right]
\end{aligned}
$$
This is similar to the REINFORCE learning gradient method with reward $C(z)$.
However, this gradient is not calculable directly on quantum computer since $\log P(z|\theta)$ cannot be computed efficiently. ($2^n$ terms). And it does not use the result of measure $J(\theta)$ directly.

On the other hand, if we sampling through
$$
\lim_{\epsilon \to 0} \frac{J(\theta + \epsilon) - J(\theta)}{\epsilon} = \nabla_\theta J(\theta)
$$
This suffers from high variance issue. The high variance is due to the high dimensionality of the output space.

## Parameter Shift Rule

Remember that in Denoising Score Matching, we can avoid calculating the score function directly by introducing noise and learning to denoise, then the score function can be computed directly. And under Gaussian noise, the score function has a closed form.

This inspires us to introduce a particular parameterization of $U(\theta)$ in QAOA such that we can compute the gradient of $J(\theta)$ directly.

For $U(\theta) = e^{-i \frac{\theta}{2} G}$, where $G$ is a Hermitian operator (generator), we have:
$$
\frac{d}{d\theta} U(\theta) = -\frac{i}{2} G U(\theta) = - \frac{i}{2} U(\theta) G
$$
this uses the property that $G$ commutes with $U(\theta)$. 

Thus,
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \langle \psi_0 | U^\dagger(\theta) H_C U(\theta) | \psi_0 \rangle \\
= & \langle \psi_0 | \left( \frac{d}{d\theta} U^\dagger(\theta) \right) H_C U(\theta) | \psi_0 \rangle + \langle \psi_0 | U^\dagger(\theta) H_C \left( \frac{d}{d\theta} U(\theta) \right) | \psi_0 \rangle \\
= & \langle \psi_0 | \left( \frac{i}{2} U^\dagger(\theta) G \right) H_C U(\theta) | \psi_0 \rangle + \langle \psi_0 | U^\dagger(\theta) H_C \left( -\frac{i}{2} G U(\theta) \right) | \psi_0 \rangle \\
= & \frac{i}{2} \langle \psi(\theta) | [G, H_C] | \psi(\theta) \rangle
\end{aligned}
$$
where $[G, H_C] = G H_C - H_C G$ is the commutator of $G$ and $H_C$, $\psi(\theta) = U(\theta) \ket{\psi_0}$.

Now, note that 
$$
\begin{aligned}
H_+ &= (e^{-i \frac{\pi}{4} G})^\dagger H_C (e^{-i \frac{\pi}{4} G}) = e^{i \frac{\pi}{4} G} H_C e^{-i \frac{\pi}{4} G}\\
&= \frac{1}{2} (I + iG) H_C (I - iG) \\
&= \frac{1}{2} (H_C - i H_C G + i G H_C + G H_C G)\\
H_- &= (e^{i \frac{\pi}{4} G})^\dagger H_C (e^{i \frac{\pi}{4} G}) = e^{-i \frac{\pi}{4} G} H_C e^{i \frac{\pi}{4} G}\\
&= \frac{1}{2} (H_C + i H_C G - i G H_C + G H_C G)
\end{aligned}
$$
Thus,
$$
H_+ - H_- = i (G H_C - H_C G) = i [G, H_C]
$$
Therefore, we have:
$$
\nabla_\theta J(\theta) = \frac{1}{2} \langle \psi(\theta) | (H_+ - H_-) | \psi(\theta) \rangle
= \frac{1}{2} \left( J(\theta + \frac{\pi}{2}) - J(\theta - \frac{\pi}{2}) \right)
$$
This means we can compute the gradient of $J(\theta)$ by evaluating $J$ at two shifted parameter values, avoiding the need to compute the score function directly.

## Parameterized Quantum Circuit in QAOA

In QAOA, the parameterized quantum circuit is constructed using two types of unitary operators: the cost unitary $U_C(\gamma)$ and the mixing unitary $U_B(\beta)$. The overall circuit for $p$ layers is given by:
$$
U(\boldsymbol{\gamma}, \boldsymbol{\beta}) = U_B(\beta_p) U_C(\gamma_p) \ldots U_B(\beta_1) U_C(\gamma_1)
$$

The cost unitary is defined as:
$$
U_C(\gamma) = e^{-i \gamma H_C}
$$
and the mixing unitary is defined as:
$$
U_B(\beta) = e^{-i \beta H_B}
$$
where $H_B = \sum_{i=1}^n X_i$ with $X_i = I^{\otimes (i-1)} \otimes X \otimes I^{\otimes (n-i)}$ being the Pauli-X operator acting on qubit $i$.

The initial state is typically chosen as the uniform superposition state:
$$
\ket{\psi_0} = \frac{1}{\sqrt{2^n}} \sum_{z \in \{0,1\}^n} \ket{z}
$$
which is the ground state of the mixing Hamiltonian $H_B$.

Reason (Can be omitted, I don't know very well about this quantum physics part):

>The Adiabatic Theorem suggests that if we vary the parameters $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$ slowly enough, the system will remain in its ground state, allowing us to approximate the optimal solution to the original combinatorial optimization problem.
> $$
> H(t) = (1 - \frac{t}{T}) H_B + \frac{t}{T} H_C
> $$
> The time evolution operator can be approximated using the Trotter-Suzuki decomposition:
> $$
> e^{-i (A+B) t} \approx \left( e^{-i A \frac{t}{n}} e^{-i B \frac{t}{n}} \right)^n
> $$

## Algorithm Summary

1. Initialize the quantum state $\ket{\psi_0}$ as the uniform superposition state.
2. Choose the number of layers $p$ and initialize the parameters $\boldsymbol{\gamma}, \boldsymbol{\beta}$.
3. Construct the parameterized quantum circuit $U(\boldsymbol{\gamma}, \boldsymbol{\beta})$.
4. Measure the expectation value $J(\boldsymbol{\gamma}, \boldsymbol{\beta}) = \langle \psi(\boldsymbol{\gamma}, \boldsymbol{\beta}) | H_C | \psi(\boldsymbol{\gamma}, \boldsymbol{\beta}) \rangle$.
5. Use the parameter shift rule to compute the gradients $\nabla_{\boldsymbol{\gamma}} J$ and $\nabla_{\boldsymbol{\beta}} J$.
6. Update the parameters $\boldsymbol{\gamma}, \boldsymbol{\beta}$ using a classical optimization algorithm (e.g., gradient descent).
7. Repeat steps 3-6 until convergence or a stopping criterion is met.
8. Measure the final state to obtain a bit string $z$ that approximates the optimal solution to the original problem.
