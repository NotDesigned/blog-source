---
title: Statistical Mechanics
date: 2026-04-14 22:03:02
tags:
- Statistical Mechanics
categories:
- Physics
math: true
---

## Thermodynamics

Thermodynamics, essentially, is a phenomenological description of macroscopic system in thermal equilibrium.

**How to idealize the system**

View it as some ensemble such as:
- closed system: often descripted as a closed box with adiabatic walls. (Microcanonical)
- open system: with exchanges
    - canonical: exchanges heat. (Two system with diabatic wall)
    - grandcanonical: exchanges both heat and particles, etc.

**How to describe the system**
- State function
- Thermodynamics coordinates.
- State Variable: $P$ (Pressure), $V$ (Volume), $\mu$ (Chemical Potential), etc.


#### The Zeroth Law of Thermodynamics:

The system is usually of $10^{20+}$ particles and the possible states are at least of $\mathcal{O}(\exp(10^{20}))$. How we can use a marco parameter such as temperature to describe a great portion of such possible states? 

In common sense, marco parameters are a property shared by physical systems. To make this statement well-defined, we should introduce the zeroth law of Thermodynamics:

> (The transitivity of equilibrium) if system A is in equilibrium with system B and system B is in equilibrium with system C. Then A is in equilibrium with system C.

Therefore, a system in thermoequilibrium can be described in a state function that is purely described by its thermocoordinate (things that equals when in thermoequilibrium), as equivalence class in math.

#### The First Law of Thermodynamics


> The amount of work required to change the state of an otherwise adiabatically isolated system depends only on the initial and final states, and not on the means by which the work is performed, or on the intermediate stages through which the system passes.

For the thermodynamics system we can construct a state function, the internal energy $U(X)$. The amount of $U$ can be derived from amount of work $W$ needed for an adiabatic transformation from an inital state $X_i$ to a final state $X_f$ using $\delta W= U(X_i)-U(X_f)$.



What if the transformation is not adiabetic (which is the most of the cases)?

It is observed that $dU \neq -\delta W$, We call the extra part that cannot be explained by $\delta W$ as $\delta Q$

$$
\delta Q = \delta W + dU \Rightarrow dU = \delta Q -\delta W
$$

$$
\text{Measure}\ W_{\text{adiabatic}} \xrightarrow{\text{Define}} U \xrightarrow{\text{Define the difference}} \delta Q \equiv dU + \delta W
$$

Sometimes we use generalized force and displacement to charactize the work done:

| System | Force | Displacement |
| -------- | -------- | -------- |
| Wire    | tension $F$ | length $L$ |
| Fluid   | pressure $-P$ | volume $V$     |
| Chemical reaction | chemical potential $\mu$ | particle number $N$|

Ideal gas:
$$
U(T, V) = U(T)
$$
T does not change during expansion.

Response functions:

Charactizing the macroscopic behaviour of the system:

Heat capacities:
$$
C_V = \frac{\bar d Q}{d T}\bigg|_V = \frac{d E + P dV}{d T}\bigg|_V= \frac{\partial E}{\partial T}\bigg|_V\\

C_P = \frac{\bar d Q}{d T}\bigg|_P = \frac{d E + P dV}{d T}\bigg|_P= P\frac{\partial V}{\partial T}\bigg|_P + \frac{\partial E}{\partial V}\bigg|_P
$$
For ideal gas $$C_P-C_V = P \frac{\partial V}{\partial T}\bigg|_P= Nk_B$$

Isothermal compressibility:
$$
\kappa_T \equiv -\frac{1}{V}\frac{\partial V}{\partial P}\bigg|_T
$$

#### The Second Law of Thermodynamics

The impetus for the second law of thermodynamics is the advent of heat engines.

**Heat engine**: a device that receives energy in the form of heat from a hot reservoir, delivers work to its surroundings, and discharges energy in the form of heat to a cold reservoir.

**Heat pump**: a device that uses work to delivers energy in the form of heat to a hot reservoir, and receives energy in the form of heat from a cold reservoir.

```
-------                      ------
|Source|--Qh->|Engine|--Q_c->|Sink|
-------         |            ------
                |
                v
               Work

               Work
                |
                |
                v
-------                       -------
|Icebox|--Qc->|Fridge| --Qh->|Exhaust|
-------                       ------- 
```

$$
Q_h = W + Q_c
$$
Heat engine efficiency:
$$
\eta = \frac{W}{Q_h} = 1 - \frac{Q_c}{Q_h} \leq 1 
$$

Heat pump efficiency:
$$
\eta = \frac{Q_c}{W} 
$$

The second law of thermodynamics states that the efficiency of a heat engine can never be 100%.

**Kelvin's Statement**: No process is possible whose sole result is the complete conversion of heat into work.

**Clausius's Statement**: No process is possible whose sole result is the transfer of heat from a cooler to a hotter body.

The best heat engine is the Carnot engine, which consists of two isothermal processes and two adiabatic processes. 

1. Isothermal expansion: the system is in contact with the hot reservoir at temperature $T_h$, and expands from volume $V_1$ to $V_2$.
2. Adiabatic expansion: the system is isolated and expands from volume $V_2$ to $V_3$.
3. Isothermal compression: the system is in contact with the cold reservoir at temperature $T_c$, and compresses from volume $V_3$ to $V_4$.
4. Adiabatic compression: the system is isolated and compresses from volume $V_4$ to $V_1$.

See https://galileo.phys.virginia.edu/classes/152.mf1i.spring02/CarnotEngine.htm for more details.

We argue that the efficiency of the Carnot engine is the upper bound of all heat engines, otherwise we can use the more efficient engine to run the Carnot engine in reverse to create a perpetual motion machine of the second kind (Pump heat from cold to hot without work input).

Use the ideal gas as an example, we can calculate the efficiency of the Carnot engine is 
$$
\eta = 1 - \frac{T_c}{T_h}
$$
Therefore, the efficiency of any heat engine is bounded by the efficiency of the Carnot engine:
$$
\frac{\delta W}{\delta Q} \leq 1 - \frac{T_c}{T_h}
$$

#### The Third Law of Thermodynamics

For an infinismal process that draw $\delta Q$ from a heat reservoir at temperature $T_h$

W.r.t. the whole system:
$$
\oint dU =0 \Rightarrow \oint \delta Q = \oint \delta W 
$$

We can decompose any process into a series of infinitismal processes, and for each process, the system exchange heats with a system with almost identical temperature. We know that:
$$
\delta W_i \leq \left(1 - \frac{T_c}{T_i}\right) \delta Q_i
$$
where $T_i$ is the temperature of the system during the $i$-th process. Therefore, we have:
$$
\begin{gather*}
\sum_{i=1}^\infty \delta W_i \leq \sum_{i=1}^\infty \left(1 - \frac{T_c}{T_i}\right) \delta Q_i \leq \sum_{i=1}^\infty \delta Q_i - \frac{T_c}{T_i} \delta Q_i = \sum_{i=1}^\infty \delta Q_i - T_c \sum_{i=1}^\infty \frac{\delta Q_i}{T_i} \\
T_c \sum_{i=1}^\infty \frac{\delta Q_i}{T_i} \leq 0
\end{gather*}
$$

Which leads to **Clausius’s theorem**
$$
\oint \frac{\delta Q}{T} \leq 0
$$

For a reversible cycle, we infer that $\frac{\delta Q}{T}=0$.
And we can define a function of state $S$, only dependent on two end-points:
$$
S(A)-S(B) = \int_B^A \frac{dQ_{rev}}{T}
$$

## Kinetic Theory of Gas

### Liouville's Theorem

### BBGKY Hierarchy

### Boltzmann's Equation

### H Theorem

### Equilibrium and Conservation Laws

### Hydrodynamics

## (Classical) Statistical Mechanics

The idea is that we start from the simplest situation (microcanonical). Then we gradually allow extensive quantity to fluctuate but add intensive quantity in control.

#### Microcanonical System

Particles system where $(E, V, N)$ is fixed.

The system is equally probable in any state that satisfy the energy constraint.

$$
P(\rho) =\begin{cases}
\frac{1}{\Omega}\quad U(\rho) =E \\
0 \quad \text{otherwise}
\end{cases}
$$

#### Canonical System

We now fix $T$ instead of $E$, i.e. $\langle E\rangle$ is fixed.

But the system can be in energy $E$ with probability $\propto\exp(-\beta H(\sigma))$

$$
\propto 1 \quad \text{Equally weighted}
$$

#### Grandcanonical System

We now fix $\mu$ instead of $N$.

The system can be in energy $E$, particle number $N$ with probability 
$$
\propto \exp(-\beta (E -\mu N))
$$

#### Gibbs System

Fix $(T,P,N)$, change $V$ to $P$, $E$ to $T$.

$$
\propto \exp(-\beta(E+PV))
$$

### Partition Function and the Calculations

$$
Z = \int d \omega p(\omega) 
$$
where $p(\omega)$ is unnormalized distribution.

In microcanonical ensemble:

$$
\begin{aligned}
\Omega(E) &= \frac{1}{h^{3N} N!}\int d \omega \delta(H(\omega ) -E) \\
&= \frac{1}{h^{3N} N!}\int \prod_{i=1}^N d^{3}q_i d^{3} p_i \delta \left(\sum_{i=1}^N \frac{p_i^2}{2m} - E\right) \\
&= \frac{V^{N}}{h^{3N} N!}  \int \prod_{i=1}^N d^{3} p_i \delta \left(\sum_{i=1}^N \frac{p_i^2}{2m} - E\right) \\
&= \frac{V^{N}}{h^{3N} N!} 2m \int \prod_{i=1}^N d^{3} p_i \delta \left(\sum_{i=1}^N p_i^2 - 2mE\right) \\
&= \frac{V^{N}}{h^{3N} N!} 2m \frac{\pi^{3N/2}}{\Gamma(3N/2)}(2mE)^{\frac{3}{2}N-1}\\
\end{aligned}
$$


In canonical ensemble:

$$
\begin{aligned}
Z(\beta, V, N) &= \frac{1}{h^{3N} N!} \int d\omega \exp(-\beta E(\omega))\\
&=\int_0^\infty \Omega(E) \exp(-\beta E) dE \\
&=\mathcal{L}\{\Omega(E)\}(\beta)
\end{aligned}
$$

> The Legrendre transform from $E$ to $\beta$ (T) correspond to the Laplace transform in the partition function.

$$
\begin{aligned}
\Omega(E) &= C E^{\frac{3N}{2}-1} \\
Z(\beta, V, N) &= C \int E^{3N/2-1} \exp(-\beta E) dE\\
&=C \cdot \frac{\Gamma(3N/2)}{\beta^{3N/2}}\\
&= \frac{V^{N}}{h^{3N} N!} \frac{\pi^{3N/2}}{\Gamma(3N/2)} (2m)^{\frac{3}{2}N} \frac{\Gamma(3N/2)}{\beta^{3N/2}}\\
&= \frac{V^{N}}{h^{3N} N!} \left(\frac{2\pi m}{\beta}\right)^{3N/2}
\end{aligned}
$$

Note:
$$
\int x^{\alpha-1}\exp(-\beta x) dx = \frac{\Gamma(\alpha)}{\beta^{\alpha}}
$$

In grandcanonical ensemble:

$$
\begin{aligned}
\Xi(\beta, \mu, V) &= \sum_{N=0}^{\infty} \frac{C_1^N}{\beta^{3N/2} N!} \exp\left(\beta\mu N \right)  \\
&= \exp\left(\frac{V}{\lambda^3}e^{\beta\mu}\right)
\end{aligned}
$$

where $\lambda =  \frac{h}{\sqrt{2\pi m k_B T}}$ is thermal de Broglie wavelength.

In Gibbs ensemble:

$$
\begin{aligned}
G(\beta, P, N) &= \int_0^\infty dV \, e^{-\beta P V} Z_{\text{can}}(\beta, V, N) \\
&= \int_0^\infty dV \, e^{-\beta P V} \cdot \frac{V^N}{h^{3N} N!} \left(\frac{2\pi m}{\beta}\right)^{3N/2} \\
&= \frac{1}{h^{3N} N!}\left(\frac{2\pi m}{\beta}\right)^{3N/2} \int_0^\infty V^N e^{-\beta P V} dV\\
&= \frac{1}{h^{3N}}\left(\frac{2\pi m}{\beta}\right)^{3N/2} \frac{1}{(\beta P)^{N+1}}
\end{aligned}
$$


### Implications

