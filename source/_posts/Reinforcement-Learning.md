---
title: Reinforcement Learning
date: 2026-07-23 15:46:39
categories:
- Machine Learning
tags:
- Stochastic Optimal Control
- Reinforcement Learning
- REINFORCE
---

## Prerequisites

Probability theory and Markov Decision Processes

Basic knowledge of linear algebra and calculus

## Notation

In a Markov Decision Process:

$$
\begin{align*}
&s, s' & \text{states} \\
&y  & \text{action} \\
&r  & \text{a reward} \\
&p(s', r| s,a ) & \text{transition probability}\\
&\mathcal{S} & \text{set of all non-terminal states} \\
&\mathcal{S^+} & \text{set of all states} \\
&\mathcal{A}(s) & \text{set of all action available in state $s$} \\
&\mathcal{R} & \text{set of all possible rewards} \\
&\mathcal{G} & \text{cumulative reward for a trajectory} \\
&t & \text{discrete time t} \\
&A_t, S_t, R_t & \text{action, state, reward at time $t$}\\
&\pi, \pi(s), \pi(a|s) & \text{policy, deterministic/stochastic policy in state $s$}
\end{align*}
$$

## Policy, Value function

$$
\begin{align*}
&G_t := \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} & \text{discounted reward with rate $\gamma$ }\\
&v_{\pi}(s) := \mathbb E_{\pi} [G_t | S_t = s] & \text{value function}\\
&q_{\pi}(s,a) := \mathbb E_{\pi} [G_t | S_t = s, A_t = a] & \text{action-value function}
\end{align*}
$$

We can represent $v_\pi$ in terms of $q_{\pi}$ and $\pi$, and in reverse using $p$

$$
\begin{align*}
v_{\pi}(s) = \sum_{a} \pi(a|s) q_{\pi}(s,a)\\
q_{\pi}(s,a) = \sum_{s',r} p(s',r | s,a) (r+\gamma v_{\pi}(s'))
\end{align*}
$$

### Policy Evaluation

$$
\begin{aligned}
v_{\pi}(s) &\doteq \mathbb E_{\pi} [ G_t | S_t = s] \\
&= \mathbb E_{\pi} [ R_t + \gamma G_{t+1} | S_t = s] \\
&= \mathbb E_{\pi} [ R_t + v_{\pi} (S_{t+1}) | S_t = s]\\
&=\sum_{a} \pi(a|s) \sum_{s', r} p(s', r | a, s) [r+ \gamma v_{\pi}(s')]
\end{aligned}
$$

Can be iteratively computed using DP if the environment is completely known.

Fail: 
Intractable for large dimension problem. Not possible for unknown environment

#### Value Iteration

![Value Iteration](Value_Iteration.png)
