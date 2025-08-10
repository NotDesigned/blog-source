---
title: Flow Matching
date: 2025-08-09 13:49:15
tags:
- Flow Matching
categories:
- Machine Learning
- Deep Learning
- Generative Models
math: true
---

## Reference

[Flow Matching For Generative Modeling](https://arxiv.org/pdf/2210.02747)

## Preliminaries

The objective of generative modeling is to learn the underlying distribution $p_{data}(x)$ of the training data $x$. This is typically achieved by training a generative model $p_{model}(x)$ to approximate $p_{data}(x)$, allowing for the generation of new samples from the learned distribution.

Let $X$ be a complete and separable metric space such as $\Omega \subseteq \mathbb R^n$. Then, the space of probability measures $\mathcal{P}(X)$ is defined as the set of all Borel probability measures on $X$. 

Generative models usually aims to learn a mapping from a simple distribution (e.g., Gaussian) to the complex data distribution $p_{data}(x)$ by transforming samples from the simple distribution into samples from the data distribution.
