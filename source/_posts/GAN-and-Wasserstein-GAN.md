---
title: GAN and Wasserstein GAN
date: 2025-07-22 14:17:15
tags:
- Generative Adversarial Networks
- Wasserstein GAN
categories:
- Machine Learning
- Deep Learning
- Generative Models
---

# Introduction

Generative Adversarial Networks is invented in 2014 by Ian J.Goodfellow, et. al. as a generative models, its idea derives from the game theory where two player compete against one another.

So they design an architecture that has two network where one is responible for generating fake data that is similar from the data and another one is responible for discriminating between fake and true data.

# Mathematics Formulas

GAN is formulated as a min-max problem 

$$
\min_{G} \max_{D} V(D,G) 
$$
$$
V(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{x \sim p_{g}(x)}[\log(1 - D(x))]
$$

Where $D$ is the discriminator, $G$ is the generator, $p_{data}(x)$ is the distribution of the real data, and $p_{g}(x)$ is the distribution of the noise input to the generator.

$V(D,G)$ can be interpreted as the value function that represents the "distance" between the real data distribution and the generated data distribution.

# Network Architecture

The architecture of GAN consists of two main components:

1. **Generator (G)**: 
- Noise input $z$ is sampled from a prior distribution (e.g., Gaussian or uniform).
- Hidden layers transform the noise into a data sample.
- Output layer generates a data sample (e.g., image).

2. **Discriminator (D)**:
- Takes a data sample (either real or generated).
- Hidden layers process the sample.
- Output layer produces a probability score indicating whether the sample is real or fake.

# Training Process
The training process of GAN involves alternating between training the discriminator and the generator:
1. **Train Discriminator**:
- Sample a batch of real data from the dataset with label 1.
- Sample a batch of noise and generate fake data using the generator with label 0.
- Update the discriminator to maximize its ability to distinguish real from fake data.
2. **Train Generator**:
- Sample a batch of noise and generate fake data using the generator.
- Update the generator to minimize the discriminator's ability to distinguish fake data from real data.
3. **Repeat**:
- Continue alternating between training the discriminator and the generator until convergence or for a fixed number of epochs

# Merits and Limitations

## Merits
- **High-Quality Samples**: GANs can generate high-quality samples that are often indistinguishable from real data.
- **Versatile Applications**: GANs can be applied to various domains, including image generation
- **Unsupervised Learning**: GANs can learn from unlabeled data, making them suitable for unsupervised learning tasks.

## Limitations
- **Training Instability**: GANs can be difficult to train due to issues like mode collapse, vanishing gradients, and oscillations.
- **Sensitive to Hyperparameters**: GANs require careful tuning of hyperparameters, such as learning rates and batch sizes, to achieve stable training.
- **Lack of Diversity**: GANs may generate samples that lack diversity, especially if the training data is limited or biased.

# Analysis

## Mode Collapse
Mode collapse is a common issue in GAN training where the generator produces a limited variety of outputs, often focusing on a few modes of the data distribution. This can lead to a lack of diversity in generated samples.

This is because the manifold of data is not necessarily connected (MNIST), but the transformation from the noise space to the data space is continuous. So it either gives up generating some data or generates some data that is not actually in the data manifold. (In the vacant middle of two number manifold).

See [AE-OT-GAN](https://arxiv.org/abs/2001.03698).

## Training Difficulties
Training GANs can be challenging due to the adversarial nature of the training process. The more powerful the discriminator becomes, the harder it is for the generator to produce samples that fool the discriminator. 

This is because the loss function has the problem of vanishing gradients when the discriminator is too good at its job.

Fix $G$, the optimal $D$ is given by:
$$
D^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}
$$

Now we observe the loss of $G$ has become:
$$
2JS(p_{data}||p_{g})-2\log 2
$$

But when the support of $p_{data}$ and $p_{g}$ has no intersection, the JS loss shall be constant. So the network cannot be trained since there is no meaningful information can be acquired.

# Wasserstein GAN (WGAN)

Wasserstein GAN (WGAN) is a variant of GAN that addresses some of the limitations of the original GAN formulation, particularly the issues related to training stability and mode collapse. WGAN uses the Wasserstein distance (also known as Earth Mover's Distance) to measure the distance between the real data distribution and the generated data distribution.

See this article for more about Wasserstein distance: [Wasserstein Distance and Optimal Transport](https://notdesigned.github.io/2025/07/16/Wasserstein-Distance-and-Optimal-Transport/)

The important properties of Wasserstein distance are:
1. **Continuity**: The Wasserstein distance is continuous with respect to the distributions.
2. **Metric**: The Wasserstein distance satisfies the triangle inequality and is a true metric.
3. **Independence of Support**: When the supports of two distributions have no overlap, Wasserstein distance still provides a suitable gradient for the generator to update.



## Formulation

The original Wasserstein is difficult to compute, so WGAN uses a simplified version of the 1-Wasserstein distance, which can be computed using the Kantorovich-Rubinstein duality:

$$
W(p_{data}, p_{g}) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_{data}}[f(x)] - \mathbb{E}_{x \sim p_{g}}[f(x)]
$$
Where $f$ is a 1-Lipschitz function, $\|f\|_L \leq 1$.

Actually if $f$ is a K-Lipschitz function, just divide the result by $K$. So long as the function is finite Lipschitz, it can be used to estimate the Wasserstein distance.


## Modification of GAN

The loss function for the discriminator (critic) is modified to use the Wasserstein distance instead of the original GAN loss. So the objective becomes:
$$
\max_{D} L = \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{x \sim p_{g}}[D(x)]
$$
The discriminator is now no longer trying to output probabilities, but rather to output a real-valued score that reflects the "realness" of the input data.

To ensure that $D$ is a K-Lipschitz function, WGAN applies weight clipping to the discriminator's weights. This means that the weights of the discriminator are constrained to lie within a certain range (e.g., [-0.01, 0.01]). So it must be limited to a finite Lipschitz constant.

For the generator, since the first term is not dependent on $G$, the objective becomes:
$$
\min_{G} L = -\mathbb{E}_{x \sim p_{g}}[D(x)]
$$
