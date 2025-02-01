# Generative AI for Inverse Problems in Physics

This repository includes implementations of deep learning methods for conditional generation to solve inverse problems in physics. It also includes solvers physics problems to generate datasets of pairwise (x, y) data and MCMC implementations to sample the target distributions for texting examples and evaluate the quality of the predictions.

## Conditional generative methods

The conditional generative methods implemented include:

- Conditional Diffusion Models: DDPM and SBLD.
- Conditional Wasserstein GANs with full gradient penalty.

## Inverse physics problems

The inverse physics problems implemented include:

- Unsteady heat equation with a single source in a square domain.
- Unsteady heat equation with a two sources in a square domain.
- Helmholtz equation in a square domain modelling a simplified geometry of the optic nerve head.