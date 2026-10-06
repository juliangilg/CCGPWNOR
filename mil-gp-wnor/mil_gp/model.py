"""Sparse variational multi-output GP for MIL with a weighted Noisy-OR aggregation.

Each instance has two latent functions:
  * f_p -> instance probability  p_ij = sigmoid(f_p)
  * f_w -> instance weight       w_ij = softmax_j(f_w)  (within the bag)
The bag probability is  P(Y_i = 1) = 1 - prod_j (1 - p_ij) ** w_ij.
"""

import numpy as np
import torch
import gpytorch
from torch import nn
from gpytorch.distributions import base_distributions
from gpytorch.likelihoods import Likelihood


class GPLayer(gpytorch.models.ApproximateGP):
    """Two-task SVGP: 'independent' (one GP per task) or 'correlated' (LMC)."""

    def __init__(self, inducing_points, dimension, mode="independent", num_latents=2):
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )
        base_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        if mode == "correlated":
            strategy = gpytorch.variational.LMCVariationalStrategy(
                base_strategy, num_tasks=2, num_latents=num_latents, latent_dim=-1
            )
        else:
            strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                base_strategy, num_tasks=2
            )
        super().__init__(strategy)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents]))
        # Heuristic lengthscale initialisation (same as the original experiments).
        self.covar_module.lengthscale = torch.tensor(
            np.sqrt(dimension) * (np.random.rand(num_latents) + 0.01) * 0.2 * np.log(dimension)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class WeightedNoisyORLikelihood(Likelihood):
    """Monte Carlo weighted Noisy-OR likelihood.

    `B` (bags), `N` (max instances per bag) and `mask` (B, N) must be set
    before each call; the trainer does this in `MILTrainer._forward`.
    """

    def __init__(self, num_samples=50, diag_sampling=True, elbo_scale=1.0):
        super().__init__()
        self.num_samples = num_samples
        self.diag_sampling = diag_sampling  # True: ignore cross-instance covariances (faster)
        self.elbo_scale = elbo_scale        # multiplies the expected log-likelihood term
        self.B, self.N, self.mask = None, None, None

    def forward(self, function_samples):
        return None

    def _sample(self, function_dist):
        """Draw S samples with shape (S, B*N, 2)."""
        if self.diag_sampling:
            function_dist = base_distributions.Independent(
                base_distributions.Normal(function_dist.mean, function_dist.variance.sqrt()), 1
            )
        return function_dist.rsample(torch.Size([self.num_samples]))

    def _p_and_w(self, function_dist):
        """Instance probabilities and normalised weights, both (S, B, N)."""
        S, B, N = self.num_samples, self.B, self.N
        samples = self._sample(function_dist)
        p = torch.sigmoid(samples[..., 0]).reshape(S, B, N)
        w = samples[..., 1].reshape(S, B, N)
        w = w.masked_fill(~self.mask.unsqueeze(0), float("-inf"))  # padded instances get w = 0
        return p, torch.softmax(w, dim=2)

    def expected_log_prob(self, target, input, *args, **kwargs):
        p, w = self._p_and_w(input)
        p = p.clamp(1e-6, 1 - 1e-6)
        bag_p = (1 - torch.prod((1 - p) ** w, dim=2)).clamp(1e-6, 1 - 1e-6)

        log_pos = torch.log(bag_p).mean(0)                         # E[log P(Y=1)]
        log_neg = torch.sum(w * torch.log(1 - p), dim=2).mean(0)   # E[log P(Y=0)]
        if self.elbo_scale:
           return self.N * torch.sum(target * log_pos + (1 - target) * log_neg)
        else:
           return 1 * torch.sum(target * log_pos + (1 - target) * log_neg)

    def marginal(self, function_dist, *args, **kwargs):
        """Predictive means/variances at bag and instance level."""
        p, w = self._p_and_w(function_dist)
        bag = 1 - torch.prod((1 - p) ** w, dim=2)

        def mean_var(x):
            m = x.mean(0)
            return m, x.pow(2).mean(0) - m.pow(2)

        y_hat, var_bag = mean_var(bag)
        E_p, Var_p = mean_var(p)
        E_w, Var_w = mean_var(w)
        return y_hat, var_bag, E_p, Var_p, E_w, Var_w


class MILGP(nn.Module):
    """Linear+ReLU feature projection -> bounded scaling -> two-task SVGP."""

    def __init__(
        self,
        M=50,                 # number of inducing points
        input_dim=2048,       # dimension of the pre-extracted features
        num_dim=16,           # dimension of the projected space
        mode="independent",   # 'independent' or 'correlated'
        num_latents=2,
        num_samples=50,       # Monte Carlo samples in the likelihood
        minis=-3,             # inducing points grid range
        maxis=3,
        grid_bounds=(-1.0, 1.0),
        diag_sampling=True,
        elbo_scale=1.0,
    ):
        super().__init__()
        assert mode in ("independent", "correlated")
        if mode == "independent":
            num_latents = 2

        self.feature_extractor = nn.Sequential(nn.Linear(input_dim, num_dim), nn.ReLU())
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(*grid_bounds)

        # Inducing points: a linspace per dimension, randomly permuted (except the first).
        grid = np.linspace(minis, maxis, M)
        Z = np.stack([grid] + [grid[np.random.permutation(M)] for _ in range(num_dim - 1)], axis=1)
        Z = torch.tensor(np.tile(Z, (num_latents, 1, 1)), dtype=torch.float32)

        self.gp_layer = GPLayer(Z, dimension=num_dim, mode=mode, num_latents=num_latents)
        self.likelihood = WeightedNoisyORLikelihood(num_samples, diag_sampling, elbo_scale)

    def forward(self, x):
        return self.gp_layer(self.scale_to_bounds(self.feature_extractor(x)))
