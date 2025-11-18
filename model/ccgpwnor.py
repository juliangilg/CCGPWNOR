import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import Tensor
import torch

import gpytorch
from gpytorch.likelihoods import Likelihood
from gpytorch.distributions import base_distributions, MultitaskMultivariateNormal
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D

import matplotlib.pyplot as plt
import numpy as np
import random

# DNN as feature extractor
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__() # Add this line to call the parent class initializer
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size), # The input layer has 28*28 units.
            nn.ReLU(),
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


# Correlated Gaussian Processes
class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_latents, num_classes, inducing_p, dimension):
        # Let's use a different set of inducing points for each latent function
        inducing_points = inducing_p

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_classes])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_classes,
            num_latents=num_classes,
            latent_dim=-1
        )


        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_classes]))
        self.covar_module = gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_classes]))
        self.covar_module.lengthscale = torch.tensor(np.sqrt(dimension) * (np.random.rand(num_classes) + 0.01)*0.2*np.log(dimension))


    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
      

class DKLModel(gpytorch.Module):
  def __init__(self, feature_extractor, input_dim, num_dim, GPslayer, num_clases,
               inducing_p, grid_bounds=(-10., 10.)):
      super(DKLModel, self).__init__()
      self.feature_extractor = feature_extractor(input_dim, num_dim)
      self.gp_layer = MultitaskGPModel(num_dim, num_clases, inducing_p, num_dim)
      self.grid_bounds = grid_bounds
      self.num_dim = num_dim

      # This module will scale the NN features so that they're nice values
      self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

  def forward(self, x):
      features = self.feature_extractor(x)
      features = self.scale_to_bounds(features)
      # print(features.shape)
      # Pass features directly to the GP layer. The shape should be [batch_size, num_dim]
      # The MultitaskGPModel with LMC handles the batching over latent dimensions internally.
      res = self.gp_layer(features)

      return res


class MIL(Likelihood):

  def __init__(self, ins_batch: int) -> None:
    super().__init__()
    self.N = ins_batch
    self.quadrature = GaussHermiteQuadrature1D(30)
    self.mask = None

  def forward(self, function_samples: Tensor):
    return None

  def expected_log_prob(self, target: Tensor, input) -> Tensor:


    # Distribution related to the instance-level label.
    Dk = input[:,:1]
    # Distribution related to the instances' reliabilities.
    Dr = input[:,1:]

    # print(Dk.mean)

    B = self.B
    # print(Dk.mean.shape[0], B)

    # Expected First Term
    samples_p_w = self._draw_likelihood_samples_nOR(input, 100)# (200, N*B, 2)
    # print(samples_p_w.shape, B)
    samples_p = torch.nn.functional.sigmoid(samples_p_w[:,:,:1])# 200xN*Bx1
    samples_p = samples_p.view(100, B, self.N) # (200, B, N)
    eps = 1e-6
    samples_p = torch.clip(samples_p,min=eps, max=1-eps)


    samples_w = samples_p_w[:,:,1:]# 200xN*Bx1
    samples_w = samples_w.view(100, B, self.N) # (200, B, N)
    mask = self.mask.bool().unsqueeze(0).repeat(100, 1, 1)
    # print(mask)
    samples_w = samples_w.masked_fill(~mask, float('-inf'))
    samples_w = torch.nn.functional.softmax(samples_w, dim=2)
    # print(samples_w)

    safe = 1 - torch.prod((1 - samples_p)**(samples_w), dim=2)
    safe = torch.clip(safe,min=eps, max=1-eps)
    E_f_t = torch.log(safe).mean(dim=0) #(B,)


    # Fisrt term
    term_A = target*E_f_t

    # Expected Second Term: E[\sum_j w_ij*log 1-p(x_ij)]
    # func = lambda x: torch.log(1 - torch.nn.functional.sigmoid(x))
    log_p_ij = torch.log(1 - samples_p) # (200, B, N)
    # w_ij
    w_ij =  samples_w # (200, B, N)
    # Second term
    # print(w_ij.shape, log_p_ij.shape)
    E_sec_term = torch.sum(w_ij*log_p_ij, dim=2).mean(dim=0) # (200, B)
    term_B = (1-target)*E_sec_term # (B,)


    return ((term_A + term_B)).mean()


  def marginal(self, function_dist: MultitaskMultivariateNormal):
    # Distribution related to the Ground truth.
    Dk = function_dist[:,:1]
    # Distribution related to the annotators' reliabilities.
    Dr = function_dist[:,1:]

    B = Dk.mean.shape[0]//self.N


    # Bag-level prediction
    samples_p_w = self._draw_likelihood_samples_nOR(function_dist, 50)# (200, N*B, 2)
    samples_p = torch.nn.functional.sigmoid(samples_p_w[:,:,:1])# 200xN*Bx1
    samples_p = samples_p.view(50, B, self.N) # (200, B, N)
    E_p = samples_p.mean(dim=0) # (B, N)

    samples_w = samples_p_w[:,:,1:]# 200xN*Bx1
    samples_w = samples_w.view(50, B, self.N) # (200, B, N)
    mask = self.mask.bool().unsqueeze(0).repeat(50, 1, 1)
    samples_w = samples_w.masked_fill(~mask, float('-inf'))

    samples_w = torch.nn.functional.softmax(samples_w, dim=2)
    E_w = samples_w.mean(dim=0) # (B, N)


    y_hat = (1 - torch.prod((1 - samples_p)**(samples_w), dim=2)).mean(dim=0) # (B,)

    return y_hat, E_p, E_w

  def _draw_likelihood_samples_nOR(
      self, function_dist: MultitaskMultivariateNormal, num_likelihood_samples)-> Tensor:
      sample_shape = torch.Size([num_likelihood_samples] +
        [1] * (self.max_plate_nesting - len(function_dist.batch_shape) - 1))
      # print(function_dist.variance)
      if self.training:
          num_event_dims = len(function_dist.event_shape)
          function_dist = base_distributions.Normal(function_dist.mean, function_dist.variance.sqrt())
          function_dist = base_distributions.Independent(function_dist, num_event_dims - 1)
      function_samples = function_dist.rsample(sample_shape)
      return function_samples
