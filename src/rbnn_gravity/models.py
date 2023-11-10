# Author(s): Justice Mason, CAB
# Project: RBNN + Gravity
# Date: 11/09/23

import torch
import numpy as np

from torch import nn

from utils.math import hat_map, vee_map, pd_matrix

# Model classes

class MLP(nn.Module):
  def __init__(self, indim, hiddim, outdim):
    super(MLP, self).__init__()

    self.linear1 = nn.Linear(indim, hiddim)
    self.linear2 = nn.Linear(hiddim, hiddim)
    self.linear3 = nn.Linear(hiddim, outdim)
    self.nonlinear = nn.ReLU()

  def forward(self, x):
    h1 = self.nonlinear(self.linear1(x))
    h2 = self.nonlinear(self.linear2(h1))
    out = self.linear3(h2)

    return out

# RBNN with gravity - Low Dimensional
class rbnn_gravity(nn.Module):
  def __init__(self,
                integrator,
                in_dim: int = 9,
                hidden_dim: int = 50,
                out_dim: int = 1,
                tau: int = 2,
                dt: float = 1e-3,
                I_diag: torch.Tensor = None,
                I_off_diag: torch.Tensor = None,
                V = None):
    """
    ...
    """
    super().__init__()
    self.integrator = integrator
    self.tau = tau
    self.dt =dt

    self.V = MLP(in_dim, hidden_dim, out_dim) if V is None else V

    # Moment-of-inertia tensor -- assert that requires_grad is on for learning
    self.I_diag = torch.rand(3, requires_grad=True) / torch.sqrt(3) if I_diag is None else I_diag
    self.I_off_diag = torch.rand(3, requires_grad=True) / torch.sqrt(3) if I_off_diag is None else I_off_diag

    assert self.I_diag.requires_grad == True and self.I_off_diag.requires_grad == True

  def calc_moi(self):
    """
    ...

    """
    # Calculate moment-of-inertia matrix as a symmetric postive definite matrix 
    moi = pd_matrix(diag=self.I_diag, off_diag=self.I_off_diag)
    return moi
  
  def forward(self, R_seq: torch.Tensor, pi_seq: torch.Tensor, seq_len: int = 100):
    """
    ...

    """
    # Calculate moment-of-inertia tensor
    moi = self.calc_moi()

    # Grab initial conditions
    R_init = R_seq[:, 0, ...][:, None, ...]
    pi_init = pi_seq[:, 0, ...][:, None, ...]

    # Integrate full trajectory
    R_pred, pi_pred = self.integrator.integrate(pi_init=pi_init, R_init=R_init, moi=moi, timestep=self.dt, traj_len=seq_len)

    return R_pred, pi_pred

# RBNN with gravity - High Dimensional
class HD_RBNN_gravity(rbnn_gravity):
  def __init__(self, encoder,
                decoder,
                integrator,
                estimator):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.integrator = integrator
    self.estimator = MLP(self.in_dim * self.tau, self.hidden_dim, 3) if estimator is None else estimator

    self.I_diag = torch.rand(3, requires_grad=True) / torch.sqrt(3)
    self.I_off_diag = torch.rand(3, requires_grad=True) / torch.sqrt(3)  
  
  def calc_I(self):
    """
    """
    moi = pd_matrix(diag=self.I_diag, off_diag=self.I_off_diag)
    return moi
  
  def encode(self, x: torch.Tensor) -> torch.Tensor:
    """
    Method to encode from image space to SE(3) latent space
    """
    pass

  def decode(self):
    pass

  def forward(self):
    pass

  def velocity_estimator(self):
    pass

  def state_update(self):
    pass

# Gravity potential function
def build_V_gravity(m: float,
                    R: torch.Tensor, 
                    e_3: torch.Tensor, 
                    rho_gt: torch.Tensor,
                    g: float = 9.81):
  """
  Potential energy function for gravity.

  ...

  Parameters
  ----------
  m : [1]
    mass of object
  g :  [1]
    gravity constant
  e_3 : [3]
    z-direction
  R : [bs, 3, 3]
    Rotation matrix
  rho_gt : 

  """
  bs = R.shape[0]
  R_reshape = R.reshape(bs, 3, 3)
  R_times_rho_gt = torch.einsum('bij, j -> bi', R_reshape, rho_gt.squeeze().float())
  e3T_times_R_times_rho_gt = torch.einsum('j, bj -> b', e_3.squeeze().float(), R_times_rho_gt)
  Vg = -m * g * e3T_times_R_times_rho_gt
  
  return Vg