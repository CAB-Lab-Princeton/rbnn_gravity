# Author(s): Justice Mason, CAB
# Project: RBNN + Gravity
# Date: 11/09/23

import torch
import numpy as np

from torch import nn

from utils.math_utils import hat_map, vee_map, pd_matrix

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
    self.dt = dt
    self.in_dim = in_dim
    self.hidden_dim = hidden_dim
    self.out_dim = out_dim
    self.device = None

    self.V = MLP(self.in_dim, self.hidden_dim, self.out_dim) if V is None else V

    # Moment-of-inertia tensor
    self.I_diag = torch.nn.Parameter(torch.rand(3) / np.sqrt(3), requires_grad=True) if I_diag is None else I_diag
    self.I_off_diag = torch.nn.Parameter(torch.rand(3) / np.sqrt(3), requires_grad=True) if I_off_diag is None else I_off_diag

  def calc_moi(self):
    """
    ...

    """
    # Calculate moment-of-inertia matrix as a symmetric postive definite matrix 
    moi = pd_matrix(diag=self.I_diag, off_diag=self.I_off_diag).to(self.device)
    return moi
  
  def forward(self, R_seq: torch.Tensor, omega_seq: torch.Tensor, seq_len: int = 100):
    """
    ...

    """
    # Calculate moment-of-inertia tensor and pi_seq
    moi = self.calc_moi()
    pi_seq = torch.einsum('ij, btj -> bti', moi, omega_seq)

    # Grab initial conditions
    R_init = R_seq[:, 0, ...]
    pi_init = pi_seq[:, 0, ...]

    # Integrate full trajectory
    R_pred, pi_pred = self.integrator.integrate(pi_init=pi_init, R_init=R_init, V=self.V, moi=moi, timestep=self.dt, traj_len=seq_len)

    # Calc omega_pred
    moi_inv = torch.linalg.inv(moi)
    omega_pred = torch.einsum('ij, btj -> bti', moi_inv, pi_pred)

    return R_pred, omega_pred


# RBNN with gravity - High Dimensional
class rbnn_gravity_hd(rbnn_gravity):
  """
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
  def __init__(self,
              encoder,
              decoder,
              estimator,
              integrator,
              in_dim: int = 9,
              hidden_dim: int = 50,
              out_dim: int = 1,
              tau: int = 2,
              dt: float = 1e-3,
              I_diag: torch.Tensor = None,
              I_off_diag: torch.Tensor = None,
              V = None):
    super().__init__(integrator,
                    in_dim,
                    hidden_dim,
                    out_dim,
                    tau,
                    dt,
                    I_diag,
                    I_off_diag,
                    V)
    
    self.encoder = encoder
    self.decoder = decoder
    self.indices2 = None
    self.indices4 = None
    self.estimator = MLP(self.in_dim * self.tau, self.hidden_dim, 3) if estimator is None else estimator
  
  def data_preprocess(self, x: torch.Tensor):
    """"""
    # Unfold data
    bs, seq_len, C, W, H = x.shape    
    pbs = bs * seq_len

    x_unfolded = x.unfold(dimension=1, size=1, step=1)
    x_perm = x_unfolded.permute(0, 1, 5, 2, 3, 4)
    x_rs = x_perm.reshape(pbs, 1, C, W, H)

    return x_rs

  def data_postprocess(self, xhat: torch.Tensor, batch_size: int, seq_len: int = 2):
    """"""
    # Reshape data
    pbs, C, W, H = xhat.shape
    xhat_rs = xhat.reshape(batch_size, seq_len, C, W, H)
    return xhat_rs

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    """
    Method to encode from image space to SO(3) latent space
    """
    # Encode images
    z, indices2, indices4 = self.encoder(x)

    # Set indices 
    self.indices2 = indices2
    self.indices4 = indices4

    return z

  def decode(self, z: torch.Tensor):
    """"""
    # Reshape indices
    if self.indices2.shape[0] != z.shape[0]:
        batch_size = int(z.shape[0]/self.indices2.shape[0])
        indices2 = self.indices2.repeat([batch_size, 1, 1, 1])
    else:
        indices2 = self.indices2
    
    if self.indices4.shape[0] != z.shape[0]:
        batch_size = int(z.shape[0]/self.indices4.shape[0])
        indices4 = self.indices4.repeat([batch_size, 1, 1, 1])
    else:
        indices4 = self.indices4
    
    # Decode latent state
    xhat = self.decoder(z, indices2, indices4)
    return xhat

  def forward(self, x: torch.Tensor, seq_len: int = 2):
    """"""
    # Shape of input
    bs, input_seq_len, _, _, _ = x.shape 
    # Calculate moment-of-inertia
    moi = self.calc_moi()
    moi_inv = torch.linalg.inv(moi)

    # Encode input sequence
    x_obs = self.data_preprocess(x=x)
    R_enc = self.encode(x=x_obs) 
    R_enc_rs = R_enc.reshape(bs, -1, 3, 3)

    # Estimate angular velocity
    omega_enc = self.velocity_estimator(z_seq=R_enc_rs)
    pi_enc = torch.einsum('ij, btj -> bti', moi, omega_enc)

    # Define the intial condition
    R_0 = R_enc_rs[:, 0, ...]
    pi_0 = pi_enc[:, 0, ...]

    # Integrate full trajectory
    R_dyn, pi_dyn = self.integrator.integrate(pi_init=pi_0, R_init=R_0, V=self.V, moi=moi, timestep=self.dt, traj_len=seq_len)
    omega_dyn = torch.einsum('ij, btj -> bti', moi_inv, pi_dyn)

    # Reshape dynamics-based prediction
    R_dyn_rs = R_dyn.reshape(-1, 3, 3)

    # Decode autoencoder-based and dynamics-based predictions
    xhat_dyn_ = self.decode(z=R_dyn_rs)
    xhat_recon_ = self.decode(z=R_enc)

    # Post-process step
    xhat_dyn = self.data_postprocess(xhat=xhat_dyn_, batch_size=bs, seq_len=seq_len)
    xhat_recon = self.data_postprocess(xhat=xhat_recon_, batch_size=bs, seq_len=seq_len)

    return xhat_dyn, xhat_recon, R_dyn, omega_dyn, R_enc_rs, omega_enc

  def velocity_estimator(self, z_seq: torch.Tensor, full_seq: bool = True):
    """"""
    bs, seq_len, W, H = z_seq.shape
    omega_est_ = []

    if full_seq:
      for t in range(seq_len - self.tau):
        z = z_seq[:, t:t+self.tau, ...].reshape(bs, self.tau*W*H)
        omega_ = self.estimator(z)
        omega_est_.append(omega_)
      
      omega_est = torch.stack(omega_est_, dim=1)
    else:
      z = z_seq[:, :self.tau, ...].reshape(bs, self.tau*W*H)
      omega_est = self.estimator(z_seq)

    return omega_est


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