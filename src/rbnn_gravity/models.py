import torch
from torch import nn
import numpy as np
import rbnn_gravity.utils.utils
from rbnn_gravity.utils.utils import hat_map, vee_map, pd_matrix


class MLP(nn.Module):
  def __init__(self, indim, hiddim, outdim):
    super(MLP, self).__init__()

    self.linear1 = nn.Linear(indim, hiddim)
    self.linear2 = nn.Linear(hiddim, hiddim)
    self.linear3 = nn.Linear(hiddim, outdim)
    self.nonlinear = nn.ReLU()

  def forward(self, x):
    h1 = self.nonlinear( self.linear1(x) )
    h2 = self.nonlinear( self.linear2(h1) )
    out = self.linear3(h2)

    return out


# You can check if  I is none then assign them the ground truth values
class RBNN(MLP):
  def __init__(self, indim=9, hiddim=50, outdim=1, dt=1e-3, I=None, V=None):
    #super(RBNN, self).__init__()
    super(RBNN, self).__init__(indim, hiddim, outdim)

    self.I = torch.nn.Parameter(torch.randn((3, 3)) / np.sqrt(9), requires_grad=True) if I is None else I
    self.V = MLP(indim, hiddim, outdim) if V is None else V
    self.dt = dt

  def forward(self, R, omega):
    # import pdb; pdb.set_trace()
    # TODO: Make this also work when it can take a sequence length as input instead of just one step

    # an example of computing gradients of a function wrt a learnable parameter 
    # is given in https://github.com/greydanus/hamiltonian-nn/blob/bcc362235dc623ffe48f22ccc22417e02e9803b4/hnn.py#L45
    # some documentation/tutorial on autograd in pytorch
    # is here https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

    # some things to check:
    # is require_grad True for R
    # it might that R should be in vector form to do this
    # there is an implementation that does this here: https://github.com/thaipduong/LieFVIN/blob/899eec2899c6fbb37af4d6ac602df2f3cde4df5a/LieFVIN/SO3FVIN.py#L79
    I33 = torch.eye(3, dtype=torch.float32)
    I_inv = torch.inverse(self.I)
    alpha = 0.5
    #
    p = torch.mv(self.I, omega)
    M = self.explicit_update(R)
    # print(M.shape)
    # Solving the nonlinear vector equation
    newton_step = 5
    a = self.dt * p + self.dt**2 * M * (1 - alpha)
    # print(p.shape, M.shape, a.shape)
    f = torch.zeros_like(a, dtype=torch.float32)
    # import pdb; pdb.set_trace()
    for i in range(newton_step):
        aTf = torch.dot(a, f)
        phi = a + torch.cross(a, f) + f * aTf - 2 * torch.mv(self.I, f)
        # print(phi.shape)
        dphi = hat_map(a, requires_grad=False) + aTf * I33 - 2 * self.I + torch.outer(f, a)
        # print(dphi.shape)
        dphi_inv = torch.inverse(dphi)
        f = f - torch.mv(dphi_inv, phi)
        # print(f.shape)

    # Need to import hat_map and vee_map 
    F = torch.matmul((I33 + hat_map(f)), torch.inverse((I33 - hat_map(f))))
    Ft = torch.transpose(F, 0, 1)
    R_next = torch.matmul(R,F)
    # print(torch.matmul(R_next,torch.transpose(R_next,0,1)))

    # Write out the full explicit update for omega_next using the equation on overleaf
    M_next = self.explicit_update(R_next)
    p_next = torch.mv(Ft, p) + alpha * self.dt * torch.mv(Ft, M) + (1 - alpha) * self.dt * M_next
    omega_next = torch.mv(I_inv, p_next)

    return R_next, omega_next

  def explicit_update(self, R):
    q = torch.flatten(R)
    V_q = self.V(q)
    dV =  torch.autograd.grad(V_q.sum(), q, create_graph=True)[0]
    dV = dV.view(3, 3)
    SM = torch.matmul(torch.transpose(dV, 0, 1), R) - torch.matmul(torch.transpose(R, 0, 1), dV)
    M = torch.stack((SM[2, 1], SM[0, 2], SM[1, 0]), dim=0)

    return M

class RBNN_gravity(nn.Module):
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


class HD_RBNN_gravity(RBNN_gravity):
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

def build_V_gt(m, g, e_3, R, rho_gt):
  # print(m, g, e_3, R.shape, rho_gt)
  # print(type(m), type(g), e_3.dtype, R.dtype, rho_gt.dtype)
  R_times_rho_gt = torch.mv(R.reshape(3,3), rho_gt)
  # print(e_3.shape, R_times_rho_gt.shape)
  e3T_times_R_times_rho_gt = torch.dot(e_3.squeeze(), R_times_rho_gt)
  m_times_g = m * g
  return -m_times_g * e3T_times_R_times_rho_gt