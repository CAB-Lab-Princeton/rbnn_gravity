import torch
from rbnn_gravity.models import RBNN
from rbnn_gravity.configuration import config
from rbnn_gravity.models import build_V_gt
import numpy as np


N = 10
T = 10
m = 1.0
g = 10.0
dt = 1e-3

e_3 = torch.tensor([[0], [0], [1]], dtype=torch.float32)
I_gt = torch.diag(torch.tensor([1, 2.8, 2], dtype=torch.float32))
rho_gt = torch.tensor([0, 0, 1], dtype=torch.float32)

omega_0 = torch.tensor([0.5, -0.5, 0.4], dtype=torch.float32)
R_0 = torch.eye(3, requires_grad=True, dtype=torch.float32)

V_gt = lambda R: build_V_gt(m, g, e_3, R, rho_gt)

# TODO: put integrator
gt_rbnn = RBNN(indim=9, hiddim=50, outdim=1, I=I_gt, V=V_gt)

R = R_0  # Convert R_0 to a tensor
# print(R_0)
omega = omega_0  # Convert omega_0 to a tensor
# print(omega_0)
R=R.unsqueeze(0)
omega=omega.unsqueeze(0)
# TO DO: we need to verify the integrator works well first so import data
# from MATLAB and then compute or plot
for i in range(N * T-1):
    R_next, omega_next = gt_rbnn(R[-1,:], omega[-1,:])
    R = torch.cat((R, R_next.unsqueeze(0)), dim=0)  # Concatenate R_next to the tensor R
    omega = torch.cat((omega, omega_next.unsqueeze(0)), dim=0)
    # print(R_next)

R_np = R.detach().numpy()
omega_np = omega.detach().numpy()
t_np = np.linspace(0, N*T*dt-dt, num=N*T)

with open('/home/ca15/projects/rbnn_gravity/src/rbnn_gravity/data/data.npz', 'wb') as outfile:
    np.savez(outfile, R=R_np, omega=omega_np)