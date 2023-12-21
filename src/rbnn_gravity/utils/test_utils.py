# Author(s): Justice Mason
# Project: RBNN + Gravity
# Date: 12/15/23

import os, sys
import numpy as np
import torch

import matplotlib.pyplot as plt

# Append parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models import build_V_gravity
from data.data_utils import generate_lowdim_dataset_gyroscope
from utils.general import setup_reproducibility
from utils.integrators import LieGroupVaritationalIntegrator
from utils.math_utils import group_matrix_to_eazyz

# Test Gyroscope 
def test_gyroscope(n_sample: int = 10):
    """"""
    # Initialize integrator
    seed = 0
    integrator = LieGroupVaritationalIntegrator()
    
    # Potential properties 
    moi = torch.diag(torch.tensor([1., 1., 2.8]))
    moi_inv = torch.linalg.inv(moi)

    dt = 1e-3
    traj_len = 1000
    
    # Constants
    mass = 1.0
    e_3 = torch.tensor([[0., 0., 1.]]).T
    rho_gt = torch.tensor([[0., 0., 1.]])

    V_grav = lambda R: build_V_gravity(m=mass, R=R, e_3=e_3, rho_gt=rho_gt)

    R_sample, pi_sample = generate_lowdim_dataset_gyroscope(MOI=moi, n_samples=n_sample, integrator=integrator, timestep=dt, traj_len=traj_len, V=V_grav, seed=seed)
    
    omega_sample = torch.einsum('ij, btj -> bti', moi_inv, pi_sample)
    ea_samples = group_matrix_to_eazyz(R_sample)
    return ea_samples, omega_sample

if __name__ == "__main__":
    setup_reproducibility(seed=0)
    ea, omega = test_gyroscope(10)

    ea = ea.detach().numpy()
    omega = omega.detach().numpy()

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 30))

    # Euler angles

    # Phi
    for i in range(ea.shape[0]):
        axes[0, 0].plot(ea[i, :, 0])
    axes[0, 0].grid()
    # axes[0, 0].legend(['Train', 'Validation'])
    axes[0, 0].set_xlabel('Timestep [t]')
    axes[0, 0].set_ylabel('Value [Radians]')
    axes[0, 0].set_title('Phi $\phi$')

    # Theta
    for i in range(ea.shape[0]):
        axes[0, 1].plot(ea[i, :, 1])
    axes[0, 1].grid()
    # axes[0, 1].legend(['Train', 'Validation'])
    axes[0, 1].set_xlabel('Timestep [t]')
    axes[0, 1].set_ylabel('Value [Radians]')
    axes[0, 1].set_title('Theta $\Theta$')

    # Psi
    for i in range(ea.shape[0]):
        axes[0, 2].plot(ea[i, :, 2])
    axes[0, 2].grid()
    # axes[0, 2].legend(['Train', 'Validation'])
    axes[0, 2].set_xlabel('Timestep [t]')
    axes[0, 2].set_ylabel('Value [Radians]')
    axes[0, 2].set_title('Psi $\psi$')

    # Oemga
    for i in range(omega.shape[0]):
        axes[1, 0].plot(omega[i, :, 0])
    axes[1, 0].grid()
    # axes[1, 0].legend(['Train', 'Validation'])
    axes[1, 0].set_xlabel('Timestep [t]')
    axes[1, 0].set_ylabel('Value [Rad/s]')
    axes[1, 0].set_title('$\Omega_1$')

    for i in range(omega.shape[0]):
        axes[1, 1].plot(omega[i, :, 1])
    axes[1, 1].grid()
    # axes[1, 1].legend(['Train', 'Validation'])
    axes[1, 1].set_xlabel('Timestep [t]')
    axes[1, 1].set_ylabel('Value [Rad/s]')
    axes[1, 1].set_title('$\Omega_2$')

    for i in range(omega.shape[0]):
        axes[1, 2].plot(omega[i, :, 2])
    axes[1, 2].grid()
    # axes[1, 2].legend(['Train', 'Validation'])
    axes[1, 2].set_xlabel('Timestep [t]')
    axes[1, 2].set_ylabel('Value [Rad/s]')
    axes[1, 2].set_title('$\Omega_3$')

    save_dir = f'src/rbnn_gravity/high_dim/results/test/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + 'ea_omega_test_plot.pdf')


