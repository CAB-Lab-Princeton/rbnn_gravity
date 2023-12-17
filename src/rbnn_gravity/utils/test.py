# Author(s): Justice Mason
# Project: RBNN + Gravity
# Date: 12/15/23

import os, sys
import numpy as np
import torch 

# Append parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models import build_V_gravity
from data.data_utils import generate_lowdim_dataset_gyroscope
from utils.integrators import LieGroupVaritationalIntegrator

# Test Gyroscope 
def test_gyroscope(n_sample: int = 10):
    """"""
    # Initialize integrator
    seed = 0
    integrator = LieGroupVaritationalIntegrator()
    
    # Potential properties 
    moi = np.diag([1., 1., 2.8])
    dt = 1e-3
    traj_len = 1000
    
    # Constants
    mass = 1.0
    e_3 = torch.tensor([[0., 0., 1.]]).T
    rho_gt = torch.tensor([[0., 0., 1.]])

    V_grav = lambda R: build_V_gravity(m=mass, R=R, e_3=e_3, rho_gt=rho_gt)

    R_sample, pi_sample = generate_lowdim_dataset_gyroscope(MOI=moi, n_samples=n_sample, integrator=integrator, timestep=dt, traj_len=traj_len, V=V_grav, seed=seed)
    return R_sample, pi_sample

if __name__ == "__main__":
    print('Hey, king, I started!')
    # R, pi = test_gyroscope(10)
    # print(R.shape)