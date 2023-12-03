# Author(s): Justice Mason
# Project: RBNN + Gravity
# Date: 11/07/23
import sys, os

import argparse
import torch
import numpy as np

# Append parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models import build_V_gravity
from data.data_utils import *
from utils.math import *
from utils.general import *
from utils.integrators import LieGroupVaritationalIntegrator

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        help="set filename",
        required=True,
    )
    parser.add_argument(
        "--date",
        type=str,
        help="set date",
        required=True,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="set directory where data is saved",
        required=True,
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=100,
        help="set number of examples, default: 100",
    )
    parser.add_argument(
        "--traj_len",
        type=int,
        default=100,
        help="set trajectory length, default: 100",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=10,
        help="set sample sequence length, default: 10",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1e-3,
        help="set dt, default: 1e-3",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=1.0,
        help="set mass, default: 1.0",
    )
    parser.add_argument(
        "--moi_diag_gt",
        type=float,
        nargs='+',
        default=None,
        required=True,
        help="set ground-truth moi diagonal entries, default: None",
    )
    parser.add_argument(
        "--moi_off_diag_gt",
        type=float,
        nargs='+',
        default=None,
        required=True,
        help="set ground-truth moi off-diagonal entries, default: None",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=50.0,
        help="set angular momentum sphere radius, default: 50.0",
    )
    parser.add_argument(
        "--R_ic_type",
        choices=['stable', 'unstable', 'uniform'],
        default='stable',
        help="set R ic type for sampling group matrices on SO(3)",
    )
    parser.add_argument(
        "--pi_ic_type",
        choices=['random', 'unstable', 'desired'],
        default='random',
        help="set pi ic types for sample angular momentum sphere",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="set random seed"
    ) 

    args = parser.parse_args()
    return args

def main():
    """
    """
    # Initialize arguments
    args = get_args()

    # Seed computation
    setup_reproducibility(args.seed)

    # Constants
    g = 9.81 # [m/s2]
    e_3 = torch.tensor([[0., 0., 1.]]).T
    rho_gt = torch.tensor([[0., 0., 1.]])

    # Initialize values
    moi = pd_matrix(diag=torch.sqrt(torch.tensor(args.moi_diag_gt)), off_diag=torch.tensor(args.moi_off_diag_gt))
    print(f'\n Ground-truth moment of inertia: \n {moi} \n')

    # Initialize potential function
    V_gravity = lambda R: build_V_gravity(m=args.mass, g=g, e_3=e_3, R=R, rho_gt=rho_gt)

    # Integrator
    integrator = LieGroupVaritationalIntegrator()

    # Random sampling ICs + integrating
    print(f'\n Generating dataset -- n_samples:{args.n_examples} --traj_len:{args.traj_len} \n')

    data_R, data_pi = generate_lowdim_dataset_3DP(MOI=moi, 
                            V=V_gravity,
                            radius=args.radius, 
                            n_samples=args.n_examples, 
                            integrator=integrator,
                            timestep=args.dt,
                            traj_len=args.traj_len,
                            R_ic_type=args.R_ic_type,
                            pi_ic_type=args.pi_ic_type,
                            seed=args.seed)
    
    # Convert to omega, numpy, and save
    moi_inv = torch.linalg.inv(moi)
    data_omega = torch.einsum('ij, btj -> bti', moi_inv, data_pi)
    name = args.name + "-n_examples-" + str(args.n_examples)

    print(f'\n Saving generated dataset in directory: {args.save_dir} \n')
    save_data(data_R=data_R, data_omega=data_omega, filename=name, date=args.date, save_dir=args.save_dir)

if __name__ == "__main__":
    main()

