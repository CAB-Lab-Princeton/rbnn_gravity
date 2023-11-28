# Author(s): Justice Mason
# Project: RBNN + Gravity
# Date: 11/27/23

import sys, os

import argparse
import numpy as np

# Append parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.train import run_experiment_hd
from utils.general import setup_reproducibility_hd

def get_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        help="set experiment filename",
        required=True,
    )
    parser.add_argument(
        "--date",
        type=str,
        help="set experiment date",
        required=True,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="directory where data is stored",
        required=True,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="directory where trained models/checkpoints are saved",
        required=True,
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="The default GPU ID to use. Set -1 to use cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set random seed",
        required=True
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="Set number of epochs for training",
        required=True
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Set learning rate",
        required=True
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Set learning rate",
        required=True
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1e-3,
        help="Set timestep for integration",
        required=True
    )
    parser.add_argument(
        "--tau",
        type=int,
        default=2,
        help="Set input time horizon",
        required=True
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=10,
        help="Set sequence length"
    )
    parser.add_argument(
        "--moi_diag",
        type=float,
        nargs="+",
        default=None,
        help="Set initial value for learned moi_diag"
    )
    parser.add_argument(
        "--moi_off_diag",
        type=float,
        nargs="+",
        default=None,
        help="Set initial value for learned moi_off_diag"
    )
    parser.add_argument(
        "--V_in_dims",
        type=int,
        default=9,
        help="Set in_dim for potential energy MLP",
        required=True
    )
    parser.add_argument(
        "--V_hidden_dims",
        type=int,
        default=50,
        help="Set hidden_dim for potential energy MLP",
        required=True
    )
    parser.add_argument(
        "--V_out_dims",
        type=int,
        default=1,
        help="Set out_dim for potential energy MLP",
        required=True
    )
    parser.add_argument(
        "--lambda_loss",
        type=float,
        nargs="+",
        default=None,
        help="Set relative weights for loss terms",
        required=True
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="Print loss value every n epochs.",
        required=True
    )
    parser.add_argument(
        "--retrain_model",
        action='store_false',
        help="Retrain model boolean",
        required=True
    )
    parser.add_argument(
        "--save_model",
        action='store_true',
        help="Save model boolean",
        required=True
    )

    args = parser.parse_args()
    return args

def main():
    """"""
    # Get args
    args = get_args()

    # Setup reproducibility
    setup_reproducibility_hd(seed=args.seed)

    # Run experiment
    run_experiment_hd(args=args)


if __name__ == "__main__":
    main()