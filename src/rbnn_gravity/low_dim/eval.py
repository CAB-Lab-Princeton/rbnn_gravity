# Author(s): Justice Mason
# Project: RBNN + Gravity
# Date: 11/24/2023
import sys, os
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt

# Append parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models import rbnn_gravity, build_V_gravity
from utils.integrators import LieGroupVaritationalIntegrator
from utils.general import setup_reproducibility
from utils.math import pd_matrix

def load_experiment(model, optimizer, filepath: str):
    """"""
    checkpoint = torch.load(filepath)

    # Load model
    model.load_state_dict(checkpoint['model_state_dict'])
    moi_lr_diag = checkpoint['moi_diag']
    moi_lr_off_diag = checkpoint['moi_off_diag']

    # Load optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load training loss
    train_loss = checkpoint['loss']

    # Epoch number
    num_epoch = checkpoint['epoch']
    
    return model, optimizer, moi_lr_diag, moi_lr_off_diag, train_loss, num_epoch

def get_learned_moi(model):
    """"""
    print(f'\n The learned moment-of-inertia is {model.calc_moi()} \n')

def generate_loss_plot(train_loss: torch.Tensor, n_epoch: int , figname: str = 'exp-X-loss'):
    """"""
    # Define the parent directory
    save_dir = 'src/rbnn_gravity/low_dim/saved_models/' 

    # Quick maths
    n_epochs = n_epoch
    n_iterations = len(train_loss)
    n_batches = int(n_iterations/n_epochs)

    # Generate semilogy plot of training loss
    fig_loss = plt.figure(num=0)
    ax = fig_loss.add_subplot(1, 1, 1)

    ax.semilogy(train_loss[::n_batches])
    ax.grid()
    ax.legend(['Train'])
    ax.set_title('RBNN + Gravity Training Loss Curve')
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('$R + \Pi$ Loss Value')
    
    fig_loss.savefig(save_dir + figname + '.pdf')

def evaluate_lr_model(model, data_dir: str):
    """"""
    pass

def evaluate_V(model, data_dir: str, figname: str = 'PE_comparision', seed: int = 0):
    """"""
    # Seed
    save_dir = 'src/rbnn_gravity/low_dim/saved_models/'
    setup_reproducibility(seed=seed)

    # Load data
    filename = glob.glob(os.path.join(data_dir, "*.npz"))[0]

    # Load files
    npzfiles = np.load(filename)
    data_R = npzfiles['R']

    # Choose R
    n_rand = np.random.randint(low=0, high=data_R.shape[0])
    R = data_R[n_rand, ...]
    R_tensor = torch.tensor(R, requires_grad=True).unsqueeze(0)

    # Ground-truth Potential Energy
    mass = 1.
    g = 9.81 # [m/s2]
    e_3 = torch.tensor([[0., 0., 1.]]).T
    rho_gt = torch.tensor([[0., 0., 1.]])

    V_fcn = lambda R: build_V_gravity(m=mass, g=g, e_3=e_3, R=R, rho_gt=rho_gt)
    V_gt = V_fcn(R_tensor.squeeze())

    # Learned Potential Energy
    V_lr = model.V(R_tensor.reshape(-1, 9))

    # Generate plot for V_gt v. V_lr
    fig = plt.figure(num=1)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(V_gt.detach().numpy(), 'k-')
    ax.plot(V_lr.detach().numpy(), 'b-')
    ax.grid()
    ax.legend(['GT', 'Learned'])
    ax.set_title('RBNN + Gravity Potential Energy Comparison')
    ax.set_xlabel('Timestep [t]')
    ax.set_ylabel('Potential Value [V]')

    fig.savefig(save_dir + figname + '.pdf')

if __name__ == "__main__":
    # Experiment type
    experiment = 'stable'

    # Load saved experiment
    print('\n Loading trained experiment ... \n')
    filename = f"src/rbnn_gravity/low_dim/saved_models/experiment-{experiment}_ic-n_epochs-001000-date-120323.pth"
    data_dir = f"src/rbnn_gravity/data/generated_datasets/{experiment}/"

    integrator = LieGroupVaritationalIntegrator()
    model = rbnn_gravity(integrator=integrator)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    model_tr, optimizer_tr, moi_lr_diag, moi_lr_off_diag, train_loss, num_epoch = load_experiment(model=model, optimizer=optimizer, filepath=filename)
    print('\n DONE! \n ')

    # Generate figures
    print(f'\n Generating Training Loss Curve \n')
    generate_loss_plot(train_loss=train_loss, n_epoch=num_epoch, figname=f'{experiment}_train_loss_120323')
    print(f'\n Generating Potential Energy Curves \n')
    evaluate_V(model=model, data_dir=data_dir, figname=f'{experiment}_PE_comparison', seed=0)
    print('\n DONE! \n')

    # Calculate the learned moment of inertia
    get_learned_moi(model=model_tr)