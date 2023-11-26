# Author(s): Justice Mason
# Project: RBNN + Gravity
# Date: 11/24/2023

import numpy as np
import torch
import matplotlib.pyplot as plt

from models import rbnn_gravity
from utils.integrators import LieGroupVaritationalIntegrator
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
    save_dir = 'src/rbnn_gravity/saved_models/' 

    # Quick maths
    n_epochs = n_epoch
    n_iterations = len(train_loss)
    n_batches = int(n_iterations/n_epochs)

    # Generate semilogy plot of training loss
    plt.semilogy(train_loss[::n_batches])
    plt.grid()
    plt.legend(['Train'])
    plt.title('RBNN + Gravity Training Loss Curve')
    plt.xlabel('Epoch Number')
    plt.ylabel('$R + \Pi$ Loss Value')
    
    plt.savefig(save_dir + figname + '.pdf')

def evaluate_lr_model(model, data_dir: str):
    pass

if __name__ == "__main__":
    
    # Load saved experiment
    print('\n Loading trained experiment ... \n')
    filename = "src/rbnn_gravity/saved_models/experiment-exp-3-n_epochs-000100-date-112523.pth"
    integrator = LieGroupVaritationalIntegrator()
    model = rbnn_gravity(integrator=integrator)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    model_tr, optimizer_tr, moi_lr_diag, moi_lr_off_diag, train_loss, num_epoch = load_experiment(model=model, optimizer=optimizer, filepath=filename)
    print('\n DONE! \n ')

    # Generate figures
    print(f'\n Generating Training Loss Curve \n')
    generate_loss_plot(train_loss=train_loss, n_epoch=num_epoch, figname='exp-3-112523')
    print('\n DONE! \n')

    # Calculate the learned moment of inertia
    get_learned_moi(model=model_tr)