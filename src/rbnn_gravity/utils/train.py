# Author(s): Justice Mason
# Project: RBNN + Gravity
# Date: 11/10/23

import sys, os

import numpy as np
import torch

from data.dataset import build_dataloader
from models import rbnn_gravity, MLP
from utils.integrators import LieGroupVaritationalIntegrator
from utils.general import setup_reproducibility

# Training functions
def train_epoch(args, model, dataloader, optimizer, loss_fcn):
    """
    Function that trains model for one epoch.

    ...
    """
    # Set model to train mode
    model.train()
    epoch_loss = []

    # Iterate over elements of the dataloader and train
    for _, data in enumerate(dataloader):
        # Extract data
        data_R, data_omega = data
       
        # Check requires_grad -- for autograd/backprop
        if not data_R.requires_grad:
            data_R.requires_grad = True
        
        if not data_omega.requires_grad:
            data_omega.requires_grad = True

        # Load data onto device
        data_R = data_R.to(model.device)
        data_omega = data_omega.to(model.device)

        # Data should be dtype float
        data_R.type(torch.float)
        data_omega.type(torch.float)

        # Forward pass and calculate loss
        data_R_recon, data_omega_recon = model(R_seq=data_R, omega_seq=data_omega, seq_len=args.seq_len)
        loss = loss_fcn(R_gt=data_R, R_recon=data_R_recon, omega_gt=data_omega, omega_recon=data_omega_recon, lambda_loss=args.lambda_loss)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward() # retain_graph=True)
        optimizer.step()

        # Append loss
        epoch_loss.append(loss.item())
    
    return epoch_loss

def eval_epoch(args, model, dataloader, loss_fcn):
    """
    Evaluate model for one epoch without backpropagation.
    ...

    """
    # Set model to eval mode
    model.eval()
    epoch_loss = []

    # Iterate over elements of the dataloader and train
    for _, data in enumerate(dataloader):
        # Extract data
        data_R, data_omega = data
        
        # Check requires_grad for R -- for M calculation
        if not data_R.requires_grad:
            data_R.requires_grad = True
        
        # Load data onto device
        data_R.to(model.device)
        data_omega.to(model.device)
        
        # Data shoudl be dtype float
        data_R.type(torch.float)
        data_omega.type(torch.float)

        # Forward pass and calculate loss
        data_R_recon, data_omega_recon = model(R_seq=data_R, omega_seq=data_omega, seq_len=args.seq_len)
        loss = loss_fcn(R_gt=data_R, R_recon=data_R_recon, omega_gt=data_omega, omega_recon=data_omega_recon, lambda_loss=args.lambda_loss)

        # Append loss
        epoch_loss.append(loss.item())

    return epoch_loss

def train(args, model, traindataloader, loss_fcn):
    """
    """
    # Initialize training loss
    training_loss = []

    # Initialize optimizer -- NEEDS TO CHANGE IF WE ARE CONTINUING TRAINING
    params = model.parameters()
    optim = torch.optim.Adam(params=params, lr=args.lr)
    
    # To Do: Add a lr scheduler
    # Iterate through epochs
    for epoch in range(args.n_epochs):
        # Run train_epoch function
        epoch_loss = train_epoch(args=args, model=model, dataloader=traindataloader, optimizer=optim, loss_fcn=loss_fcn)

        # Concat epoch loss onto training loss
        training_loss += epoch_loss

        if epoch % args.print_every == 0:
            print(f'\n Epoch Number: {epoch + 1} ; Training Loss: {training_loss[-1]:.4e} \n')
    
    return training_loss, optim
        
# Auxilary function for experiments
def run_experiment(args):
    """
    """
    # Set up reproducibility
    print(f'\n Setting up reproducibility with seed: {args.seed} ... \n')
    setup_reproducibility(seed=args.seed)

    # Set up GPU
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else "cpu")
    
    # Initialize potential energy function as an MLP
    V_learned = MLP(args.V_in_dims, args.V_hidden_dims, args.V_out_dims)

    # Initialize integrator
    lgvi_integrator = LieGroupVaritationalIntegrator()

    # Initialize model and optimizer
    model = rbnn_gravity(integrator=lgvi_integrator,
                        in_dim=args.V_in_dims,
                        hidden_dim=args.V_hidden_dims,
                        out_dim=args.V_out_dims,
                        tau=args.tau, 
                        dt=args.dt, 
                        I_diag=args.moi_diag, 
                        I_off_diag=args.moi_off_diag, 
                        V=V_learned)
    model.to(device)
    model.device = device

    # Create dataloaders
    print('\n Building the dataloaders ... \n')
    train_dataloader = build_dataloader(args=args)

    # Loss function
    loss = low_dim_loss

    # Retrain model
    if args.retrain_model:
        args.save_dir = args.save_dir[:-4] + "-retrain.pth"
        print(f'\n New save directory for re-trained model: {args.save_dir} ... \n')

    # Train model
    print(f'\n Training model on device ({device}) ... \n')
    train_loss, optim = train(args=args, model=model, traindataloader=train_dataloader, loss_fcn=loss)

    # Save model
    if args.save_model:
        print(f'\n Saving experiment {args.exp_name} from date {args.date} ... \n')
        save_experiment(args=args, model=model, optimizer=optim, loss=train_loss)    
    
def save_experiment(args, model, optimizer, loss):
    """
    """
    # Check if save directory exists
    if not os.path.exists(args.save_dir):
        print(f'\n Making save directory for models {args.save_dir} ... \n')
        os.makedirs(args.save_dir)

    # Save model and optimizer
    checkpoint_path = args.save_dir + f'/experiment-{args.exp_name}-n_epochs-{args.n_epochs:06}-date-{args.date}.pth'
    checkpoint = {
            'epoch': args.n_epochs,
            'model_state_dict': model.state_dict(),
            'moi_inv_diag': model.I_diag,
            'moi_inv_off_diag': model.I_off_diag,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }
    torch.save(checkpoint, checkpoint_path)

# Loss functions
def low_dim_loss(R_gt: torch.Tensor, 
                 omega_gt: torch.Tensor, 
                 R_recon: torch.Tensor, 
                 omega_recon: torch.Tensor, 
                 lambda_loss: torch.Tensor = torch.ones(2)):
    """"""
    R_loss = R_loss_fcn(R_gt=R_gt, R_recon=R_recon)
    omega_loss = omega_loss_fcn(omega_gt=omega_gt, omega_recon=omega_recon)

    total_loss = lambda_loss[0] * R_loss + lambda_loss[1] * omega_loss
    return total_loss

def R_loss_fcn(R_gt: torch.Tensor, R_recon: torch.Tensor):
    """"""
    # Loss function definition
    loss_fcn = torch.nn.MSELoss()

    # Grab shape of ground-truth R
    bs, seq_len, _, _ = R_gt.shape

    # Make identity elem
    I_n = torch.eye(3, device=R_gt.device)[None, None, ...].repeat(bs, seq_len, 1, 1)
    I_n_recon = torch.einsum('btji, btjk -> btik', R_gt, R_recon)

    loss = loss_fcn(I_n, I_n_recon)
    return loss

def omega_loss_fcn(omega_gt: torch.Tensor, omega_recon: torch.Tensor):
    """"""
    # Loss function definition 
    loss_fcn = torch.nn.MSELoss()
    loss = loss_fcn(omega_gt, omega_recon)
    return loss

