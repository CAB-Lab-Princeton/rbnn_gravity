# Author(s): Justice Mason
# Project: RBNN + Gravity
# Date: 11/10/23

import sys, os

import numpy as np
import torch
import time

from data.dataset import build_dataloader, build_dataloader_hd
from models import rbnn_gravity, rbnn_gravity_hd, MLP
from autoencoder import EncoderRBNN_gravity, DecoderRBNN_gravity
from utils.integrators import LieGroupVaritationalIntegrator, Harsh_LGVI
from utils.general import setup_reproducibility, setup_reproducibility_hd

# Low-dimensional training functions
def train_epoch(args, model, dataloader, optimizer, loss_fcn):
    """
    Function that trains model for one epoch.

    ...
    """
    # Set model to train mode
    epoch_loss = []
    
    # Set tau -- should be an input
    tau = 1 # equals 1 for low dimension

    # Iterate over elements of the dataloader and train
    for idx, data in enumerate(dataloader):
        # Extract data
        data_R, data_omega = data
    
        # Check requires_grad -- for autograd/backprop
        if not data_R.requires_grad:
            data_R.requires_grad = True
        
        # Load data onto device
        data_R = data_R.to(model.device)
        data_omega = data_omega.to(model.device)

        # Data should be dtype float
        data_R.type(torch.float)
        data_omega.type(torch.float)

        # Forward pass and calculate loss
        data_R_recon, data_omega_recon = model(R_seq=data_R, omega_seq=data_omega, seq_len=args.seq_len)
        loss = loss_fcn(R_gt=data_R[:, tau:, ...], R_recon=data_R_recon[:, tau:, ...], omega_gt=data_omega[:, tau:, ...], omega_recon=data_omega_recon[:, tau:, ...], lambda_loss=args.lambda_loss)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward() # retain_graph=True)
        optimizer.step()

        # Append loss
        if idx % args.print_every == 0:
            batch_loss = loss.item()
            epoch_loss.append(batch_loss)
    
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
        
# Auxilary function for low-dimensional experiments
def run_experiment(args):
    """"""
    # Set up reproducibility
    print(f'\n Setting up reproducibility with seed: {args.seed} ... \n')
    setup_reproducibility(seed=args.seed)

    # Set up GPU
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else "cpu")
    
    # Initialize potential energy function as an MLP
    V_learned = MLP(args.V_in_dims, args.V_hidden_dims, args.V_out_dims)

    # Initialize integrator
    lgvi_integrator = LieGroupVaritationalIntegrator() # Harsh_LGVI()

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
    t0 = time.time()
    train_loss, optim = train(args=args, model=model, traindataloader=train_dataloader, loss_fcn=loss)
    
    train_time = time.time() - t0
    print(f'\n Training time: {train_time} \n')

    # Save model
    if args.save_model:
        print(f'\n Saving experiment {args.exp_name} from date {args.date} ... \n')
        save_experiment(args=args, model=model, optimizer=optim, loss=train_loss)
      
# High-dimensional training functions
def train_epoch_hd(args, model, dataloader, optimizer, loss_fcn):
    """
    Function that trains model for one epoch.

    ...
    """
    # Set model to train mode
    epoch_loss = []

    # Iterate over elements of the dataloader and train
    for idx, data in enumerate(dataloader):
        # Extract data
        data_x = data
    
        # Check requires_grad -- for autograd/backprop
        if not data_x.requires_grad:
            data_x.requires_grad = True
        
        # Load data onto device
        data_x = data_x.to(model.device)

        # Data should be dtype float
        data_x.type(torch.float)

        # Forward pass and calculate loss
        data_x_dyn, data_x_recon, R_dyn, omega_dyn, R_enc, omega_enc  = model(x=data_x, seq_len=args.seq_len)
        loss = loss_fcn(x_gt=data_x,\
                        x_dyn=data_x_dyn,\
                        x_enc=data_x_recon,\
                        R_dyn=R_dyn,\
                        R_enc=R_enc,\
                        omega_dyn=omega_dyn,\
                        omega_enc=omega_enc,\
                        tau=model.tau,\
                        lambda_loss=args.lambda_loss)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Append loss
        if idx % args.print_every == 0:
            batch_loss = loss.item()
            epoch_loss.append(batch_loss)
    
    return epoch_loss

def eval_epoch_hd(args, model, dataloader, loss_fcn):
    """"""
    model.eval()
    epoch_loss = []

    # Iterate over elements of the dataloader and train
    for _, data in enumerate(dataloader):
        # Extract data
        data_x = data
    
        # Check requires_grad -- for autograd/backprop
        if not data_x.requires_grad:
            data_x.requires_grad = True
        
        # Load data onto device
        data_x = data_x.to(model.device)

        # Data should be dtype float
        data_x.type(torch.float)

        # Forward pass and calculate loss
        data_x_dyn, data_x_recon, R_dyn, omega_dyn, R_enc, omega_enc  = model(x=data_x, seq_len=args.seq_len)
        loss = loss_fcn(x_gt=data_x,\
                        x_dyn=data_x_dyn,\
                        x_recon=data_x_recon,\
                        R_dyn=R_dyn,\
                        R_enc=R_enc,\
                        omega_dyn=omega_dyn,\
                        omega_enc=omega_enc,\
                        tau=model.tau,\
                        lambda_loss=args.lambda_loss)
        
        # Append loss
        epoch_loss.append(loss.item())
    
    return epoch_loss

def train_hd(args, model, traindataloader,  loss_fcn):
    """"""
    # Initialize training loss
    training_loss = []

    # Initialize optimizer -- NEEDS TO CHANGE IF WE ARE CONTINUING TRAINING
    params = model.parameters()
    optim = torch.optim.Adam(params=params, lr=args.lr)
    
    # To Do: Add a lr scheduler
    # Iterate through epochs
    for epoch in range(args.n_epochs):
        # Run train_epoch function
        epoch_loss = train_epoch_hd(args=args, model=model, dataloader=traindataloader, optimizer=optim, loss_fcn=loss_fcn)

        # Concat epoch loss onto training loss
        training_loss += epoch_loss

        if epoch % args.print_every == 0:
            print(f'\n Epoch Number: {epoch + 1} ; Training Loss: {training_loss[-1]:.4e} \n')
    
    return training_loss, optim

# Auxilary function for high-dimensional experiments
def run_experiment_hd(args):
    """"""
    # Set up reproducibility
    print(f'\n Setting up reproducibility with seed: {args.seed} ... \n')
    setup_reproducibility_hd(seed=args.seed)

    # Set up GPU
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else "cpu")
    
    # Initialize potential energy function as an MLP
    V_learned = MLP(args.V_in_dims, args.V_hidden_dims, args.V_out_dims)

    # Initialize angular estimator
    omega_estimator = MLP(args.V_in_dims * args.tau, args.V_hidden_dims, 3)

    # Initialize integrator
    lgvi_integrator = LieGroupVaritationalIntegrator()

    # Initialize encoder and decoder
    encoder = EncoderRBNN_gravity()
    decoder = DecoderRBNN_gravity()

    # Initialize model and optimizer
    model = rbnn_gravity_hd(encoder=encoder,
                        decoder=decoder,
                        estimator=omega_estimator,
                        integrator=lgvi_integrator,
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
    train_dataloader = build_dataloader_hd(args=args)

    # Loss function
    loss = high_dim_loss

    # Retrain model
    if args.retrain_model:
        args.save_dir = args.save_dir[:-4] + "-retrain.pth"
        print(f'\n New save directory for re-trained model: {args.save_dir} ... \n')

    # Train model
    print(f'\n Training model on device ({device}) ... \n')
    t0 = time.time()
    train_loss, optim = train_hd(args=args, model=model, traindataloader=train_dataloader, loss_fcn=loss)
    
    train_time = time.time() - t0
    print(f'\n Training time: {train_time} \n')

    # Save model
    if args.save_model:
        print(f'\n Saving experiment {args.exp_name} from date {args.date} ... \n')
        save_experiment(args=args, model=model, optimizer=optim, loss=train_loss) 

# General auxilary functions
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
            'moi_diag': model.I_diag,
            'moi_off_diag': model.I_off_diag,
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

def high_dim_loss(x_gt: torch.Tensor,
                  x_dyn: torch.Tensor,
                  x_enc: torch.Tensor,
                  R_dyn: torch.Tensor,
                  R_enc: torch.Tensor,
                  omega_dyn: torch.Tensor,
                  omega_enc: torch.Tensor,
                  tau: int = 2,
                  lambda_loss: torch.Tensor = torch.ones(2)):
    """"""
    R_loss = R_loss_fcn(R_gt=R_enc[:, 1:, ...], R_recon=R_dyn[:, 1:, ...])
    omega_loss = omega_loss_fcn(omega_gt=omega_enc[:, 1:, ...], omega_recon=omega_dyn[:, 1:-tau, ...])
    ae_loss = ae_recon_loss(x_gt=x_gt, x_enc=x_enc)
    dyn_loss = dyn_recon_loss(x_gt=x_gt, x_dyn=x_dyn)

    total_loss = lambda_loss[0] * R_loss + lambda_loss[1] * omega_loss + lambda_loss[2] * ae_loss + lambda_loss[3] * dyn_loss
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

def ae_recon_loss(x_gt: torch.Tensor, x_enc: torch.Tensor):
    """"""
    loss_ = torch.nn.MSELoss()
    loss = loss_(x_gt, x_enc)
    return loss

def dyn_recon_loss(x_gt: torch.Tensor, x_dyn: torch.Tensor):
    """"""
    loss_ = torch.nn.MSELoss()
    loss = loss_(x_gt[:, 1:, ...], x_dyn[:, 1:, ...])
    return loss

