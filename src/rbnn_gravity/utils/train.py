# Author(s): Justice Mason
# Project: RBNN + Gravity
# Date: 11/10/23

import sys, os

import numpy as np
import torch
import time
import glob

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
    epoch_loss = {'train total': [], 'train R': [], 'train omega': []}
    
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
        loss, R_loss, omega_loss = loss_fcn(R_gt=data_R[:, tau:, ...], R_recon=data_R_recon[:, tau:, ...], omega_gt=data_omega[:, tau:, ...], omega_recon=data_omega_recon[:, tau:, ...], lambda_loss=args.lambda_loss)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward() # retain_graph=True)
        optimizer.step()

        # Append loss
        if idx % args.print_every == 0:
            batch_loss_total = loss.item()
            batch_R_loss = R_loss.item()
            batch_omega_loss = omega_loss.item()

            epoch_loss['train total'].append(batch_loss_total)
            epoch_loss['train R'].append(batch_R_loss)
            epoch_loss['train omega'].append(batch_omega_loss)
    
    return epoch_loss

def eval_epoch(args, model, dataloader, loss_fcn):
    """
    Evaluate model for one epoch without backpropagation.
    ...

    """
    # Set model to eval mode
    model.eval()
    epoch_loss = {'val total': [], 'val R': [], 'val omega': []}

    # Iterate over elements of the dataloader and train
    for idx, data in enumerate(dataloader):
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
        loss, R_loss, omega_loss = loss_fcn(R_gt=data_R, R_recon=data_R_recon, omega_gt=data_omega, omega_recon=data_omega_recon, lambda_loss=args.lambda_loss)

        # Append loss
        if idx % args.print_every == 0:
            batch_loss_total = loss.item()
            batch_R_loss = R_loss.item()
            batch_omega_loss = omega_loss.item()

            epoch_loss['val total'].append(batch_loss_total)
            epoch_loss['val R'].append(batch_R_loss)
            epoch_loss['val omega'].append(batch_omega_loss)

    return epoch_loss

def train(args, model, traindataloader, valdataloader, loss_fcn, early_stopper, start_epoch: int = 0, stats: dict = None):
    """
    """
    # Initialize training loss
    # Log stats in dictionary
    if not stats:
        stats = {'train total': [], 'train R': [], 'train omega': [],\
             'val total': [], 'val R': [], 'val omega': []}

    # Initialize optimizer -- NEEDS TO CHANGE IF WE ARE CONTINUING TRAINING
    params = model.parameters()
    optim = torch.optim.Adam(params=params, lr=args.lr)
    
    # To Do: Add a lr scheduler
    # Iterate through epochs
    for epoch in range(start_epoch, args.n_epochs):
        # Run train_epoch function
        epoch_loss_tr = train_epoch(args=args, model=model, dataloader=traindataloader, optimizer=optim, loss_fcn=loss_fcn)

        # Concat epoch loss onto training loss
        stats['train total'] += epoch_loss_tr['train total']
        stats['train R'] += epoch_loss_tr['train R']
        stats['train omega'] += epoch_loss_tr['train omega']

        if epoch % args.print_every == 0:
            # Log validation losses
            epoch_loss_val = eval_epoch(args=args, model=model, dataloader=valdataloader, loss_fcn=loss_fcn)
            stats['val total'] += epoch_loss_val['val total']
            stats['val R'] += epoch_loss_val['val R']
            stats['val omega'] += epoch_loss_val['val omega']

            print(f'\n Epoch Number: {epoch + 1} ; Training Loss: {stats["train total"][-1]:.4e} ; Validation Loss: {stats["val total"][-1]:.4e} \n')

            # Early stopper criterion
            if early_stopper.early_stop(stats['val total'][-1]):             
                break
    
    return stats, optim
        
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
    epoch_loss = {'train total': [], 'train recon': [], 'train dyn': [], 'train R': [], 'train omega': []}

    # Iterate over elements of the dataloader and train
    for idx, data in enumerate(dataloader):
        # Extract data
        data_x = data
    
        # Check requires_grad -- for autograd/backprop
        if not data_x.requires_grad:
            data_x.requires_grad = True
        
        # Load data onto device
        data_x = data_x.to(model.device).float()

        # Data should be dtype float
        # data_x.type(torch.float)

        # Forward pass and calculate loss
        data_x_dyn, data_x_recon, R_dyn, omega_dyn, R_enc, omega_enc  = model(x=data_x, seq_len=args.seq_len)
        loss, R_loss, omega_loss, recon_loss, dyn_loss = loss_fcn(x_gt=data_x,\
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
            batch_loss_total = loss.item()
            batch_loss_R = R_loss.item()
            batch_loss_omega = omega_loss.item()
            batch_loss_recon = recon_loss.item()
            batch_loss_dyn = dyn_loss.item()

            epoch_loss['train total'].append(batch_loss_total)
            epoch_loss['train R'].append(batch_loss_R)
            epoch_loss['train omega'].append(batch_loss_omega)
            epoch_loss['train recon'].append(batch_loss_recon)
            epoch_loss['train dyn'].append(batch_loss_dyn)
    
    return epoch_loss

def eval_epoch_hd(args, model, dataloader, loss_fcn):
    """"""
    model.eval()
    epoch_loss = {'val total': [], 'val recon': [], 'val dyn': [], 'val R': [], 'val omega': []}

    # Iterate over elements of the dataloader and train
    for idx, data in enumerate(dataloader):
        # Extract data
        data_x = data
    
        # Check requires_grad -- for autograd/backprop
        if not data_x.requires_grad:
            data_x.requires_grad = True
        
        # Load data onto device
        data_x = data_x.to(model.device).float()

        # Data should be dtype float
        # data_x.type(torch.float)

        # Forward pass and calculate loss
        data_x_dyn, data_x_recon, R_dyn, omega_dyn, R_enc, omega_enc  = model(x=data_x, seq_len=args.seq_len)
        loss, R_loss, omega_loss, recon_loss, dyn_loss  = loss_fcn(x_gt=data_x,\
                                                    x_dyn=data_x_dyn,\
                                                    x_enc=data_x_recon,\
                                                    R_dyn=R_dyn,\
                                                    R_enc=R_enc,\
                                                    omega_dyn=omega_dyn,\
                                                    omega_enc=omega_enc,\
                                                    tau=model.tau,\
                                                    lambda_loss=args.lambda_loss)
        
        # Append loss
        if idx % args.print_every == 0:
            batch_loss_total = loss.item()
            batch_loss_R = R_loss.item()
            batch_loss_omega = omega_loss.item()
            batch_loss_recon = recon_loss.item()
            batch_loss_dyn = dyn_loss.item()

            epoch_loss['val total'].append(batch_loss_total)
            epoch_loss['val R'].append(batch_loss_R)
            epoch_loss['val omega'].append(batch_loss_omega)
            epoch_loss['val recon'].append(batch_loss_recon)
            epoch_loss['val dyn'].append(batch_loss_dyn)
    
    return epoch_loss

def train_hd(args, model, traindataloader, valdataloader, loss_fcn, early_stopper, start_epoch: int = 0, stats: dict = None):
    """"""
    # Log stats in dictionary
    if not stats:
        stats = {'train total': [], 'train recon': [], 'train dyn': [], 'train R': [], 'train omega': [],\
             'val total': [], 'val recon': [], 'val dyn': [], 'val R': [], 'val omega': []}

    # Initialize optimizer -- NEEDS TO CHANGE IF WE ARE CONTINUING TRAINING
    params = model.parameters()
    optim = torch.optim.Adam(params=params, lr=args.lr)
    
    # To Do: Add a lr scheduler
    # Iterate through epochs
    for epoch in range(start_epoch, args.n_epochs):
        # Run train_epoch function
        epoch_loss_tr = train_epoch_hd(args=args, model=model, dataloader=traindataloader, optimizer=optim, loss_fcn=loss_fcn)

        # Concat epoch loss onto training loss
        stats['train total'] += epoch_loss_tr['train total']
        stats['train recon'] += epoch_loss_tr['train recon']
        stats['train dyn'] += epoch_loss_tr['train dyn']
        stats['train R'] += epoch_loss_tr['train R']
        stats['train omega'] += epoch_loss_tr['train omega']

        if epoch % args.print_every == 0:
            # Log validation losses 
            epoch_loss_val = eval_epoch_hd(args=args, model=model, dataloader=valdataloader, loss_fcn=loss_fcn)
            stats['val total'] += epoch_loss_val['val total']
            stats['val recon'] += epoch_loss_val['val recon']
            stats['val dyn'] += epoch_loss_val['val dyn']
            stats['val R'] += epoch_loss_val['val R']
            stats['val omega'] += epoch_loss_val['val omega']

            print(f'\n Epoch Number: {epoch + 1} ; Training Loss: {stats["train total"][-1]:.4e} ; Validation Loss: {stats["val total"][-1]:.4e} \n')

        # Save checkpoint
        if epoch % args.save_every == 1:
            save_checkpoint(model=model, stats=stats, optimizer=optim, epoch=epoch, loss=stats['val total'], path=args.checkpoint_dir+f'/{args.date}/')
        
        # Early stopping
        if epoch % args.print_every == 0:   
            # Early stopper criterion
            if early_stopper.early_stop(stats['val total'][-1]):             
                break
    
    return stats, optim

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

    # Early stopper
    early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)

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
    train_dataloader, test_dataloader, val_dataloader = build_dataloader_hd(args=args)

    # Loss function
    loss = high_dim_loss

    # Retrain model
    if args.retrain_model:
        args.save_dir = args.save_dir[:-4] + "-retrain.pth"
        print(f'\n New save directory for re-trained model: {args.save_dir} ... \n')

    # Train model
    print(f'\n Training model on device ({device}) ... \n')
    t0 = time.time()
    train_loss, optim = train_hd(args=args, model=model, traindataloader=train_dataloader, valdataloader=val_dataloader, loss_fcn=loss, early_stopper=early_stopper)
    
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
    return total_loss, R_loss, omega_loss

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
    return total_loss, R_loss, omega_loss, ae_loss, dyn_loss

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

# Auxiliary Functions -- Checkpointing
def latest_checkpoint(model,
                      optimizer,
                      chkptdir):
    """
    Function to load the latest checkpoint of model and optimizer in given checkpoint directory.
    
    ...
    
    Parameters
    ----------
    model : torch.nn.Module
        Untrained model
    
    optimizer : torch.nn.optim
        Optimizer
    
    checkpointdir : str
        Checkpoint directory
        
    Returns
    -------
    model : torch.nn.Module
        Untrained model
    
    optimizer : torch.nn.optim
        Optimizer
    
    checkpointdir : str
        Checkpoint directory
    
    stats : list
        List of statistics for analysis.
        
    start_epoch : int
        Starting epoch for training.

    Notes
    -----
    
    """
    if not os.path.exists(chkptdir):
        print('\n Log directory does not exist. Making it ... \n')
        os.makedirs(chkptdir)
        
    filenames = glob(chkptdir + '*.pth')
    
    if not filenames:
        latest = 'not-found'
    else:
        latest = sorted(filenames)[-1]
    
    model, optimizer, stats, start_epoch = load_checkpoint(model, optimizer, latest)
    
    return model, optimizer, stats, start_epoch

def load_checkpoint(model,
                    optimizer,
                    filename):
    """
    Function to load checkpoints.
    
    ...
    
    Parameters
    ----------
    model
        Neural network work model used for training.
        
    optimizer
        Optimizer used during training.
        
    filename : str
        String that contains the filename of the checkpoint.
        
    Returns
    -------
    model
        Neural network work model used for training with updated states.
        
    optimizer
         Optimizer used during training with updated states.
         
    start_epoch : int
        Starting epoch for training.
        
    Notes
    -----
    
    """
    start_epoch = 0
    stats = []
    
    if os.path.isfile(filename):
        print(f"\n Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location='cpu')
        
        start_epoch = checkpoint['epoch']
        print(f'\n Starting at epoch {start_epoch} ... \n')
        
        stats = checkpoint['stats']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.I_diag = checkpoint['moi_diag']
        model.I_off_diag = checkpoint['moi_off_diag']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(model.device)
        print("\n Loaded checkpoint '{}' (epoch {}) ... \n".format(filename, checkpoint['epoch']))
    else:
        print("\n No checkpoint found at '{}' ... \n".format(filename))
    
    return model, optimizer, stats, start_epoch

def save_checkpoint(model,
                    stats: list,
                    optimizer,
                    epoch: int ,
                    loss: float,
                    path: str):
    """
    Saves checkpoint after every number of epochs.
    
    ...
    
    Parameters
    ----------
    model
    
    stats : list
    
    optimizer
    
    epoch : int
        Epoch of the checkpoint.
        
    loss : float
        Loss value for checkpoint.
        
    path : str
        Directory for saving checkpoints.
        
    Returns
    -------
    
    Notes
    -----
    
    """
    if not os.path.exists(path):
        print(f'\n Directory not found at "{path}"; creating directory ... \n')
        os.makedirs(path)
        
    ckpt_path = path + f'/checkpoint-{epoch:06}.pth'
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'moi_diag': model.I_diag,
            'moi_off_diag': model.I_off_diag,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'stats': stats,
            }
    torch.save(checkpoint, ckpt_path)

# Auxiliary Functions -- Early Stopping
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

