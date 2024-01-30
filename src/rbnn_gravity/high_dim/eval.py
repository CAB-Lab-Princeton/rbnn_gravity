# Author(s): Justice Mason
# Project: RBNN - Gravity
# Date: 12/07/23

import sys, os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

# Append parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models import rbnn_gravity_hd, build_V_gravity, MLP
from autoencoder import EncoderRBNN_gravity, DecoderRBNN_gravity
from utils.integrators import LieGroupVaritationalIntegrator
from utils.general import setup_reproducibility_hd, gen_gif
from utils.math_utils import pd_matrix, mean_confidence_interval
from utils.train import latest_checkpoint, load_checkpoint
from data.dataset import shuffle_and_split

# Define the arguments
def get_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_type",
        choices=['ucube', 'nucube', 'uprism', 'nuprism', 'calipso', 'cloudsat', '3dpend', 'gyroscope'],
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
        "--select_checkpoint",
        type=int,
        default=None,
        help="selected checkpoint"
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
        "--n_epoch",
        type=int,
        default=100,
        help="Set number of epochs for training",
        required=True
    )
    
    args = parser.parse_args()
    return args

# Auxiliary functions for loading trained experiments
def load_experiment(args, model, optimizer):
    """"""
    # Pull out needed args
    experiment_name = args.experiment_type
    date = args.date
    select_cp = args.select_checkpoint

    # Create checkpoint directory
    checkpoint_dir  = f'src/rbnn_gravity/high_dim/checkpoints/{experiment_name}/{date}/'

    # Check for selected checkpoint 
    if select_cp:
        filepath = checkpoint_dir + f'checkpoint-{select_cp:06}.pth'
        model, _, stats, _ = load_checkpoint(model=model, optimizer=optimizer, filename=filepath)

    else:
        model, _, stats, _ = latest_checkpoint(model=model, optimizer=optimizer, chkptdir=checkpoint_dir)
    
    return model, stats

def get_learned_moi(model):
    """"""
    print(f'\n The learned moment-of-inertia is {model.calc_moi()} \n')

def generate_loss_plots(args, stats: torch.Tensor):
    """"""
    # Name of experiment
    experiment = args.experiment_type
    date = args.date
    n_epoch = args.n_epoch

    # Define the parent directory
    save_dir = f'src/rbnn_gravity/high_dim/results/{experiment}/{date}/' 
    os.makedirs(save_dir, exist_ok=True)

    # Unpack losses
    tr_loss_total = stats['train total']
    tr_loss_R = stats['train R']
    tr_loss_omega = stats['train omega']
    tr_loss_dyn = stats['train dyn']
    tr_loss_recon = stats['train recon']

    val_loss_total = stats['val total']
    val_loss_R = stats['val R']
    val_loss_omega = stats['val omega']
    val_loss_dyn = stats['val dyn']
    val_loss_recon = stats['val recon']

    # Quick maths
    n_epochs = n_epoch
    tv_ratio  = int(len(tr_loss_total)/len(val_loss_total))
    n_iterations = len(tr_loss_total)
    n_batches = int(n_iterations/n_epochs)

    # Generate semilogy plot of training loss
    fig_total_loss = plt.figure()

    ax = fig_total_loss.add_subplot(1, 1, 1)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # Generate total loss curve for validation and training
    ax.semilogy(tr_loss_total[::n_batches*tv_ratio])
    ax.semilogy(val_loss_total[::n_batches])
    ax.grid()
    ax.legend(['Train', 'Validation'])
    ax.set_title('RBNN + Gravity Loss Curves')
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Loss Value')

    # Generatre loss plots for sublosses
    # Dyn
    axes[0, 0].semilogy(tr_loss_dyn[::n_batches*tv_ratio])
    axes[0, 0].semilogy(val_loss_dyn[::n_batches])
    axes[0, 0].grid()
    axes[0, 0].legend(['Train', 'Validation'])
    axes[0, 0].set_xlabel('Epoch Number')
    axes[0, 0].set_ylabel('Loss Value')
    axes[0, 0].set_title('Dynamics-based Recon')

    # Enc
    axes[0, 1].semilogy(tr_loss_recon[::n_batches*tv_ratio])
    axes[0, 1].semilogy(val_loss_recon[::n_batches])
    axes[0, 1].grid()
    axes[0, 1].legend(['Train', 'Validation'])
    axes[0, 1].set_xlabel('Epoch Number')
    axes[0, 1].set_ylabel('Loss Value')
    axes[0, 1].set_title('AE-based Recon')

    # Latent R
    axes[1, 0].semilogy(tr_loss_R[::n_batches*tv_ratio])
    axes[1, 0].semilogy(val_loss_R[::n_batches])
    axes[1, 0].grid()
    axes[1, 0].legend(['Train', 'Validation'])
    axes[1, 0].set_xlabel('Epoch Number')
    axes[1, 0].set_ylabel('Loss Value')
    axes[1, 0].set_title('Latent $R$')

    # Latent Omega
    axes[1, 1].semilogy(tr_loss_omega[::n_batches*tv_ratio])
    axes[1, 1].semilogy(val_loss_omega[::n_batches])
    axes[1, 1].grid()
    axes[1, 1].legend(['Train', 'Validation'])
    axes[1, 1].set_xlabel('Epoch Number')
    axes[1, 1].set_ylabel('Loss Value')
    axes[1, 1].set_title('Latent $\Omega$')
    
    fig_total_loss.savefig(save_dir + 'total_loss.pdf')
    plt.savefig(save_dir + 'losses_plot.pdf')

def gen_recon_fig(x_gt, x_pred, img_interval: int = 5):
    """"""
    fig, axes = plt.subplots(nrows=2, ncols=int(x_gt.shape[1]/img_interval), figsize=(80, 8), subplot_kw={'xticks': [], 'yticks': []})
    n = 0 
    fontsize = 42
    
    axes[0, 0].text(-0.75, 0.5, 'True', fontsize=fontsize, horizontalalignment='center', verticalalignment='center', transform=axes[0, 0].transAxes)
    axes[1, 0].text(-0.75, 0.5, 'Predict', fontsize=fontsize, horizontalalignment='center', verticalalignment='center', transform=axes[1, 0].transAxes)
    
    for i in range(int(x_gt.shape[1]/img_interval)):
        axes[0, i].imshow(x_gt[n, img_interval*i,  :, :, :].transpose(1, 2, 0))
        axes[1, i].imshow(x_pred[n, img_interval*i,  :, :, :].transpose(1, 2, 0))
        axes[1, i].text(14.0, 35.0, r"$\tau$ = {}".format(i * img_interval), ha="center", fontsize=fontsize)
        
    return fig, axes

def reconstruction_eval(args, model):
    """"""
    # Pull out needed args
    data_dir = args.data_dir
    date = args.date

    dataset = args.experiment_type
    select_cp = args.select_checkpoint

    # Load and split data
    raw_data = np.load(data_dir, allow_pickle=True)
    rd_split = shuffle_and_split(raw_data=raw_data, test_split=0.2, val_split=0.1)
    traj_len = raw_data.shape[1]

    if traj_len > 100:
        train_dataset = rd_split[0][:, :100, ...]
        val_dataset = rd_split[1][:, :100, ...]
        test_dataset = rd_split[2][:, :100, ...]
    else:
        train_dataset = rd_split[0]
        val_dataset = rd_split[1]
        test_dataset = rd_split[2] 

    # Generate random sample for evaluation
    x_train = torch.tensor(train_dataset[:int(0.3*train_dataset.shape[0]), ...], device=model.device)
    x_test = torch.tensor(test_dataset, device=model.device)
    x_val = torch.tensor(val_dataset, device=model.device)

    # Evaluate models for train, test, and val
    x_train_dyn, x_train_recon, _, _, _, _ = model(x=x_train.float(), seq_len=x_train.shape[1])
    x_test_dyn, x_test_recon, _, _, _, _ = model(x=x_test.float(), seq_len=x_test.shape[1])
    x_val_dyn, x_val_recon, _, _, _, _ = model(x=x_val.float(), seq_len=x_val.shape[1])

    # Calculate confidence intervals
    loss = torch.nn.MSELoss(reduction='none')

    train_mse = torch.mean(loss(x_train, x_train_dyn), dim=(1, 2, 3, 4))
    tr_idx = torch.argmin(train_mse)

    test_mse = torch.mean(loss(x_test, x_test_dyn), dim=(1, 2, 3, 4))
    ts_idx = torch.argmin(test_mse)

    val_mse = torch.mean(loss(x_val, x_val_dyn), dim=(1, 2, 3, 4))
    val_idx = torch.argmin(val_mse)
    
    # mean_confidence_interval()    

    # import pdb; pdb.set_trace()

    fig_tr, _ = gen_recon_fig(x_gt=x_train[tr_idx, ...][None, ...].detach().cpu().numpy(), x_pred=x_train_dyn[tr_idx, ...][None, ...].detach().cpu().numpy())
    fig_ts, _ = gen_recon_fig(x_gt=x_test[ts_idx, ...][None, ...].detach().cpu().numpy(), x_pred=x_test_dyn[ts_idx, ...][None, ...].detach().cpu().numpy())
    fig_val, _ = gen_recon_fig(x_gt=x_val[val_idx, ...][None, ...].detach().cpu().numpy(), x_pred=x_val_dyn[val_idx, ...][None, ...].detach().cpu().numpy())

    # Generate the GIFs

    # Save figures
    SAVE_DIR = f'src/rbnn_gravity/high_dim/results/{dataset}/{date}/'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Ground-truth

    gen_gif(save_dir=SAVE_DIR + f"train_dyn_gt.gif", img_array=x_train[tr_idx, ...][None, ...].detach().cpu().numpy())
    gen_gif(save_dir=SAVE_DIR + f"test_dyn_gt.gif", img_array=x_test[ts_idx, ...][None, ...].detach().cpu().numpy())
    gen_gif(save_dir=SAVE_DIR + f"val_dyn_gt.gif", img_array=x_val[val_idx, ...][None, ...].detach().cpu().numpy())
    
    if select_cp:
        gen_gif(save_dir=SAVE_DIR + f"train_dyn_recon_{select_cp:06}.gif", img_array=x_train_dyn[tr_idx, ...][None, ...].detach().cpu().numpy())
        gen_gif(save_dir=SAVE_DIR + f"test_dyn_recon_{select_cp:06}.gif", img_array=x_test_dyn[ts_idx, ...][None, ...].detach().cpu().numpy())
        gen_gif(save_dir=SAVE_DIR + f"val_dyn_recon_{select_cp:06}.gif", img_array=x_val_dyn[val_idx, ...][None, ...].detach().cpu().numpy())
        

    else:
        gen_gif(save_dir=SAVE_DIR + f"train_dyn_recon_end.gif", img_array=x_train_dyn[tr_idx, ...][None, ...].detach().cpu().numpy())
        gen_gif(save_dir=SAVE_DIR + f"test_dyn_recon_end.gif", img_array=x_test_dyn[ts_idx, ...][None, ...].detach().cpu().numpy())
        gen_gif(save_dir=SAVE_DIR + f"val_dyn_recon_end.gif", img_array=x_val_dyn[val_idx, ...][None, ...].detach().cpu().numpy())

    if select_cp:
        fig_tr.savefig(SAVE_DIR + f"train_dyn_recon_{select_cp:06}.pdf", format="pdf", bbox_inches="tight")
        fig_ts.savefig(SAVE_DIR + f"test_dyn_recon_{select_cp:06}.pdf", format="pdf", bbox_inches="tight")
        fig_val.savefig(SAVE_DIR + f"val_dyn_recon_{select_cp:06}.pdf", format="pdf", bbox_inches="tight")
    else:
        fig_tr.savefig(SAVE_DIR + f"train_dyn_recon_end.pdf", format="pdf", bbox_inches="tight")
        fig_ts.savefig(SAVE_DIR + f"test_dyn_recon_end.pdf", format="pdf", bbox_inches="tight")
        fig_val.savefig(SAVE_DIR + f"val_dyn_recon_end.pdf", format="pdf", bbox_inches="tight")

def potential_energy_eval(args, model):
    """
    Evaluation function for learned potential energy.

    ...
    """
    # Load files
    data_dir = args.data_dir
    raw_data = np.load(data_dir)

    # Pull out needed args
    dataset = args.experiment_type
    date = args.date
    select_cp = args.select_checkpoint

    SAVE_DIR = f'src/rbnn_gravity/high_dim/results/{dataset}/{date}/'
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    rd_split = shuffle_and_split(raw_data=raw_data, test_split=0.2, val_split=0.1)
    data_test = rd_split[2]

    # Choose random test sample
    n_rand = np.random.randint(low=0, high=data_test.shape[0])
    x = data_test[n_rand, ...]

    x_tensor = torch.tensor(x, requires_grad=True).unsqueeze(0).to(model.device)

    # Run model on x_tensor
    _, _, R_dyn, omega_dyn, R_enc, omega_enc = model(x=x_tensor.float(), seq_len=x_tensor.shape[1])

    # Ground-truth Potential Energy
    mass = 1.
    g = 9.81 # [m/s2]
    e_3 = torch.tensor([[0., 0., 1.]], device=model.device).T
    rho_gt = torch.tensor([[0., 0., 1.]], device=model.device)

    V_fcn = lambda R: build_V_gravity(m=mass, g=g, e_3=e_3, R=R, rho_gt=rho_gt)
    
    # Autoencoder-based trajectory
    V_gt_ae = V_fcn(R=R_enc.squeeze())
    V_gt_ae_ = V_gt_ae - V_gt_ae[0]

    V_lr_ae = model.V(R_enc.reshape(-1, 9)).squeeze()
    V_lr_ae_ = V_lr_ae - V_lr_ae[0]

    # Dynamics-based trajectory
    V_gt_dyn = V_fcn(R=R_dyn.squeeze())
    V_gt_dyn_ = V_gt_dyn - V_gt_dyn[0]

    V_lr_dyn = model.V(R_dyn.reshape(-1, 9)).squeeze()
    V_lr_dyn_ = V_lr_dyn - V_lr_dyn[0]

    # Embedding-based kinetic energy
    moi_lr = model.calc_moi()
    T_lr_ae = 0.5 * torch.einsum('btj, jk, btk -> bt', omega_enc, moi_lr, omega_enc).squeeze()
    T_lr_ae_ = T_lr_ae - T_lr_ae[0]

    # Dynamics-based prediction kinetic energy
    T_lr_dyn = 0.5 * torch.einsum('btj, jk, btk -> bt', omega_dyn, moi_lr, omega_dyn).squeeze()
    T_lr_dyn_ = T_lr_dyn - T_lr_dyn[0]

    # Total energy
    E_lr_ae = T_lr_ae_ + V_lr_ae_[:-model.tau]
    E_lr_dyn = T_lr_dyn_ + V_lr_dyn_

    # Generate figure
    fig_v, axes_v = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))
    # axes_v[0].plot(V_gt_ae_.detach().cpu().numpy(), 'k-')

    axes_v[0, 0].plot(V_lr_ae_.detach().cpu().numpy(), 'b-')
    axes_v[0, 0].grid()
    axes_v[0, 0].legend(['Learned']) #['GT', 'Learned'])
    axes_v[0, 0].set_title('Encoding-based V(R) - V($R_0$)')

    # axes_v[1].plot(V_gt_dyn_.detach().cpu().numpy(), 'k-')
    axes_v[0, 1].plot(V_lr_dyn_.detach().cpu().numpy(), 'b-')
    axes_v[0, 1].grid()
    axes_v[0, 1].legend(['Learned']) #['GT', 'Learned'])
    axes_v[0, 1].set_title('Dynamics-based V(R) - V($R_0$)')

    axes_v[1, 0].plot(T_lr_ae_.detach().cpu().numpy(), 'b-')
    axes_v[1, 0].grid()
    axes_v[1, 0].legend(['Learned']) #['GT', 'Learned'])
    axes_v[1, 0].set_title('Encoding-based T($\Omega$) - T($\Omega_0$)')

    # axes_v[1].plot(V_gt_dyn_.detach().cpu().numpy(), 'k-')
    axes_v[1, 1].plot(T_lr_dyn_.detach().cpu().numpy(), 'b-')
    axes_v[1, 1].grid()
    axes_v[1, 1].legend(['Learned']) #['GT', 'Learned'])
    axes_v[1, 1].set_title('Dynamics-based T($\Omega$) - T($\Omega_0$)')

    axes_v[2, 0].plot(E_lr_ae.detach().cpu().numpy(), 'b-')
    axes_v[2, 0].grid()
    axes_v[2, 0].legend(['Learned']) #['GT', 'Learned'])
    axes_v[2, 0].set_title('Encoding-based E(R, $\Omega$) - E($R_0$, $\Omega_0$)')

    # axes_v[1].plot(V_gt_dyn_.detach().cpu().numpy(), 'k-')
    axes_v[2, 1].plot(E_lr_dyn.detach().cpu().numpy(), 'b-')
    axes_v[2, 1].grid()
    axes_v[2, 1].legend(['Learned']) #['GT', 'Learned'])
    axes_v[2, 1].set_title('Dynamics-based E(R, $\Omega$) - E($R_0$, $\Omega_0$)')
    
    if select_cp:
        plt.savefig(SAVE_DIR + f'energy_functions_{select_cp:06}.pdf')
    else:
        plt.savefig(SAVE_DIR + f'energy_functions_end.pdf')

if __name__ == "__main__":
    # Load args
    args = get_args()

    # Setup reporducibility
    setup_reproducibility_hd(seed=args.seed, cnn_reprod=True)

    # Set up GPU
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else "cpu")

    # Initialize integrator
    lgvi_integrator = LieGroupVaritationalIntegrator()

    # Initialize encoder and decoder
    encoder = EncoderRBNN_gravity()
    decoder = DecoderRBNN_gravity()

    # Initialize model and optimizer
    model = rbnn_gravity_hd(encoder=encoder,
                        decoder=decoder,
                        estimator=None,
                        integrator=lgvi_integrator,
                        in_dim=9,
                        hidden_dim=50,
                        out_dim=1,
                        tau=2, 
                        dt=1e-3, 
                        I_diag=None, 
                        I_off_diag=None, 
                        V=None)
    
    # Load optimizer
    optim = torch.optim.Adam(params=model.parameters())

    # Load trained model
    model_tr, stats = load_experiment(args=args, model=model, optimizer=optim)
    model_tr.to(device)
    model_tr.device = device
    
    model.eval()
    
    # Diagonalize MOI
    e, v = torch.linalg.eig(model_tr.calc_moi())

    # Learned MOI
    print(f'\n Learned MOI: {model_tr.calc_moi()} and scale: {torch.linalg.norm(model_tr.calc_moi())}\n')

    # Plot losses
    print('\n Generating Loss Plots \n')
    generate_loss_plots(args=args, stats=stats)
    
    # Plot potential 
    print('\n Generating Energy Plots \n')
    potential_energy_eval(args=args, model=model_tr)

    # Reconstruction plots
    print('\n Generating Dynamics-based Reconstruction Plot \n')
    reconstruction_eval(args=args, model=model_tr)






