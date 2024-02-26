# Author(s): Justice Mason
# Project: RBNN - Appeareance
# Date: 02/08/24

import sys, os
import argparse
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

# Append parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models import rbnn_gravity_content, build_V_gravity, MLP
from autoencoder import EncoderRBNN_content, DecoderRBNN_content
from utils.integrators import LieGroupVaritationalIntegrator
from utils.general import setup_reproducibility_hd, gen_gif
from utils.math_utils import pd_matrix, mean_confidence_interval, group_matrix_to_quaternions
from utils.train import latest_checkpoint, load_checkpoint
from data.dataset import shuffle_and_split

from high_dim.eval import gen_recon_fig

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
    checkpoint_dir  = f'src/rbnn_gravity/content_info/checkpoints/{experiment_name}/{date}/'

    # Check for selected checkpoint 
    if select_cp:
        filepath = checkpoint_dir + f'checkpoint-{select_cp:06}.pth'
        model, _, stats, _ = load_checkpoint(model=model, optimizer=optimizer, filename=filepath)

    else:
        print('here')
        model, _, stats, _ = latest_checkpoint(model=model, optimizer=optimizer, chkptdir=checkpoint_dir)
    
    return model, stats

# Generate Datasets
def gen_datasets_eval(args):
    """"""
    data_dir = args.data_dir
    date = args.date

    dataset = args.experiment_type
    select_cp = args.select_checkpoint

    # Load and split data
    raw_data_original = np.load(data_dir, allow_pickle=True)
    rd_split = shuffle_and_split(raw_data=raw_data_original, test_split=0.2, val_split=0.1)
    traj_len = raw_data_original.shape[1]

    if traj_len > 100:
        train_dataset = rd_split[0][:, :100, ...]
        val_dataset = rd_split[1][:, :100, ...]
        test_dataset = rd_split[2][:, :100, ...]
    else:
        train_dataset = rd_split[0]
        val_dataset = rd_split[1]
        test_dataset = rd_split[2] 

    # Generate random sample for evaluation
    x_train = torch.tensor(train_dataset, device=model.device)
    x_test = torch.tensor(test_dataset, device=model.device)
    x_val = torch.tensor(val_dataset, device=model.device)

    return x_train, x_test, x_val

# Visualize the latent space using quaternions
def latent_space_viz(args, model, x_test):
    """"""
    # Select a batch of images
    n_batch = torch.randperm(x_test.shape[0])[:10]

    # load args
    dataset = args.experiment_type
    date = datetime.today().strftime('%m%d%Y')

    # load all test sets for comparison
    x_eval = x_test[n_batch, ...]
    _, _, R_dyn, omega_dyn, R_enc, omega_enc = model(x_eval.float(), seq_len=x_eval.shape[1])

    q_dyn = group_matrix_to_quaternions(R_dyn)
    q_enc = group_matrix_to_quaternions(R_enc)

    import pdb; pdb.set_trace()

    # Generate semilogy plot of training loss
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # run model 
    
    
# Content vector swamp in same dataset

# Evaluation functions
def content_dynamics_swap(args, model, x_test):
    # Load trained model
    model_trained = model
    dataset = args.name
    date = args.date

    # Load all datasets for comparison
    x_eval = x_test[:2, ...]

    print('\n Loading Data ... \n')
    # CLOUDSAT DATA
    cloudsat_data = np.load('../data/devs_datasets_long/cloudsat/data.npy', allow_pickle=True)
    cloudsat_example = torch.tensor(cloudsat_data[0, :100, ...][None, ...], device=model_trained.device)

    # UCUUBE
    ucube_data = np.load('../data/devs_datasets_stand/uniform_cube/2022-05-15_uniform_cube_1k/uniform_cube_1k.npy', allow_pickle=True)
    ucube_example = torch.tensor(ucube_data[0, ...][None, ...], device=model_trained.device)

    # UPRISM
    uprism_data = np.load('../data/devs_datasets_stand/uniform_prism/2022-05-13_uniform_prism_FIXED/uniform_prism_1k.npy', allow_pickle=True)
    uprism_example = torch.tensor(uprism_data[0, ...][None, ...], device=model_trained.device)

    # NUPRISM
    nuprism_data = np.load('../data/devs_datasets_stand/nonuniform_prism/2022-05-13_non_uniform_prism_FIXED/non_uniform_prism_1k.npy', allow_pickle=True)
    nuprism_example = torch.tensor(nuprism_data[0, ...][None, ...], device=model_trained.device)

    # NUCUBE
    nucube_data = np.load('../data/devs_datasets_stand/nonuniform_cube/2022-05-15_non_uniform_cube_1k/non_uniform_cube_1k.npy', allow_pickle=True)
    nucube_example = torch.tensor(nucube_data[0, ...][None, ...], device=model_trained.device)

    print('\n Compute dynamics code ... \n')

    # Compute dynamics code for the 2 trajectories from the trained dataset

    xhat_dyn_cal0, _, R_dyn_cal0, _, _, _ = model_trained(x_eval[0, ...][None, ...].float(), seq_len=x_eval.shape[1])
    xhat_dyn_cal1, _, R_dyn_cal1, _, _, _ = model_trained(x_eval[1, ...][None, ...].float(), seq_len=x_eval.shape[1])

    R_dyn_cal0_rs = R_dyn_cal0.reshape(-1, 3, 3)
    R_dyn_cal1_rs = R_dyn_cal1.reshape(-1, 3, 3)

    xhat_dyn_cal0 = xhat_dyn_cal0.reshape(-1, x_eval.shape[1], 3, 28, 28)
    xhat_dyn_cal1 = xhat_dyn_cal1.reshape(-1, x_eval.shape[1], 3, 28, 28)

    print('\n Compute content codes and decoding ... \n')

    # Cloudsat content code and image prediction

    zC0_cs, _ = model_trained.encode(cloudsat_example[:, 0, ...][:, None, ...])  # [bs, 1, content_dim]
    zC_cs = zC0_cs.repeat(1, x_eval.shape[1], 1) # [bs, traj_len, content_dim]
    zC_cs_rs = zC_cs.reshape(-1, model_trained.encoder.content_dim) # [pbs, content_dim]
    
    xhat_cs0 = model_trained.decode(z_content=zC_cs_rs, z_dyn=R_dyn_cal0_rs).reshape(-1, x_eval.shape[1], 3, 28, 28)
    xhat_cs1 = model_trained.decode(z_content=zC_cs_rs, z_dyn=R_dyn_cal1_rs).reshape(-1, x_eval.shape[1], 3, 28, 28)

    # Ucube content code and image prediction
    zC0_uc, _ = model_trained.encode(ucube_example[:, 0, ...][:, None, ...])  # [bs, 1, content_dim]
    zC_uc = zC0_uc.repeat(1, x_eval.shape[1], 1) # [bs, traj_len, content_dim]
    zC_uc_rs = zC_uc.reshape(-1, model_trained.encoder.content_dim) # [pbs, content_dim]

    xhat_uc0 = model_trained.decode(z_content=zC_uc_rs, z_dyn=R_dyn_cal0_rs).reshape(-1, x_eval.shape[1], 3, 28, 28)
    xhat_uc1 = model_trained.decode(z_content=zC_uc_rs, z_dyn=R_dyn_cal1_rs).reshape(-1, x_eval.shape[1], 3, 28, 28)

    # Uprism content code and image prediction
    zC0_up, _ = model_trained.encode(uprism_example[:, 0, ...][:, None, ...])  # [bs, 1, content_dim]
    zC_up = zC0_up.repeat(1, x_eval.shape[1], 1) # [bs, traj_len, content_dim]
    zC_up_rs = zC_up.reshape(-1, model_trained.encoder.content_dim) # [pbs, content_dim]

    xhat_up0 = model_trained.decode(z_content=zC_up_rs, z_dyn=R_dyn_cal0_rs).reshape(-1, x_eval.shape[1], 3, 28, 28)
    xhat_up1 = model_trained.decode(z_content=zC_up_rs, z_dyn=R_dyn_cal1_rs).reshape(-1, x_eval.shape[1], 3, 28, 28)

    # NUcube content code and image prediction
    zC0_nuc, _ = model_trained.encode(nucube_example[:, 0, ...][:, None, ...])  # [bs, 1, content_dim]
    zC_nuc = zC0_nuc.repeat(1, x_eval.shape[1], 1) # [bs, traj_len, content_dim]
    zC_nuc_rs = zC_nuc.reshape(-1, model_trained.encoder.content_dim) # [pbs, content_dim]

    xhat_nuc0 = model_trained.decode(z_content=zC_nuc_rs, z_dyn=R_dyn_cal0_rs).reshape(-1, x_eval.shape[1], 3, 28, 28)
    xhat_nuc1 = model_trained.decode(z_content=zC_nuc_rs, z_dyn=R_dyn_cal1_rs).reshape(-1, x_eval.shape[1], 3, 28, 28)

    # Uprism content code and image prediction
    zC0_nup, _ = model_trained.encode(nuprism_example[:, 0, ...][:, None, ...])  # [bs, 1, content_dim]
    zC_nup = zC0_nup.repeat(1, x_eval.shape[1], 1) # [bs, traj_len, content_dim]
    zC_nup_rs = zC_nup.reshape(-1, model_trained.encoder.content_dim) # [pbs, content_dim]

    xhat_nup0 = model_trained.decode(z_content=zC_nup_rs, z_dyn=R_dyn_cal0_rs).reshape(-1, x_eval.shape[1], 3, 28, 28)
    xhat_nup1 = model_trained.decode(z_content=zC_nup_rs, z_dyn=R_dyn_cal1_rs).reshape(-1, x_eval.shape[1], 3, 28, 28)

    print('\n Generate results plots ... \n')

    # Compute image prediction for each dataset using their content code but with the dynamics code
    # Visualize as layout
    fig_cs0, _ = gen_recon_fig(xhat_dyn_cal0.detach().cpu().numpy(), xhat_cs0.detach().cpu().numpy(), img_interval=10) 
    fig_cs1, _ = gen_recon_fig(xhat_dyn_cal1.detach().cpu().numpy(), xhat_cs1.detach().cpu().numpy(), img_interval=10)

    fig_uc0, _ = gen_recon_fig(xhat_dyn_cal0.detach().cpu().numpy(), xhat_uc0.detach().cpu().numpy(), img_interval=10) 
    fig_uc1, _ = gen_recon_fig(xhat_dyn_cal1.detach().cpu().numpy(), xhat_uc1.detach().cpu().numpy(), img_interval=10)

    fig_up0, _ = gen_recon_fig(xhat_dyn_cal0.detach().cpu().numpy(), xhat_up0.detach().cpu().numpy(), img_interval=10) 
    fig_up1, _ = gen_recon_fig(xhat_dyn_cal1.detach().cpu().numpy(), xhat_up1.detach().cpu().numpy(), img_interval=10)

    fig_nuc0, _ = gen_recon_fig(xhat_dyn_cal0.detach().cpu().numpy(), xhat_nuc0.detach().cpu().numpy(), img_interval=10) 
    fig_nuc1, _ = gen_recon_fig(xhat_dyn_cal1.detach().cpu().numpy(), xhat_nuc1.detach().cpu().numpy(), img_interval=10)

    fig_nup0, _ = gen_recon_fig(xhat_dyn_cal0.detach().cpu().numpy(), xhat_nup0.detach().cpu().numpy(), img_interval=10) 
    fig_nup1, _ = gen_recon_fig(xhat_dyn_cal1.detach().cpu().numpy(), xhat_nup1.detach().cpu().numpy(), img_interval=10)

    # Save figures
    SAVE_DIR = f'src/rbnn_gravity/content_info/results/{dataset}/{date}/'
    os.makedirs(SAVE_DIR, exist_ok=True)

    fig_cs0.savefig(SAVE_DIR + f"cloudsat_content_dyn_0.pdf", format="pdf", bbox_inches="tight")
    fig_cs1.savefig(SAVE_DIR + f"cloudsat_content_dyn_1.pdf", format="pdf", bbox_inches="tight")
    
    fig_uc0.savefig(SAVE_DIR + f"ucube_content_dyn_0.pdf", format="pdf", bbox_inches="tight")
    fig_uc1.savefig(SAVE_DIR + f"ucube_content_dyn_1.pdf", format="pdf", bbox_inches="tight")

    fig_up0.savefig(SAVE_DIR + f"uprism_content_dyn_0.pdf", format="pdf", bbox_inches="tight")
    fig_up1.savefig(SAVE_DIR + f"uprism_content_dyn_1.pdf", format="pdf", bbox_inches="tight")

    fig_nuc0.savefig(SAVE_DIR + f"nucube_content_dyn_0.pdf", format="pdf", bbox_inches="tight")
    fig_nuc1.savefig(SAVE_DIR + f"nucube_content_dyn_1.pdf", format="pdf", bbox_inches="tight")

    fig_nup0.savefig(SAVE_DIR + f"nuprism_content_dyn_0.pdf", format="pdf", bbox_inches="tight")
    fig_nup1.savefig(SAVE_DIR + f"nuprism_content_dyn_1.pdf", format="pdf", bbox_inches="tight")

    print('\n Generate results gif ... \n')
    
    # Generate .gifs
    gen_gif(save_dir=SAVE_DIR + f"calipso_0.gif", img_array=xhat_dyn_cal0.detach().cpu().numpy())
    gen_gif(save_dir=SAVE_DIR + f"calipso_1.gif", img_array=xhat_dyn_cal1.detach().cpu().numpy())

    gen_gif(save_dir=SAVE_DIR + f"cloudsat_0.gif", img_array=xhat_cs0.detach().cpu().numpy())
    gen_gif(save_dir=SAVE_DIR + f"cloudsat_1.gif", img_array=xhat_cs1.detach().cpu().numpy())

    gen_gif(save_dir=SAVE_DIR + f"ucube_0.gif", img_array=xhat_uc0.detach().cpu().numpy())
    gen_gif(save_dir=SAVE_DIR + f"ucube_1.gif", img_array=xhat_uc1.detach().cpu().numpy())

    gen_gif(save_dir=SAVE_DIR + f"uprism_0.gif", img_array=xhat_up0.detach().cpu().numpy())
    gen_gif(save_dir=SAVE_DIR + f"uprism_1.gif", img_array=xhat_up1.detach().cpu().numpy())

    gen_gif(save_dir=SAVE_DIR + f"nucube_0.gif", img_array=xhat_nuc0.detach().cpu().numpy())
    gen_gif(save_dir=SAVE_DIR + f"nucube_1.gif", img_array=xhat_nuc1.detach().cpu().numpy())

    gen_gif(save_dir=SAVE_DIR + f"nuprism_0.gif", img_array=xhat_nup0.detach().cpu().numpy())
    gen_gif(save_dir=SAVE_DIR + f"nuprism_1.gif", img_array=xhat_nup1.detach().cpu().numpy())

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
    encoder = EncoderRBNN_content(content_dim=15)
    decoder = DecoderRBNN_content(content_dim=15)

    # Initialize model and optimizer
    model = rbnn_gravity_content(encoder=encoder,
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

    # Generate data for evaluation 
    x_train, x_test, x_val = gen_datasets_eval(args)
    
    # Visualize latent states with cotent information
    latent_space_viz(args=args, model=model_tr, x_test=x_test)
    # Generate content-dynamics plots
    content_dynamics_swap(args=args, model=model_tr, x_test=x_test)