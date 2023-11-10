# Author(s): Justice Mason
# Project: RBNN + Gravity
# Date: 11/08/23

import os
import numpy as np
import torch

# Functions for generating datasets
def euler_eigvec(MOI: np.ndarray, radius: float) -> np.ndarray:
    """
    Function to calculate the eigenvectors of the Euler dynamics, linearized about the intermediate axis.
    
    ...
    
    Parameters
    ----------
    MOI : np.ndarray
        Moment of intertia tensor for the system.
        
    radius : float
        Radius of the angular momentum sphere.
    
    Returns
    -------
    eigvec : np.ndarray
        Eigenvectors correpsonding to the dynamics after they're linearized about the intermediate axis.
        
    Notes
    -----
    
    """
    MOI = np.diag(MOI)
    beta = (MOI[0] - MOI[1])/(MOI[0] * MOI[1]) # factor used for linearization
    gamma = (MOI[1] - MOI[2])/(MOI[1] * MOI[2]) # factor used for linearization
    
    euler_umatrix = np.array([[0, 0, beta * radius], [0, 0, 0], [gamma * radius, 0 , 0]]) # linearize dyns
    _, eigvec = np.linalg.eig(euler_umatrix) # calculate the eigenvalues and eigenvectors 
    
    return eigvec

def calc_hetero_angle(eigv: np.ndarray) -> np.ndarray:
    """
    """
    e3 = np.array([0., 0., 1.]).reshape((3,))
    v1 = eigv[:, 0]
    
    # Calculate angle using the first eigenvalue and z-axis 
    cos_theta = np.max(np.min((np.dot(v1, e3)/(np.linalg.norm(v1) * np.linalg.norm(e3))), axis=0), axis=-1)
    angle = np.real(np.arccos(cos_theta))
    return angle

def sample_init_conds(MOI: np.ndarray, radius: float, seed: int = 0, ic_type: str = 'random', n_samples: int = 10, desired_samples: np.ndarray = None, bandwidth_us: float = 5.0) -> np.ndarray:
    """
    Function to sample from the body angular momentum sphere.
    
    ...
    
    """
    np.random.seed(seed=seed)
    ic_type = ic_type.lower()
    eps = 1e-3
    
    if ic_type == 'random':
        theta = np.random.uniform(low=0.0, high=np.pi+eps, size=(n_samples, 1))
        phi = np.random.uniform(low=0.0, high=2*np.pi, size=(n_samples, 1))
    
    elif ic_type == 'unstable':
        assert bandwidth_us < 10
        bw_rad = np.deg2rad(bandwidth_us)
        ev = euler_eigvec(MOI=MOI, radius=radius)
        heteroclinic_angle = calc_hetero_angle(eigv=ev)
        
        theta = np.random.uniform(low = heteroclinic_angle - (0.5 * bw_rad), high = heteroclinic_angle + (0.5 * bw_rad), size=(n_samples, 1))
        phi = np.zeros((n_samples, 1))
        
    elif ic_type =='desired':
        theta = desired_samples[0, ...]
        phi = desired_samples[1, ...]
    else:
        raise ValueError('Use the allowed ic_type.')
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    samples = np.concatenate((x, y, z), axis=-1)
    return samples

def generate_lowdim_dataset(MOI: np.ndarray, radius: float, n_samples: int, integrator, timestep: float = 1e-3, traj_len: int = 100, bandwidth_us: float = 5., desired_samples: np.ndarray = None, ic_type: str = 'random', seed: int = 0) -> np.ndarray:
    """"""
    # sample initial conditions 
    # body angular momentum sphere
    pi_samples = sample_init_conds(MOI=MOI, radius=radius, ic_type=ic_type, n_samples=n_samples, bandwidth_us=bandwidth_us)
    pi_samples_tensor = torch.tensor(pi_samples, device=MOI.device)
    
    # group element matrices
    R_samples = random_group_matrices(n=n_samples)
    
    # integrate trajectories
    data_R, data_pi = integrator.integrate(pi_init=pi_samples_tensor, moi=MOI, R_init=R_samples, timestep=timestep, traj_len=traj_len)
    return data_R, data_pi


# Auxilary function for dataset generation
def random_quaternions(n, dtype=torch.float32, device=None):
    u1, u2, u3 = torch.rand(3, n, dtype=dtype, device=device)
    return torch.stack((
        torch.sqrt(1-u1) * torch.sin(2 * np.pi * u2),
        torch.sqrt(1-u1) * torch.cos(2 * np.pi * u2),
        torch.sqrt(u1) * torch.sin(2 * np.pi * u3),
        torch.sqrt(u1) * torch.cos(2 * np.pi * u3),
    ), 1)

def quaternions_to_group_matrix(q):
    """Normalises q and maps to group matrix."""
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return torch.stack([
        r*r - i*i - j*j + k*k, 2*(r*i + j*k), 2*(r*j - i*k),
        2*(r*i - j*k), -r*r + i*i - j*j + k*k, 2*(i*j + r*k),
        2*(r*j + i*k), 2*(i*j - r*k), -r*r - i*i + j*j + k*k
    ], -1).view(*q.shape[:-1], 3, 3)

def random_group_matrices(n, dtype=torch.float32, device=None):
    return quaternions_to_group_matrix(random_quaternions(n, dtype, device))

# Functions to save generated
def save_data(data_R: np.ndarray, data_omega: np.ndarray, filename: str, save_dir: str = '/home/jmason/rbnn_gravity/src/rbnn_gravity/data'):
    """
    Function to save data file.

    ...

    """
    print(f'\n Checking if save directory {save_dir} exists already ... \n')
    if not os.path.exists(save_dir):
        print('\n Save directory does NOT exist yet. Creating it ... \n')
        os.makedirs(save_dir)
    else:
        print('\n Save directory already exists! Yay! \n')
    
    print(f'\n Now saving data as {save_dir + "/" + filename + ".npz"} ... \n')
    with open(save_dir + f'/{filename}.npz', 'wb') as outfile:
        np.savez(outfile, R=data_R, omega=data_omega)

