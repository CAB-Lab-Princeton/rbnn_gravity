# Author(s): Justice Mason
# Project: RBNN + Gravity
# Date: 11/08/23

import os, sys
import numpy as np
import torch

from utils.math_utils import eazyz_to_group_matrix, eazxz_to_group_matrix

# Functions for generating datasets for freely rotating RBD
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

def generate_lowdim_dataset(MOI: np.ndarray, radius: float, n_samples: int, integrator, timestep: float = 1e-3, traj_len: int = 100, bandwidth_us: float = 5., desired_samples: np.ndarray = None, ic_type: str = 'random', V = None, seed: int = 0) -> np.ndarray:
    """"""
    # sample initial conditions 
    # body angular momentum sphere
    pi_samples = sample_init_conds(MOI=MOI, radius=radius, ic_type=ic_type, n_samples=n_samples, bandwidth_us=bandwidth_us)
    pi_samples_tensor = torch.tensor(pi_samples, device=MOI.device)
    
    # group element matrices
    R_samples = random_group_matrices(n=n_samples)
    
    # Integrate trajectories
    data_R, data_pi = integrator.integrate(pi_init=pi_samples_tensor, moi=MOI, V=V, R_init=R_samples, timestep=timestep, traj_len=traj_len)
    return data_R, data_pi

# Functions for data generation for 3D Pendulum and Physical Pendulum
def sample_group_matrices_3DP(radius: float, ic_type: str = 'stable', n_samples: int = 100, scale: float = 0.1, specified_samples: np.ndarray = None):
    """"""
    ic_type = ic_type.lower()
    eps = 1e-3

    if ic_type == 'stable':
        stable  = np.array([0., 0., 0.])[None, ...].repeat(n_samples, 1)
        alpha = np.random.uniform(low=-scale*np.pi, high=scale*np.pi, size=(n_samples, 1)) + stable[:, 0]
        beta = np.random.uniform(low=-0.5*scale*np.pi, high=0.5*scale*np.pi, size=(n_samples, 1)) + stable[:, 1]
        gamma = np.random.uniform(low=-scale*np.pi, high=scale*np.pi, size=(n_samples, 1)) + stable[:, 2]
    elif ic_type == 'unstable':
        unstable  = np.array([0., np.pi, 0.])[None, ...].repeat(n_samples, 1)
        alpha = np.random.uniform(low=-scale*np.pi, high=scale*np.pi, size=(n_samples, 1)) + unstable[:, 0]
        beta = np.random.uniform(low=-0.5*scale*np.pi, high=0.5*scale*np.pi, size=(n_samples, 1)) + unstable[:, 1]
        gamma = np.random.uniform(low=-scale*np.pi, high=scale*np.pi, size=(n_samples, 1)) + unstable[:, 2]
    elif ic_type == 'uniform':
        alpha = np.random.uniform(low=0.0, high=2*np.pi-eps, size=(n_samples, 1))
        beta = np.random.uniform(low=-0.5*scale*np.pi, high=0.5*scale*np.pi, size=(n_samples, 1))
        gamma = np.random.uniform(low=-scale*np.pi, high=scale*np.pi, size=(n_samples, 1))
    else:
        raise ValueError(f"Use 'ic_type' from the allowed set: {['stable', 'unstable', 'uniform']}")

    R = eazyz_to_group_matrix(alpha=alpha, beta=beta, gamma=gamma).squeeze()
    samples = R.transpose(2, 0, 1)

    return samples

def sample_group_matrix_gyroscope(MOI: np.ndarray, n_samples: int,  mass: float = 1.0, l: float = 1.0, general_flag: bool = False):
    """"""
    if general_flag:
        # Euler angles for gyroscope
        eulers = (np.random.rand(n_samples, 2, 3) - 0.5) * 3 #3

        # Edit euler angles to be in a desired range (should be psidot >> thetadot >> phidot)
        eulers[:,1,0]*=3 # phidot [magnitude 4.5]
        eulers[:,1,1]*=.2 # thetadot [magnitude 0.3]
        eulers[:,1,2] = (np.random.randint(2, size=(n_samples, )) * 2. - 1) * (np.random.randn(n_samples) + 7) * 1.5 # psidot [manigtude 7]

        # Assign Euler angles -- general
        phi = eulers[:, 0, 0]
        theta = eulers[:, 0, 1]
        psi = eulers[:, 0, 2]

        # Calculate omega in the body-fixed frame
        phidot = eulers[:, 1, 0]
        thetadot = eulers[:, 1, 1]
        psidot = eulers[:, 1, 2]

        w3 = (phidot * ct) + psidot

    else:

        # Assign Euler angles -- steady top
        g = 9.8 #gravity
        I_1 = MOI[0, 0]
        I_3 = MOI[2, 2]

        eulers = (np.random.rand(n_samples, 3) - 0.5) * (np.pi - 0.1)
        phi = eulers[:, 0]
        theta = eulers[:, 1]
        psi = eulers[:, 2] 

        # Thetadot initialized to zero
        thetadot = np.zeros((n_samples))
    
        # Phidot samples from range
        w3_min = (2./I_3) * np.sqrt(mass * g * l * I_1 * np.cos(theta))
        w3 = np.einsum('b, b -> b', (np.random.rand((n_samples)) + 1.1), w3_min)

        phidot_slow = (mass * g * l)/ (I_3 * w3)  
        phidot_fast = (I_3 * w3)/(I_1 * np.cos(theta))

        # Psidot is calculated
        phidot = phidot_slow
        psidot = ((mass * g * l + (I_1 - I_3) * (phidot ** 2) * np.cos(theta))/(I_3 * phidot))
    

    st = np.sin(theta)
    ct = np.cos(theta)

    sp = np.sin(psi)
    cp = np.cos(psi)

    # Calculate pi in the body frame -- Goldstein 
    w1 = (phidot * st * sp) + (thetadot * cp)
    w2 = (phidot * st * cp) - (thetadot * sp)

    omega = np.stack([w1, w2, w3], axis=-1)
    pi_samples = np.einsum('ij, bj -> bi', MOI, omega)

    # Convert the Euler angles to rotation matrices using the ZYZ convention
    R_samples = eazyz_to_group_matrix(alpha=phi, beta=theta, gamma=psi)
    # R_samples = eazxz_to_group_matrix(alpha=phi, beta=theta, gamma=psi)

    return R_samples, pi_samples

def generate_lowdim_dataset_3DP(MOI: np.ndarray, radius: float, n_samples: int, integrator, timestep: float = 1e-3, traj_len: int = 100, bandwidth_us: float = 5., desired_samples: np.ndarray = None, R_ic_type: str = 'stable', pi_ic_type: str = 'random', V = None, seed: int = 0):
    """"""
    # sample initial conditions 
    # body angular momentum sphere
    pi_samples = sample_init_conds(MOI=MOI, radius=radius, ic_type=pi_ic_type, n_samples=n_samples, bandwidth_us=bandwidth_us)
    pi_samples_tensor = torch.tensor(pi_samples, device=MOI.device)
    
    # group element matrices
    R_samples = sample_group_matrices_3DP(radius=radius, ic_type=R_ic_type, n_samples=n_samples)
    R_samples_tensor = torch.tensor(R_samples, device=MOI.device, requires_grad=True)
    
    # integrate trajectories
    data_R, data_pi = integrator.integrate(pi_init=pi_samples_tensor, moi=MOI, V=V, R_init=R_samples_tensor, timestep=timestep, traj_len=traj_len)
    return data_R, data_pi

def generate_lowdim_dataset_gyroscope(MOI: np.ndarray, mass: float, l3: float, n_samples: int, integrator, general_flag: bool = False, timestep: float = 1e-3, traj_len: int = 100, V = None, seed: int = 0):
    """"""
    # Sample initial conditions
    R_samples, pi_samples = sample_group_matrix_gyroscope(MOI=MOI, n_samples=n_samples, mass=mass, l=l3, general_flag=False)

    # Make samples tensors
    R_samples_tensor = torch.tensor(R_samples, requires_grad=True).permute(2, 0, 1)
    pi_samples_tensor = torch.tensor(pi_samples)

    # Integrate trajectories
    data_R, data_pi = integrator.integrate(pi_init=pi_samples_tensor, moi=MOI, V=V, R_init=R_samples_tensor, timestep=timestep, traj_len=traj_len)

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
def save_data(data_R: np.ndarray, data_omega: np.ndarray, filename: str, date: str, save_dir: str = '/home/jmason/rbnn_gravity/src/rbnn_gravity/data'):
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
    
    print(f'\n Now saving data as {save_dir + "/" + filename + "-date-" + date + ".npz"} ... \n')
    with open(save_dir + f'/{filename}.npz', 'wb') as outfile:
        data_R = data_R.detach().numpy()
        data_omega = data_omega.detach().numpy()
        np.savez(outfile, R=data_R, omega=data_omega)

