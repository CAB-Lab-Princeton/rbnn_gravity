import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
# from rbnn_gravity.configuration import config

# Measurement dataset class
class LowDimDataset(Dataset):
    """
    Dataset class specfic to the rigid body ODE network.
    
    ...
    
    Attributes
    ----------
    data : np.ndarray
        Full trajectories generated using Euler's equations
        
    seq_len : int, default=2
        The length of each sample trajectory in the dataset
        
    Methods
    -------
    __len__()
    __getitem__(idx)
    
    Notes
    -----
    
    """
    def __init__(self, data_dir, seq_len: int = 2):
        self.data_dir = data_dir
        #import pdb; pdb.set_trace()
        filename = glob.glob(os.path.join(self.data_dir, "*.npz"))[0]
        self.seq_len = seq_len

        # Load files
        npzfiles = np.load(filename)
        self.data_R = npzfiles['R']
        self.data_omega = npzfiles['omega']
        
    def __len__(self):
        """
        Computes total number of trajectory samples of length seq_len in the dataset.
        
        ...
        
        """
        num_traj, traj_len, _, _  = self.data_R.shape
        num_samples = num_traj * (traj_len - self.seq_len + 1)
        
        return num_samples
    
    def __getitem__(self, idx):
        """
        Returns specific trajectory subsequence corresponding to idx.
        
        ...
        
        Parameters
        ----------
        idx : int
            Sample index.
        
        Returns
        -------
        sample : torch.Tensor
            Returns sample of length seq_len with 
        
        Notes
        -----
        
        """
        assert idx < self.__len__(), "Index is out of range."
        _, traj_len, _, _  = self.data_R.shape
        
        traj_idx, seq_idx = divmod(idx, traj_len - self.seq_len + 1)
        sample_R = self.data_R[traj_idx, seq_idx:seq_idx+self.seq_len, ...]
        sample_omega = self.data_omega[traj_idx, seq_idx:seq_idx+self.seq_len, ...]
        
        sample = (sample_R, sample_omega)
        
        return sample

# Auxiliary functions
def build_dataloader(args):
    """
    Wrapper to build dataloader.

    ...

    Args:
        args (ArgumentParser): experiment parameters

    """
    dataset = get_dataset(args)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    return dataloader

def get_dataset(args):
    """
    builds a dataloader

    Args:
        args (ArgumentParser): experiment parameters

    Returns:
        Dataset
    """
    return LowDimDataset(args.data_dir, args.seq_len)

# Archive
class Measurement(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        filename = glob.glob(os.path.join(self.data_dir, "*.npz"))[0]
        # import pdb; pdb.set_trace()
        npzfiles = np.load(filename)
        
        self.R = npzfiles['R']
        self.omega = npzfiles['omega']

    def __len__(self):
        return len(self.R) - 1

    def __getitem__(self, idx):
        return self.R[idx:idx+2], self.omega[idx:idx+2]