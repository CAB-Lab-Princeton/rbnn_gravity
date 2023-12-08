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

# Image data class
class RBNNdataset(Dataset):
    """
    Dataset class for the DEVS project.
    
    ...
    
    Attributes
    ----------
    data : torch.Tensor
        N-D array of images for training/testing.
        
    seq_len : int, default=3
        Number of observations representating a sequence of images -- input to the network.
        
        
    Methods
    -------
    __len__()
    __getitem__()
    
    Notes
    -----
    
    """
    def __init__(self, data: np.ndarray, seq_len: int = 3):
        super().__init__()

        self.data = data
        self.seq_len = seq_len
        
    def __len__(self):
        """
        """
        num_traj, traj_len, _, _, _ = self.data.shape
        length = num_traj * (traj_len - self.seq_len + 1)
            
        return length
        
    def __getitem__(self, idx):
        """
        """
        assert idx < self.__len__(),  "Index is out of range."
        num_traj, traj_len, _, _, _ = self.data.shape
        
        traj_idx, seq_idx = divmod(idx, traj_len - self.seq_len + 1)
        
        sample = self.data[traj_idx, seq_idx:seq_idx+self.seq_len,...]
        
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

def build_dataloader_hd(args):
    """
    Wrapper to build dataloader.

    ...

    Args:
        args (ArgumentParser): experiment parameters

    """
    # Construct datasets
    trainds, testds, valds = get_dataset_hd(args)
    
    # Dataset kwargs 
    train_kwargs = {'batch_size': args.batch_size}
    val_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    
    cuda_kwargs = {'num_workers': 4}
        
    train_kwargs.update(cuda_kwargs)
    val_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    # Create dataloaders
    traindataloader = DataLoader(
        trainds, **train_kwargs, drop_last=True, shuffle=True, pin_memory=True
    )
    testdataloader = DataLoader(
        testds, **test_kwargs, drop_last=True, shuffle=True, pin_memory=True
    )
    valdataloader = DataLoader(
        valds, **val_kwargs, drop_last=True, shuffle=True, pin_memory=True
    )

    return traindataloader, testdataloader, valdataloader

def get_dataset_hd(args):
    """
    builds a dataloader

    Args:
        args (ArgumentParser): experiment parameters

    Returns:
        Dataset
    """
    # Load data from file
    raw_data = np.load(args.data_dir, allow_pickle=True)

    num_traj, traj_len, _, _, _ = raw_data.shape
    
    test_split = args.test_split
    val_split = args.val_split
    
    test_len = int(test_split * num_traj)
    val_len = int((1. - test_split) * val_split * num_traj)
    train_len = int((1. - test_split) * (1. - val_split) * num_traj)
    
    np.random.shuffle(raw_data)
        
    rd_split = np.split(raw_data.astype(float), [train_len, train_len + val_len, train_len + val_len + test_len], axis=0)
    
    if traj_len > 100:
        train_dataset = rd_split[0][:, :100, ...]
        val_dataset = rd_split[1][:, :100, ...]
        test_dataset = rd_split[2][:, :100, ...]
    else:
        train_dataset = rd_split[0]
        val_dataset = rd_split[1]
        test_dataset = rd_split[2]

    trainds_rbnn = RBNNdataset(data=train_dataset, seq_len=args.seq_len)
    testds_rbnn = RBNNdataset(data=test_dataset, seq_len=args.seq_len)
    valds_rbnn = RBNNdataset(data=val_dataset, seq_len=args.seq_len)

    return trainds_rbnn, testds_rbnn, valds_rbnn

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