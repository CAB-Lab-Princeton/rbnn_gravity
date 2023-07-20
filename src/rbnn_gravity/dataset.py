import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from rbnn_gravity.configuration import config


class Measurement(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        filename = glob.glob(os.path.join(self.data_dir, "*.npz"))[0]
        # import pdb; pdb.set_trace()
        npzfiles = np.load(filename)
        
        self.R = npzfiles['R']
        self.omega = npzfiles['omega']

    def __len__(self):
        return len(self.R)

    def __getitem__(self, idx):
        return self.R[idx], self.omega[idx]


def build_dataloader(args):
    videos_dataset = get_dataset(args)
    videos_dataloader = DataLoader(
        videos_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    return videos_dataloader


def get_dataset(args):
    """
    builds a dataloader

    Args:
        args (ArgumentParser): experiment parameters

    Returns:
        Dataset
    """
    return Measurement(args.data_dir)