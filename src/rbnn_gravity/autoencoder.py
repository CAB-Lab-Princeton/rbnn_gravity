# Project Name: RBNN + Gravity
# File Purpose: ``
# Creation Date:
# Creator: Justice Mason (jjmason@princeton.edu)

import sys, os
import torch
from torch import nn
from rbnn_gravity.utils.math import *

# Todo: 
#   - Get rid of the maxpooling layers
#   - ELU activation layers might need to be changed 
#   - ELU requires using Kaiming/He initialization -- right now using Xavier initialization

class EncoderNetSE3(nn.Module):
    """
    Implementation of an convolutional encoder network for DEVS to map to the SO(3) latent space.
    
    ...
    
    """
    def __init__(self,
                in_channels: int = 3,
                obs_len: int = 2,
                latent_dim = 9) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of channels on the input layer of the encoder network.
            
        obs_len : int, default=3
            Observation length of each input in the train/test dataset.
            
        latent_dim : int, default=9
            Latent dimension size. Chosen to be 9 to reconstruct the state (R, \Pi) \in T*SO(3).
            
        Notes
        -----
        
        """
        super().__init__()
        self.obs_len = obs_len

        self.conv1 = nn.Conv2d(in_channels= in_channels * self.obs_len, out_channels=16, kernel_size=3)
        self.relu1 = nn.ELU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.relu2 = nn.ELU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu3 = nn.ELU()
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.relu4 = nn.ELU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.bn4 =  nn.BatchNorm2d(32)
        self.flatten4 = nn.Flatten(start_dim=1)
        self.linear5 = nn.Linear(in_features=32*4*4, out_features=120)
        self.relu5 = nn.ELU()
        self.bn5 = nn.BatchNorm1d(120)
        self.linear6 = nn.Linear(in_features=120, out_features=84)
        self.relu6 = nn.ELU()
        self.linear7 = nn.Linear(in_features=84, out_features=latent_dim, bias=False)
    
    def map_s2s2_so3(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        """
        # assert z1.shape[-1] == 3 and z2.shape[-1] == 3, "Both input vectors must be in R^{3}."
        # assert torch.any(z1.isnan()) == False and torch.any(z1.isinf()) == False
        # assert torch.any(z2.isnan()) == False and torch.any(z2.isinf()) == False

        z1_norm = z1 / z1.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
        z2_norm = z2 / z2.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
        
        assert torch.linalg.norm(z1_norm - z2_norm) > 1e-1 # we don't want z1/z2 to be aligned
        enc_R = s2s2_gram_schmidt(v1=z1_norm, v2=z2_norm)
        
        return enc_R
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the encoding for a sequence of images.
        
        ...
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor for encoding -- size (B, obs_len, C, W, H).

        Returns
        -------
        x_enc_SO3
        x_enc_trans
        indices1
        indices2
        
        Notes
        -----
        
        """
        # assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        batch_size, obs_len, channels, w, h = x.shape
        x_ = x.reshape(batch_size, obs_len*channels, w, h)
        
        h1 = self.relu1(self.conv1(x_))
        h2, indices2 = self.maxpool2(self.relu2(self.conv2(h1)))
        h2 = self.bn2(h2)
        h3 = self.relu3(self.conv3(h2))
        h4, indices4 =  self.maxpool4(self.relu4(self.conv4(h3)))
        h4 = self.flatten4(self.bn4(h4))
        h5 = self.bn5(self.relu5(self.linear5(h4)))
        h6 = self.relu6(self.linear6(h5))
        x_enc = self.linear7(h6)
        
        z1_enc = x_enc[:, :3]
        z2_enc = x_enc[:, 3:6]

        x_enc_trans = x_enc[:, 6:]    
        x_enc_SO3 = self.map_s2s2_so3(z1=z1_enc, z2=z2_enc)

        return x_enc_SO3, x_enc_trans, indices2, indices4

class DecoderNetSE3(nn.Module):
    """
    Implementation of an deconvolutional decoder network for DEVS to map from the desired latent space to image space.
    
    ...
    
    """
    def __init__(self,
                in_channels: int = 9) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of channels on the input layer of the encoder network.
            
        Notes
        -----
        
        """
        super().__init__()
        
        self.linear7 = nn.Linear(in_features=in_channels, out_features=84)
        self.relu6 = nn.ELU()
        self.linear6 = nn.Linear(in_features=84, out_features=120)
        self.bn5 = nn.BatchNorm1d(120)
        self.relu5 = nn.ELU()
        self.linear5 = nn.Linear(in_features=120, out_features=32*4*4)
        self.unflatten4 = nn.Unflatten(dim=1, unflattened_size=(32, 4, 4))
        self.bn4 = nn.BatchNorm2d(32)
        self.maxunpool4 = nn.MaxUnpool2d(2, 2)
        self.relu4 = nn.ELU()
        self.conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3)
        self.relu3 = nn.ELU()
        self.conv3 = nn.ConvTranspose2d(32, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.maxunpool2 = nn.MaxUnpool2d(2, 2)
        self.relu2 = nn.ELU()
        self.conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3)
        self.relu1 = nn.ELU()
        self.conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3)
                
    def forward(self, x: torch.Tensor, indices2, indices4) -> torch.Tensor:
        """
        Compute the encoding for a sequence of images.
        
        ...
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor for encoding -- x_enc_SO3 and x_enc_trans concat. 
            
        Returns
        -------
        x_dec : torch.Tensor
            Encoded tensor.
        
        Notes
        -----
        The input should be the concatenation of the transformed x_enc_SO3 (SO3 -> S2S2) and x_enc_trans.
        
        """
        # assert torch.any(x.isnan()) == False and torch.any(x.isinf()) == False
        x_ = x.reshape(-1, 9)
        
        h7 = self.linear7(x_)
        h6 = self.linear6(self.relu6(h7))
        h5 = self.linear5(self.relu5(self.bn5(h6)))
        h4 = self.bn4(self.unflatten4(h5))
        
        h4 = self.conv4(self.relu4(self.maxunpool4(h4, indices4)))
        h3 = self.conv3(self.relu3(h4))
        h2 = self.bn2(h3)
        h2 = self.conv2(self.relu2(self.maxunpool2(h2, indices2)))
        x_dec = self.conv1(self.relu1(h2))
        return x_dec