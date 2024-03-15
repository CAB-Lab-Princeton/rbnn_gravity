# Project Name: RBNN + Gravity
# File Purpose:
# Creation Date:
# Creator: Justice Mason (jjmason@princeton.edu)

import sys, os
import torch
from torch import nn
from utils.math_utils import *

# Todo: 
#   - Get rid of the maxpooling layers
#   - ELU activation layers might need to be changed 
#   - ELU requires using Kaiming/He initialization -- right now using Xavier initialization

class EncoderRBNN(nn.Module):
    """
    Implementation of an convolutional encoder network for DEVS to map to the SO(3) latent space.
    
    ...
    
    """
    def __init__(self,
                in_channels: int = 3,
                obs_len: int = 1,
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
        z2_enc = x_enc[:, 3:]
  
        x_enc_SO3 = self.map_s2s2_so3(z1=z1_enc, z2=z2_enc)

        # List for maxpooling indices
        indices = dict((f'indices{2*i}', None) for i in range(1, 3))
        indices['indices2'] = indices2
        indices['indices4'] = indices4

        return x_enc_SO3, indices

class DecoderRBNN(nn.Module):
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
    
# Encoder and Decoder with GELU

class EncoderRBNN_gravity(nn.Module):
    """
    Implementation of an convolutional encoder network for DEVS to map to the SO(3) latent space.
    
    ...
    
    """
    def __init__(self,
                in_channels: int = 3,
                obs_len: int = 1,
                latent_dim: int  = 6,
                nonlinearity = torch.nn.GELU()) -> None:
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
        self.nonlin = nonlinearity

        self.conv1 = nn.Conv2d(in_channels= in_channels * self.obs_len, out_channels=16, kernel_size=3) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.bn4 =  nn.BatchNorm2d(32)
        self.flatten4 = nn.Flatten(start_dim=1)
        self.linear5 = nn.Linear(in_features=32*5*5, out_features=120)
        self.bn5 = nn.BatchNorm1d(120)
        self.linear6 = nn.Linear(in_features=120, out_features=84)
        self.linear7 = nn.Linear(in_features=84, out_features=latent_dim, bias=False)
    
    def map_s2s2_so3(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """"""
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
        batch_size, obs_len, channels, w, h = x.shape
        x_ = x.reshape(batch_size, obs_len*channels, w, h)
        
        h1 = self.nonlin(self.conv1(x_))
        h2, indices2 = self.maxpool2(self.nonlin(self.conv2(h1)))
        h2 = self.bn2(h2)
        h3 = self.nonlin(self.conv3(h2))
        h4, indices4 =  self.maxpool4(self.nonlin(self.conv4(h3)))
        h4 = self.flatten4(self.bn4(h4))
        h5 = self.bn5(self.nonlin(self.linear5(h4)))
        h6 = self.nonlin(self.linear6(h5))
        x_enc = self.linear7(h6)
        
        z1_enc = x_enc[:, :3]
        z2_enc = x_enc[:, 3:]
   
        x_enc_SO3 = self.map_s2s2_so3(z1=z1_enc, z2=z2_enc)

        # List for maxpooling indices
        indices = dict((f'indices{2*i}', None) for i in range(1, 3))
        indices['indices2'] = indices2
        indices['indices4'] = indices4

        return x_enc_SO3, indices
    
class DecoderRBNN_gravity(nn.Module):
    """
    Implementation of an deconvolutional decoder network for DEVS to map from the desired latent space to image space.
    
    ...
    
    """
    def __init__(self,
                in_channels: int = 6,
                nonlinearity = torch.nn.GELU()) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of channels on the input layer of the encoder network.
            
        Notes
        -----
        
        """
        super().__init__()
        self.nonlin = nonlinearity
        
        self.linear7 = nn.Linear(in_features=in_channels, out_features=84)
        self.linear6 = nn.Linear(in_features=84, out_features=120)
        self.bn5 = nn.BatchNorm1d(120)
        self.linear5 = nn.Linear(in_features=120, out_features=32*5*5)
        self.unflatten4 = nn.Unflatten(dim=1, unflattened_size=(32, 5, 5))
        self.bn4 = nn.BatchNorm2d(32)
        self.maxunpool4 = nn.MaxUnpool2d(2, 2)
        self.conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = nn.ConvTranspose2d(32, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.maxunpool2 = nn.MaxUnpool2d(2, 2)
        self.conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3)
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
        sig = torch.nn.Sigmoid()
        x_ = x[:, :2, ...].reshape(-1, 6)
        
        h7 = self.linear7(x_)
        h6 = self.linear6(self.nonlin(h7))
        h5 = self.linear5(self.nonlin(self.bn5(h6)))
        h4 = self.bn4(self.unflatten4(h5))
        
        h4 = self.conv4(self.nonlin(self.maxunpool4(h4, indices4)))
        h3 = self.conv3(self.nonlin(h4))
        h2 = self.bn2(h3)
        h2 = self.conv2(self.nonlin(self.maxunpool2(h2, indices2)))
        x_dec = sig(self.conv1(self.nonlin(h2)))
        return x_dec    

# Content + Dynamics Autoencoders

class EncoderRBNN_content(nn.Module):
    """
    Implementation of an convolutional encoder network for DEVS to map to the SO(3) latent space.
    
    ...
    
    """
    def __init__(self,
                in_channels: int = 3,
                obs_len: int = 1,
                content_dim: int = 50,
                dynamics_dim: int  = 6, 
                nonlinearity = torch.nn.GELU()) -> None:
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
        self.nonlin = nonlinearity
        self.content_dim = content_dim
        self.dynanmics_dim = dynamics_dim
        self.latent_dim = content_dim + dynamics_dim

        self.conv1 = nn.Conv2d(in_channels= in_channels * self.obs_len, out_channels=16, kernel_size=3) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.bn4 =  nn.BatchNorm2d(32)
        self.flatten4 = nn.Flatten(start_dim=1)
        self.linear5 = nn.Linear(in_features=32*5*5, out_features=120)
        self.bn5 = nn.BatchNorm1d(120)
        self.linear6 = nn.Linear(in_features=120, out_features=84)
        self.linear7 = nn.Linear(in_features=84, out_features=self.latent_dim, bias=False)
    
    def map_s2s2_so3(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """"""
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
        batch_size, obs_len, channels, w, h = x.shape
        x_ = x.reshape(batch_size, obs_len*channels, w, h)
        
        h1 = self.nonlin(self.conv1(x_))
        h2, indices2 = self.maxpool2(self.nonlin(self.conv2(h1)))
        h2 = self.bn2(h2)
        h3 = self.nonlin(self.conv3(h2))
        h4, indices4 =  self.maxpool4(self.nonlin(self.conv4(h3)))
        h4 = self.flatten4(self.bn4(h4))
        h5 = self.bn5(self.nonlin(self.linear5(h4)))
        h6 = self.nonlin(self.linear6(h5))
        x_enc = self.linear7(h6)
        
        z_content = x_enc[:, :self.content_dim]
        z1_enc = x_enc[:, self.content_dim:self.content_dim+3]
        z2_enc = x_enc[:, self.content_dim+3:]
   
        z_enc_SO3 = self.map_s2s2_so3(z1=z1_enc, z2=z2_enc)

        # List for maxpooling indices
        indices = dict((f'indices{2*i}', None) for i in range(1, 3))
        indices['indices2'] = indices2
        indices['indices4'] = indices4

        return z_content, z_enc_SO3, indices
    
class DecoderRBNN_content(nn.Module):
    """
    Implementation of an deconvolutional decoder network for DEVS to map from the desired latent space to image space.
    
    ...
    
    """
    def __init__(self,
                content_dim: int = 50,
                dynamics_dim: int = 6,
                nonlinearity = torch.nn.GELU()) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of channels on the input layer of the encoder network.
            
        Notes
        -----
        
        """
        super().__init__()
        self.nonlin = nonlinearity
        in_channels = content_dim + dynamics_dim
        
        self.linear7 = nn.Linear(in_features=in_channels, out_features=84)
        self.linear6 = nn.Linear(in_features=84, out_features=120)
        self.bn5 = nn.BatchNorm1d(120)
        self.linear5 = nn.Linear(in_features=120, out_features=32*5*5)
        self.unflatten4 = nn.Unflatten(dim=1, unflattened_size=(32, 5, 5))
        self.bn4 = nn.BatchNorm2d(32)
        self.maxunpool4 = nn.MaxUnpool2d(2, 2)
        self.conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = nn.ConvTranspose2d(32, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.maxunpool2 = nn.MaxUnpool2d(2, 2)
        self.conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3)
                
    def forward(self, x_content: torch.Tensor, x_dynamics: torch.Tensor, indices2 = None, indices4 = None) -> torch.Tensor:
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
        sig = torch.nn.Sigmoid()
        
        x_C = x_content
        x_D = x_dynamics[:, :2, ...].reshape(-1, 6)
        x_ = torch.cat((x_C, x_D), dim=-1)
        
        h7 = self.linear7(x_)
        h6 = self.linear6(self.nonlin(h7))
        h5 = self.linear5(self.nonlin(self.bn5(h6)))
        h4 = self.bn4(self.unflatten4(h5))
        
        h4 = self.conv4(self.nonlin(self.maxunpool4(h4, indices4)))
        h3 = self.conv3(self.nonlin(h4))
        h2 = self.bn2(h3)
        h2 = self.conv2(self.nonlin(self.maxunpool2(h2, indices2)))
        x_dec = sig(self.conv1(self.nonlin(h2)))
        return x_dec

# Higher Resolution Autoencoders
    
class EncoderRBNN_HR(nn.Module):
    """
    Encoder class for images of size 256 x 256 pixels.

    ...

    """
    def __init__(self,
                 in_channels,
                 obs_len: int = 1,
                 latent_dim: int  = 6,
                nonlinearity = torch.nn.GELU()) -> None:
        
        super().__init__()
        self.nonlin = nonlinearity
        self.obs_len = obs_len

        # Define layers
        self.conv1 = nn.Conv2d(in_channels= in_channels * self.obs_len, out_channels=16, kernel_size=3) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.bn4 =  nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.bn6 =  nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.maxpool8 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.bn8 =  nn.BatchNorm2d(128)

        self.flatten8 = nn.Flatten(start_dim=1)
        self.linear9 = nn.Linear(in_features=128*10*10, out_features=120)
        self.bn9 = nn.BatchNorm1d(120)
        self.linear10 = nn.Linear(in_features=120, out_features=84)
        self.linear11 = nn.Linear(in_features=84, out_features=latent_dim, bias=False)

    def map_s2s2_so3(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """"""
        z1_norm = z1 / z1.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
        z2_norm = z2 / z2.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
        
        assert torch.linalg.norm(z1_norm - z2_norm) > 1e-1 # we don't want z1/z2 to be aligned
        enc_R = s2s2_gram_schmidt(v1=z1_norm, v2=z2_norm)
        
        return enc_R
    
    def forward(self, x: torch.Tensor):
        # Reshape input to (batch_size, observation length * channels, w, h)
        batch_size, obs_len, channels, w, h = x.shape
        x_ = x.reshape(batch_size, obs_len*channels, w, h)

        # Push input tensor through convolution layers
        h1 = self.nonlin(self.conv1(x_))
        h2, indices2 = self.maxpool2(self.nonlin(self.conv2(h1)))
        h2 = self.bn2(h2)
        h3 = self.nonlin(self.conv3(h2))
        h4, indices4 =  self.maxpool4(self.nonlin(self.conv4(h3)))
        h4 = self.bn4(h4)
        h5 = self.nonlin(self.conv5(h4))
        h6, indices6 = self.maxpool6(self.nonlin(self.conv6(h5)))
        h6 = self.bn6(h6)
        h7 = self.nonlin(self.conv7(h6))
        h8, indices8 = self.maxpool8(self.nonlin(self.conv8(h7)))

        # Push feature map through linear layers
        h8 = self.flatten8(self.bn8(h8))
        h9 = self.bn9(self.nonlin(self.linear9(h8)))
        h10 = self.nonlin(self.linear10(h9))
        x_enc = self.linear11(h10)
        
        # Map from S2 \times S2 to SO(3)
        z1_enc = x_enc[:, :3]
        z2_enc = x_enc[:, 3:]
        x_enc_SO3 = self.map_s2s2_so3(z1=z1_enc, z2=z2_enc)

        # List for maxpooling indices
        indices = dict((f'indices{2*i}', None) for i in range(1, 5))
        indices['indices2'] = indices2
        indices['indices4'] = indices4
        indices['indices6'] = indices6
        indices['indices8'] = indices8

        return x_enc_SO3, indices

class DecoderRBNN_HR(nn.Module):
    def __init__(self,
                  in_channels: int = 6,
                    nonlinearity = torch.nn.GELU()) -> None:
        super().__init__()
        self.nonlin = nonlinearity
        
        self.linear11 = nn.Linear(in_features=in_channels, out_features=84)
        self.linear10 = nn.Linear(in_features=84, out_features=120)
        self.bn9 = nn.BatchNorm1d(120)
        self.linear9 = nn.Linear(in_features=120, out_features=128*10*10)
        self.unflatten8 = nn.Unflatten(dim=1, unflattened_size=(128, 10, 10))
        
        self.bn8 = nn.BatchNorm2d(128)
        self.maxunpool8 = nn.MaxUnpool2d(2, 2)
        self.conv8 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv7 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3)

        self.bn6 = nn.BatchNorm2d(64)
        self.maxunpool6 = nn.MaxUnpool2d(2, 2)
        self.conv6 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3)

        self.bn4 = nn.BatchNorm2d(32)
        self.maxunpool4 = nn.MaxUnpool2d(2, 2)
        self.conv4 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3)

        self.bn2 = nn.BatchNorm2d(16)
        self.maxunpool2 = nn.MaxUnpool2d(2, 2)
        self.conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=5) # increase filter size to make sure output is 224 x 224
        self.conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=5)

    def forward(self, x: torch.Tensor, indices2, indices4, indices6, indices8):
        """"""
        sig = torch.nn.Sigmoid()
        x_ = x[:, :2, ...].reshape(-1, 6)

        # Push through reverse linear layers
        h11 = self.linear11(x_)
        h10 = self.linear10(self.nonlin(h11))
        h9 = self.linear9(self.nonlin(self.bn9(h10)))
        h8_ = self.bn8(self.unflatten8(h9))
        
        # Push through transpose convolution layers
        h8 = self.conv8(self.nonlin(self.maxunpool8(h8_, indices8)))
        h7 = self.conv7(self.nonlin(h8))
        h6_ = self.bn6(h7)

        h6 = self.conv6(self.nonlin(self.maxunpool6(h6_, indices6)))
        h5 = self.conv5(self.nonlin(h6))
        h4_ = self.bn4(h5)

        h4 = self.conv4(self.nonlin(self.maxunpool4(h4_, indices4)))
        h3 = self.conv3(self.nonlin(h4))
        h2_ = self.bn2(h3)

        h2 = self.conv2(self.nonlin(self.maxunpool2(h2_, indices2)))
        x_dec = sig(self.conv1(self.nonlin(h2)))

        return x_dec

