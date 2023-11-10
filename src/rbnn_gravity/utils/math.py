# Author(s): CAB, Justice Mason
# Project: RBNN + Gravity
# Date: 11/09/23

import random
import torch
import numpy as np

# Math Utility Functions
def vee_map(R, device = None, mode = "torch", requires_grad=True):
    """
    Performs the vee mapping from a rotation matrix to a vector
    """
    if mode == "torch":
        a_hat = torch.tensor([-R[1, 2], R[0, 2], -R[0, 1]], device=device, dtype=torch.float32, requires_grad=requires_grad)
    else:
        arr_out = np.zeros(3)
        arr_out[0] = -R[1, 2]
        arr_out[1] = R[0, 2]
        arr_out[2] = -R[0, 1]
    return arr_out


def hat_map(a, device = None, mode = "torch", requires_grad=True):
    if mode == "torch":
        a_hat = torch.tensor([[0, -a[2], a[1]],
                          [a[2], 0, -a[0]],
                          [-a[1], a[0], 0]], device=device, dtype=torch.float32, requires_grad=requires_grad)
    else:
        a_hat = np.array([[0, -a[2], a[1]],
                      [a[2], 0, -a[0]],
                      [-a[1], a[0], 0]])
    return a_hat

def s2s2_gram_schmidt(v1, v2):
    """Normalise 2 3-vectors. Project second to orthogonal component.
    Take cross product for third. Stack to form SO matrix."""
    # assert torch.any(v1.isnan()) == False and torch.any(v1.isinf()) == False
    # assert torch.any(v2.isnan()) == False and torch.any(v2.isinf()) == False
    u1 = v1
    e1 = u1 / u1.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / u2.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    
    e3 = torch.cross(e1, e2)
    return torch.stack([e1, e2, e3], 1)

def pd_matrix(diag: torch.Tensor, off_diag: torch.Tensor) -> torch.Tensor:
    """
    Function constructing postive-definite matrix from diag/off-diag entries.
    
    ...
    
    Parameters
    ----------
    diag : torch.Tensor
        Diagonal elements of PD matrix.
        
    off-diag: torch.Tensor
        Off-diagonal elements of PD matrix.
        
    Returns
    -------
    matrix_pd : torch.Tensor
        Calculated PD matrix.
        
    Notes
    -----
    
    """
    diag_dim = diag.shape[0]
    
    L = torch.diag_embed(diag)
    ind = np.tril_indices(diag_dim, k=-1)
    flat_ind  = np.ravel_multi_index(ind, (diag_dim, diag_dim))
    
    L = torch.flatten(L, start_dim=0)
    L[flat_ind] = off_diag
    L = torch.reshape(L, (diag_dim, diag_dim))
    
    matrix_pd = L @ L.T + (0.001 * torch.eye(3, device=diag.device))
    
    return matrix_pd
