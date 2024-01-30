# Author(s): CAB, Justice Mason
# Project: RBNN + Gravity
# Date: 11/09/23

import random
import torch
import numpy as np
import scipy.stats

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

def eazyz_to_group_matrix(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """"""
    assert len(alpha) == len(beta) and len(beta) == len(gamma)
    t1 = alpha
    t2 = beta
    t3 = gamma

    c1 = np.cos(t1)
    c2 = np.cos(t2)
    c3 = np.cos(t3)

    s1 = np.sin(t1)
    s2 = np.sin(t2)
    s3 = np.sin(t3)

    DCM = np.array([[c1*c2*c3 - s1*s3, -c3*s1 - c1*c2*s3, c1*s2],
                    [c1*s3 + c2*c3*s1, c1*c3 - c2*s1*s3, s1*s2],
                    [-c3*s2, s2*s3, c2]])
    
    return DCM

def eazxz_to_group_matrix(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """"""
    assert len(alpha) == len(beta) and len(beta) == len(gamma)
    t1 = alpha
    t2 = beta
    t3 = gamma

    c1 = np.cos(t1)
    c2 = np.cos(t2)
    c3 = np.cos(t3)

    s1 = np.sin(t1)
    s2 = np.sin(t2)
    s3 = np.sin(t3)

    DCM = np.array([[c1*c3 - c2*s1*s3, -c1*s3 - c2*c3*s1, s1*s2],
                    [c3*s1 + c1*c2*s3, c1*c2*c3 - s1*s3, -c1*s2],
                    [s2*s3, s2*c3, c2]])
    
    return DCM

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

def mean_confidence_interval(data, confidence=0.95):
    """Calculates mean and confidence interval from samples such that they lie within m +/- h 
    with the given confidence.

    Args:
        data (np.array): Sample to calculate the confidence interval.
        confidence (float): Confidence of the interval (betwen 0 and 1).
    """
    n = len(data)
    m, se = np.mean(data), scipy.stats.sem(data)
    print(se.shape)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    print(se.shape)
    return m, h

# Function from Lie-VAE (Homeomorphic VAE)
def group_matrix_to_quaternions(r):
    """Map batch of SO(3) matrices to quaternions."""
    batch_dims = r.shape[:-2]
    assert list(r.shape[-2:]) == [3, 3], 'Input must be 3x3 matrices'
    r = r.view(-1, 3, 3)
    n = r.shape[0]

    diags = [r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]]
    denom_pre = torch.stack([
        1 + diags[0] - diags[1] - diags[2],
        1 - diags[0] + diags[1] - diags[2],
        1 - diags[0] - diags[1] + diags[2],
        1 + diags[0] + diags[1] + diags[2]
    ], 1)
    denom = 0.5 * torch.sqrt(1E-6 + torch.abs(denom_pre))

    case0 = torch.stack([
        denom[:, 0],
        (r[:, 0, 1] + r[:, 1, 0]) / (4 * denom[:, 0]),
        (r[:, 0, 2] + r[:, 2, 0]) / (4 * denom[:, 0]),
        (r[:, 1, 2] - r[:, 2, 1]) / (4 * denom[:, 0])
    ], 1)
    case1 = torch.stack([
        (r[:, 0, 1] + r[:, 1, 0]) / (4 * denom[:, 1]),
        denom[:, 1],
        (r[:, 1, 2] + r[:, 2, 1]) / (4 * denom[:, 1]),
        (r[:, 2, 0] - r[:, 0, 2]) / (4 * denom[:, 1])
    ], 1)
    case2 = torch.stack([
        (r[:, 0, 2] + r[:, 2, 0]) / (4 * denom[:, 2]),
        (r[:, 1, 2] + r[:, 2, 1]) / (4 * denom[:, 2]),
        denom[:, 2],
        (r[:, 0, 1] - r[:, 1, 0]) / (4 * denom[:, 2])
    ], 1)
    case3 = torch.stack([
        (r[:, 1, 2] - r[:, 2, 1]) / (4 * denom[:, 3]),
        (r[:, 2, 0] - r[:, 0, 2]) / (4 * denom[:, 3]),
        (r[:, 0, 1] - r[:, 1, 0]) / (4 * denom[:, 3]),
        denom[:, 3]
    ], 1)

    cases = torch.stack([case0, case1, case2, case3], 1)

    quaternions = cases[torch.arange(n, dtype=torch.long),
                        torch.argmax(denom.detach(), 1)]
    return quaternions.view(*batch_dims, 4)

def quaternions_to_eazyz(q):
    """Map batch of quaternion to Euler angles ZYZ. Output is not mod 2pi."""
    batch_dims = q.shape[:-1]
    assert q.shape[-1] == 4, 'Input must be 4 dim vectors'
    q = q.view(-1, 4)

    eps = 1E-6
    return torch.stack([
        torch.atan2(q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3],
                    q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3]),
        torch.acos(torch.clamp(q[:, 3] ** 2 - q[:, 0] ** 2
                               - q[:, 1] ** 2 + q[:, 2] ** 2,
                               -1.0+eps, 1.0-eps)),
        torch.atan2(q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2],
                    q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    ], 1).view(*batch_dims, 3)


def group_matrix_to_eazyz(r):
    """Map batch of SO(3) matrices to Euler angles ZYZ."""
    return quaternions_to_eazyz(group_matrix_to_quaternions(r))