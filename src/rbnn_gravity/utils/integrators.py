# Name(s): Justice Mason
# Project: RBNN + Gravity
# Date: 10/31/23

import numpy as np
import torch

from utils.math import vee_map, hat_map

# 3D Pendulum LGVI
class LieGroupVaritationalIntegrator():
    """
    
    """
    def __init__(self):
        super().__init__()
        
    def skew(self, v: torch.Tensor):
        
        S = torch.zeros([v.shape[0], 3, 3], device=v.device)
        S[:, 0, 1] = -v[..., 2]
        S[:, 1, 0] = v[..., 2]
        S[:, 0, 2] = v[..., 1]
        S[:, 2, 0] = -v[..., 1]
        S[:, 1, 2] = -v[..., 0]
        S[:, 2, 1] = v[..., 0]
    
        return S
    
    def calc_M(self, R: torch.Tensor, V) -> torch.Tensor:
        """"""
        bs, _, _ = R.shape

        # Calc V(q)
        if not R.requires_grad:
            R.requires_grad = True
        
        q = R.reshape(bs, 9)
        V_q = V(q)

        # Calc gradient 
        dV =  torch.autograd.grad(V_q.sum(), q, create_graph=True)[0]
        dV = dV.reshape(bs, 3, 3)

        # Calc skew(M) and extract M
        SM = torch.bmm(torch.transpose(dV, -2, -1), R) - torch.bmm(torch.transpose(R, -2, -1), dV)
        M = torch.stack((SM[..., 2, 1], SM[..., 0, 2], SM[..., 1, 0]), dim=-1).float()
        return M

    def cayley_transx(self, fc: torch.Tensor):
        """
        """
        F = torch.einsum('bij, bjk -> bik', (torch.eye(3, device=fc.device) + self.skew(fc)), torch.linalg.inv(torch.eye(3, device=fc.device) - self.skew(fc)))
        return F
    
    def calc_fc_init(self, a_vec: torch.Tensor, moi:torch.Tensor) -> torch.Tensor:
        """
        """
        fc_init = torch.einsum('bij, bj -> bi', torch.linalg.inv(2 * moi - self.skew(a_vec)), a_vec)
        return fc_init
    
    def calc_Ac(self, a_vec: torch.Tensor, moi: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        """
        """
        
        Ac = a_vec + torch.einsum('bij, bj -> bi', self.skew(a_vec), fc) + torch.einsum('bj, b -> bj', fc, torch.einsum('bj, bj -> b', a_vec, fc)) - (2 * torch.einsum('ij, bj -> bi', moi, fc))
        return Ac
        
    def calc_grad_Ac(self, a_vec: torch.Tensor, moi: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        """
        """
        grad_Ac = self.skew(a_vec) + torch.einsum('b, bij -> bij', torch.einsum('bi, bi -> b', a_vec, fc), torch.unsqueeze(torch.eye(3, device=a_vec.device), 0).repeat(fc.shape[0], 1, 1)) + torch.einsum('bi, bj -> bij', fc, a_vec) - (2 * moi)
        return grad_Ac
    
    def optimize_fc(self, R_vec: torch.Tensor, pi_vec: torch.Tensor, moi: torch.Tensor, V, fc_list: list = [], timestep: float = 1e-3, max_iter: int = 1000, tol: float = 1e-3) -> list:
        """
        """
        it = 0
        M_vec = self.calc_M(R=R_vec, V=V)

        if not fc_list:
            a_vec = timestep * (pi_vec + (0.5 * timestep) * M_vec)
            fc_list.append(self.calc_fc_init(a_vec=a_vec, moi=moi))
        
        eps = torch.ones(fc_list[-1].shape[0])
        
        while  torch.any(eps > tol) and it < max_iter:
            
            fc_i = fc_list[-1]
            a_vec = timestep * (pi_vec + (0.5 * timestep) * M_vec)
            
            Ac = self.calc_Ac(a_vec=a_vec, moi=moi, fc=fc_i)
            grad_Ac = self.calc_grad_Ac(a_vec=a_vec, moi=moi, fc=fc_i)
           
            fc_ii = fc_i - torch.einsum('bij, bj -> bi', torch.linalg.inv(grad_Ac),  Ac)
            
            eps = torch.linalg.norm(fc_ii - fc_i, axis=-1)
            fc_list.append(fc_ii)
            it += 1
            
        return fc_list
    
    def step(self, R_i: torch.Tensor, pi_i: torch.Tensor, moi: torch.Tensor, V, fc_list: list = [], timestep: float = 1e-3):
        """
        """
        fc_list = self.optimize_fc(R_vec=R_i, pi_vec=pi_i, moi=moi, timestep=timestep, fc_list=fc_list, V=V)
        
        fc_opt = fc_list[-1]
        F_i = self.cayley_transx(fc=fc_opt)
        R_ii = torch.einsum('bij, bjk -> bik', R_i, F_i)
        
        M_i = self.calc_M(R=R_i, V=V)
        M_ii = self.calc_M(R=R_ii, V=V)
        pi_ii = torch.einsum('bji, bj -> bi', F_i, pi_i) + torch.einsum('bji, bj -> bi', 0.5 * timestep * F_i, M_i) + (0.5 * timestep) * M_ii
        
        return R_ii, pi_ii, fc_list
    
    def integrate(self, pi_init: torch.Tensor, R_init: torch.Tensor, moi: torch.Tensor, V, timestep: float = 1e-3, traj_len: int = 100):
        """
        """
        pi_list = [pi_init.float()]
        R_list = [R_init.float()]
        
        for it in range(1, traj_len):
            fc_list = []
            R_i = R_list[-1]
            pi_i = pi_list[-1]
            
            R_ii, pi_ii, fc_list = self.step(R_i=R_i, pi_i=pi_i, moi=moi, V=V, fc_list=fc_list, timestep=timestep)
            
            R_list.append(R_ii)
            pi_list.append(pi_ii)
        
        R_traj = torch.stack(R_list, axis=1)
        pi_traj = torch.stack(pi_list, axis=1)
        return R_traj, pi_traj
    
class Harsh_LGVI():
    def __init__(self):
        super().__init__()

    def step(self, R: torch.Tensor, omega: torch.Tensor, moi: torch.Tensor):
        """"""
        I33 = torch.eye(3, dtype=torch.float32, device=R.device)
        moi_inv = torch.linalg.inv(moi)
        alpha = 0.5

        p = torch.einsum('ij, bj', moi, omega)
        M = self.explicit_update(R)

        newton_step = 5
        a = self.dt * p + self.dt**2 * M * (1 - alpha)
        # print(p.shape, M.shape, a.shape)
        f = torch.zeros_like(a, dtype=torch.float32)
        # import pdb; pdb.set_trace()
        for i in range(newton_step):
            aTf = torch.dot(a, f)
            phi = a + torch.cross(a, f) + f * aTf - 2 * torch.mv(self.I, f)
            # print(phi.shape)
            dphi = hat_map(a, requires_grad=False) + aTf * I33 - 2 * self.I + torch.outer(f, a)
            # print(dphi.shape)
            dphi_inv = torch.inverse(dphi)
            f = f - torch.mv(dphi_inv, phi)
            # print(f.shape)

        # Need to import hat_map and vee_map 
        F = torch.matmul((I33 + hat_map(f)), torch.inverse((I33 - hat_map(f))))
        Ft = torch.transpose(F, 0, 1)
        R_next = torch.matmul(R,F)
        # print(torch.matmul(R_next,torch.transpose(R_next,0,1)))

        # Write out the full explicit update for omega_next using the equation on overleaf
        M_next = self.explicit_update(R_next)
        p_next = torch.mv(Ft, p) + alpha * self.dt * torch.mv(Ft, M) + (1 - alpha) * self.dt * M_next
        omega_next = torch.mv(moi_inv, p_next)

        return R_next, omega_next

    def explicit_update(self, R: torch.Tensor):
        """"""
        q = torch.flatten(R, start_dim=-3) # [bs, 9] 
        V_q = self.V(q) 
        dV =  torch.autograd.grad(V_q.sum(), q, create_graph=True)[0]
        dV = dV.view(-1, 3, 3)
        SM = torch.matmul(torch.transpose(dV, -1, -2), R) - torch.matmul(torch.transpose(R, -1, -2), dV)
        M = torch.stack((SM[:, 2, 1], SM[:, 0, 2], SM[:, 1, 0]), dim=-2)
        
        return M
    
    def integrate(self, R: torch.Tensor, omega: torch.Tensor, moi: torch.Tensor, seq_len: int = 2):
            """"""
            output_R = [R]
            output_omega = [omega]

            for step in range(1, seq_len):
                R_i = output_R[-1]
                omega_i = output_omega[-1]

                R_ii, omega_ii = self.step(R=R_i, omega=omega_i, moi=moi)
                output_R.append(R_ii)
                output_omega.append(omega_ii)

            R_traj = torch.stack(output_R, axis=1)
            omega_traj = torch.stack(output_omega, axis=1) 
            return R_traj, omega_traj
    
