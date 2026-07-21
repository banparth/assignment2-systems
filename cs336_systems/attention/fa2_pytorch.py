from sympy.core.symbol import Boolean
import torch
import math
from einops import einsum, einops
from typing import Any


class FA2PytorchFunc(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                is_causal: bool = False):
        
        device = Q.device
        # assert Q.ndim == 2, f"Expected 2D Q, got shape {Q.shape}"
        # assert K.ndim == 2, f"Expected 2D K, got shape {K.shape}"
        # assert V.ndim == 2, f"Expected 2D V, got shape {V.shape}"
        Nq = Q.shape[-2]
        Nk = K.shape[-2]
        d = Q.shape[-1]
        # ctx.Q_TILE_SIZE = 16
        # ctx.K_TILE_SIZE = 16
        Bq = 16
        Bk = 16
        
        print(f"Q.shape: {Q.shape} {str(V.dtype)}")
        print(f"Q2.shape: {Q[...,0:Bq,:].shape}")
        Tq = (Nq+Bq-1)//Bq
        Tk = (Nk+Bk-1)//Bk
        
        O = torch.empty(Q.shape, device=device)
        L = torch.empty(Q.shape[:-1], device=device)
        
        for i in range(Tq):
            Oi = torch.zeros(*Q.shape[:-2], Bq, d, device=device)
            li = torch.zeros(*Q.shape[:-2], Bq, device=device)
            mi = torch.full((*Q.shape[:-2], Bq,), -math.inf, device=device)
            
            for j in range(Tk):
                Si = (Q[...,i*(Bq): i*Bq + Bq,:] @ K[...,j*Bk: j*Bk+Bk,:].transpose(-2, -1))/math.sqrt(d)
                mi_new = torch.max(mi, Si.amax(dim=-1))
                Pibar = torch.exp(Si-mi_new.unsqueeze(-1))
                li_new = torch.exp(mi-mi_new)*li + Pibar.sum(-1)
                Vj = V[...,j*Bk: j*Bk+Bk,:]
                Oi_new = torch.exp(mi - mi_new).unsqueeze(-1)* Oi + einsum(Pibar, Vj, "... bq bk, ... bk d -> ... bq d")
                
                mi = mi_new
                li = li_new
                Oi = Oi_new
            
            Oi = Oi/li.unsqueeze(-1)
            Li = mi + torch.log(li)
            
            start = i*Bq
            end = (i+1)*Bq
            
            O[...,start:end, :] = Oi
            L[...,start:end] = Li
            
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, dO: torch.Tensor):
        Q: torch.Tensor
        K: torch.Tensor
        V: torch.Tensor
        L: torch.Tensor
        O: torch.Tensor
        Q, K, V, L, O = ctx.saved_tensors 
        device = Q.device
        d = Q.shape[-1]
        # assert isinstance(Q, torch.Tensor)
        S = einsum(Q, K, "... q d, ... k d -> ... q k")/math.sqrt(d)
        if ctx.is_causal:
            # query[:, None] >= key[None, :]
            query = torch.arange(0, S.shape[-2], device=device)
            key = torch.arange(0, S.shape[-1], device=device)
            print(f"{query.shape=} {key.shape=}")
            mask = query.unsqueeze(0)[..., :, None] >= key.unsqueeze(0)[..., None, :]
            print(mask.shape)
            S = torch.where(
                mask,
                input=S,
                other=-1e6
            )
        P = torch.exp(S - L.unsqueeze(-1))
        
        dV = einsum(P, dO, "... q k, ... q d -> ... k d")
        dP = einsum(dO, V, "... q d, ... k d -> ... q k")
        
        D = torch.sum(O*dO, -1)
        dS = P*(dP - D.unsqueeze(-1))
        
        dQ = einsum(dS, K, "... q k, ... k d -> ... q d")/math.sqrt(d)
        dK = einsum(dS, Q, "... q k, ... q d -> ... k d")/math.sqrt(d)
        return dQ, dK, dV, None
            
            
                
        
        
        
        