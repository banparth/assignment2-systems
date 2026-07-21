from einops import einsum
import triton
import torch
import triton.language as tl
from typing import Any
import math

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr = False,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index*stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index*stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index*stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0,0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index*stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index*Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    query = query_tile_index*Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    
    
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE, ), -math.inf, tl.float32)
    
    qi = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
    
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        
        key = j*K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        kj = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
        vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        sij = tl.dot(qi, tl.trans(kj)) * scale
        
        if is_causal:
            mask = query[:, None] >= key[None, :]
            sij = tl.where(mask, sij, -1e6)
        
        mi_new = tl.maximum(mi, tl.max(sij, -1))
        pi_bar = tl.exp(sij - mi_new[:, None])
        li_new = tl.exp(mi - mi_new)*li + tl.sum(pi_bar, -1)
        
        # Oi_new = tl.exp(mi - mi_new)[:, None]*Oi + tl.dot(pi_bar, vj)
        
        Oi = tl.exp(mi-mi_new)[:, None]*Oi
        Oi = tl.dot(pi_bar, vj, acc=Oi)
        
    
        
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        li = li_new
        mi = mi_new
        
    Oi = Oi / li[:, None]
    Li = mi + tl.log(li)
    
    tl.store(O_block_ptr,Oi, boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))
    
    
        
    
    
    

class FA2TritonFunc(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                is_causal: bool = False):
        
        device = Q.device
        assert Q.ndim == 3, f"Expected 3D Q, got shape {Q.shape}"
        assert K.ndim == 3, f"Expected 3D K, got shape {K.shape}"
        assert V.ndim == 3, f"Expected 3D V, got shape {V.shape}"
        Nq = Q.shape[-2]
        Nk = K.shape[-2]
        d = Q.shape[-1]
        batch_size = Q.shape[0]
        # ctx.Q_TILE_SIZE = 16
        # ctx.K_TILE_SIZE = 16
        Bq = 16
        Bk = 16
        scale = 1 / math.sqrt(d)        
        Tq = (Nq+Bq-1)//Bq
        Tk = (Nk+Bk-1)//Bk
        
        O = torch.empty(Q.shape, device=device)
        L = torch.empty(Q.shape[:-1], device=device)

        
        flash_fwd_kernel[(Tq, batch_size)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk, scale,
            d, Bq, Bk,
            is_causal
        )
            
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
            
            
                
        
        
        
        