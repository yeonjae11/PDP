# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Qwen model implementation for torchtitan

from dataclasses import dataclass
from typing import Optional, Tuple
import contextlib

import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import record_function
from torchtitan.protocols.train_spec import BaseModelArgs, ModelProtocol
from torchtitan.config_manager import JobConfig
from torchtitan.models.attention import build_attention, init_attention_mask
from torchtitan.components.tokenizer import Tokenizer


@dataclass
class TransformerModelArgs(BaseModelArgs):
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    intermediate_size: int = 27648  # If None, computed as 4*dim
    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    initializer_range: float = 0.02  # Standard deviation for weight initialization

    max_seq_len: int = 2048
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True
    # Qwen specific params
    use_sliding_window: bool = False
    sliding_window: Optional[int] = 4096
    max_window_layers: int = 0

    def update_from_config(self, job_config: JobConfig, tokenizer: Tokenizer) -> None:
        self.vocab_size = tokenizer.n_words
        self.max_seq_len = job_config.training.seq_len
        self.eos_id = tokenizer.eos_id

        if job_config.activation_checkpoint.mode == "selective" and self.use_flex_attn:
            raise ValueError(
                "FlexAttention is not compatible with selective AC yet. "
                "See https://github.com/pytorch/pytorch/issues/147879"
            )

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise ValueError(
                "FlexAttention is not compatible with CP yet. "
                "We are still working on this."
            )

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )

        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.dim // self.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

        return nparams, num_flops_per_token



def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, model_args: TransformerModelArgs, layer_idx: int):
        super().__init__()
        self.model_args = model_args
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads if model_args.n_kv_heads is not None else model_args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = nn.Linear(model_args.dim, model_args.n_heads * self.head_dim, bias=True)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=True)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=True)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.dim, bias=False)

        self.layer_idx = layer_idx
        
        self.is_causal = True
        
        self.use_sliding_window = (model_args.use_sliding_window and 
                                model_args.sliding_window is not None and 
                                layer_idx >= model_args.max_window_layers)
        self.sliding_window = model_args.sliding_window if self.use_sliding_window else None

    def init_weights(self, init_std: float):
        for linear in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        bs, seqlen, _ = x.shape
        
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)
        
        # Apply rotary position embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # Repeat key and value states if needed
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)
        
        # Transpose for attention computation
        xq = xq.transpose(1, 2)  # (batch_size, n_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (batch_size, n_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (batch_size, n_heads, seqlen, head_dim)
        
        # Sliding window attention mask 적용 (필요한 경우)
        attention_mask = None
        if self.use_sliding_window and attention_mask is None:
            # sliding_window 크기에 맞는 마스크 생성
            # 시퀀스 길이가 sliding_window보다 클 경우에만 적용
            if seqlen > self.sliding_window:
                # 토큰 위치에 따른 마스크 생성
                rows = torch.arange(seqlen, device=x.device)
                cols = torch.arange(seqlen, device=x.device)
                
                # sliding window 범위 밖의 토큰에 대해 마스킹 (큰 음수값)
                mask = (rows[:, None] - cols[None, :]) > self.sliding_window
                mask = mask.to(torch.float32) * -10000.0
                # [batch_size, 1, seq_len, seq_len] 형태로 확장
                attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(bs, 1, -1, -1)
        
        # scaled_dot_product_attention 사용
        # 입력: (batch_size, num_heads, seq_len, head_dim)
        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=attention_mask,
            # sliding window 사용 시 is_causal=False로 설정
            is_causal=not self.use_sliding_window and self.is_causal and attention_mask is None
        )
        
        # 출력 형태 변환 (Llama 방식으로 일관성 유지)
        output = output.transpose(1, 2).contiguous()  # (batch_size, seqlen, n_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
        self, 
        dim: int,
        intermediate_size: int
        ):
        super().__init__()
        
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in [self.w2, self.w3]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: TransformerModelArgs):
        super().__init__()
        self.dim = model_args.dim
        self.attention = Attention(model_args=model_args, layer_idx=layer_id)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            intermediate_size=model_args.intermediate_size
        )
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
    
        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

    def forward(self, x, freqs_cis):
        attn_norm = self.attention_norm(x)
        with record_function(f"attention_layer_{self.layer_id}"):
            h = x + self.attention(attn_norm, freqs_cis)
        ffn_norm = self.ffn_norm(h)
        with record_function(f"feed_forward_{self.layer_id}"):
            out = h + self.feed_forward(ffn_norm)
        return out
    
    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Transformer(nn.Module):
    def __init__(self, model_args: TransformerModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        
        # Register freqs_cis buffer
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)
        
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)
        
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        
        self.init_weights()
        
    def init_weights(self, buffer_device: Optional[torch.device] = None):
        """
        Initialize model weights and buffers.
        
        Args:
            buffer_device (Optional[torch.device]): Device to store buffers.
        """
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
                self.freqs_cis = self._precompute_freqs_cis()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        # Initialize layers
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
    
        if self.norm is not None:
            self.norm.reset_parameters()
    
        # Initialize output projection
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0, 
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )
            
    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )
            
    def forward(self, tokens: torch.Tensor):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.
        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output
    
    @classmethod
    def from_model_args(cls, model_args: TransformerModelArgs):
        return cls(model_args)
