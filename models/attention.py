"""
This file contains an implementation of a MultiheadAttention layer.

Explicit differences from nn.MultiheadAttention:


- No need for key_padding_mask with nested tensors :)
- add_bias_kv is not supported: this argument creates a trainable bias parameters
  and cats it to the key and value inputs at dim 0. Supporting this is left as an
  exercise for the reader.
- add_zero_attn is not supported: this argument adds a batch of zeros to the key
  and value inputs at dim 1. Supporting this is left as an exercise for the reader.
- Only supports batch_first=True: nested tensors do not support seq_len as the
  first dimension
- Does not support need_weights=True or average_attn_weights: These do not work
  with scaled_dot_product_attention which is required for performance
- unnecessary fast path logic is removed

Modified from: https://github.com/mikaylagawarecki/transformer_tutorial_accompaniment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // n_heads
        n_heads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        n_heads: int,
        dropout: float = 0.0,
        is_causal: bool = False,
        bias: bool = False,
        qk_norm: type[nn.Module] | None = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.is_causal = is_causal
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(E_total, E_q, bias=bias, **factory_kwargs)
        assert E_total % n_heads == 0, "Embedding dim is not divisible by n_heads"
        self.E_head = E_total // n_heads
        self.bias = bias

        if qk_norm is not None:
            self.q_norm = qk_norm(self.E_head)
            self.k_norm = qk_norm(self.E_head)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (N, L_q, E_qk)
            key (torch.Tensor): key of shape (N, L_kv, E_qk)
            value (torch.Tensor): value of shape (N, L_kv, E_v)
            attn_mask (torch.Tensor, optional): attention mask of shape (N, L_q, L_kv) to pass to sdpa. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, n_heads, E_head) -> (N, n_heads, L_t, E_head)
        query = query.unflatten(-1, [self.n_heads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, n_heads, E_head) -> (N, n_heads, L_s, E_head)
        key = key.unflatten(-1, [self.n_heads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, n_heads, E_head) -> (N, n_heads, L_s, E_head)
        value = value.unflatten(-1, [self.n_heads, self.E_head]).transpose(1, 2)

        query = self.q_norm(query)
        key = self.k_norm(key)

        # Step 3. Run SDPA
        # (N, n_heads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=self.is_causal,
        )
        # (N, n_heads, L_t, E_head) -> (N, L_t, n_heads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output


class MultiHeadSelfAttention(nn.Module):
    """
    Computes multi-head self attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // n_heads
        n_heads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: False
        is_causal (bool, optional): Whether to apply causal mask. Default: False
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        is_causal: bool = False,
        bias: bool = False,
        qk_norm: type[nn.Module] | None = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.is_causal = is_causal
        self.packed_proj = nn.Linear(
            embed_dim, embed_dim * 3, bias=bias, **factory_kwargs
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        assert embed_dim % n_heads == 0, "Embedding dim is not divisible by n_heads"
        self.E_head = embed_dim // n_heads
        self.bias = bias

        if qk_norm is not None:
            self.q_norm = qk_norm(self.E_head)
            self.k_norm = qk_norm(self.E_head)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(
        self,
        input: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (N, L_q, E_qk)
            key (torch.Tensor): key of shape (N, L_kv, E_qk)
            value (torch.Tensor): value of shape (N, L_kv, E_v)
            attn_mask (torch.Tensor, optional): attention mask of shape (N, L_q, L_kv) to pass to sdpa. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        result = self.packed_proj(input)
        query, key, value = torch.chunk(result, 3, dim=-1)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, n_heads, E_head) -> (N, n_heads, L_t, E_head)
        query = query.unflatten(-1, [self.n_heads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, n_heads, E_head) -> (N, n_heads, L_s, E_head)
        key = key.unflatten(-1, [self.n_heads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, n_heads, E_head) -> (N, n_heads, L_s, E_head)
        value = value.unflatten(-1, [self.n_heads, self.E_head]).transpose(1, 2)

        query = self.q_norm(query)
        key = self.k_norm(key)

        # Step 3. Run SDPA
        # (N, n_heads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=self.is_causal,
        )
        # (N, n_heads, L_t, E_head) -> (N, L_t, n_heads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output
