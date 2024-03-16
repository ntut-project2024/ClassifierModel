from typing import Optional

import torch
from torch import nn, Tensor

from utils.DevConf import DevConf

class MHABlock(nn.Module):
    def __init__(self, 
            hidDim: int,
            nHead: int,
            batch_first: bool = True,
            device: str = 'cpu',
            dtype: torch.dtype = torch.float32
        )->None:
        super(MHABlock, self).__init__()

        self._mha = nn.MultiheadAttention(
            hidDim,
            nHead,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

        self.outProj = nn.Linear(hidDim, hidDim, device=device, dtype=dtype)
        self.norm = nn.LayerNorm(hidDim, device=device, dtype=dtype)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False
        )->tuple[Tensor, Optional[Tensor]]:

        query, attnWeight = self._mha(
                query=query,
                key=key,
                value=value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        
        query = self.outProj(query)
        query = self.norm(query)

        return query, attnWeight