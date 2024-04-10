from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(
            self,
            hidDim: int,
            nHead: int,
            headDim: int = None,
            nKVHead: int = 1,
            # batch_first: bool = True,
            device: str = 'cpu',
            dtype: torch.dtype = torch.float32) -> None:
        super(Attention, self).__init__()

        assert nHead % nKVHead == 0
        self.nHead = nHead
        self.nKVHead = nKVHead
        self.nQPerKvHead = nHead // nKVHead

        self.headDim = hidDim // nHead if headDim is None else headDim
        self.hidDim = hidDim

        self.scale = headDim ** -0.5

        # self.batch_first = batch_first
        
        self.qProj = nn.Linear(
            hidDim,
            nHead*headDim,
            device=device,
            dtype=dtype)

        self.kvProj = nn.Linear(
            hidDim,
            (2*nKVHead) * headDim,
            device=device,
            dtype=dtype)
        
        self.outProj = nn.Linear(
            nHead*headDim,
            hidDim,
            device=device,
            dtype=dtype)

    def forward(
        self,
        query: Tensor,
        kv: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        # need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        # average_attn_weights: bool = True,
        # is_causal : bool = False
    )->Tensor:
        
        assert query.dim() == 3
        assert kv.dim() == 3
        batch, len, _ = query.shape

        q: Tensor = self.qProj(query)
        kv = self.kvProj(kv)
        k, v = torch.chunk(kv, 2, dim=-1)

        q = q.view(batch, -1, self.nHead, self.headDim).transpose(1, 2)
        k = k.view(batch, -1, self.nKVHead, self.nQPerKvHead, self.headDim).transpose(1, 2)
        v = v.view(batch, -1, self.nKVHead, self.nQPerKvHead, self.headDim).transpose(1, 2)

        attn_output = torch.matmul(q, k.transpose(2, 3)) / (self.headDim ** 0.5)

        if attn_mask is not None:
            attn_output += attn_mask

        if key_padding_mask is not None:
            attn_output = attn_output.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9)

        attn_output = F.softmax(attn_output, dim=-1)

        attn_output = torch.matmul(attn_output, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch, len, -1)
        output = self.outProj(attn_output)

        return output