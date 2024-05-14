from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from utils.AttnBlocksConf import AttnBlocksConf

class Attention(nn.Module):
    def __init__(
            self,
            # hidDim: int,
            # nHead: int,
            # headDim: int = None,
            # nKVHead: int = None,
            attnConf: AttnBlocksConf = None,
            # batch_first: bool = True,
            device: str = 'cpu',
            dtype: torch.dtype = torch.float32
    ) -> None:
        super(Attention, self).__init__()

        self._SetVariables(attnConf)

        # self.batch_first = batch_first
        
        self.qProj = nn.Linear(
            self.hidDim,
            self.nHead * self.headDim,
            device=device,
            dtype=dtype)

        self.kvProj = nn.Linear(
            self.hidDim,
            (2*self.nKVHead) * self.headDim,
            device=device,
            dtype=dtype)
        
        self.outProj = nn.Linear(
            self.nHead * self.headDim,
            self.hidDim,
            device=device,
            dtype=dtype)

    def _SetVariables(self, attnConf: AttnBlocksConf) -> None:
        self.hidDim = attnConf.hidDim
        self.nHead = attnConf.nHead
        self.headDim = attnConf.headDim
        self.nKVHead = attnConf.nKVHead
        if attnConf.nQPerKvHead is not None:
            self.nQPerKvHead = attnConf.nQPerKvHead
        self.scale = attnConf.scale

    def forward(
        self,
        query: Tensor,
        kv: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        # average_attn_weights: bool = True,
        # is_causal : bool = False
    ) -> tuple[Tensor, Optional[Tensor]]:
        assert query.dim() == 3
        assert kv.dim() == 3
        batch, len, _ = query.shape

        q: Tensor = self.qProj(query)
        kv = self.kvProj(kv)
        k, v = torch.chunk(kv, 2, dim=-1)

        q, k, v = self._TransformQKV(q, k, v, batch)

        qk = torch.matmul(q, k.transpose(2, 3)) * self.scale

        if attn_mask is not None:
            qk += attn_mask

        if key_padding_mask is not None:
            qk = qk.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9)

        qk = F.softmax(qk.float(), dim=-1).type_as(q)

        attn_output = torch.matmul(qk, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch, len, -1)
        output = self.outProj(attn_output)

        return output, qk if need_weights else None
    
    def _TransformQKV(self, q: Tensor, k: Tensor, v: Tensor, batch: int) -> tuple[Tensor, Tensor, Tensor]:
        q = q.view(batch, -1, self.nHead, self.headDim)
        k = k.view(batch, -1, self.nKVHead, self.headDim)
        v = v.view(batch, -1, self.nKVHead, self.headDim)

        if self.nKVHead != self.nHead:
            k = k.repeat_interleave(self.nQPerKvHead, dim=2)
            v = v.repeat_interleave(self.nQPerKvHead, dim=2)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v