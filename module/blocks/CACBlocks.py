from typing import Optional

from torch import nn, Tensor
from transformers.modeling_outputs import BaseModelOutput

from utils.AttnBlocksConf import AttnBlocksConf
from utils.DevConf import DevConf
from utils.AttnBlocks import AttnBlocks

class CACBlocks(AttnBlocks):
    def __init__(self,
            layerNum: int,
            conf: AttnBlocksConf,
            devConf: DevConf = DevConf()
        )->None:
        super(CACBlocks, self).__init__(
            layerNum=layerNum,
            conf=conf,
            devConf=devConf
        )

    def forward(
            self,
            query: Tensor,
            kv: BaseModelOutput,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False
        )->tuple[Tensor, Optional[Tensor]]:

        kvs = [
            tensor
            for tensor in kv.hidden_states]
        kvLen = len(kvs) - 1
        if kvLen < self._layerNum:
            print(f'Warning: kvLen: {kvLen} < layerNum, the rest of the layers will operate as self-attention')
        tensor = kvs[kvLen]
        query, attnWeight = self._mha[0](
                query=query,
                key=tensor,
                value=tensor,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        
        for i in range(1, self._layerNum):
            if i > kvLen:
                tensor = query
            else:
                tensor = kvs[kvLen - i]
            query, _ = self._mha[i].forward(
                query=query,
                key=tensor,
                value=tensor,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)

        return query, attnWeight