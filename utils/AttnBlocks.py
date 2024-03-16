from typing import Optional

from torch import nn, Tensor
from transformers.modeling_outputs import BaseModelOutput

from utils.AttnBlocksConf import AttnBlocksConf
from utils.DevConf import DevConf
from utils.MHABlock import MHABlock

class AttnBlocks(nn.Module):
    def __init__(self, 
            conf: AttnBlocksConf,
            devConf: DevConf = DevConf()
        )->None:
        super(AttnBlocks, self).__init__()

        self._mha = [MHABlock(
            conf.hidDim,
            conf.nHead,
            batch_first=True,
            device=devConf.device,
            dtype=devConf.dtype,
        ) for _ in range(conf.layerNum)]

        self._layerNum = conf.layerNum
        self._devConf = devConf

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

        raise NotImplementedError