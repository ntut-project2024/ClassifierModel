from typing import Optional

import torch
from torch import nn, Tensor
from einops import repeat
from transformers.modeling_outputs import BaseModelOutput

from utils.const import BlockType
from utils.AttnBlocksConf import AttnBlocksConf
from utils.DevConf import DevConf
from utils.AttnBlocks import AttnBlocks
from module.blocks.CALBlocks import CALBlocks
from module.blocks.CACBlocks import CACBlocks
from module.blocks.CAPBlocks import CAPBlocks


class SentiClassifier(nn.Module):
    def __init__(
            self,
            layerNum: int,
            conf: AttnBlocksConf,
            blockType: str=BlockType.LAST,
            devConf: DevConf=DevConf()
        ):
        super(SentiClassifier, self).__init__()

        if layerNum < 1:
            raise ValueError('layerNum must be greater than 0')
        else:
            self.mapper = MapperFactory(layerNum=layerNum, conf=conf, blockType=blockType, devConf=devConf)
        self.IsNeedHiddenState = not (blockType == BlockType.LAST)
        self._Q = nn.Parameter(torch.randn(1, conf.hidDim, device=devConf.device, dtype=devConf.dtype))
        
        self._layerNum = layerNum
    
    def forward(self,
            input: BaseModelOutput,
            returnAttnWeight: bool=False
        )->tuple[Tensor, Optional[Tensor]]:

        batch = input.last_hidden_state.size(0)
        sentVec, attnWeight = self.mapper.forward(repeat(self._Q, "l d -> b l d", b=batch), input, need_weights=True)
        
        if returnAttnWeight:
            return sentVec.squeeze(1), attnWeight
        return sentVec.squeeze(1)
    
def MapperFactory(
        layerNum: int,
        conf: AttnBlocksConf,
        blockType: str,
        devConf: DevConf,
    )->nn.Module:
    if blockType == BlockType.LAST:
        return CALBlocks(layerNum, conf, devConf)
    elif blockType == BlockType.CROSS:
        return CACBlocks(layerNum, conf, devConf)
    elif blockType == BlockType.PARALLEL:
        return CAPBlocks(layerNum, conf, devConf)
    else:
        raise ValueError('blockType must be either "last", "cross" or "parallel"')
    