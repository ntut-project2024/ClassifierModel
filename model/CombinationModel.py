from typing import Optional

import torch
from torch import nn, Tensor
from transformers import DistilBertModel, BatchEncoding

from utils.DevConf import DevConf
from model.BertDecoder.SentiClassifier import SentiClassifier

class CombinationModel(nn.Module):
    def __init__(self,
            nClass: int,
            decoder: nn.Module,
            distilBert: DistilBertModel=None,
            devConf: DevConf = DevConf()
        ):
        
        super(CombinationModel, self).__init__()
        if distilBert is None:
            self.distilBert = DistilBertModel.from_pretrained('distilbert/distilbert-base-multilingual-cased', cache_dir="./cache/model").to(devConf.device).to(devConf.dtype)
        else:
            self.distilBert = distilBert.to(devConf.device).to(devConf.dtype)

        self.decoder = decoder.to(devConf.device).to(devConf.dtype)
        self.outProj = nn.Linear(768, nClass, device=devConf.device, dtype=devConf.dtype)
        self.activate = nn.Sigmoid().to(devConf.device).to(devConf.dtype)
    
    def forward(self,
            input_ids: Tensor,
            attention_mask: Tensor,
            NoGradBert: bool=True,
            NoGradDecoder: bool=False,
            returnAttnWeight: bool=False
        )->tuple[Tensor, Tensor] | Tensor:

        if NoGradBert:
            with torch.no_grad():
                output = self._getBertOutput(input_ids=input_ids, attention_mask=attention_mask)
        else:
            output = self._getBertOutput(input_ids=input_ids, attention_mask=attention_mask)

        if NoGradDecoder:
            with torch.no_grad():
                output, attnWeig = self.decoder(output, returnAttnWeight=True)
        else:
            output, attnWeig = self.decoder(output, returnAttnWeight=True)

        if returnAttnWeight:
            return self.activate(self.outProj(output)), attnWeig
        return self.activate(self.outProj(output))
    
    def _getBertOutput(self, input_ids: Tensor, attention_mask: Tensor)->BatchEncoding:
        return self.distilBert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states= self.decoder.IsNeedHiddenState)
