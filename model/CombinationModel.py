from typing import Optional

import torch
from torch import nn, Tensor
from transformers import DistilBertModel, BatchEncoding

from utils.DevConf import DevConf
from model.BertDecoder.SentiClassifier import SentiClassifier

class CombinationModel(nn.Module):
    def __init__(self,
            distilBert: DistilBertModel,
            decoder: nn.Module,
            outputProject: nn.Linear
        ):
        
        super(CombinationModel, self).__init__()
        self.distilBert = distilBert
        self.decoder = decoder
        self.outProj = outputProject
        self.softmax = nn.Softmax(dim=1)
    
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
            return self.softmax(self.outProj(output)), attnWeig
        return self.softmax(self.outProj(output))
    
    def _getBertOutput(self, input_ids: Tensor, attention_mask: Tensor)->tuple[Tensor, Optional[Tensor]]:
        return self.distilBert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states= self.decoder.IsNeedHiddenState)
