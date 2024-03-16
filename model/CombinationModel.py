from typing import Optional

import torch
from torch import nn, Tensor
from transformers import DistilBertModel, AutoTokenizer

from utils.DevConf import DevConf
from model.BertDecoder.SentiClassifier import SentiClassifier

class CombinationModel(nn.Module):
    def __init__(self,
            tokenizer: AutoTokenizer,
            distilBert: DistilBertModel,
            decoder: nn.Module,
            outputProject: nn.Linear
        ):
        
        super(CombinationModel, self).__init__()
        self.tokenizer = tokenizer
        self.distilBert = distilBert
        self.decoder = decoder
        self.outProj = outputProject
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,
            input: str,
            NoGradBert: bool=True,
            NoGradDecoder: bool=False,
            returnAttnWeight: bool=False
        )->tuple[Tensor, Tensor] | Tensor:

        if NoGradBert:
            with torch.no_grad():
                output = self._getBertOutput(input, )
        else:
            output = self._getBertOutput(input)

        if NoGradDecoder:
            with torch.no_grad():
                output, attnWeig = self.decoder(output, returnAttnWeight=True)
        else:
            output, attnWeig = self.decoder(output, returnAttnWeight=True)

        if returnAttnWeight:
            return self.softmax(self.outProj(output)), attnWeig
        return self.softmax(self.outProj(output))
    
    def _getBertOutput(self, input: str)->tuple[Tensor, Optional[Tensor]]:
        seqTensor = self.tokenizer(input, return_tensors='pt', padding=True, truncation=True)
        return self.distilBert(**seqTensor, output_hidden_states= self.decoder.IsNeedHiddenState)
