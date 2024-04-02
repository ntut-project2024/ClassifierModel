import sys
sys.path.append('.')
import unittest

import torch
from transformers.modeling_outputs import BaseModelOutput

from model.BertDecoder.SentiClassifier import SentiClassifier
from utils.AttnBlocksConf import AttnBlocksConf

class SentiClassifierTesting(unittest.TestCase):
    def test_Forward(self):
        testClass = SentiClassifier(AttnBlocksConf(4, 2, 2))
        input = BaseModelOutput(last_hidden_state=torch.randn(2, 3, 4))
        output = testClass.forward(input)
        self.assertEqual(output.shape, torch.Size([2, 4]))
    
    def test_Init(self):
        testClass = SentiClassifier(AttnBlocksConf(4, 2, 1))
        self.assertEqual(testClass._layerNum, 1)
        self.assertEqual(testClass._Q.shape, torch.Size([1, 4]))
        output = testClass.forward(BaseModelOutput(last_hidden_state=torch.randn(2, 3, 4)))
        self.assertEqual(output.shape, torch.Size([2, 4]))

if __name__ == '__main__':
    unittest.main()