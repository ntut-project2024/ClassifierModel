import sys
sys.path.append('.')
from model.BertDecoder.SentiClassifier import SentiClassifier
import torch

testClass = SentiClassifier(4, 2, 2, torch.device('cpu'))

output = testClass.forward(torch.randn(2, 3, 4))
