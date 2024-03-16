import sys
sys.path.append('.')
from model.blocks.CALBlocks import CALBlocks

import torch
from torch import Tensor
testClass = CALBlocks(4, 2, 2)

output, weight = testClass.forward(torch.randn(2, 3, 4), torch.randn(2, 3, 4), torch.randn(2, 3, 4), need_weights=True)

if not isinstance(output, Tensor):
    raise ValueError(f'forward() test failed, output type is not correct {type(output)} {output}')

if output.shape != torch.Size([2, 3, 4]):
    raise ValueError(f'forward() test failed, output size is not correct {output.shape}')

if not isinstance(weight, Tensor):
    raise ValueError(f'forward() test failed, weight type is not correct {type(weight)} {weight}')

if weight.shape != torch.Size([2, 3, 3]):
    raise ValueError(f'forward() test failed, weight size is not correct {weight.shape}')