import torch
from torch.autograd import Function
from typing import Tuple

from .utils import is_triton_available


if is_triton_available:

    from .triton.spmm import 
    class CsrCscTritonMatmul(Function):
        @staticmethod
        def forward(self, 
                    A_val:torch.Tensor, 
                    A_rowptr:torch.Tensor, 
                    A_col:torch.Tensor,
                    A_shape:Tuple[int, int]):
            from .triton.spmv import spmv_csr
