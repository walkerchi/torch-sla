import torch
from torch.autograd import Function
from .utils import is_triton_available


class SparseMatricVectorMulCOOTriton(Function):
    @staticmethod
    def forward(ctx, val, row, col, shape, b):
        if not is_triton_available:
            raise RuntimeError("Triton is not available")
        from .triton.spmv import spmv_coo
        o = spmv_coo(val, row, col, shape, b)
        if torch.is_grad_enabled():
            ctx.save_for_backward(val, row, col, b)
            ctx.shape = shape 
        return o

    @staticmethod 
    def backward(ctx, grado):
        if not torch.is_grad_enabled():
            raise RuntimeError("SparseMatricVectorMulCOOTriton: Backward called in no_grad or inference_mode context")
        val, row, col, b = ctx.saved_tensors
        shape = ctx.shape
        if not is_triton_available:
            raise RuntimeError("Triton is not available")
        from .triton.spmv import spmv_coo
        gradb   = spmv_coo(val, col, row, (shape[1], shape[0]), grado)
        gradval = b[col] * grado[row]
        return gradval, None, None, None, gradb

class SparseMatricVectorMulCSRTriton(Function):
    @staticmethod
    def forward(ctx, val, rowptr, col, shape, b):
        if not is_triton_available:
            raise RuntimeError("Triton is not available")
        from .triton.spmv import spmv_csr
        o = spmv_csr(val, rowptr, col, shape, b)
        if torch.is_grad_enabled():
            ctx.save_for_backward(val, rowptr, col, b)
            ctx.shape = shape 
        return o

    @staticmethod 
    def backward(ctx, grado):
        if not torch.is_grad_enabled():
            raise RuntimeError("SparseMatricVectorMulCOOTriton: Backward called in no_grad or inference_mode context")
        val, row, col, b = ctx.saved_tensors
        shape = ctx.shape
        if not is_triton_available:
            raise RuntimeError("Triton is not available")
        from .triton.spmv import spmv_coo
        gradb   = spmv_coo(val, col, row, (shape[1], shape[0]), grado)
        gradval = b[col] * grado[row]
        return gradval, None, None, None, gradb