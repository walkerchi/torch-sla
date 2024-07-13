import os 
import warnings
import torch 
from torch.autograd.function import Function
from torch.utils.cpp_extension import load
from typing import Tuple

_spsolve_cpp = load(
    name="spsolve", sources=[
        os.path.abspath(os.path.join(__file__,"..","..","csrc","spsolve","spsolve.cpp"))], 
        verbose=True)

class SparseLinearSolveCG(Function):

    @staticmethod
    def forward(ctx, 
                val:torch.Tensor, 
                row:torch.Tensor, 
                col:torch.Tensor, 
                shape:Tuple[int, int],
                b:torch.Tensor, 
                atol:float,
                maxiter:float):
        u = _spsolve_cpp.cg(torch.stack([row, col],0), val, shape[0], shape[1], b, atol, maxiter)
        ctx.save_for_backward(val, row, col, u)
        ctx.A_shape = shape
        ctx.atol    = atol
        ctx.maxiter = maxiter
        return u

    @staticmethod
    def backward(ctx, gradu):
        val, row, col, u = ctx.saved_tensors
        m, n   = ctx.A_shape
        atol   = ctx.atol
        maxiter= ctx.maxiter
        gradb  = _spsolve_cpp.cg(torch.stack([col, row],0), val, n, m, gradu, atol, maxiter)
        gradval= - gradb[row] * u[col]

        return gradval, None, None, None, gradb, None, None

class SparseLinearSolveBiCGStab(Function):

    @staticmethod
    def forward(ctx, 
                val:torch.Tensor, 
                row:torch.Tensor, 
                col:torch.Tensor, 
                shape:Tuple[int, int],
                b:torch.Tensor, 
                atol:float,
                maxiter:float):
        u = _spsolve_cpp.bicgstab(torch.stack([row, col],0), val, shape[0], shape[1], b, atol, maxiter)
        ctx.save_for_backward(val, row, col, u)
        ctx.A_shape = shape
        ctx.atol    = atol
        ctx.maxiter = maxiter
        return u

    @staticmethod
    def backward(ctx, gradu):
        val, row, col, u = ctx.saved_tensors
        m, n   = ctx.A_shape
        atol   = ctx.atol
        maxiter= ctx.maxiter
        gradb  = _spsolve_cpp.bicgstab(torch.stack([col, row],0), val, n, m, gradu, atol, maxiter)
        gradval= - gradb[row] * u[col]

        return gradval, None, None, None, gradb, None, None



def spsolve(val:torch.Tensor, 
            row:torch.Tensor, 
            col:torch.Tensor, 
            shape:Tuple[int,int], 
            b:torch.Tensor, 
            method:str="bicgstab", 
            atol:float=1e-10, 
            maxiter:int=10000)->torch.Tensor:
    """Solve the Sparse Linear Equation of Pytorch represented in COO format with gradient support

    Only the val and b can receive graident 

    .. math::
        Ax = b


    Parameters
    ----------
    val : torch.Tensor
        [nnz]
    row : torch.Tensor
        [nnz]
    col : torch.Tensor
        [nnz]
    shape : Tuple[int, int]
        (m, n)
    b : torch.Tensor
        [m]
    method : str, optional
        {'cg', 'bicgstab'}, by default "bicgstab"
    atol : float, optional
        , by default 1e-5
    maxiter : int, optional
        , by default 100
    
    Returns
    -------
    torch.Tensor
        [n]
    """

    # assertion 
    assert val.dim() == 1, f"val must be 1D tensor, got {val.dim()}"
    assert row.dim() == 1, f"row must be 1D tensor, got {row.dim()}"
    assert col.dim() == 1, f"col must be 1D tensor, got {col.dim()}"
    assert b.dim() == 1, f"b must be 1D tensor, got {b.dim()}"
    assert shape[0] > 0, f"shape[0] must be positive, got {shape[0]}"
    assert shape[1] > 0, f"shape[1] must be positive, got {shape[1]}"
    assert val.size(0) == row.size(0), f"val and row must have same size, got {val.size(0)} and {row.size(0)}"
    assert val.size(0) == col.size(0), f"val and col must have same size, got {val.size(0)} and {col.size(0)}"
    assert b.size(0) == shape[0], f"b and shape[0] must have same size, got {b.size(0)} and {shape[0]}"
    assert {
        "cg", "bicgstab"
    }.__contains__(method), f"method must be one of 'cg' or 'bicgstab', got {method}"
    assert atol > 0, f"atol must be positive, got {atol}"
    assert maxiter > 0, f"maxiter must be positive, got {maxiter}"
    assert val.dtype == b.dtype, f"val and b must have same dtype, got {val.dtype} and {b.dtype}"
    if val.dtype != torch.float64:
        warnings.warn("You'd better use float64 to maintain good precision")

    return {
        "cg": SparseLinearSolveCG.apply,
        "bicgstab": SparseLinearSolveBiCGStab.apply
    }[method](val, row, col, shape, b, atol, maxiter)
    