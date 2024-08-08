import torch
import triton 
import triton.language as tl 
from typing import Tuple

@triton.jit
def _spmv_coo_kernel(
    val, row, col, b, o,
    nnz: int, BLK: tl.constexpr
    ):
    idx = tl.program_id(axis=0)
    
    offsets = idx * BLK + tl.arange(0, BLK) # [BLK]
    
    mask = offsets < nnz # [BLK]
    offsets = tl.where(mask, offsets, 0) # [BLK]
     
    row_block = tl.load(row + offsets, mask=mask) # [BLK]
    col_block = tl.load(col + offsets, mask=mask) # [BLK]
    val_block = tl.load(val + offsets, mask=mask) # [BLK]
    
    b_block   = tl.load(b + col_block, mask=mask) # [BLK] 

    o_block   = val_block * b_block
    
    tl.atomic_add(o + row_block, o_block, mask=mask) # [BLK]


@triton.jit 
def _spmv_csr_kernel(
    val, rowptr, col, b, o,
    BLK: tl.constexpr
    ):
    row_idx = tl.program_id(axis=0)
    
    start = tl.load(rowptr + row_idx) # scalar
    end   = tl.load(rowptr + row_idx + 1) # scalar
    
    result = 0.0
    for col_block_idx in tl.cdiv(end-start, BLK):
        col_idx = tl.arange(0, BLK) + start + col_block_idx * BLK
        mask = col_idx < end # [BLK]
        col_block = tl.load(col + col_idx, mask=mask) # [BLK]
        val_block = tl.load(val + col_idx, mask=mask) # [BLK]
        b_block   = tl.load(b + col_block, mask=mask) # [BLK]
        result += tl.sum(val_block * b_block)

    tl.store(o + row_idx, result)

@triton.jit
def _spmv_csc_kernel(
    val, row, colptr, b, o,
    BLK:tl.constexpr
    ):
    col_idx = tl.program_id(axis=0)

    start = tl.load(colptr + col_idx) # scalar
    end   = tl.load(colptr + col_idx + 1) # scalar

    result = 0.0
    for row_block_idx in tl.cdiv(end-start, BLK):
        row_idx = tl.arange(0, BLK) + start + row_block_idx * BLK
        mask = row_idx < end # [BLK]
        col_block = tl.load(col + col_idx, mask=mask) # [BLK]
        val_block = tl.load(val + col_idx, mask=mask) # [BLK]
        b_block   = tl.load(b + col_block, mask=mask) # [BLK]
        result += tl.sum(val_block * b_block)

def spmv_coo(
        val:torch.Tensor, 
        row:torch.Tensor, 
        col:torch.Tensor, 
        shape:Tuple[int, int],
        b:torch.Tensor):
    """
    Parameters
    ----------
    val : torch.Tensor
        1D [nnz] Values of the matrix
    row : torch.Tensor
        1D [nnz] Row indices of the matrix
    col : torch.Tensor
        1D [nnz] Column indices of the matrix
    shape : Tuple[int, int]
        (m,n) Shape of the matrix
    b : torch.Tensor
        1D [n] Vector to multiply with

    Returns
    -------
    torch.Tensor
        1D [n] Result of the matrix-vector multiplication
    """
    # assertion 
    assert val.ndim == 1, "val must be 1D"
    assert row.ndim == 1, "row must be 1D"
    assert col.ndim == 1, "col must be 1D"
    assert b.ndim == 1, "b must be 1D"
    assert val.shape[0] == row.shape[0], "val and row must have the same length"
    assert val.shape[0] == col.shape[0], "val and col must have the same length"
    assert shape[0] == b.shape[0], "shape[0] and b must have the same length"

    output = torch.zeros(shape[0], device=val.device, dtype=val.dtype)
    nnz    = val.shape[0]
    BLOCK_SIZE = 1024
    grid = (nnz + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    _spmv_coo_kernel[grid](
        val, row, col, b, output,
        nnz=nnz, BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def spmv_csr(
        val:torch.Tensor, 
        rowptr:torch.Tensor, 
        col:torch.Tensor, 
        shape:Tuple[int, int],
        b:torch.Tensor):
    """
    Parameters
    ----------
    val : torch.Tensor
        1D [nnz] Values of the matrix
    rowptr : torch.Tensor
        1D [m+1] Row pointers of the matrix
    col : torch.Tensor
        1D [nnz] Column indices of the matrix
    shape : Tuple[int, int]
        (m,n) Shape of the matrix
    b : torch.Tensor
        1D [n] Vector to multiply with

    Returns
    -------
    torch.Tensor
        1D [n] Result of the matrix-vector multiplication
    """
    # assertion 
    assert val.ndim == 1, "val must be 1D"
    assert rowptr.ndim == 1, "rowptr must be 1D"
    assert col.ndim == 1, "col must be 1D"
    assert b.ndim == 1, "b must be 1D"
    assert val.shape[0] == col.shape[0], "val and col must have the same length"
    assert shape[0] == rowptr.shape[0] - 1, "shape[0] and rowptr must have the same length"
    assert shape[1] == b.shape[0], "shape[1] and b must have the same length"

    output = torch.zeros(shape[0], device=val.device, dtype=val.dtype)
    BLOCK_SIZE = 1024
    grid = shape[0]
    
    _spmv_csr_kernel[grid](
        val, rowptr, col, b, output,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output