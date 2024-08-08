import torch
import triton 
import triton.language as tl
from typing import Tuple

"""
A Systematic Survey of General Sparse Matrix-Matrix
Multiplication
https://arxiv.org/pdf/2002.11273
"""

@triton.jit 
def _csr_csc_matmul_nnz_kernel(
    A_val, A_rowptr, A_col, 
    col_nnz,
    BLK: tl.constexpr, **meta
    ):
    row = tl.program_id(0)

    row_start = A_rowptr[row]
    row_end   = A_rowptr[row + 1]
    col_start = B_colptr[col]
    col_end   = B_colptr[col + 1]
    
    sum = tl.zeros(1, dtype=tl.float32)
    
    # Pointers for A and B
    a_ptr = row_start
    b_ptr = col_start
    nnz  

    while a_ptr < row_end and b_ptr < col_end:
        a_col = A_col[a_ptr]
        b_row = B_row[b_ptr]

        # Move along A's row and B's column
        if a_col == b_row:
            sum += A_val[a_ptr] * B_val[b_ptr]
            a_ptr += 1
            b_ptr += 1
        elif a_col < b_row:
            a_ptr += 1
        else:
            b_ptr += 1

    # Write the result to the output matrix C
    tl.store(C[row, col], sum)

@triton.jit
def _csr_csc_matmul_kernel(
    A_val, A_rowptr, A_col, 
    B_val, B_row, B_colptr,
    BLK: tl.constexpr, **meta
    ):
    row = tl.program_id(0)
    col = tl.program_id(1)

    row_start = A_rowptr[row]
    row_end   = A_rowptr[row + 1]
    col_start = B_colptr[col]
    col_end   = B_colptr[col + 1]
    
    sum = tl.zeros(1, dtype=tl.float32)
    
    # Pointers for A and B
    a_ptr = row_start
    b_ptr = col_start

    while a_ptr < row_end and b_ptr < col_end:
        a_col = A_col[a_ptr]
        b_row = B_row[b_ptr]

        # Move along A's row and B's column
        if a_col == b_row:
            sum += A_val[a_ptr] * B_val[b_ptr]
            a_ptr += 1
            b_ptr += 1
        elif a_col < b_row:
            a_ptr += 1
        else:
            b_ptr += 1

    # Write the result to the output matrix C
    tl.store(C[row, col], sum)


def sparse_sparse_matmul_nnz(A_col:torch.Tensor, B_row:torch.Tensor, k:int)->int:
    """
    Sparse matrix-matrix multiplication of CSR and CSC matrices

    A Systematic Survey of General Sparse Matrix-Matrix Multiplication
    https://arxiv.org/pdf/2002.11273

    Parameters
    ----------
    A_col : torch.Tensor
        [nnz1] column indices of the CSR matrix
    B_row : torch.Tensor
        [nnz2] row indices of the CSC matrix
    k : int 
        the reduction dimension
    BLK : int, optional
        Block size, by default 128

    Returns
    -------
    int The number of nnz in the resulting matrix
    """
    assert A_col.dim() == 1, f"A_col must be 1D, got {A_col.shape}"
    assert B_row.dim() == 1, f"B_row must be 1D, got {B_row.shape}"
    assert k > 0, f"k must be > 0, got {k}"
    assert A_col.dtype in (torch.int64, torch.int32), f"A_col must be int64 or int32, got {A_col.dtype}"
    assert B_row.dtype in (torch.int64, torch.int32), f"B_row must be int64 or int32, got {B_row.dtype}"

    B_row_nnz = B_row.bincount(minlength=k) # [k]
    total_nnz = A_col[B_row_nnz].sum().item()
    return total_nnz


def csr_csc_matmul(A_val:torch.Tensor,
                   A)