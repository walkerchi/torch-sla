import torch 
from typing import Tuple
from .check import check_coo, check_csr, check_csc
from .sort import lexsort
#################
# coo, csr, csc 
#################

def coo2csr(val:torch.Tensor, 
            row:torch.Tensor, 
            col:torch.Tensor, 
            shape:tuple
            )->Tuple[torch.Tensor, 
                     torch.Tensor, 
                     torch.Tensor, 
                     Tuple[int, int]]:
    """
    Convert COO format to CSR format

    Parameters
    ----------
        val: torch.Tensor
            [n, ...] values of the sparse matrix
        row: torch.Tensor
            [n] row indices of the sparse matrix
        col: torch.Tensor
            [n] column indices of the sparse matrix
        shape: tuple
            (m,n) shape of the sparse matrix
    Returns
    -------
        val: torch.Tensor
            [n, ...] values of the sparse matrix
        rowptr: torch.Tensor
            [m+1] rowptr of the sparse matrix
        col: torch.Tensor
            [n] column indices of the sparse matrix
        shape: tuple
            (m,n) shape of the sparse matrix
    """
    check_coo(val, row, col, shape)

    m, n   = shape
    
    arg    = lexsort([row, col])
    # arg    = torch.argsort(row)
    row    = row[arg]
    col    = col[arg]
    val    = val[arg]
    rowptr = torch.zeros(shape[0] + 1, dtype=row.dtype, device=val.device)
    rowcount   = torch.bincount(row, minlength=m)
    rowptr[1:] = torch.cumsum(rowcount, 0)
    
    return val, rowptr, col, shape

def csr2coo(val:torch.Tensor,
            rowptr:torch.Tensor, 
            col:torch.Tensor,
            shape:tuple
            )->Tuple[torch.Tensor, 
                     torch.Tensor, 
                     torch.Tensor, 
                     Tuple[int, int]]: 
    """
    Convert CSR format to COO format

    Parameters
    ----------
        val: torch.Tensor
            [n, ...] values of the sparse matrix
        rowptr: torch.Tensor
            [m+1] rowptr of the sparse matrix
        col: torch.Tensor
            [n] column indices of the sparse matrix
        shape: tuple
            (m,n) shape of the sparse matrix

    Returns
    -------
        val: torch.Tensor
            [n, ...] values of the sparse matrix
        row: torch.Tensor
            [n] row indices of the sparse matrix
        col: torch.Tensor
            [n] column indices of the sparse matrix
        shape: tuple
            (m,n) shape of the sparse matrix
    """

    check_csr(val, rowptr, col, shape)
    
    m, n = shape
    row  = torch.repeat_interleave(
        torch.arange(m, dtype=rowptr.dtype, device=val.device),
        rowptr[1:] - rowptr[:-1]
    )
    return val, row, col, shape

def coo2csc(val:torch.Tensor, 
            row:torch.Tensor, 
            col:torch.Tensor, 
            shape:tuple
            )->Tuple[torch.Tensor, 
                     torch.Tensor, 
                     torch.Tensor, 
                     Tuple[int, int]]:
    """
    Convert COO format to CSC format

    Parameters
    ----------
        val: torch.Tensor
            [n, ...] values of the sparse matrix
        row: torch.Tensor
            [n] row indices of the sparse matrix
        col: torch.Tensor
            [n] column indices of the sparse matrix
        shape: tuple
            (m,n) shape of the sparse matrix
    Returns
    -------
        val: torch.Tensor
            [n, ...] values of the sparse matrix
        row: torch.Tensor
            [n] row indices of the sparse matrix
        colptr: torch.Tensor
            [n+1] colptr of the sparse matrix
        shape: tuple
            (m,n) shape of the sparse matrix
    """
    check_coo(val, row, col, shape)
    
    m, n   = shape
    arg    = lexsort([col, row])
    # arg    = torch.argsort(col)
    row    = row[arg]
    col    = col[arg]
    val    = val[arg]
    colptr = torch.zeros(shape[1] + 1, dtype=col.dtype, device=val.device)
    colcount   = torch.bincount(col, minlength=n)
    colptr[1:] = torch.cumsum(colcount, 0)
    
    return val, row, colptr, shape

def csc2coo(val:torch.Tensor,
            row:torch.Tensor,
            colptr:torch.Tensor,
            shape:tuple
            )->Tuple[torch.Tensor, 
                     torch.Tensor, 
                     torch.Tensor, 
                     Tuple[int, int]]:
    """
    Convert CSC format to COO format

    Parameters
    ----------
        val: torch.Tensor
            [n, ...] values of the sparse matrix
        row: torch.Tensor
            [n] row indices of the sparse matrix
        colptr: torch.Tensor
            [n+1] colptr of the sparse matrix
        shape: tuple
            (m,n) shape of the sparse matrix

    Returns
    -------
        val: torch.Tensor
            [n, ...] values of the sparse matrix
        row: torch.Tensor
            [n] row indices of the sparse matrix
        col: torch.Tensor
            [n] column indices of the sparse matrix
        shape: tuple
            (m,n) shape of the sparse matrix
    """
    check_csc(val, row, colptr, shape)

    m, n = shape
    col  = torch.repeat_interleave(
        torch.arange(n, dtype=colptr.dtype, device=val.device),
        colptr[1:] - colptr[:-1]
    )
    return val, row, col, shape

def csr2csc(val:torch.Tensor,
            rowptr:torch.Tensor, 
            col:torch.Tensor,
            shape:Tuple[int, int]
            )->Tuple[torch.Tensor, 
                     torch.Tensor, 
                     torch.Tensor, 
                     Tuple[int, int]]:
    """
    Convert CSR format to CSC format

    Parameters
    ----------
        val: torch.Tensor
            [n, ...] values of the sparse matrix
        rowptr: torch.Tensor
            [m+1] rowptr of the sparse matrix
        col: torch.Tensor
            [n] column indices of the sparse matrix
        shape: tuple
            (m,n) shape of the sparse matrix

    Returns
    -------
        val: torch.Tensor
            [n, ...] values of the sparse matrix
        row: torch.Tensor
            [n] row indices of the sparse matrix
        colptr: torch.Tensor
            [n+1] colptr of the sparse matrix
        shape: tuple
            (m,n) shape of the sparse matrix
    
    """
    check_csr(val, rowptr, col, shape)

    m, n = shape

    row  = torch.repeat_interleave(
        torch.arange(m, dtype=rowptr.dtype, device=val.device),
        rowptr[1:] - rowptr[:-1]
    )
    arg    = torch.argsort(col)
    row    = row[arg]
    col    = col[arg]
    val    = val[arg]
    colptr = torch.zeros(shape[1] + 1, dtype=col.dtype, device=val.device)
    colcount   = torch.bincount(col, minlength=n)
    colptr[1:] = torch.cumsum(colcount, 0)
   
    return val, row, colptr, shape

def csc2csr(val:torch.Tensor,
            row:torch.Tensor,
            colptr:torch.Tensor,
            shape:Tuple[int, int]
            )->Tuple[torch.Tensor, 
                     torch.Tensor, 
                     torch.Tensor, 
                     Tuple[int, int]]:
    """
    Convert CSC format to CSR format

    Parameters
    ----------
        val: torch.Tensor
            [n, ...] values of the sparse matrix
        row: torch.Tensor
            [n] row indices of the sparse matrix
        colptr: torch.Tensor
            [n+1] colptr of the sparse matrix
        shape: tuple
            (m,n) shape of the sparse matrix


    Returns
    -------
        val: torch.Tensor
            [n, ...] values of the sparse matrix
        rowptr: torch.Tensor
            [m+1] rowptr of the sparse matrix
        col: torch.Tensor
            [n] column indices of the sparse matrix
        shape: tuple
            (m,n) shape of the sparse matrix
    """

    check_csc(val, row, colptr, shape)

    m, n = shape
    col  = torch.repeat_interleave(
        torch.arange(n, dtype=colptr.dtype, device=val.device),
        colptr[1:] - colptr[:-1]
    )
    arg    = torch.argsort(row)
    row    = row[arg]
    col    = col[arg]
    val    = val[arg]
    rowptr = torch.zeros(shape[0] + 1, dtype=row.dtype, device=val.device)
    rowcount   = torch.bincount(row, minlength=m)
    rowptr[1:] = torch.cumsum(rowcount, 0)

    return val, rowptr, col, shape


######################
# dense
######################
def dense2coo(dense:torch.Tensor
              )->Tuple[torch.Tensor, 
                     torch.Tensor, 
                     torch.Tensor, 
                     Tuple[int, int]]:
    """
    Convert dense matrix to COO format

    Parameters
    ----------
    dense: torch.Tensor 
        [m, n, ...] dense matrix

    Returns
    -------
    val: torch.Tensor
        [nnz, ...] values of the sparse matrix
    row: torch.Tensor
        [nnz] row indices of the sparse matrix
    col: torch.Tensor
        [nnz] column indices of the sparse matrix
    shape: tuple
        (m,n) shape of the sparse matrix
    """
    m, n = dense.shape[:2]

    row, col = torch.where(dense.view(m, n, -1)[:, :, 0])

    val = dense[row, col] # [nnz]

    check_coo(val, row, col, (m, n))

    return val, row, col, (m, n)

def coo2dense(val:torch.Tensor,
              row:torch.Tensor, 
              col:torch.Tensor,
              shape:Tuple[int, int]
              )->torch.Tensor:
    
    """
    Convert COO format to dense matrix

    Parameters
    ----------
    val: torch.Tensor
        [nnz, ...] values of the sparse matrix

    row: torch.Tensor
        [nnz] row indices of the sparse matrix

    col: torch.Tensor
        [nnz] column indices of the sparse matrix

    shape: tuple
        (m,n) shape of the sparse matrix

    Returns
    -------
    dense: torch.Tensor
        [m, n, ...] dense matrix
    
    """



    check_coo(val, row, col, shape)

    m, n = shape

    dense = torch.zeros(m, n, *val.shape[1:], dtype=val.dtype, device=val.device)

    dense[row, col] = val

    return dense




