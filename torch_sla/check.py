import torch 


class ShapeException(Exception):
    def __init__(self, name, shape, expected_shape):
        self.name = name
        self.shape = shape
        self.expected_shape = expected_shape
        super().__init__(f"{name} has shape {shape} expected {expected_shape}")


def check_coo(val:torch.Tensor,
              row:torch.Tensor, 
              col:torch.Tensor,
              shape:tuple
              ):
    """
    Check the COO format

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

    """
    if not row.ndim == 1:
        raise ShapeException("row", row.shape, "[nnz]")
    if not col.ndim == 1:
        raise ShapeException("col", col.shape, "[nnz]")
    if not val.shape[0] == row.shape[0]:
        raise ShapeException("val", val.shape, "[nnz, ...]")
    if not val.shape[0] == col.shape[0]:
        raise ShapeException("val", val.shape, "[nnz, ...]")
    if not shape[0] > 0 and shape[1] > 0:
        raise ShapeException("shape", shape, "(m,n)")

def check_csr(val:torch.Tensor,
              rowptr:torch.Tensor,
              col:torch.Tensor,
              shape:tuple):
    """
    Check the CSR format

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
    """
    m, n = shape
    if not (rowptr.ndim == 1 and rowptr.shape[0] == m+1):
        raise ShapeException("rowptr", rowptr.shape, f"[{m+1}]")
    if not col.ndim == 1:
        raise ShapeException("col", col.shape, "[nnz]")
    if not val.shape[0] == rowptr[-1]:
        raise ShapeException("val", val.shape, "[nnz, ...]")
    if not val.shape[0] == col.shape[0]:
        raise ShapeException("val", val.shape, "[nnz, ...]")
    if not shape[0] > 0 and shape[1] > 0:
        raise ShapeException("shape", shape, "(m,n)")

def check_csc(val:torch.Tensor, 
              row:torch.Tensor, 
              colptr:torch.Tensor, 
              shape:tuple):
    """
    Check the CSC format

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
    """
    m, n = shape
    if not row.ndim == 1:
        raise ShapeException("row", row.shape, "[nnz]")
    if not (colptr.ndim == 1 and colptr.shape[0] == n+1):
        raise ShapeException("colptr", colptr.shape, f"[{n+1}]")
    if not val.shape[0] == colptr[-1]:
        raise ShapeException("val", val.shape, "[nnz, ...]")
    if not val.shape[0] == row.shape[0]:
        raise ShapeException("val", val.shape, "[nnz, ...]")
    if not shape[0] > 0 and shape[1] > 0:
        raise ShapeException("shape", shape, "(m,n)")
    