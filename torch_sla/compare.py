import torch  
from .sort import lexsort
from .check import check_coo


def coo_eq(
    val1:torch.Tensor,
    row1:torch.Tensor, 
    col1:torch.Tensor,
    shape1:tuple,
    val2:torch.Tensor,
    row2:torch.Tensor,
    col2:torch.Tensor,
    shape2:tuple
    )->bool:

    """
    Parameters
    ----------
    val1: torch.Tensor
        [n, ...] values of the sparse matrix
    row1: torch.Tensor
        [n1] row indices of the sparse matrix
    col1: torch.Tensor
        [n1] column indices of the sparse matrix
    shape1: tuple
        (m1,n1) shape of the sparse matrix

    val2: torch.Tensor
        [n, ...] values of the sparse matrix
    row2: torch.Tensor
        [n2] row indices of the sparse matrix
    col2: torch.Tensor
        [n2] column indices of the sparse matrix
    shape2: tuple
        (m2,n2) shape of the sparse matrix
    
    Returns
    -------
    bool 
        whether the layout of two matrix is the same

    """
    if shape1 != shape2:
        return False

    check_coo(val1, row1, col1, shape1)
    check_coo(val2, row2, col2, shape2)

    n1 = val1.shape[0]
    n2 = val2.shape[0]

    if n1 != n2: # not same number of nnz
        return False  

    indices1 = lexsort([row1, col1])

    val1 = val1[indices1]
    row1 = row1[indices1]
    col1 = col1[indices1]

    indices2 = lexsort([row2, col2])

    val2 = val2[indices2]
    row2 = row2[indices2]
    col2 = col2[indices2]

    return (val1 == val2).all() and (row1 == row2).all() and (col1 == col2).all()
