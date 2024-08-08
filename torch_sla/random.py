import torch 
from typing import Tuple

def coo(shape, 
        density:float=0.1, 
        definite:str=">0",
        device=torch.device('cpu'),
        dtype=torch.float32
        )->Tuple[torch.Tensor, 
                 torch.Tensor, 
                 torch.Tensor, 
                 Tuple[int, int]]:
    """
    random COO matrix generator

    Parameters
    ----------
    shape : tuple
        (m,n) shape of the matrix
    density : float, optional
        Density of the matrix, by default 0.1
    definite : str, optional
        Definite of the matrix, by default ">0"
    device : torch.device, optional
        Device of the matrix, by default torch.device('cpu')
    dtype : torch.dtype, optional
        Data type of the matrix, by default torch.float32
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]
        val: torch.Tensor
            [nnz, ...] values of the sparse matrix
        row: torch.Tensor
            [nnz] row indices of the sparse matrix
        col: torch.Tensor
            [nnz] column indices of the sparse matrix
        shape: tuple
            (m,n) shape of the sparse matrix
    """

    assert definite in [">0", ">=0", "<0", "<=0"], "definite must be '>=0', '<=0', '>0', or '<0'"

    m, n = shape[:2]
    nnz = int(m * n * density)
    row = torch.randint(0, m, (nnz,), device=device)
    col = torch.randint(0, n, (nnz,), device=device)
    val = torch.randn(nnz, shape[2:], device=device, dtype=dtype)
    return val, row, col, shape