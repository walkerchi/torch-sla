import torch
from typing import Sequence

def lexsort(keys:Sequence[torch.Tensor], dim=-1)->torch.Tensor:
    """ Multi level sort 
    https://discuss.pytorch.org/t/numpy-lexsort-equivalent-in-pytorch/47850/4

    
    Parameters
    ----------
    keys: Sequence[torch.Tensor]
        list of tuples of ND Tensor, 
    
    dim: int 
        the dimension for sorting


    Returns
    -------
    indices: torch.Tensor
        the sorted indices
        
    """
    if keys.ndim < 2:
        raise ValueError(f"keys must be at least 2 dimensional, but {keys.ndim=}.")
    if len(keys) == 0:
        raise ValueError(f"Must have at least 1 key, but {len(keys)=}.")
    
    idx = keys[0].argsort(dim=dim, stable=True)
    for k in keys[1:]:
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))
    
    return idx