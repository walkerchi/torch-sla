import pytest
import numpy as np
import torch
from itertools import product
import sys
sys.path.append("..")
from torch_sla.sort import lexsort

@pytest.mark.parametrize(
    ['n', 'nlevel', 'device'], 
    product([16, 128, 512, 4096],
            [2, 3, 4, 5, 6],
            ['cpu'] + ['cuda'] if torch.cuda.is_available() else [])
    )
def test_lexsort(n:int, nlevel:int, device):
    keys_np    = [
        np.random.rand(n) for _ in range(nlevel)
    ]
    keys_torch = [
        torch.from_numpy(x).to(device) for x in keys_np
    ]
    indices_np    = np.lexsort(keys_np)
    indices_torch = lexsort(keys_torch)

    torch.testing.assert_close(torch.from_numpy(indices_np).to(device), indices_torch)