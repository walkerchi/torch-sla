import pytest
import torch 
import numpy as np 
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg as sp_cg
from itertools import product
import sys
sys.path.append("..")
from torch_sla import spsolve


@pytest.mark.parametrize(
    ['n', 'method', 'device'], 
    product([16, 128, 512],
            ['cg', 'bicgstab'],
            ['cpu'] + ['cuda'] if torch.cuda.is_available() else [])
    )
def test_spsolve(n, method, device):

    A = torch.rand(n, n).double().to(device) 
    A = A @ A.T 
    A[A<0.5] = 0
    A = A.to_sparse_coo()
    
    
    b = torch.randn(n).double().to(device)
    x = spsolve(A.values(), A.indices()[0], A.indices()[1], A.shape, b, method=method, atol=1e-10, maxiter=10000)
    A_scipy = csc_matrix(A.to_dense().cpu().numpy())
    b_scipy = b.cpu().numpy()
    x2, _ = sp_cg(A_scipy, b_scipy, atol=1e-10, maxiter=10000)
    x2 = torch.tensor(x2).to(device)
    torch.testing.assert_close(x, x2, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    ['n', 'method', 'device'], 
    product([16, 128, 512],
            ['cg', 'bicgstab'],
            ['cpu'] + ['cuda'] if torch.cuda.is_available() else [])
    )
def test_spsolve_gradient(n, method, device):

    A_dense = torch.rand(n, n).double().to(device) 
    A_dense = A_dense @ A_dense.T 
    A_dense[A_dense<0.5] = 0
    A = A_dense.to_sparse_coo()
    
    b = torch.randn(n).double().to(device)
    b_dense = b.clone()
    val = A.values()
    val.requires_grad_(True)
    b.requires_grad_(True)
    A_dense.requires_grad_(True)
    b_dense.requires_grad_(True)
    x = spsolve(val, A.indices()[0], A.indices()[1], A.shape, b, method=method, atol=1e-10, maxiter=10000)
    x.sum().backward()
    x2 = torch.linalg.solve(A_dense, b_dense)
    x2.sum().backward()

    A_grad = torch.sparse_coo_tensor(A.indices(), val.grad, A.shape).to_dense()

    torch.testing.assert_close(x, x2, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(A_dense.grad, A_grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(b_dense.grad, b.grad, rtol=1e-3, atol=1e-3)




if __name__ == '__main__':
    # test_spsolve(1000, 'bicgstab', 'cuda')
    test_spsolve_gradient(100, 'cg', 'cpu')
