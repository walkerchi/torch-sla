import torch
import torch_sla as sla


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = 10
    A = torch.rand(n, n).double().to(device) 
    A = A @ A.T 
    A[A<0.5] = 0
    A = A.to_sparse_coo()
    
    b = torch.randn(n).double().to(device)
    x = sla.spsolve(A.values(), A.indices()[0], A.indices()[1], A.shape, b, method='bicgstab', atol=1e-10, maxiter=10000)
   
    print(f"x={x}")