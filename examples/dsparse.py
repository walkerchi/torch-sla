#!/usr/bin/env python
"""
DSparseTensor: Distributed Sparse Tensor Example

Demonstrates:
- Creating DSparseTensor with automatic partitioning
- SparseTensor.partition() and DSparseTensor.gather()
- Distributed matrix-vector product
- Distributed solve
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_sla import DSparseTensor, SparseTensor


def create_2d_poisson(nx: int, ny: int, dtype=torch.float64):
    """Create 2D Poisson matrix with 5-point stencil."""
    N = nx * ny
    idx = torch.arange(N)
    i, j = idx // nx, idx % nx
    
    entries = [
        (idx, idx, torch.full((N,), 4.0, dtype=dtype)),
        (idx[i > 0], idx[i > 0] - nx, torch.full(((i > 0).sum(),), -1.0, dtype=dtype)),
        (idx[i < ny-1], idx[i < ny-1] + nx, torch.full(((i < ny-1).sum(),), -1.0, dtype=dtype)),
        (idx[j > 0], idx[j > 0] - 1, torch.full(((j > 0).sum(),), -1.0, dtype=dtype)),
        (idx[j < nx-1], idx[j < nx-1] + 1, torch.full(((j < nx-1).sum(),), -1.0, dtype=dtype)),
    ]
    
    row = torch.cat([e[0] for e in entries])
    col = torch.cat([e[1] for e in entries])
    val = torch.cat([e[2] for e in entries])
    
    return val, row, col, (N, N)


def main():
    print("=" * 60)
    print("DSparseTensor Example")
    print("=" * 60)
    
    val, row, col, shape = create_2d_poisson(4, 4)
    
    # 1. Create DSparseTensor
    print("\n1. Creating DSparseTensor:")
    D = DSparseTensor(val, row, col, shape, num_partitions=2, verbose=True)
    print(f"   {D}")
    
    # 2. Create from SparseTensor.partition()
    print("\n2. SparseTensor.partition():")
    A = SparseTensor(val, row, col, shape)
    D2 = A.partition(num_partitions=2, verbose=False)
    print(f"   Original: {A}")
    print(f"   Partitioned: {D2}")
    
    # 3. Gather back
    print("\n3. DSparseTensor.gather():")
    A_gathered = D2.gather()
    print(f"   Gathered: {A_gathered}")
    
    # 4. Distributed matvec
    print("\n4. Distributed Matrix-Vector Product:")
    x = torch.randn(shape[1], dtype=val.dtype)
    y_ref = A @ x
    y_dist = D @ x
    error = torch.norm(y_dist - y_ref) / torch.norm(y_ref)
    print(f"   D @ x error: {error:.2e} {'✓' if error < 1e-10 else '✗'}")
    
    # 5. Scatter/Gather vectors
    print("\n5. Scatter/Gather Vectors:")
    x_local = D.scatter_local(x)
    y_local = D.matvec_all(x_local)
    y_gathered = D.gather_global(y_local)
    error = torch.norm(y_gathered - y_ref) / torch.norm(y_ref)
    print(f"   scatter → matvec → gather error: {error:.2e} {'✓' if error < 1e-10 else '✗'}")
    
    # 6. Verify round-trip solve
    print("\n6. Round-trip Verification:")
    b = torch.ones(shape[0], dtype=val.dtype)
    x1 = A.solve(b)
    x2 = A_gathered.solve(b)
    diff = torch.norm(x1 - x2).item()
    print(f"   Solution difference: {diff:.2e} {'✓' if diff < 1e-10 else '✗'}")
    
    # 7. Device management
    print("\n7. Device Management:")
    print(f"   Original device: {D.device}")
    if torch.cuda.is_available():
        D_cuda = D.cuda()
        print(f"   After .cuda(): {D_cuda.device}")


if __name__ == "__main__":
    main()

