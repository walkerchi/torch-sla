#!/usr/bin/env python
"""
Persistence (I/O) Example

Demonstrates saving and loading SparseTensor/DSparseTensor using safetensors format:
- SparseTensor.save() / SparseTensor.load()
- save_distributed() / load_partition()
- Cross-format loading
"""

import torch
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_sla import SparseTensor, DSparseTensor, DSparseMatrix
from torch_sla.io import load_sparse_as_partition, load_distributed_as_sparse


def create_2d_poisson(n: int, dtype=torch.float64):
    """Create 2D Poisson matrix."""
    N = n * n
    idx = torch.arange(N)
    i, j = idx // n, idx % n
    
    entries = [
        (idx, idx, torch.full((N,), 4.0, dtype=dtype)),
        (idx[i > 0], idx[i > 0] - n, torch.full(((i > 0).sum(),), -1.0, dtype=dtype)),
        (idx[i < n-1], idx[i < n-1] + n, torch.full(((i < n-1).sum(),), -1.0, dtype=dtype)),
        (idx[j > 0], idx[j > 0] - 1, torch.full(((j > 0).sum(),), -1.0, dtype=dtype)),
        (idx[j < n-1], idx[j < n-1] + 1, torch.full(((j < n-1).sum(),), -1.0, dtype=dtype)),
    ]
    
    row = torch.cat([e[0] for e in entries])
    col = torch.cat([e[1] for e in entries])
    val = torch.cat([e[2] for e in entries])
    
    return val, row, col, (N, N)


def main():
    print("=" * 60)
    print("Persistence (I/O) Example")
    print("=" * 60)
    
    val, row, col, shape = create_2d_poisson(4)
    A = SparseTensor(val, row, col, shape)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. SparseTensor save/load
        print("\n1. SparseTensor save/load:")
        path = os.path.join(tmpdir, "matrix.safetensors")
        A.save(path)
        A_loaded = SparseTensor.load(path)
        print(f"   Saved: {A}")
        print(f"   Loaded: {A_loaded}")
        
        # Verify
        x = torch.randn(shape[1], dtype=val.dtype)
        error = torch.norm(A @ x - A_loaded @ x)
        print(f"   Matvec match: {'✓' if error < 1e-10 else '✗'}")
        
        # 2. Save as distributed partitions
        print("\n2. Save distributed, load partitions:")
        dist_dir = os.path.join(tmpdir, "matrix_dist")
        A.save_distributed(dist_dir, num_partitions=2)
        print(f"   Saved 2 partitions to {dist_dir}/")
        
        # Each rank loads its partition
        p0 = DSparseMatrix.load(dist_dir, rank=0, world_size=2)
        p1 = DSparseMatrix.load(dist_dir, rank=1, world_size=2)
        print(f"   Rank 0: owned={p0.num_owned}, halo={p0.num_halo}")
        print(f"   Rank 1: owned={p1.num_owned}, halo={p1.num_halo}")
        
        # 3. Load distributed as single SparseTensor
        print("\n3. Load distributed as SparseTensor:")
        A_gathered = load_distributed_as_sparse(dist_dir)
        print(f"   {A_gathered}")
        error = torch.norm(A @ x - A_gathered @ x)
        print(f"   Matvec match: {'✓' if error < 1e-10 else '✗'}")
        
        # 4. Load single file as partition
        print("\n4. Load single file as partition:")
        p = load_sparse_as_partition(path, rank=0, world_size=2)
        print(f"   From single file → Partition: owned={p.num_owned}, halo={p.num_halo}")
        
        # 5. DSparseTensor save/load
        print("\n5. DSparseTensor save/load:")
        D = A.partition(num_partitions=2, verbose=False)
        dsparse_dir = os.path.join(tmpdir, "dsparse")
        D.save(dsparse_dir)
        D_loaded = DSparseTensor.load(dsparse_dir)
        print(f"   Saved: {D}")
        print(f"   Loaded: {D_loaded}")
        
        error = torch.norm(D @ x - D_loaded @ x)
        print(f"   Matvec match: {'✓' if error < 1e-10 else '✗'}")
    
    print("\n" + "=" * 60)
    print("Cross-Format Conversion Summary:")
    print("=" * 60)
    print("""
    | Save Format                    | Load as SparseTensor           | Load as DSparseMatrix          |
    |--------------------------------|--------------------------------|--------------------------------|
    | A.save("f.safetensors")        | SparseTensor.load("f")         | load_sparse_as_partition("f")  |
    | A.save_distributed("dir", n)   | load_distributed_as_sparse()   | DSparseMatrix.load("dir",rank) |
    | D.save("dir")                  | load_distributed_as_sparse()   | DSparseTensor.load("dir")      |
    """)


if __name__ == "__main__":
    main()

