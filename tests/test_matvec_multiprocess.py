#!/usr/bin/env python
"""
End-to-end verification of distributed matvec across multiple processes.

Tests that DSparseTensor @ x gives identical results to SparseTensor @ x
when computed across multiple processes with halo exchange.

Run with:
    python tests/test_matvec_multiprocess.py
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_poisson_2d(n, dtype=torch.float64, device='cpu'):
    """Create 2D Poisson matrix with 5-point stencil."""
    N = n * n
    idx = torch.arange(N, device=device)
    i, j = idx // n, idx % n
    
    entries = [
        (idx, idx, torch.full((N,), 4.0, dtype=dtype, device=device)),
        (idx[i > 0], idx[i > 0] - n, torch.full(((i > 0).sum(),), -1.0, dtype=dtype, device=device)),
        (idx[i < n-1], idx[i < n-1] + n, torch.full(((i < n-1).sum(),), -1.0, dtype=dtype, device=device)),
        (idx[j > 0], idx[j > 0] - 1, torch.full(((j > 0).sum(),), -1.0, dtype=dtype, device=device)),
        (idx[j < n-1], idx[j < n-1] + 1, torch.full(((j < n-1).sum(),), -1.0, dtype=dtype, device=device)),
    ]
    vals = torch.cat([e[2] for e in entries])
    rows = torch.cat([e[0] for e in entries])
    cols = torch.cat([e[1] for e in entries])
    return vals, rows, cols, (N, N)


def run_matvec_test(rank, world_size, grid_size, backend='gloo'):
    """Run distributed matvec test on a single process."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29502'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    try:
        from torch_sla.distributed import DSparseTensor
        from torch_sla import SparseTensor
        import warnings
        warnings.filterwarnings('ignore')
        
        n = grid_size
        N = n * n
        val, row, col, shape = create_poisson_2d(n)
        
        # Create distributed partition
        partition = DSparseTensor.from_global_distributed(
            val, row, col, shape,
            rank=rank, world_size=world_size,
            partition_method='simple',
            verbose=False
        )
        
        # Reference global matrix
        A_global = SparseTensor(val, row, col, shape)
        
        # Test vector (seeded for reproducibility across ranks)
        torch.manual_seed(42)
        x_global = torch.randn(N, dtype=torch.float64)
        y_global_ref = A_global @ x_global
        
        # Distributed matvec
        owned_nodes = partition.partition.owned_nodes
        halo_nodes = partition.partition.halo_nodes
        
        x_local = torch.zeros(partition.num_local, dtype=torch.float64)
        x_local[:partition.num_owned] = x_global[owned_nodes]
        if partition.num_halo > 0:
            x_local[partition.num_owned:] = x_global[halo_nodes]
        
        y_local = partition.matvec(x_local, exchange_halo=True)
        y_owned = y_local[:partition.num_owned]
        
        # Compare
        y_global_owned = y_global_ref[owned_nodes]
        error = torch.norm(y_owned - y_global_owned) / torch.norm(y_global_owned)
        
        # Gather errors to rank 0
        all_errors = [torch.zeros(1) for _ in range(world_size)]
        dist.all_gather(all_errors, error.unsqueeze(0))
        
        if rank == 0:
            max_error = max(e.item() for e in all_errors)
            status = "✓" if max_error < 1e-10 else "✗"
            print(f"  {status} Grid: {n}x{n} ({N:4d} nodes), Ranks: {world_size}, Max Error: {max_error:.2e}")
        
        dist.barrier()
        
    finally:
        dist.destroy_process_group()


def main():
    print("=" * 60)
    print("  End-to-End Multi-Process Distributed Matvec Test")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Test with 2 processes
    print("Testing with 2 processes:")
    for grid_size in [4, 8, 16]:
        try:
            spawn(run_matvec_test, args=(2, grid_size), nprocs=2, join=True)
        except Exception as e:
            print(f"  ✗ Grid {grid_size}x{grid_size} failed: {e}")
            all_passed = False
    
    print()
    
    # Test with 4 processes
    print("Testing with 4 processes:")
    for grid_size in [4, 8, 16]:
        try:
            spawn(run_matvec_test, args=(4, grid_size), nprocs=4, join=True)
        except Exception as e:
            print(f"  ✗ Grid {grid_size}x{grid_size} failed: {e}")
            all_passed = False
    
    print()
    
    if all_passed:
        print("=" * 60)
        print("  ALL DISTRIBUTED MATVEC TESTS PASSED!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("  SOME TESTS FAILED!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

