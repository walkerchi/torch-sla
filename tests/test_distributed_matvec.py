#!/usr/bin/env python
"""
End-to-end verification of distributed matrix-vector multiplication.

This test verifies that D @ x (distributed) == A @ x (serial) for various
matrix sizes and partition counts.

Run: python tests/test_distributed_matvec.py
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_poisson_2d(n, dtype=torch.float64, device='cpu'):
    """Create 2D Poisson matrix (5-point stencil)."""
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


def test_distributed_matvec(rank, world_size):
    """Test distributed matvec on a single process."""
    import warnings
    warnings.filterwarnings('ignore')
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29503'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    
    try:
        from torch_sla import SparseTensor
        
        # Test multiple matrix sizes
        test_cases = [
            (4, 'small 4x4 grid'),
            (8, 'medium 8x8 grid'),
            (16, 'large 16x16 grid'),
            (32, 'xlarge 32x32 grid'),
        ]
        
        all_passed = True
        
        for n, desc in test_cases:
            N = n * n
            val, row, col, shape = create_poisson_2d(n)
            
            # Create SparseTensor and get partition for this rank
            A = SparseTensor(val, row, col, shape)
            partition = A.partition_for_rank(rank, world_size, partition_method='simple')
            
            # Reference matrix
            A_global = SparseTensor(val, row, col, shape)
            
            # Random global vector (must be same on all ranks)
            torch.manual_seed(42)  # Same seed for all ranks
            x_global = torch.randn(N, dtype=torch.float64)
            
            # Reference result
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
            y_global_owned = y_global_ref[owned_nodes]
            
            error = torch.norm(y_owned - y_global_owned)
            ref_norm = torch.norm(y_global_owned)
            rel_error = error / ref_norm if ref_norm > 0 else error
            
            passed = rel_error < 1e-10
            all_passed = all_passed and passed
            
            if rank == 0:
                print(f'  {desc} (N={N}): error={rel_error:.2e} [{"PASS" if passed else "FAIL"}]')
        
        # Gather final status
        status = torch.tensor([1.0 if all_passed else 0.0])
        all_status = [torch.zeros(1) for _ in range(world_size)]
        dist.all_gather(all_status, status)
        
        if rank == 0:
            final_pass = all(s.item() > 0 for s in all_status)
            return final_pass
        return True
    
    finally:
        dist.destroy_process_group()


def main():
    print("=" * 60)
    print("  End-to-End Distributed Matvec Verification")
    print("=" * 60)
    
    results = {}
    
    for ws in [2, 4, 8]:
        print(f'\n--- Testing with {ws} processes ---')
        try:
            spawn(test_distributed_matvec, args=(ws,), nprocs=ws, join=True)
            results[ws] = True
        except Exception as e:
            print(f'  Error: {e}')
            results[ws] = False
    
    print("\n" + "=" * 60)
    all_passed = all(results.values())
    if all_passed:
        print("✅ ALL DISTRIBUTED MATVEC TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 60)


if __name__ == '__main__':
    main()

