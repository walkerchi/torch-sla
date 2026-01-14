#!/usr/bin/env python
"""
Real distributed I/O test using torchrun.

Run with:
    torchrun --nproc_per_node=2 tests/test_io_distributed.py
    torchrun --nproc_per_node=4 tests/test_io_distributed.py
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_poisson_2d(n, dtype=torch.float64):
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
    vals = torch.cat([e[2] for e in entries])
    rows = torch.cat([e[0] for e in entries])
    cols = torch.cat([e[1] for e in entries])
    return vals, rows, cols, (N, N)


def main():
    # Initialize distributed
    dist.init_process_group(backend='gloo')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("=" * 60)
        print(f"  Real Distributed I/O Test (world_size={world_size})")
        print("=" * 60)
    
    import warnings
    warnings.filterwarnings('ignore')
    
    from torch_sla import SparseTensor, DSparseMatrix
    from torch_sla.io import load_metadata
    
    all_passed = True
    
    # Create temp directory (only rank 0 creates, then broadcast path)
    if rank == 0:
        tmpdir = tempfile.mkdtemp()
    else:
        tmpdir = None
    
    # Broadcast tmpdir path
    if rank == 0:
        path_tensor = torch.zeros(256, dtype=torch.uint8)
        path_bytes = tmpdir.encode('utf-8')
        path_tensor[:len(path_bytes)] = torch.tensor(list(path_bytes), dtype=torch.uint8)
    else:
        path_tensor = torch.zeros(256, dtype=torch.uint8)
    
    dist.broadcast(path_tensor, src=0)
    
    if rank != 0:
        path_bytes = bytes(path_tensor.tolist())
        tmpdir = path_bytes.rstrip(b'\x00').decode('utf-8')
    
    dist.barrier()
    
    # Test 1: Rank 0 saves, all ranks load their partitions
    if rank == 0:
        print("\n--- Test 1: Save on rank 0, load partitions on all ranks ---")
        n = 6
        val, row, col, shape = create_poisson_2d(n)
        A = SparseTensor(val, row, col, shape)
        A.save_distributed(tmpdir, num_partitions=world_size)
        print(f"  Saved {world_size} partitions to {tmpdir}")
    
    dist.barrier()
    
    # All ranks load their partition
    partition = DSparseMatrix.load(tmpdir, rank, world_size)
    
    if rank == 0:
        print(f"  All ranks loaded their partitions")
    
    # Verify: compute matvec with loaded partition
    n = 6
    N = n * n
    torch.manual_seed(42)
    x_global = torch.randn(N, dtype=torch.float64)
    
    # Get reference on rank 0
    if rank == 0:
        val, row, col, shape = create_poisson_2d(n)
        A = SparseTensor(val, row, col, shape)
        y_ref = A @ x_global
    else:
        y_ref = None
    
    # Compute local matvec
    owned = partition.partition.owned_nodes
    halo = partition.partition.halo_nodes
    
    x_local = torch.zeros(partition.num_local, dtype=torch.float64)
    x_local[:partition.num_owned] = x_global[owned]
    if partition.num_halo > 0:
        x_local[partition.num_owned:] = x_global[halo]
    
    y_local = partition.matvec(x_local, exchange_halo=False)
    y_owned = y_local[:partition.num_owned]
    
    # Gather to rank 0 for verification
    # First gather sizes
    local_size = torch.tensor([partition.num_owned], dtype=torch.int64)
    all_sizes = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    
    max_size = max(s.item() for s in all_sizes)
    
    # Gather owned nodes indices
    owned_padded = torch.zeros(max_size, dtype=torch.int64)
    owned_padded[:partition.num_owned] = owned
    all_owned = [torch.zeros(max_size, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(all_owned, owned_padded)
    
    # Gather y values
    y_padded = torch.zeros(max_size, dtype=torch.float64)
    y_padded[:partition.num_owned] = y_owned
    all_y = [torch.zeros(max_size, dtype=torch.float64) for _ in range(world_size)]
    dist.all_gather(all_y, y_padded)
    
    if rank == 0:
        # Reconstruct global y
        y_result = torch.zeros(N, dtype=torch.float64)
        for r_idx in range(world_size):
            size = all_sizes[r_idx].item()
            y_result[all_owned[r_idx][:size]] = all_y[r_idx][:size]
        
        error = torch.norm(y_result - y_ref) / torch.norm(y_ref)
        passed = error < 1e-10
        all_passed = all_passed and passed
        print(f"  Matvec error: {error:.2e} [{'PASS' if passed else 'FAIL'}]")
    
    # Test 2: Verify partition metadata
    if rank == 0:
        print("\n--- Test 2: Verify metadata ---")
        meta = load_metadata(tmpdir)
        shape_ok = meta["shape"] == [N, N]
        parts_ok = meta["num_partitions"] == world_size
        passed = shape_ok and parts_ok
        all_passed = all_passed and passed
        print(f"  Shape: {meta['shape']} (expected [{N}, {N}]) [{'PASS' if shape_ok else 'FAIL'}]")
        print(f"  Partitions: {meta['num_partitions']} (expected {world_size}) [{'PASS' if parts_ok else 'FAIL'}]")
    
    # Cleanup
    dist.barrier()
    if rank == 0:
        import shutil
        shutil.rmtree(tmpdir)
    
    # Final status
    status = torch.tensor([1.0 if all_passed else 0.0])
    all_status = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(all_status, status)
    
    dist.barrier()
    
    if rank == 0:
        final_pass = all(s.item() > 0 for s in all_status)
        print("\n" + "=" * 60)
        if final_pass:
            print("✅ ALL DISTRIBUTED I/O TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED!")
        print("=" * 60)
    
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

