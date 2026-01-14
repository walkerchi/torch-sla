#!/usr/bin/env python
"""
Real distributed test using torchrun.

Run with:
    torchrun --nproc_per_node=2 tests/test_real_distributed.py
    torchrun --nproc_per_node=4 tests/test_real_distributed.py
"""

import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_poisson_2d(n, dtype=torch.float64):
    """Create 2D Poisson matrix (5-point stencil)."""
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
        print(f"  Real Distributed Test (world_size={world_size})")
        print("=" * 60)
    
    import warnings
    warnings.filterwarnings('ignore')
    
    from torch_sla import SparseTensor
    
    all_passed = True
    
    # Test multiple sizes
    for n in [4, 8, 16]:
        N = n * n
        val, row, col, shape = create_poisson_2d(n)
        
        # Create SparseTensor and partition for this rank
        A = SparseTensor(val, row, col, shape)
        partition = A.partition_for_rank(rank, world_size, partition_method='simple')
        
        # Global vector (same on all ranks)
        torch.manual_seed(42)
        x_global = torch.randn(N, dtype=torch.float64)
        
        # Reference
        y_ref = A @ x_global
        
        # Distributed matvec
        owned_nodes = partition.partition.owned_nodes
        halo_nodes = partition.partition.halo_nodes
        
        x_local = torch.zeros(partition.num_local, dtype=torch.float64)
        x_local[:partition.num_owned] = x_global[owned_nodes]
        if partition.num_halo > 0:
            x_local[partition.num_owned:] = x_global[halo_nodes]
        
        y_local = partition.matvec(x_local, exchange_halo=True)
        y_owned = y_local[:partition.num_owned]
        y_ref_owned = y_ref[owned_nodes]
        
        error = torch.norm(y_owned - y_ref_owned) / torch.norm(y_ref_owned)
        passed = error < 1e-10
        all_passed = all_passed and passed
        
        if rank == 0:
            print(f"\n{n}x{n} grid (N={N}):")
            print(f"  Matvec error: {error:.2e} [{'PASS' if passed else 'FAIL'}]")
    
    # Test distributed CG solve
    if rank == 0:
        print("\n--- Distributed CG Solve ---")
    
    n = 8
    N = n * n
    val, row, col, shape = create_poisson_2d(n)
    A = SparseTensor(val, row, col, shape)
    partition = A.partition_for_rank(rank, world_size, partition_method='simple')
    
    b_global = torch.ones(N, dtype=torch.float64)
    x_ref = A.solve(b_global)
    
    # Prepare for distributed CG
    owned_nodes = partition.partition.owned_nodes
    halo_nodes = partition.partition.halo_nodes
    
    local_size = torch.tensor([partition.num_owned], dtype=torch.int64)
    all_sizes = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(s.item() for s in all_sizes)
    
    owned_padded = torch.zeros(max_size, dtype=torch.int64)
    owned_padded[:partition.num_owned] = owned_nodes
    all_owned = [torch.zeros(max_size, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(all_owned, owned_padded)
    
    # Initialize
    x_local = torch.zeros(partition.num_local, dtype=torch.float64)
    b_local = b_global[owned_nodes]
    
    Ax_local = partition.matvec(x_local, exchange_halo=True)
    r_local = torch.zeros(partition.num_local, dtype=torch.float64)
    r_local[:partition.num_owned] = b_local - Ax_local[:partition.num_owned]
    p_local = r_local.clone()
    
    rs_local = torch.dot(r_local[:partition.num_owned], r_local[:partition.num_owned])
    rs_global = rs_local.clone()
    dist.all_reduce(rs_global, op=dist.ReduceOp.SUM)
    
    # CG iterations
    atol, maxiter = 1e-10, 500
    for iteration in range(maxiter):
        # Gather p globally
        p_global = torch.zeros(N, dtype=torch.float64)
        p_padded = torch.zeros(max_size, dtype=torch.float64)
        p_padded[:partition.num_owned] = p_local[:partition.num_owned]
        all_p = [torch.zeros(max_size, dtype=torch.float64) for _ in range(world_size)]
        dist.all_gather(all_p, p_padded)
        
        for r_idx in range(world_size):
            size = all_sizes[r_idx].item()
            p_global[all_owned[r_idx][:size]] = all_p[r_idx][:size]
        
        if partition.num_halo > 0:
            p_local[partition.num_owned:] = p_global[halo_nodes]
        
        Ap_local = partition.matvec(p_local, exchange_halo=False)
        
        pAp_local = torch.dot(p_local[:partition.num_owned], Ap_local[:partition.num_owned])
        pAp_global = pAp_local.clone()
        dist.all_reduce(pAp_global, op=dist.ReduceOp.SUM)
        
        if pAp_global.abs() < 1e-30:
            break
        
        alpha = rs_global / pAp_global
        x_local[:partition.num_owned] += alpha * p_local[:partition.num_owned]
        r_local[:partition.num_owned] -= alpha * Ap_local[:partition.num_owned]
        
        rs_new = torch.dot(r_local[:partition.num_owned], r_local[:partition.num_owned])
        rs_global_new = rs_new.clone()
        dist.all_reduce(rs_global_new, op=dist.ReduceOp.SUM)
        
        if rs_global_new.sqrt() < atol:
            break
        
        beta = rs_global_new / rs_global
        p_local[:partition.num_owned] = r_local[:partition.num_owned] + beta * p_local[:partition.num_owned]
        rs_global = rs_global_new
    
    # Gather solution
    x_final = torch.zeros(N, dtype=torch.float64)
    x_padded = torch.zeros(max_size, dtype=torch.float64)
    x_padded[:partition.num_owned] = x_local[:partition.num_owned]
    all_x = [torch.zeros(max_size, dtype=torch.float64) for _ in range(world_size)]
    dist.all_gather(all_x, x_padded)
    
    for r_idx in range(world_size):
        size = all_sizes[r_idx].item()
        x_final[all_owned[r_idx][:size]] = all_x[r_idx][:size]
    
    residual = torch.norm(A @ x_final - b_global) / torch.norm(b_global)
    solve_passed = residual < 1e-6
    all_passed = all_passed and solve_passed
    
    if rank == 0:
        print(f"  8x8 grid CG solve: residual={residual:.2e} [{'PASS' if solve_passed else 'FAIL'}]")
    
    # Final status
    status = torch.tensor([1.0 if all_passed else 0.0])
    all_status = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(all_status, status)
    
    dist.barrier()
    
    if rank == 0:
        final_pass = all(s.item() > 0 for s in all_status)
        print("\n" + "=" * 60)
        if final_pass:
            print("✅ ALL REAL DISTRIBUTED TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED!")
        print("=" * 60)
    
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

