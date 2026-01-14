#!/usr/bin/env python
"""
Test DSparseTensor distributed operations: matvec and solve.

This test verifies that:
1. Distributed matrix-vector product (A @ x) gives the same result as serial
2. Distributed solve (A^{-1} b) converges to the correct solution
3. DSparseTensor @ DTensor/Tensor multiplication works correctly

Run with:
    python -m torch.distributed.launch --nproc_per_node=2 tests/test_distributed_multiprocess.py
    
Or with torchrun:
    torchrun --nproc_per_node=2 tests/test_distributed_multiprocess.py
    
Or simply:
    python tests/test_distributed_multiprocess.py
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_poisson_2d(n: int, dtype=torch.float64, device='cpu'):
    """
    Create 2D Poisson matrix with 5-point stencil using vectorized operations.
    
    For an n x n grid, creates the discrete Laplacian matrix:
        A[i,i] = 4 (diagonal)
        A[i,j] = -1 for j in {i-n, i+n, i-1, i+1} (neighbors)
    """
    N = n * n
    
    # Node indices as 2D grid
    idx = torch.arange(N, device=device)
    i = idx // n  # Row in grid
    j = idx % n   # Column in grid
    
    # Diagonal entries: all nodes
    diag_row = idx
    diag_col = idx
    diag_val = torch.full((N,), 4.0, dtype=dtype, device=device)
    
    # Up neighbor (i-1): valid when i > 0
    up_mask = i > 0
    up_row = idx[up_mask]
    up_col = idx[up_mask] - n
    up_val = torch.full((up_mask.sum(),), -1.0, dtype=dtype, device=device)
    
    # Down neighbor (i+1): valid when i < n-1
    down_mask = i < n - 1
    down_row = idx[down_mask]
    down_col = idx[down_mask] + n
    down_val = torch.full((down_mask.sum(),), -1.0, dtype=dtype, device=device)
    
    # Left neighbor (j-1): valid when j > 0
    left_mask = j > 0
    left_row = idx[left_mask]
    left_col = idx[left_mask] - 1
    left_val = torch.full((left_mask.sum(),), -1.0, dtype=dtype, device=device)
    
    # Right neighbor (j+1): valid when j < n-1
    right_mask = j < n - 1
    right_row = idx[right_mask]
    right_col = idx[right_mask] + 1
    right_val = torch.full((right_mask.sum(),), -1.0, dtype=dtype, device=device)
    
    # Concatenate all entries
    rows = torch.cat([diag_row, up_row, down_row, left_row, right_row])
    cols = torch.cat([diag_col, up_col, down_col, left_col, right_col])
    vals = torch.cat([diag_val, up_val, down_val, left_val, right_val])
    
    return vals, rows, cols, (N, N)


def run_distributed_test(rank: int, world_size: int, backend: str = 'gloo'):
    """Run comprehensive distributed tests on a single process."""
    # Initialize process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    print(f"[Rank {rank}] Initialized (world_size={world_size})")
    
    try:
        from torch_sla import SparseTensor
        import warnings
        warnings.filterwarnings('ignore')
        
        # Create global matrix (same on all ranks)
        n = 4  # 4x4 grid = 16 nodes
        N = n * n
        val, row, col, shape = create_poisson_2d(n)
        
        # =====================================================================
        # Setup: Create distributed partition using SparseTensor.partition_for_rank()
        # =====================================================================
        A_global = SparseTensor(val, row, col, shape)
        partition = A_global.partition_for_rank(rank, world_size, partition_method='simple')
        
        print(f"[Rank {rank}] Partition: owned={partition.num_owned}, halo={partition.num_halo}")
        
        dist.barrier()
        
        # =====================================================================
        # Test 1: Distributed Matrix-Vector Product (DSparseTensor @ Tensor)
        # =====================================================================
        print(f"\n[Rank {rank}] === Test 1: Distributed Matvec ===")
        
        # Global vector x (same on all ranks)
        x_global = torch.arange(1.0, N + 1, dtype=torch.float64)
        
        # Reference: compute global A @ x
        y_global_ref = A_global @ x_global
        
        # Distributed: each rank computes its portion
        # Extract local portion of x (owned + halo)
        owned_nodes = partition.partition.owned_nodes
        halo_nodes = partition.partition.halo_nodes
        
        # Build local x with owned + halo values
        x_local = torch.zeros(partition.num_local, dtype=torch.float64)
        x_local[:partition.num_owned] = x_global[owned_nodes]
        if partition.num_halo > 0:
            x_local[partition.num_owned:] = x_global[halo_nodes]
        
        # Compute local matvec (with halo exchange for boundary consistency)
        y_local = partition.matvec(x_local, exchange_halo=True)
        
        # Extract only owned portion of result
        y_owned = y_local[:partition.num_owned]
        
        # Compare with global reference (only for owned nodes)
        y_global_owned = y_global_ref[owned_nodes]
        
        matvec_error = torch.norm(y_owned - y_global_owned) / torch.norm(y_global_owned)
        matvec_match = matvec_error < 1e-10
        
        print(f"[Rank {rank}] Matvec:")
        print(f"  - Local result (owned): {y_owned[:4].tolist()}")
        print(f"  - Global reference:     {y_global_owned[:4].tolist()}")
        print(f"  - Relative error: {matvec_error:.2e}")
        print(f"  - Match: {matvec_match}")
        
        assert matvec_match, f"Rank {rank}: Matvec mismatch!"
        
        dist.barrier()
        
        # =====================================================================
        # Test 2: Distributed CG Solve (A^{-1} b) with global reductions
        # =====================================================================
        print(f"\n[Rank {rank}] === Test 2: Distributed CG Solve ===")
        
        # Global RHS b (same on all ranks)
        b_global = torch.ones(N, dtype=torch.float64)
        
        # Reference: solve globally
        x_global_ref = A_global.solve(b_global)
        
        # Prepare for distributed CG
        local_size = torch.tensor([partition.num_owned], dtype=torch.int64)
        all_sizes = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        max_size = max(s.item() for s in all_sizes)
        
        owned_padded = torch.zeros(max_size, dtype=torch.int64)
        owned_padded[:partition.num_owned] = owned_nodes
        all_owned = [torch.zeros(max_size, dtype=torch.int64) for _ in range(world_size)]
        dist.all_gather(all_owned, owned_padded)
        
        # Initialize local x and r
        x_local = torch.zeros(partition.num_local, dtype=torch.float64)
        b_local = b_global[owned_nodes]
        
        Ax_local = partition.matvec(x_local, exchange_halo=True)
        r_local = torch.zeros(partition.num_local, dtype=torch.float64)
        r_local[:partition.num_owned] = b_local - Ax_local[:partition.num_owned]
        p_local = r_local.clone()
        
        # Global ||r||^2
        rs_local = torch.dot(r_local[:partition.num_owned], r_local[:partition.num_owned])
        rs_global = rs_local.clone()
        dist.all_reduce(rs_global, op=dist.ReduceOp.SUM)
        
        # CG iterations with global reductions
        atol, maxiter = 1e-10, 500
        for iteration in range(maxiter):
            # Gather p to global and update halo
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
            
            # Ap = A @ p
            Ap_local = partition.matvec(p_local, exchange_halo=False)
            
            # pAp = p^T @ Ap (global reduction)
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
        
        # Compare with reference
        x_global_owned = x_global_ref[owned_nodes]
        solve_error = torch.norm(x_local[:partition.num_owned] - x_global_owned) / torch.norm(x_global_owned)
        solve_match = solve_error < 1e-6
        
        print(f"[Rank {rank}] Distributed CG Solve:")
        print(f"  - Local solution:    {x_local[:4].tolist()}")
        print(f"  - Global reference:  {x_global_owned[:4].tolist()}")
        print(f"  - Relative error: {solve_error:.2e}")
        print(f"  - Match: {solve_match}")
        
        assert solve_match, f"Rank {rank}: Solve mismatch! Error = {solve_error:.2e}"
        
        dist.barrier()
        
        # =====================================================================
        # Test 3: Gather global solution and verify residual
        # =====================================================================
        print(f"\n[Rank {rank}] === Test 3: Gather and Verify Residual ===")
        
        # Gather solution from all ranks
        solution_padded = torch.zeros(max_size, dtype=torch.float64)
        solution_padded[:partition.num_owned] = x_local[:partition.num_owned]
        all_solutions = [torch.zeros(max_size, dtype=torch.float64) for _ in range(world_size)]
        dist.all_gather(all_solutions, solution_padded)
        
        x_reconstructed = torch.zeros(N, dtype=torch.float64)
        for r_idx in range(world_size):
            size = all_sizes[r_idx].item()
            x_reconstructed[all_owned[r_idx][:size]] = all_solutions[r_idx][:size]
        
        # Compute global residual
        residual = A_global @ x_reconstructed - b_global
        residual_norm = torch.norm(residual) / torch.norm(b_global)
        
        print(f"[Rank {rank}] Gathered solution verification:")
        print(f"  - Reconstructed solution: {x_reconstructed[:4].tolist()}")
        print(f"  - Global residual ||Ax-b||/||b||: {residual_norm:.2e}")
        
        residual_ok = residual_norm < 1e-6
        assert residual_ok, f"Rank {rank}: Residual too large! {residual_norm:.2e}"
        
        dist.barrier()
        
        if rank == 0:
            print("\n" + "=" * 60)
            print("  ALL DISTRIBUTED TESTS PASSED!")
            print("=" * 60)
            
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def main():
    """Main entry point."""
    # Default world size, can be overridden via command line
    world_size = int(os.environ.get('WORLD_SIZE_OVERRIDE', '2'))
    
    # Check if running in distributed mode already
    if 'RANK' in os.environ:
        # Running via torch.distributed.launch or torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Determine backend
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        
        run_distributed_test(rank, world_size, backend)
    else:
        # Spawn processes manually
        print("=" * 60)
        print("  Distributed Sparse Tensor Tests")
        print("=" * 60)
        print(f"Backend: gloo")
        print(f"World size: {world_size}\n")
        
        spawn(
            run_distributed_test,
            args=(world_size, 'gloo'),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    main()
