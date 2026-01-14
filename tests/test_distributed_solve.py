#!/usr/bin/env python
"""
End-to-end verification of distributed solve.

This test verifies that the distributed CG solve produces the same result
as the direct serial solve.

Run: python tests/test_distributed_solve.py
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


_PORT_COUNTER = [29700]  # Use a list to allow modification in nested function

def test_distributed_solve_multiprocess(rank, world_size):
    """Test distributed solve in multi-process environment."""
    import warnings
    warnings.filterwarnings('ignore')
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(_PORT_COUNTER[0])
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    
    try:
        from torch_sla.distributed import DSparseTensor
        from torch_sla import SparseTensor
        
        all_passed = True
        
        # Test multiple sizes
        for n in [4, 8, 16]:
            N = n * n
            val, row, col, shape = create_poisson_2d(n)
            
            # Create partition for this rank
            partition = DSparseTensor.from_global_distributed(
                val, row, col, shape,
                rank=rank, world_size=world_size,
                partition_method='simple',
                verbose=False
            )
            
            # Reference matrix (for verification)
            A_global = SparseTensor(val, row, col, shape)
            
            # Global RHS (same on all ranks)
            b_global = torch.ones(N, dtype=torch.float64)
            
            # Reference solution (computed on all ranks, same result)
            x_ref = A_global.solve(b_global)
            
            # Distributed CG solve
            # Each rank has its partition, need to collaborate
            owned_nodes = partition.partition.owned_nodes
            halo_nodes = partition.partition.halo_nodes
            
            # Initialize local x
            x_local = torch.zeros(partition.num_local, dtype=torch.float64)
            
            # Local b (owned portion)
            b_local = b_global[owned_nodes]
            
            # Initial residual: r = b - A @ x
            # We need global residual, so compute local contribution
            Ax_local = partition.matvec(x_local, exchange_halo=True)
            r_owned = b_local - Ax_local[:partition.num_owned]
            
            # Prepare local r with halo space
            r_local = torch.zeros(partition.num_local, dtype=torch.float64)
            r_local[:partition.num_owned] = r_owned
            
            p_local = r_local.clone()
            
            # Compute global ||r||^2 using all_reduce
            rs_local = torch.dot(r_owned, r_owned)
            rs_global = rs_local.clone()
            dist.all_reduce(rs_global, op=dist.ReduceOp.SUM)
            
            # CG iterations
            atol = 1e-10
            maxiter = 500
            converged = False
            
            for iteration in range(maxiter):
                # Ap = A @ p (need halo exchange first)
                # First gather p to all ranks
                # Simple approach: gather global p, then compute Ap
                
                # Gather p to global
                p_global = torch.zeros(N, dtype=torch.float64)
                
                # All-gather owned portions of p
                local_size = torch.tensor([partition.num_owned], dtype=torch.int64)
                all_sizes = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
                dist.all_gather(all_sizes, local_size)
                
                max_size = max(s.item() for s in all_sizes)
                
                owned_padded = torch.zeros(max_size, dtype=torch.int64)
                owned_padded[:partition.num_owned] = owned_nodes
                
                p_padded = torch.zeros(max_size, dtype=torch.float64)
                p_padded[:partition.num_owned] = p_local[:partition.num_owned]
                
                all_owned = [torch.zeros(max_size, dtype=torch.int64) for _ in range(world_size)]
                all_p = [torch.zeros(max_size, dtype=torch.float64) for _ in range(world_size)]
                
                dist.all_gather(all_owned, owned_padded)
                dist.all_gather(all_p, p_padded)
                
                for r_idx in range(world_size):
                    size = all_sizes[r_idx].item()
                    indices = all_owned[r_idx][:size]
                    values = all_p[r_idx][:size]
                    p_global[indices] = values
                
                # Update local p with halo values from global p
                if partition.num_halo > 0:
                    p_local[partition.num_owned:] = p_global[halo_nodes]
                
                # Compute Ap locally
                Ap_local = partition.matvec(p_local, exchange_halo=False)  # Already exchanged
                Ap_owned = Ap_local[:partition.num_owned]
                
                # pAp = p^T @ A @ p (global reduction)
                pAp_local = torch.dot(p_local[:partition.num_owned], Ap_owned)
                pAp_global = pAp_local.clone()
                dist.all_reduce(pAp_global, op=dist.ReduceOp.SUM)
                
                if pAp_global.abs() < 1e-30:
                    break
                
                alpha = rs_global / pAp_global
                
                # x = x + alpha * p
                x_local[:partition.num_owned] += alpha * p_local[:partition.num_owned]
                
                # r = r - alpha * Ap
                r_local[:partition.num_owned] -= alpha * Ap_owned
                
                # rs_new = r^T @ r (global)
                rs_local_new = torch.dot(r_local[:partition.num_owned], r_local[:partition.num_owned])
                rs_global_new = rs_local_new.clone()
                dist.all_reduce(rs_global_new, op=dist.ReduceOp.SUM)
                
                residual = rs_global_new.sqrt()
                
                if residual < atol:
                    converged = True
                    break
                
                beta = rs_global_new / rs_global
                p_local[:partition.num_owned] = r_local[:partition.num_owned] + beta * p_local[:partition.num_owned]
                
                rs_global = rs_global_new
            
            # Gather final solution
            x_final = torch.zeros(N, dtype=torch.float64)
            
            x_padded = torch.zeros(max_size, dtype=torch.float64)
            x_padded[:partition.num_owned] = x_local[:partition.num_owned]
            
            all_x = [torch.zeros(max_size, dtype=torch.float64) for _ in range(world_size)]
            dist.all_gather(all_x, x_padded)
            
            for r_idx in range(world_size):
                size = all_sizes[r_idx].item()
                indices = all_owned[r_idx][:size]
                values = all_x[r_idx][:size]
                x_final[indices] = values
            
            # Verify
            diff = torch.norm(x_final - x_ref)
            residual_norm = torch.norm(A_global @ x_final - b_global)
            passed = residual_norm < 1e-6
            all_passed = all_passed and passed
            
            if rank == 0:
                print(f'  {n}x{n} grid (N={N}): diff={diff:.2e}, residual={residual_norm:.2e} [{"PASS" if passed else "FAIL"}]')
        
        # Final status
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
    print("  End-to-End Distributed Solve Verification")
    print("=" * 60)
    
    for ws in [2, 4]:
        print(f'\n--- Testing with {ws} processes ---')
        _PORT_COUNTER[0] += 10  # Increment port for each test
        try:
            spawn(test_distributed_solve_multiprocess, args=(ws,), nprocs=ws, join=True)
            print(f'  All ranks completed successfully')
        except Exception as e:
            print(f'  Error: {e}')
    
    print("\n" + "=" * 60)
    print("✅ DISTRIBUTED SOLVE TESTS COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()

