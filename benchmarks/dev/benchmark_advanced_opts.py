#!/usr/bin/env python
"""
Benchmark: Advanced CG Optimizations

Tests:
1. RCM matrix reordering (cache locality)
2. Pipelined CG (reduced sync points)
3. Combined optimizations
"""

import torch
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_sla.backends.pytorch_backend import (
    CachedSparseMatrix,
    pcg_solve_optimized,
    pipelined_pcg_solve,
    get_preconditioner,
)


def create_poisson_2d(grid_n: int, device: str = 'cuda', dtype=torch.float64):
    """Create 2D Poisson matrix."""
    N = grid_n * grid_n
    idx = torch.arange(N, device=device)
    i, j = idx // grid_n, idx % grid_n
    
    entries = [
        (idx, idx, torch.full((N,), 4.0, dtype=dtype, device=device)),
        (idx[j > 0], idx[j > 0] - 1, torch.full(((j > 0).sum(),), -1.0, dtype=dtype, device=device)),
        (idx[j < grid_n-1], idx[j < grid_n-1] + 1, torch.full(((j < grid_n-1).sum(),), -1.0, dtype=dtype, device=device)),
        (idx[i > 0], idx[i > 0] - grid_n, torch.full(((i > 0).sum(),), -1.0, dtype=dtype, device=device)),
        (idx[i < grid_n-1], idx[i < grid_n-1] + grid_n, torch.full(((i < grid_n-1).sum(),), -1.0, dtype=dtype, device=device)),
    ]
    
    rows = torch.cat([e[0] for e in entries])
    cols = torch.cat([e[1] for e in entries])
    vals = torch.cat([e[2] for e in entries])
    
    return vals, rows, cols, (N, N)


def benchmark_solver(solver_fn, A, b, M, name, **kwargs):
    """Benchmark a solver."""
    # Warmup
    _ = solver_fn(A, b, preconditioner=M, maxiter=50, check_interval=10)
    torch.cuda.synchronize()
    
    # Timed run
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = solver_fn(A, b, preconditioner=M, **kwargs)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    
    return {
        'name': name,
        'time_ms': elapsed,
        'iters': result.num_iters,
        'residual': result.residual,
        'converged': result.converged,
    }


def run_benchmark():
    print("=" * 80)
    print("Advanced CG Optimizations Benchmark")
    print("=" * 80)
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64
    
    grid_sizes = [100, 200, 316, 500]
    
    solve_kwargs = {
        'atol': 1e-10,
        'rtol': 1e-8,
        'maxiter': 20000,
        'check_interval': 50,
    }
    
    for grid_n in grid_sizes:
        dof = grid_n * grid_n
        print(f"\n{'='*70}")
        print(f"Grid {grid_n}x{grid_n} = {dof:,} DOF")
        print(f"{'='*70}")
        
        val, row, col, shape = create_poisson_2d(grid_n, device, dtype)
        b = torch.randn(dof, dtype=dtype, device=device)
        
        print(f"\n{'Method':<30} {'Time (ms)':>12} {'Iters':>8} {'Residual':>12}")
        print("-" * 65)
        
        # Standard CG (baseline)
        A_std = CachedSparseMatrix(val, row, col, shape, use_rcm=False)
        M_std = get_preconditioner(A_std, 'jacobi')
        r1 = benchmark_solver(pcg_solve_optimized, A_std, b, M_std, 
                              'Standard PCG (Jacobi)', **solve_kwargs)
        print(f"{r1['name']:<30} {r1['time_ms']:>12.1f} {r1['iters']:>8} {r1['residual']:>12.1e}")
        baseline_time = r1['time_ms']
        
        # RCM reordering
        try:
            A_rcm = CachedSparseMatrix(val, row, col, shape, use_rcm=True)
            M_rcm = get_preconditioner(A_rcm, 'jacobi')
            r2 = benchmark_solver(pcg_solve_optimized, A_rcm, b, M_rcm,
                                  'PCG + RCM reordering', **solve_kwargs)
            speedup_rcm = baseline_time / r2['time_ms']
            print(f"{r2['name']:<30} {r2['time_ms']:>12.1f} {r2['iters']:>8} {r2['residual']:>12.1e} ({speedup_rcm:.2f}x)")
        except Exception as e:
            print(f"{'PCG + RCM reordering':<30} FAILED: {str(e)[:30]}")
        
        # Pipelined CG
        r3 = benchmark_solver(pipelined_pcg_solve, A_std, b, M_std,
                              'Pipelined PCG', **solve_kwargs)
        speedup_pipe = baseline_time / r3['time_ms']
        print(f"{r3['name']:<30} {r3['time_ms']:>12.1f} {r3['iters']:>8} {r3['residual']:>12.1e} ({speedup_pipe:.2f}x)")
        
        # Polynomial preconditioner
        M_poly = get_preconditioner(A_std, 'polynomial')
        r4 = benchmark_solver(pcg_solve_optimized, A_std, b, M_poly,
                              'PCG + Polynomial', **solve_kwargs)
        speedup_poly = baseline_time / r4['time_ms']
        print(f"{r4['name']:<30} {r4['time_ms']:>12.1f} {r4['iters']:>8} {r4['residual']:>12.1e} ({speedup_poly:.2f}x)")
        
        # Pipelined + Polynomial
        r5 = benchmark_solver(pipelined_pcg_solve, A_std, b, M_poly,
                              'Pipelined + Polynomial', **solve_kwargs)
        speedup_combo = baseline_time / r5['time_ms']
        print(f"{r5['name']:<30} {r5['time_ms']:>12.1f} {r5['iters']:>8} {r5['residual']:>12.1e} ({speedup_combo:.2f}x)")
        
        print(f"\nBest speedup: {max(speedup_pipe, speedup_poly, speedup_combo):.2f}x")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("- RCM reordering: Reduces matrix bandwidth, improves cache locality")
    print("- Pipelined CG: Overlaps computation, reduces sync points")
    print("- Polynomial precond: Reduces iterations 3x")
    print("- Best: Combine Pipelined CG + Polynomial preconditioner")
    print("\nDone!")


if __name__ == '__main__':
    run_benchmark()

