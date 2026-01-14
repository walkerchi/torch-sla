#!/usr/bin/env python
"""
Benchmark: Mixed Precision Preconditioner

Tests the speedup from using float32 preconditioners with float64 solve.

The idea:
- CG accumulation in float64 (high precision)
- Preconditioner computation in float32 (2x faster on GPU)
- Preconditioner only approximates M^{-1}, so low precision is OK
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
    get_preconditioner,
    polynomial_preconditioner,
    polynomial_preconditioner_f32,
    amg_preconditioner,
    amg_preconditioner_f32,
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


def benchmark_precond(A, b, precond_fn, name, **kwargs):
    """Benchmark a preconditioner."""
    M = precond_fn(A)
    
    # Warmup
    _ = pcg_solve_optimized(A, b, preconditioner=M, maxiter=50, check_interval=10)
    torch.cuda.synchronize()
    
    # Timed run
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = pcg_solve_optimized(A, b, preconditioner=M, **kwargs)
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
    print("Mixed Precision Preconditioner Benchmark")
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
    
    print("Comparing float64 vs float32 preconditioners (solve always in float64)")
    print()
    
    for grid_n in grid_sizes:
        dof = grid_n * grid_n
        print(f"\n{'='*70}")
        print(f"Grid {grid_n}x{grid_n} = {dof:,} DOF")
        print(f"{'='*70}")
        
        val, row, col, shape = create_poisson_2d(grid_n, device, dtype)
        A = CachedSparseMatrix(val, row, col, shape)
        b = torch.randn(dof, dtype=dtype, device=device)
        
        print(f"\n{'Preconditioner':<25} {'Time (ms)':>12} {'Iters':>8} {'Residual':>12} {'Status':<8}")
        print("-" * 70)
        
        # Polynomial f64
        r1 = benchmark_precond(A, b, lambda a: polynomial_preconditioner(a, degree=5), 
                               'Polynomial (f64)', **solve_kwargs)
        print(f"{r1['name']:<25} {r1['time_ms']:>12.1f} {r1['iters']:>8} {r1['residual']:>12.1e} "
              f"{'✓' if r1['converged'] else '✗'}")
        
        # Polynomial f32
        r2 = benchmark_precond(A, b, polynomial_preconditioner_f32, 
                               'Polynomial (f32)', **solve_kwargs)
        speedup_poly = r1['time_ms'] / r2['time_ms']
        print(f"{r2['name']:<25} {r2['time_ms']:>12.1f} {r2['iters']:>8} {r2['residual']:>12.1e} "
              f"{'✓' if r2['converged'] else '✗'}  ({speedup_poly:.2f}x)")
        
        # AMG f64
        r3 = benchmark_precond(A, b, amg_preconditioner, 'AMG (f64)', **solve_kwargs)
        print(f"{r3['name']:<25} {r3['time_ms']:>12.1f} {r3['iters']:>8} {r3['residual']:>12.1e} "
              f"{'✓' if r3['converged'] else '✗'}")
        
        # AMG f32
        r4 = benchmark_precond(A, b, amg_preconditioner_f32, 'AMG (f32)', **solve_kwargs)
        speedup_amg = r3['time_ms'] / r4['time_ms']
        print(f"{r4['name']:<25} {r4['time_ms']:>12.1f} {r4['iters']:>8} {r4['residual']:>12.1e} "
              f"{'✓' if r4['converged'] else '✗'}  ({speedup_amg:.2f}x)")
        
        # Jacobi f64 (baseline)
        r5 = benchmark_precond(A, b, lambda a: get_preconditioner(a, 'jacobi'), 
                               'Jacobi (f64)', **solve_kwargs)
        print(f"{r5['name']:<25} {r5['time_ms']:>12.1f} {r5['iters']:>8} {r5['residual']:>12.1e} "
              f"{'✓' if r5['converged'] else '✗'}")
        
        # Jacobi f32 (using mixed_precision wrapper)
        r6 = benchmark_precond(A, b, lambda a: get_preconditioner(a, 'jacobi', mixed_precision=True), 
                               'Jacobi (f32)', **solve_kwargs)
        speedup_jacobi = r5['time_ms'] / r6['time_ms']
        print(f"{r6['name']:<25} {r6['time_ms']:>12.1f} {r6['iters']:>8} {r6['residual']:>12.1e} "
              f"{'✓' if r6['converged'] else '✗'}  ({speedup_jacobi:.2f}x)")
        
        print(f"\nSummary for {dof:,} DOF:")
        print(f"  Polynomial f32 speedup: {speedup_poly:.2f}x")
        print(f"  AMG f32 speedup: {speedup_amg:.2f}x")
        print(f"  Jacobi f32 speedup: {speedup_jacobi:.2f}x")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("Mixed precision preconditioners provide speedup by computing M^{-1}r in float32")
    print("while maintaining float64 accuracy in the CG iteration.")
    print("\nDone!")


if __name__ == '__main__':
    run_benchmark()

