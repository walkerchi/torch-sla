#!/usr/bin/env python
"""
Benchmark: Preconditioner Comparison for CG

Compares iteration counts and solve times for different preconditioners:
- Jacobi (baseline)
- SSOR
- Block Jacobi
- Polynomial
- IC0 (Incomplete Cholesky)
- AMG (Algebraic Multigrid)

Key metrics:
- Iteration count (lower = better preconditioner)
- Total solve time (includes preconditioner application cost)
- Speedup vs Jacobi
"""

import torch
import time
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_sla.backends.pytorch_backend import (
    CachedSparseMatrix,
    pcg_solve_optimized,
    get_preconditioner,
)

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class PrecondResult:
    name: str
    dof: int
    time_ms: float
    num_iters: int
    residual: float
    converged: bool
    setup_time_ms: float = 0.0


def create_poisson_2d(grid_n: int, device: str = 'cuda', dtype=torch.float64):
    """Create 2D Poisson matrix (5-point stencil)."""
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


def create_anisotropic_poisson_2d(grid_n: int, anisotropy: float = 100.0, 
                                   device: str = 'cuda', dtype=torch.float64):
    """Create anisotropic 2D Poisson matrix (harder for simple preconditioners)."""
    N = grid_n * grid_n
    idx = torch.arange(N, device=device)
    i, j = idx // grid_n, idx % grid_n
    
    # Anisotropic coefficients: eps in x-direction, 1 in y-direction
    eps = 1.0 / anisotropy
    
    entries = [
        (idx, idx, torch.full((N,), 2.0 * (eps + 1.0), dtype=dtype, device=device)),
        (idx[j > 0], idx[j > 0] - 1, torch.full(((j > 0).sum(),), -eps, dtype=dtype, device=device)),
        (idx[j < grid_n-1], idx[j < grid_n-1] + 1, torch.full(((j < grid_n-1).sum(),), -eps, dtype=dtype, device=device)),
        (idx[i > 0], idx[i > 0] - grid_n, torch.full(((i > 0).sum(),), -1.0, dtype=dtype, device=device)),
        (idx[i < grid_n-1], idx[i < grid_n-1] + grid_n, torch.full(((i < grid_n-1).sum(),), -1.0, dtype=dtype, device=device)),
    ]
    
    rows = torch.cat([e[0] for e in entries])
    cols = torch.cat([e[1] for e in entries])
    vals = torch.cat([e[2] for e in entries])
    
    return vals, rows, cols, (N, N)


def benchmark_preconditioner(A: CachedSparseMatrix, b: torch.Tensor, 
                             precond_name: str, **solve_kwargs) -> PrecondResult:
    """Benchmark a single preconditioner."""
    dof = A.n
    
    try:
        # Setup timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        M = get_preconditioner(A, precond_name)
        torch.cuda.synchronize()
        setup_time = (time.perf_counter() - t0) * 1000
        
        # Warmup
        _ = pcg_solve_optimized(A, b, preconditioner=M, maxiter=50, check_interval=10)
        torch.cuda.synchronize()
        
        # Timed run
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = pcg_solve_optimized(A, b, preconditioner=M, **solve_kwargs)
        torch.cuda.synchronize()
        solve_time = (time.perf_counter() - t0) * 1000
        
        return PrecondResult(
            name=precond_name,
            dof=dof,
            time_ms=solve_time,
            num_iters=result.num_iters,
            residual=result.residual,
            converged=result.converged,
            setup_time_ms=setup_time,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return PrecondResult(
            name=precond_name,
            dof=dof,
            time_ms=float('inf'),
            num_iters=-1,
            residual=float('inf'),
            converged=False,
        )


def run_benchmark():
    print("=" * 80)
    print("Preconditioner Comparison Benchmark")
    print("=" * 80)
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64
    
    # Test configurations
    preconditioners = ['jacobi', 'ssor', 'polynomial', 'ic0', 'amg']
    grid_sizes = [100, 200, 316, 500]  # DOF: 10K, 40K, 100K, 250K
    
    solve_kwargs = {
        'atol': 1e-10,
        'rtol': 1e-8,
        'maxiter': 20000,
        'check_interval': 50,
    }
    
    results = {name: [] for name in preconditioners}
    
    # Standard Poisson
    print("\n" + "=" * 80)
    print("Test 1: Standard 2D Poisson (5-point stencil)")
    print("=" * 80)
    
    for grid_n in grid_sizes:
        dof = grid_n * grid_n
        print(f"\n--- Grid {grid_n}x{grid_n} = {dof:,} DOF ---")
        
        # Create problem
        val, row, col, shape = create_poisson_2d(grid_n, device, dtype)
        A = CachedSparseMatrix(val, row, col, shape)
        b = torch.randn(dof, dtype=dtype, device=device)
        
        print(f"{'Precond':<12} {'Iters':>8} {'Time (ms)':>12} {'Setup (ms)':>12} {'Residual':>12}")
        print("-" * 60)
        
        for precond_name in preconditioners:
            result = benchmark_preconditioner(A, b, precond_name, **solve_kwargs)
            results[precond_name].append(result)
            
            status = "✓" if result.converged else "✗"
            print(f"{precond_name:<12} {result.num_iters:>8} {result.time_ms:>12.1f} "
                  f"{result.setup_time_ms:>12.1f} {result.residual:>12.1e} {status}")
    
    # Anisotropic Poisson (harder problem)
    print("\n" + "=" * 80)
    print("Test 2: Anisotropic 2D Poisson (ratio 100:1)")
    print("=" * 80)
    
    results_aniso = {name: [] for name in preconditioners}
    
    for grid_n in grid_sizes[:3]:  # Smaller sizes for harder problem
        dof = grid_n * grid_n
        print(f"\n--- Grid {grid_n}x{grid_n} = {dof:,} DOF ---")
        
        val, row, col, shape = create_anisotropic_poisson_2d(grid_n, anisotropy=100.0, 
                                                              device=device, dtype=dtype)
        A = CachedSparseMatrix(val, row, col, shape)
        b = torch.randn(dof, dtype=dtype, device=device)
        
        print(f"{'Precond':<12} {'Iters':>8} {'Time (ms)':>12} {'Residual':>12}")
        print("-" * 50)
        
        for precond_name in preconditioners:
            result = benchmark_preconditioner(A, b, precond_name, **solve_kwargs)
            results_aniso[precond_name].append(result)
            
            status = "✓" if result.converged else "✗"
            print(f"{precond_name:<12} {result.num_iters:>8} {result.time_ms:>12.1f} "
                  f"{result.residual:>12.1e} {status}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Standard Poisson")
    print("=" * 80)
    
    # Iteration reduction table
    print("\nIteration Count:")
    print(f"{'Precond':<12}", end="")
    for grid_n in grid_sizes:
        print(f"{grid_n*grid_n:>12,}", end="")
    print()
    print("-" * (12 + 12 * len(grid_sizes)))
    
    for precond_name in preconditioners:
        print(f"{precond_name:<12}", end="")
        for r in results[precond_name]:
            print(f"{r.num_iters:>12}", end="")
        print()
    
    # Speedup vs Jacobi
    print("\nSpeedup vs Jacobi:")
    print(f"{'Precond':<12}", end="")
    for grid_n in grid_sizes:
        print(f"{grid_n*grid_n:>12,}", end="")
    print()
    print("-" * (12 + 12 * len(grid_sizes)))
    
    jacobi_times = [r.time_ms for r in results['jacobi']]
    for precond_name in preconditioners:
        print(f"{precond_name:<12}", end="")
        for i, r in enumerate(results[precond_name]):
            if jacobi_times[i] > 0:
                speedup = jacobi_times[i] / r.time_ms
                print(f"{speedup:>11.2f}x", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()
    
    # Generate plot
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax_iters, ax_time, ax_speedup = axes
        
        colors = {
            'jacobi': '#7f8c8d',
            'ssor': '#3498db',
            'polynomial': '#9b59b6',
            'ic0': '#e74c3c',
            'amg': '#2ecc71',
        }
        
        markers = {'jacobi': 'o', 'ssor': 's', 'polynomial': '^', 'ic0': 'D', 'amg': 'p'}
        
        dofs = [grid_n * grid_n for grid_n in grid_sizes]
        
        # Iterations plot
        for precond_name in preconditioners:
            iters = [r.num_iters for r in results[precond_name]]
            ax_iters.plot(dofs, iters, marker=markers[precond_name], 
                         color=colors[precond_name], linewidth=2, markersize=8,
                         label=precond_name.upper())
        
        ax_iters.set_xlabel('Degrees of Freedom', fontsize=11)
        ax_iters.set_ylabel('Iterations', fontsize=11)
        ax_iters.set_title('Iteration Count', fontsize=12, fontweight='bold')
        ax_iters.legend(loc='upper left', fontsize=9)
        ax_iters.grid(True, alpha=0.3)
        ax_iters.set_xscale('log')
        ax_iters.set_yscale('log')
        
        # Time plot
        for precond_name in preconditioners:
            times = [r.time_ms for r in results[precond_name]]
            ax_time.plot(dofs, times, marker=markers[precond_name],
                        color=colors[precond_name], linewidth=2, markersize=8,
                        label=precond_name.upper())
        
        ax_time.set_xlabel('Degrees of Freedom', fontsize=11)
        ax_time.set_ylabel('Solve Time (ms)', fontsize=11)
        ax_time.set_title('Total Solve Time', fontsize=12, fontweight='bold')
        ax_time.legend(loc='upper left', fontsize=9)
        ax_time.grid(True, alpha=0.3)
        ax_time.set_xscale('log')
        ax_time.set_yscale('log')
        
        # Speedup plot
        for precond_name in preconditioners:
            if precond_name == 'jacobi':
                continue
            speedups = [jacobi_times[i] / r.time_ms 
                       for i, r in enumerate(results[precond_name])]
            ax_speedup.plot(dofs, speedups, marker=markers[precond_name],
                           color=colors[precond_name], linewidth=2, markersize=8,
                           label=precond_name.upper())
        
        ax_speedup.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
        ax_speedup.set_xlabel('Degrees of Freedom', fontsize=11)
        ax_speedup.set_ylabel('Speedup vs Jacobi', fontsize=11)
        ax_speedup.set_title('Speedup', fontsize=12, fontweight='bold')
        ax_speedup.legend(loc='best', fontsize=9)
        ax_speedup.grid(True, alpha=0.3)
        ax_speedup.set_xscale('log')
        
        plt.suptitle('Preconditioner Comparison (2D Poisson, float64)', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = OUTPUT_DIR / "benchmark_preconditioners.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved to: {output_path}")
        
    except ImportError:
        print("\nmatplotlib not available, skipping plot")
    
    print("\nDone!")
    return results


if __name__ == '__main__':
    run_benchmark()

