#!/usr/bin/env python
"""
Benchmark: Batched CG vs Sequential CG

Compares solving multiple RHS:
1. Sequential: Solve each RHS one at a time
2. Batched: Solve all RHS together using SpMM

Expected speedup: 1.5-3x for batched due to:
- Better GPU utilization
- Fewer kernel launches
- Shared preconditioner computation
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
    batched_cg_solve,
    get_preconditioner,
)

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


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


def benchmark_sequential(A, B, preconditioner, **kwargs):
    """Solve each RHS sequentially."""
    batch_size = B.shape[1]
    X = torch.zeros_like(B)
    total_iters = 0
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    for j in range(batch_size):
        result = pcg_solve_optimized(A, B[:, j], preconditioner=preconditioner, **kwargs)
        X[:, j] = result.x
        total_iters += result.num_iters
    
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    
    avg_iters = total_iters / batch_size
    return X, elapsed, avg_iters


def benchmark_batched(val, row, col, shape, B, preconditioner_name, **kwargs):
    """Solve all RHS together."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    result = batched_cg_solve(val, row, col, shape, B, preconditioner=preconditioner_name, **kwargs)
    
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    
    avg_iters = result.num_iters.float().mean().item()
    return result.X, elapsed, avg_iters


def compute_relative_residuals(A, X, B):
    """Compute relative residual for each column."""
    AX = torch.mm(A._csr, X)
    residuals = torch.norm(AX - B, dim=0) / torch.norm(B, dim=0)
    return residuals


def run_benchmark():
    print("=" * 80)
    print("Batched CG vs Sequential CG Benchmark")
    print("=" * 80)
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64
    
    # Test configurations
    grid_sizes = [100, 200, 316]  # DOF: 10K, 40K, 100K
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    solve_kwargs = {
        'atol': 1e-10,
        'rtol': 1e-6,
        'maxiter': 10000,
        'check_interval': 50,
    }
    
    results = []
    
    for grid_n in grid_sizes:
        dof = grid_n * grid_n
        print(f"\n{'='*60}")
        print(f"Grid: {grid_n}x{grid_n} = {dof:,} DOF")
        print(f"{'='*60}")
        
        # Create matrix
        val, row, col, shape = create_poisson_2d(grid_n, device, dtype)
        A = CachedSparseMatrix(val, row, col, shape)
        M = get_preconditioner(A, 'jacobi')
        
        # Header
        print(f"\n{'Batch':<8} {'Sequential':<15} {'Batched':<15} {'Speedup':<10} {'Accuracy':<12}")
        print("-" * 60)
        
        for batch_size in batch_sizes:
            # Generate random RHS
            B = torch.randn(dof, batch_size, dtype=dtype, device=device)
            
            # Warmup
            if batch_size == 1:
                _ = pcg_solve_optimized(A, B[:, 0], preconditioner=M, maxiter=50, check_interval=10)
                _ = batched_cg_solve(val, row, col, shape, B, preconditioner='jacobi', maxiter=50, check_interval=10)
                torch.cuda.synchronize()
            
            # Sequential solve
            X_seq, time_seq, iters_seq = benchmark_sequential(A, B, M, **solve_kwargs)
            
            # Batched solve
            X_batch, time_batch, iters_batch = benchmark_batched(
                val, row, col, shape, B, 'jacobi', **solve_kwargs
            )
            
            # Compute accuracy
            res_seq = compute_relative_residuals(A, X_seq, B)
            res_batch = compute_relative_residuals(A, X_batch, B)
            
            speedup = time_seq / time_batch
            
            print(f"{batch_size:<8} {time_seq:>10.1f} ms   {time_batch:>10.1f} ms   {speedup:>7.2f}x   "
                  f"seq:{res_seq.mean():.1e} bat:{res_batch.mean():.1e}")
            
            results.append({
                'dof': dof,
                'batch_size': batch_size,
                'time_sequential_ms': time_seq,
                'time_batched_ms': time_batch,
                'speedup': speedup,
                'iters_seq': iters_seq,
                'iters_batch': iters_batch,
                'res_seq': res_seq.mean().item(),
                'res_batch': res_batch.mean().item(),
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Speedup by Batch Size")
    print("=" * 80)
    
    print(f"\n{'Batch':<8}", end="")
    for grid_n in grid_sizes:
        dof = grid_n * grid_n
        print(f"{dof:>12,} DOF", end="")
    print()
    print("-" * (8 + 15 * len(grid_sizes)))
    
    for batch_size in batch_sizes:
        print(f"{batch_size:<8}", end="")
        for grid_n in grid_sizes:
            dof = grid_n * grid_n
            for r in results:
                if r['dof'] == dof and r['batch_size'] == batch_size:
                    print(f"{r['speedup']:>14.2f}x", end="")
                    break
        print()
    
    # Generate plot
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_speedup, ax_time = axes
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for idx, grid_n in enumerate(grid_sizes):
            dof = grid_n * grid_n
            batch_data = [(r['batch_size'], r['speedup'], r['time_batched_ms']) 
                         for r in results if r['dof'] == dof]
            batch_data.sort(key=lambda x: x[0])
            
            batches = [d[0] for d in batch_data]
            speedups = [d[1] for d in batch_data]
            times = [d[2] for d in batch_data]
            
            ax_speedup.plot(batches, speedups, 'o-', color=colors[idx], 
                           linewidth=2, markersize=8, label=f'{dof:,} DOF')
            ax_time.plot(batches, times, 'o-', color=colors[idx],
                        linewidth=2, markersize=8, label=f'{dof:,} DOF')
        
        ax_speedup.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7)
        ax_speedup.set_xlabel('Batch Size (# RHS)', fontsize=12)
        ax_speedup.set_ylabel('Speedup (Batched / Sequential)', fontsize=12)
        ax_speedup.set_title('Speedup from Batched Solving', fontsize=13, fontweight='bold')
        ax_speedup.legend(loc='best', fontsize=10)
        ax_speedup.grid(True, alpha=0.3)
        ax_speedup.set_xscale('log', base=2)
        
        ax_time.set_xlabel('Batch Size (# RHS)', fontsize=12)
        ax_time.set_ylabel('Time (ms)', fontsize=12)
        ax_time.set_title('Batched Solve Time', fontsize=13, fontweight='bold')
        ax_time.legend(loc='best', fontsize=10)
        ax_time.grid(True, alpha=0.3)
        ax_time.set_xscale('log', base=2)
        ax_time.set_yscale('log')
        
        plt.suptitle('Batched CG vs Sequential CG (2D Poisson, float64)', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = OUTPUT_DIR / "benchmark_batched_cg.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved to: {output_path}")
        
    except ImportError:
        print("\nmatplotlib not available, skipping plot")
    
    print("\nDone!")
    return results


if __name__ == '__main__':
    run_benchmark()

