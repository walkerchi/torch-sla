#!/usr/bin/env python
"""
Benchmark communication-computation overlap in distributed sparse solvers.

This script measures the performance improvement from overlapping
halo exchange communication with interior node computation.

Usage:
    # Single process (simulated partitions)
    python benchmark_overlap.py
    
    # Multi-GPU distributed
    torchrun --standalone --nproc_per_node=4 benchmark_overlap.py --distributed
"""

import argparse
import time
import torch
import numpy as np

# For plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def create_2d_poisson(n: int, dtype=torch.float64, device='cpu'):
    """Create 2D Poisson matrix with 5-point stencil."""
    N = n * n
    row, col, val = [], [], []
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            # Diagonal
            row.append(idx)
            col.append(idx)
            val.append(4.0)
            # Left
            if j > 0:
                row.append(idx)
                col.append(idx - 1)
                val.append(-1.0)
            # Right
            if j < n - 1:
                row.append(idx)
                col.append(idx + 1)
                val.append(-1.0)
            # Up
            if i > 0:
                row.append(idx)
                col.append(idx - n)
                val.append(-1.0)
            # Down
            if i < n - 1:
                row.append(idx)
                col.append(idx + n)
                val.append(-1.0)
    
    return (
        torch.tensor(val, dtype=dtype, device=device),
        torch.tensor(row, dtype=torch.int64, device=device),
        torch.tensor(col, dtype=torch.int64, device=device),
        (N, N)
    )


def benchmark_matvec(dsparse, x, num_iters=100, warmup=10):
    """Benchmark matvec with and without overlap."""
    device = dsparse.device
    
    # Warmup
    for _ in range(warmup):
        _ = dsparse.matvec(x.clone())
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Benchmark without overlap
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        y = dsparse.matvec(x.clone())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_no_overlap = (time.perf_counter() - t0) / num_iters
    
    # Warmup overlap
    for _ in range(warmup):
        _ = dsparse.matvec_overlap(x.clone())
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Benchmark with overlap
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        y = dsparse.matvec_overlap(x.clone())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_overlap = (time.perf_counter() - t0) / num_iters
    
    return t_no_overlap, t_overlap


def benchmark_solve(dsparse, b, maxiter=100, warmup=3):
    """Benchmark solve with and without overlap."""
    device = dsparse.device
    
    # Warmup
    for _ in range(warmup):
        _ = dsparse.solve(b, maxiter=10, overlap=False, verbose=False)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Benchmark without overlap
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    x_no_overlap = dsparse.solve(b, maxiter=maxiter, overlap=False, verbose=False)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_no_overlap = time.perf_counter() - t0
    
    # Warmup overlap
    for _ in range(warmup):
        _ = dsparse.solve(b, maxiter=10, overlap=True, verbose=False)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Benchmark with overlap
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    x_overlap = dsparse.solve(b, maxiter=maxiter, overlap=True, verbose=False)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_overlap = time.perf_counter() - t0
    
    return t_no_overlap, t_overlap


def run_single_process_benchmark(args):
    """Run benchmark in single process mode (simulated partitions)."""
    from torch_sla import SparseTensor
    from torch_sla.distributed import DSparseTensor
    
    device = torch.device(args.device)
    dtype = torch.float64
    
    print("=" * 60)
    print("Communication-Computation Overlap Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Partitions: {args.num_partitions}")
    print()
    
    results = []
    
    for n in args.sizes:
        N = n * n
        print(f"\n--- Grid {n}x{n} = {N:,} DOF ---")
        
        # Create 2D Poisson matrix
        val, row, col, shape = create_2d_poisson(n, dtype=dtype, device='cpu')
        
        # Create SparseTensor and partition
        A = SparseTensor(val, row, col, shape).to(device)
        D = A.partition(num_partitions=args.num_partitions)
        
        # Get first partition for testing
        dsparse = D._partitions[0]
        
        # Build interior/boundary decomposition and get stats
        dsparse._build_interior_boundary_decomposition()
        stats = dsparse._overlap_stats if hasattr(dsparse, '_overlap_stats') else {}
        
        print(f"  Partition 0: owned={dsparse.num_owned}, halo={dsparse.num_halo}")
        print(f"  Interior ratio: {stats.get('interior_ratio', 0):.1%}")
        print(f"  Interior nnz: {stats.get('interior_nnz', 0):,}")
        print(f"  Boundary nnz: {stats.get('boundary_nnz', 0):,}")
        
        # Create test vectors
        x = torch.randn(dsparse.num_local, dtype=dtype, device=device)
        b = torch.randn(dsparse.num_owned, dtype=dtype, device=device)
        
        # Benchmark matvec
        t_mv_no_overlap, t_mv_overlap = benchmark_matvec(dsparse, x, num_iters=50)
        mv_speedup = t_mv_no_overlap / t_mv_overlap if t_mv_overlap > 0 else 1.0
        
        print(f"\n  Matvec:")
        print(f"    Without overlap: {t_mv_no_overlap*1000:.3f} ms")
        print(f"    With overlap:    {t_mv_overlap*1000:.3f} ms")
        print(f"    Speedup:         {mv_speedup:.2f}x")
        
        # Benchmark solve
        t_solve_no_overlap, t_solve_overlap = benchmark_solve(dsparse, b, maxiter=100)
        solve_speedup = t_solve_no_overlap / t_solve_overlap if t_solve_overlap > 0 else 1.0
        
        print(f"\n  Solve (100 CG iterations):")
        print(f"    Without overlap: {t_solve_no_overlap*1000:.1f} ms")
        print(f"    With overlap:    {t_solve_overlap*1000:.1f} ms")
        print(f"    Speedup:         {solve_speedup:.2f}x")
        
        results.append({
            'n': n,
            'N': N,
            'interior_ratio': stats.get('interior_ratio', 0),
            't_mv_no_overlap': t_mv_no_overlap,
            't_mv_overlap': t_mv_overlap,
            'mv_speedup': mv_speedup,
            't_solve_no_overlap': t_solve_no_overlap,
            't_solve_overlap': t_solve_overlap,
            'solve_speedup': solve_speedup,
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'DOF':>10} {'Interior%':>10} {'MV Speedup':>12} {'Solve Speedup':>14}")
    print("-" * 50)
    for r in results:
        print(f"{r['N']:>10,} {r['interior_ratio']:>10.1%} {r['mv_speedup']:>12.2f}x {r['solve_speedup']:>14.2f}x")
    
    # Plot if matplotlib available
    if HAS_MATPLOTLIB and len(results) > 1:
        plot_results(results, args.output)
    
    return results


def plot_results(results, output_dir):
    """Plot benchmark results."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    Ns = [r['N'] for r in results]
    
    # Plot 1: Interior ratio
    ax = axes[0]
    ax.bar(range(len(Ns)), [r['interior_ratio'] * 100 for r in results], color='steelblue')
    ax.set_xticks(range(len(Ns)))
    ax.set_xticklabels([f"{r['n']}x{r['n']}" for r in results])
    ax.set_ylabel('Interior Ratio (%)')
    ax.set_xlabel('Grid Size')
    ax.set_title('Matrix Interior Ratio')
    ax.set_ylim(0, 100)
    
    # Plot 2: Matvec speedup
    ax = axes[1]
    ax.bar(range(len(Ns)), [r['mv_speedup'] for r in results], color='forestgreen')
    ax.axhline(y=1.0, color='red', linestyle='--', label='Baseline')
    ax.set_xticks(range(len(Ns)))
    ax.set_xticklabels([f"{r['n']}x{r['n']}" for r in results])
    ax.set_ylabel('Speedup')
    ax.set_xlabel('Grid Size')
    ax.set_title('Matvec Speedup with Overlap')
    ax.legend()
    
    # Plot 3: Solve speedup
    ax = axes[2]
    ax.bar(range(len(Ns)), [r['solve_speedup'] for r in results], color='darkorange')
    ax.axhline(y=1.0, color='red', linestyle='--', label='Baseline')
    ax.set_xticks(range(len(Ns)))
    ax.set_xticklabels([f"{r['n']}x{r['n']}" for r in results])
    ax.set_ylabel('Speedup')
    ax.set_xlabel('Grid Size')
    ax.set_title('CG Solve Speedup with Overlap')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overlap_benchmark.png', dpi=150)
    print(f"\nPlot saved to {output_dir}/overlap_benchmark.png")
    plt.close()


def run_distributed_benchmark(args):
    """Run benchmark in distributed mode."""
    import torch.distributed as dist
    from torch_sla.distributed import DSparseMatrix
    
    dist.init_process_group(backend='nccl' if args.device == 'cuda' else 'gloo')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if args.device == 'cuda':
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    else:
        device = torch.device('cpu')
    
    dtype = torch.float64
    
    if rank == 0:
        print("=" * 60)
        print("Distributed Overlap Benchmark")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Device: {device}")
    
    for n in args.sizes:
        N = n * n
        
        # Create 2D Poisson matrix (on all ranks)
        val, row, col, shape = create_2d_poisson(n, dtype=dtype, device='cpu')
        
        # Create distributed matrix
        dsparse = DSparseMatrix.from_global(
            val, row, col, shape,
            num_partitions=world_size,
            my_partition=rank,
            device=device
        )
        
        # Build overlap decomposition
        dsparse._build_interior_boundary_decomposition()
        
        if rank == 0:
            print(f"\n--- Grid {n}x{n} = {N:,} DOF ---")
            stats = dsparse._overlap_stats
            print(f"  Interior ratio: {stats.get('interior_ratio', 0):.1%}")
        
        # Create test vectors
        b = torch.randn(dsparse.num_owned, dtype=dtype, device=device)
        
        # Warmup
        for _ in range(3):
            _ = dsparse.solve(b, maxiter=10, overlap=False, verbose=False)
        dist.barrier()
        
        # Benchmark without overlap
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        x_no_overlap = dsparse.solve(b, maxiter=100, overlap=False, verbose=False)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dist.barrier()
        t_no_overlap = time.perf_counter() - t0
        
        # Warmup overlap
        for _ in range(3):
            _ = dsparse.solve(b, maxiter=10, overlap=True, verbose=False)
        dist.barrier()
        
        # Benchmark with overlap
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        x_overlap = dsparse.solve(b, maxiter=100, overlap=True, verbose=False)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dist.barrier()
        t_overlap = time.perf_counter() - t0
        
        speedup = t_no_overlap / t_overlap if t_overlap > 0 else 1.0
        
        if rank == 0:
            print(f"  Without overlap: {t_no_overlap*1000:.1f} ms")
            print(f"  With overlap:    {t_overlap*1000:.1f} ms")
            print(f"  Speedup:         {speedup:.2f}x")
    
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Benchmark communication-computation overlap')
    parser.add_argument('--sizes', type=int, nargs='+', default=[50, 100, 200, 500],
                        help='Grid sizes to test (n for n×n grid)')
    parser.add_argument('--num-partitions', type=int, default=4,
                        help='Number of partitions (for single process mode)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--distributed', action='store_true',
                        help='Run in distributed mode (use with torchrun)')
    parser.add_argument('--output', type=str, default='results/overlap',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.distributed:
        run_distributed_benchmark(args)
    else:
        run_single_process_benchmark(args)


if __name__ == '__main__':
    main()

