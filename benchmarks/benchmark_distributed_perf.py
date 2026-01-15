#!/usr/bin/env python
"""
Distributed Solver Performance Benchmark

Compare different optimization strategies for DSparseTensor:
1. Baseline (no optimization)
2. CSR caching
3. Jacobi preconditioner
4. Communication-computation overlap
5. All optimizations combined

Measures:
- Solve time
- Memory usage
- Iterations to convergence
- Speedup vs baseline

Usage:
    # Single process (simulated partitions)
    python benchmark_distributed_perf.py

    # Multi-GPU distributed
    torchrun --standalone --nproc_per_node=4 benchmark_distributed_perf.py --distributed
"""

import argparse
import time
import gc
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

import torch

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    n: int
    dof: int
    num_partitions: int
    time_ms: float
    memory_mb: float
    iterations: int
    residual: float
    speedup: float = 1.0


def create_2d_poisson(n: int, dtype=torch.float64):
    """Create 2D Poisson matrix with 5-point stencil."""
    N = n * n
    row, col, val = [], [], []
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            val.append(4.0)
            row.append(idx)
            col.append(idx)
            if j > 0:
                val.append(-1.0)
                row.append(idx)
                col.append(idx - 1)
            if j < n - 1:
                val.append(-1.0)
                row.append(idx)
                col.append(idx + 1)
            if i > 0:
                val.append(-1.0)
                row.append(idx)
                col.append(idx - n)
            if i < n - 1:
                val.append(-1.0)
                row.append(idx)
                col.append(idx + n)
    
    return (
        torch.tensor(val, dtype=dtype),
        torch.tensor(row, dtype=torch.int64),
        torch.tensor(col, dtype=torch.int64),
        (N, N)
    )


def get_memory_usage(device):
    """Get current memory usage in MB."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated(device) / 1024 / 1024
    else:
        # For CPU, use process memory (approximate)
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0


def reset_memory(device):
    """Reset memory tracking."""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


class IterationCounter:
    """Count CG iterations."""
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1


def benchmark_solve(
    dsparse,
    b: torch.Tensor,
    config: Dict,
    warmup: int = 3,
    repeat: int = 5
) -> Dict:
    """
    Benchmark a solve configuration.
    
    Returns dict with time, memory, iterations, residual.
    """
    device = dsparse.device
    maxiter = config.get('maxiter', 1000)
    rtol = config.get('rtol', 1e-6)
    preconditioner = config.get('preconditioner', 'none')
    overlap = config.get('overlap', False)
    
    # Warmup
    for _ in range(warmup):
        x = dsparse.solve(
            b, 
            preconditioner=preconditioner,
            rtol=rtol,
            maxiter=min(10, maxiter),
            overlap=overlap,
            verbose=False
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Reset memory tracking
    reset_memory(device)
    mem_before = get_memory_usage(device)
    
    # Benchmark
    times = []
    for _ in range(repeat):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        x = dsparse.solve(
            b,
            preconditioner=preconditioner,
            rtol=rtol,
            maxiter=maxiter,
            overlap=overlap,
            verbose=False
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    mem_after = get_memory_usage(device)
    
    # Compute residual
    # Need to expand x to local size for matvec
    x_full = torch.zeros(dsparse.num_local, dtype=b.dtype, device=device)
    x_full[:dsparse.num_owned] = x
    dsparse.halo_exchange(x_full)
    Ax = dsparse.matvec(x_full, exchange_halo=False)
    
    b_full = torch.zeros(dsparse.num_local, dtype=b.dtype, device=device)
    b_full[:dsparse.num_owned] = b
    
    residual = (Ax[:dsparse.num_owned] - b[:dsparse.num_owned]).norm().item()
    
    return {
        'time_ms': min(times) * 1000,
        'memory_mb': max(0, mem_after - mem_before),
        'residual': residual,
    }


def run_single_process_benchmark(args):
    """Run benchmark in single process mode."""
    from torch_sla import SparseTensor
    
    device = torch.device(args.device)
    dtype = torch.float64
    
    print("=" * 70)
    print("DSparseTensor Performance Benchmark")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Partitions: {args.num_partitions}")
    print(f"Sizes: {args.sizes}")
    print()
    
    # Define optimization configurations
    configs = [
        {
            'name': 'Baseline (no precond)',
            'preconditioner': 'none',
            'overlap': False,
        },
        {
            'name': 'Jacobi Precond',
            'preconditioner': 'jacobi',
            'overlap': False,
        },
        {
            'name': 'Block Jacobi',
            'preconditioner': 'block_jacobi',
            'overlap': False,
        },
        {
            'name': 'Jacobi + Overlap',
            'preconditioner': 'jacobi',
            'overlap': True,
        },
    ]
    
    all_results = []
    
    for n in args.sizes:
        N = n * n
        print(f"\n{'='*70}")
        print(f"Grid {n}×{n} = {N:,} DOF")
        print(f"{'='*70}")
        
        # Create matrix
        val, row, col, shape = create_2d_poisson(n, dtype=dtype)
        A = SparseTensor(val, row, col, shape).to(device)
        D = A.partition(num_partitions=args.num_partitions)
        
        # Get first partition for testing
        dsparse = D._partitions[0]
        
        # Create RHS
        b = torch.ones(dsparse.num_owned, dtype=dtype, device=device)
        
        # Print partition info
        print(f"Partition 0: owned={dsparse.num_owned}, halo={dsparse.num_halo}, nnz={dsparse.nnz}")
        
        # Build decomposition for stats
        dsparse._build_interior_boundary_decomposition()
        stats = dsparse._overlap_stats
        print(f"Interior ratio: {stats.get('interior_ratio', 0):.1%}")
        print()
        
        # Header
        print(f"{'Configuration':<25} {'Time (ms)':>12} {'Memory (MB)':>12} {'Residual':>12} {'Speedup':>10}")
        print("-" * 75)
        
        baseline_time = None
        
        for config in configs:
            config['maxiter'] = args.maxiter
            config['rtol'] = args.rtol
            
            result = benchmark_solve(dsparse, b, config, warmup=3, repeat=5)
            
            if baseline_time is None:
                baseline_time = result['time_ms']
            
            speedup = baseline_time / result['time_ms'] if result['time_ms'] > 0 else 1.0
            
            print(f"{config['name']:<25} {result['time_ms']:>12.2f} {result['memory_mb']:>12.1f} "
                  f"{result['residual']:>12.2e} {speedup:>10.2f}x")
            
            all_results.append(BenchmarkResult(
                name=config['name'],
                n=n,
                dof=N,
                num_partitions=args.num_partitions,
                time_ms=result['time_ms'],
                memory_mb=result['memory_mb'],
                iterations=0,  # TODO: track iterations
                residual=result['residual'],
                speedup=speedup,
            ))
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary: Speedup vs Baseline")
    print("=" * 70)
    
    # Group by name
    from collections import defaultdict
    by_name = defaultdict(list)
    for r in all_results:
        by_name[r.name].append(r)
    
    print(f"\n{'Configuration':<25}", end='')
    for n in args.sizes:
        print(f" {n}×{n}".rjust(10), end='')
    print()
    print("-" * (25 + 10 * len(args.sizes)))
    
    for name in [c['name'] for c in configs]:
        results = by_name[name]
        print(f"{name:<25}", end='')
        for r in results:
            print(f" {r.speedup:>10.2f}x", end='')
        print()
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    with open(f'{args.output}/perf_results.json', 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {args.output}/perf_results.json")
    
    # Plot
    if HAS_MATPLOTLIB:
        plot_results(all_results, configs, args)


def plot_results(results: List[BenchmarkResult], configs: List[Dict], args):
    """Generate performance comparison plots."""
    import numpy as np
    
    sizes = args.sizes
    n_sizes = len(sizes)
    n_configs = len(configs)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Colors for each configuration
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    # Group results
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in results:
        by_config[r.name].append(r)
    
    # Plot 1: Solve Time
    ax = axes[0]
    x = np.arange(n_sizes)
    width = 0.2
    
    for i, config in enumerate(configs):
        times = [r.time_ms for r in by_config[config['name']]]
        offset = (i - n_configs/2 + 0.5) * width
        bars = ax.bar(x + offset, times, width, label=config['name'], color=colors[i])
    
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Solve Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}×{n}' for n in sizes])
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    
    # Plot 2: Speedup
    ax = axes[1]
    for i, config in enumerate(configs):
        speedups = [r.speedup for r in by_config[config['name']]]
        ax.plot(range(n_sizes), speedups, 'o-', label=config['name'], 
                color=colors[i], linewidth=2, markersize=8)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Speedup vs Baseline')
    ax.set_title('Optimization Speedup')
    ax.set_xticks(range(n_sizes))
    ax.set_xticklabels([f'{n}×{n}' for n in sizes])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Memory Usage
    ax = axes[2]
    for i, config in enumerate(configs):
        mems = [r.memory_mb for r in by_config[config['name']]]
        offset = (i - n_configs/2 + 0.5) * width
        ax.bar(x + offset, mems, width, label=config['name'], color=colors[i])
    
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Memory Usage')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}×{n}' for n in sizes])
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{args.output}/perf_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to {args.output}/perf_comparison.png")
    plt.close()
    
    # Additional plot: Scaling
    fig, ax = plt.subplots(figsize=(8, 6))
    
    dofs = [n*n for n in sizes]
    
    for i, config in enumerate(configs):
        times = [r.time_ms for r in by_config[config['name']]]
        ax.loglog(dofs, times, 'o-', label=config['name'], 
                  color=colors[i], linewidth=2, markersize=8)
    
    # Reference lines
    if len(dofs) > 1:
        # O(n) reference
        ref_times = [times[0] * (d / dofs[0]) for d in dofs]
        ax.loglog(dofs, ref_times, '--', color='gray', alpha=0.5, label='O(n)')
    
    ax.set_xlabel('DOF')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Scaling: Time vs Problem Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{args.output}/perf_scaling.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to {args.output}/perf_scaling.png")
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
        print("=" * 70)
        print("Distributed DSparseTensor Performance Benchmark")
        print("=" * 70)
        print(f"World size: {world_size}")
        print(f"Device: {device}")
    
    configs = [
        {'name': 'Baseline', 'preconditioner': 'none', 'overlap': False},
        {'name': 'Jacobi', 'preconditioner': 'jacobi', 'overlap': False},
        {'name': 'Jacobi+Overlap', 'preconditioner': 'jacobi', 'overlap': True},
    ]
    
    for n in args.sizes:
        N = n * n
        
        val, row, col, shape = create_2d_poisson(n, dtype=dtype)
        
        dsparse = DSparseMatrix.from_global(
            val, row, col, shape,
            num_partitions=world_size,
            my_partition=rank,
            device=device
        )
        
        b = torch.ones(dsparse.num_owned, dtype=dtype, device=device)
        
        if rank == 0:
            print(f"\n--- Grid {n}×{n} = {N:,} DOF ---")
            print(f"{'Config':<20} {'Time (ms)':>12} {'Speedup':>10}")
            print("-" * 45)
        
        baseline_time = None
        
        for config in configs:
            # Warmup
            for _ in range(3):
                _ = dsparse.solve(b, preconditioner=config['preconditioner'],
                                 overlap=config['overlap'], maxiter=10, verbose=False)
            dist.barrier()
            
            # Benchmark
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dist.barrier()
            
            t0 = time.perf_counter()
            x = dsparse.solve(b, preconditioner=config['preconditioner'],
                            overlap=config['overlap'], maxiter=args.maxiter,
                            rtol=args.rtol, verbose=False)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dist.barrier()
            
            elapsed = (time.perf_counter() - t0) * 1000
            
            if baseline_time is None:
                baseline_time = elapsed
            
            speedup = baseline_time / elapsed if elapsed > 0 else 1.0
            
            if rank == 0:
                print(f"{config['name']:<20} {elapsed:>12.2f} {speedup:>10.2f}x")
    
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='DSparseTensor Performance Benchmark')
    parser.add_argument('--sizes', type=int, nargs='+', default=[50, 100, 200, 500],
                        help='Grid sizes to test')
    parser.add_argument('--num-partitions', type=int, default=4,
                        help='Number of partitions')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--maxiter', type=int, default=1000,
                        help='Maximum CG iterations')
    parser.add_argument('--rtol', type=float, default=1e-6,
                        help='Relative tolerance')
    parser.add_argument('--distributed', action='store_true',
                        help='Run in distributed mode')
    parser.add_argument('--output', type=str, default='results/distributed_perf',
                        help='Output directory')
    
    args = parser.parse_args()
    
    if args.distributed:
        run_distributed_benchmark(args)
    else:
        run_single_process_benchmark(args)


if __name__ == '__main__':
    main()

