#!/usr/bin/env python
"""
4-GPU Distributed Solver Performance Benchmark

Run with:
    torchrun --standalone --nproc_per_node=4 benchmark_distributed_4gpu.py

Compare optimization strategies in true multi-GPU distributed setting:
1. Baseline: Standard distributed CG
2. +CSR Cache: Cached sparse matrix format
3. +Jacobi Precond: Diagonal preconditioner
4. +Comm Overlap: Communication-computation overlap
"""

import argparse
import time
import os
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List

import torch
import torch.distributed as dist

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class BenchmarkResult:
    config_name: str
    n: int
    dof: int
    num_gpus: int
    time_ms: float
    memory_mb: float
    residual: float
    speedup: float = 1.0


def create_2d_poisson(n: int, dtype=torch.float64):
    """Create 2D Poisson matrix (5-point stencil)."""
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


def get_gpu_memory_mb(device) -> float:
    """Get peak GPU memory in MB."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return 0


def reset_memory(device):
    """Reset memory tracking."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def benchmark_solve(
    dsparse, 
    b: torch.Tensor,
    preconditioner: str,
    overlap: bool,
    use_cache: bool,
    rtol: float,
    maxiter: int,
    warmup: int,
    repeat: int,
    rank: int,
    world_size: int
) -> tuple:
    """Benchmark a solve configuration."""
    device = dsparse.device
    
    # Clear cache if testing without it
    if not use_cache:
        dsparse._invalidate_cache()
    
    # Warmup
    for _ in range(warmup):
        if not use_cache:
            dsparse._csr_cache = None
        x = dsparse.solve(b, preconditioner=preconditioner, overlap=overlap,
                         rtol=rtol, maxiter=min(50, maxiter), verbose=False)
    
    dist.barrier()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    
    # Reset memory
    reset_memory(device)
    
    # Benchmark
    times = []
    for _ in range(repeat):
        if not use_cache:
            dsparse._csr_cache = None
        
        dist.barrier()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        
        t0 = time.perf_counter()
        x = dsparse.solve(b, preconditioner=preconditioner, overlap=overlap,
                         rtol=rtol, maxiter=maxiter, verbose=False)
        
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        dist.barrier()
        
        times.append(time.perf_counter() - t0)
    
    # Get memory
    memory_mb = get_gpu_memory_mb(device)
    
    # Compute residual
    x_full = torch.zeros(dsparse.num_local, dtype=b.dtype, device=device)
    x_full[:dsparse.num_owned] = x
    dsparse.halo_exchange(x_full)
    Ax = dsparse.matvec(x_full, exchange_halo=False)
    
    b_full = torch.zeros(dsparse.num_local, dtype=b.dtype, device=device)
    b_full[:dsparse.num_owned] = b
    local_residual = (Ax[:dsparse.num_owned] - b[:dsparse.num_owned]).norm() ** 2
    
    # Global residual
    global_residual = local_residual.clone()
    dist.all_reduce(global_residual, op=dist.ReduceOp.SUM)
    residual = global_residual.sqrt().item()
    
    return min(times) * 1000, memory_mb, residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=int, nargs='+', default=[200, 500, 1000, 2000])
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--maxiter', type=int, default=2000)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--output', type=str, default='results/distributed_4gpu')
    args = parser.parse_args()
    
    # Initialize distributed
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    from torch_sla.distributed import DSparseMatrix
    
    if rank == 0:
        print("=" * 80)
        print(f"4-GPU Distributed Solver Benchmark (World Size: {world_size})")
        print("=" * 80)
        print(f"GPUs: {world_size} x {torch.cuda.get_device_name(0)}")
        print(f"Sizes: {args.sizes}")
        print(f"rtol: {args.rtol}, maxiter: {args.maxiter}")
        print()
    
    # Configurations
    configs = [
        ("Baseline", False, 'none', False),
        ("+CSR Cache", True, 'none', False),
        ("+Jacobi", True, 'jacobi', False),
        ("+Overlap", True, 'jacobi', True),
    ]
    
    all_results = []
    
    for n in args.sizes:
        N = n * n
        
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Grid {n}×{n} = {N:,} DOF, {world_size} GPUs ({N//world_size:,} DOF/GPU)")
            print(f"{'='*80}")
        
        # Create matrix (same on all ranks)
        val, row, col, shape = create_2d_poisson(n)
        
        # Distribute
        dsparse = DSparseMatrix.from_global(
            val, row, col, shape,
            num_partitions=world_size,
            my_partition=rank,
            device=device
        )
        
        # RHS
        b = torch.ones(dsparse.num_owned, dtype=torch.float64, device=device)
        
        if rank == 0:
            print(f"Rank 0: owned={dsparse.num_owned}, halo={dsparse.num_halo}, nnz={dsparse.nnz}")
            dsparse._build_interior_boundary_decomposition()
            stats = dsparse._overlap_stats
            print(f"Interior ratio: {stats.get('interior_ratio', 0):.1%}")
            print()
            print(f"{'Config':<20} {'Time (ms)':>12} {'Memory (MB)':>12} {'Residual':>12} {'Speedup':>10}")
            print("-" * 70)
        
        dist.barrier()
        
        baseline_time = None
        
        for name, use_cache, precond, overlap in configs:
            time_ms, mem_mb, residual = benchmark_solve(
                dsparse, b, precond, overlap, use_cache,
                args.rtol, args.maxiter, args.warmup, args.repeat,
                rank, world_size
            )
            
            if baseline_time is None:
                baseline_time = time_ms
            
            speedup = baseline_time / time_ms if time_ms > 0 else 1.0
            
            if rank == 0:
                print(f"{name:<20} {time_ms:>12.2f} {mem_mb:>12.1f} {residual:>12.2e} {speedup:>10.2f}x")
            
            all_results.append(BenchmarkResult(
                config_name=name,
                n=n,
                dof=N,
                num_gpus=world_size,
                time_ms=time_ms,
                memory_mb=mem_mb,
                residual=residual,
                speedup=speedup
            ))
    
    # Summary
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"Summary: Speedup vs Baseline ({world_size} GPUs)")
        print("=" * 80)
        
        by_config = defaultdict(list)
        for r in all_results:
            by_config[r.config_name].append(r)
        
        header = f"{'Config':<20}"
        for n in args.sizes:
            header += f" {n}×{n}".rjust(12)
        print(header)
        print("-" * (20 + 12 * len(args.sizes)))
        
        for name, _, _, _ in configs:
            line = f"{name:<20}"
            for r in by_config[name]:
                line += f" {r.speedup:>10.2f}x "
            print(line)
        
        # Save results
        os.makedirs(args.output, exist_ok=True)
        with open(f'{args.output}/results_{world_size}gpu.json', 'w') as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"\nResults saved to {args.output}/results_{world_size}gpu.json")
        
        # Plot
        if HAS_MATPLOTLIB:
            plot_results(all_results, configs, args, world_size)
    
    dist.destroy_process_group()


def plot_results(results: List[BenchmarkResult], configs, args, world_size: int):
    """Generate plots."""
    sizes = args.sizes
    n_sizes = len(sizes)
    n_configs = len(configs)
    
    by_config = defaultdict(list)
    for r in results:
        by_config[r.config_name].append(r)
    
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#9b59b6']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Distributed Solver Performance ({world_size} GPUs)', fontsize=14, fontweight='bold')
    
    # Time
    ax = axes[0]
    x = np.arange(n_sizes)
    width = 0.18
    for i, (name, _, _, _) in enumerate(configs):
        times = [r.time_ms for r in by_config[name]]
        ax.bar(x + (i - n_configs/2 + 0.5) * width, times, width, 
               label=name, color=colors[i], edgecolor='white')
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Solve Time')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}×{n}' for n in sizes])
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Speedup
    ax = axes[1]
    markers = ['o', 's', '^', 'D']
    for i, (name, _, _, _) in enumerate(configs):
        speedups = [r.speedup for r in by_config[name]]
        ax.plot(range(n_sizes), speedups, marker=markers[i], linestyle='-',
                label=name, color=colors[i], linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Speedup vs Baseline')
    ax.set_title('Optimization Speedup')
    ax.set_xticks(range(n_sizes))
    ax.set_xticklabels([f'{n}×{n}' for n in sizes])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Memory
    ax = axes[2]
    for i, (name, _, _, _) in enumerate(configs):
        mems = [r.memory_mb for r in by_config[name]]
        ax.bar(x + (i - n_configs/2 + 0.5) * width, mems, width,
               label=name, color=colors[i], edgecolor='white')
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Memory/GPU (MB)')
    ax.set_title('Memory Usage')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}×{n}' for n in sizes])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{args.output}/perf_{world_size}gpu.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to {args.output}/perf_{world_size}gpu.png")
    plt.close()


if __name__ == '__main__':
    main()

