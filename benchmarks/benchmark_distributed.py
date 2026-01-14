#!/usr/bin/env python
"""
Distributed Sparse Solve Benchmark

Benchmarks DSparseMatrix.solve() on CPU (Gloo) and CUDA (NCCL).

Usage:
    # CPU with 4 processes
    torchrun --standalone --nproc_per_node=4 benchmark_distributed.py --device cpu
    
    # CUDA with 4 GPUs  
    torchrun --standalone --nproc_per_node=4 benchmark_distributed.py --device cuda
    
    # Generate plots
    python benchmark_distributed.py --only-plot
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch

IN_DISTRIBUTED = 'RANK' in os.environ


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--dofs', type=str, default=None,
                       help='Comma-separated DOFs. Default: auto-scale')
    parser.add_argument('--maxiter', type=int, default=5000)
    parser.add_argument('--atol', type=float, default=1e-8)
    parser.add_argument('--output-dir', type=str, default='results/benchmark_distributed')
    parser.add_argument('--only-plot', action='store_true')
    parser.add_argument('--no-title', action='store_true')
    return parser.parse_args()


def create_poisson_2d(n: int, device: str = 'cpu'):
    """Create 2D Poisson matrix (5-point stencil)."""
    grid = int(n ** 0.5)
    actual_n = grid * grid
    
    rows, cols, vals = [], [], []
    
    for i in range(grid):
        for j in range(grid):
            idx = i * grid + j
            rows.append(idx)
            cols.append(idx)
            vals.append(4.0)
            
            if j > 0:
                rows.append(idx)
                cols.append(idx - 1)
                vals.append(-1.0)
            if j < grid - 1:
                rows.append(idx)
                cols.append(idx + 1)
                vals.append(-1.0)
            if i > 0:
                rows.append(idx)
                cols.append(idx - grid)
                vals.append(-1.0)
            if i < grid - 1:
                rows.append(idx)
                cols.append(idx + grid)
                vals.append(-1.0)
    
    row = torch.tensor(rows, dtype=torch.int64, device=device)
    col = torch.tensor(cols, dtype=torch.int64, device=device)
    val = torch.tensor(vals, dtype=torch.float64, device=device)
    
    return val, row, col, (actual_n, actual_n), actual_n


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def benchmark_distributed(args):
    """Run distributed benchmark."""
    import torch.distributed as dist
    from torch_sla.distributed import DSparseMatrix, partition_simple
    
    # Choose backend based on device
    backend = 'nccl' if args.device == 'cuda' else 'gloo'
    dist.init_process_group(backend=backend)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if args.device == 'cuda':
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        device = 'cpu'
    
    if rank == 0:
        print("=" * 70)
        print(f"Distributed Solve Benchmark")
        print(f"  Backend: {backend}")
        print(f"  World size: {world_size}")
        print(f"  Device: {args.device}")
        if args.device == 'cuda':
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        print("=" * 70)
    
    # Auto-scale DOFs based on device
    if args.dofs:
        dofs = [int(d) for d in args.dofs.split(',')]
    elif args.device == 'cuda':
        # CUDA: try larger scales
        dofs = [10000, 100000, 1000000, 4000000, 10000000, 25000000, 50000000]
    else:
        # CPU: smaller scales
        dofs = [1000, 10000, 100000, 500000, 1000000]
    
    results = []
    
    for target_dof in dofs:
        # Create matrix on CPU first, then move
        val, row, col, shape, actual_dof = create_poisson_2d(target_dof, device='cpu')
        n = shape[0]
        
        partition_ids = partition_simple(n, world_size)
        
        try:
            if args.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            
            # Move data to device
            val_dev = val.to(device)
            row_dev = row.to(device)
            col_dev = col.to(device)
            
            # Create local matrix
            A = DSparseMatrix.from_global(
                val_dev, row_dev, col_dev, shape,
                num_partitions=world_size,
                my_partition=rank,
                partition_ids=partition_ids,
                device=device,  # Important: pass device!
                verbose=False
            )
            
            # RHS
            b_owned = torch.ones(A.num_owned, dtype=torch.float64, device=device)
            
            # Warmup
            dist.barrier()
            try:
                _ = A.solve(b_owned, atol=1e-4, maxiter=50, verbose=False)
            except Exception:
                pass
            dist.barrier()
            
            if args.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            dist.barrier()
            start = time.perf_counter()
            
            x_owned = A.solve(b_owned, atol=args.atol, maxiter=args.maxiter, verbose=False)
            
            if args.device == 'cuda':
                torch.cuda.synchronize()
            dist.barrier()
            
            elapsed = time.perf_counter() - start
            
            # Get memory per GPU
            if args.device == 'cuda':
                local_peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
                # Gather all GPUs' memory
                mem_tensor = torch.tensor([local_peak_mem], device=device)
                all_mems = [torch.zeros(1, device=device) for _ in range(world_size)]
                dist.all_gather(all_mems, mem_tensor)
                all_mems = [m.item() for m in all_mems]
                peak_mem = max(all_mems)  # Max memory across GPUs
            else:
                local_peak_mem = 0
                all_mems = [0] * world_size
                peak_mem = 0
            
            # Compute residual
            x_local = torch.zeros(A.num_local, dtype=torch.float64, device=device)
            x_local[:A.num_owned] = x_owned
            A.halo_exchange(x_local)
            Ax_local = A.matvec(x_local, exchange_halo=False)
            
            b_local = torch.zeros(A.num_local, dtype=torch.float64, device=device)
            b_local[:A.num_owned] = b_owned
            
            local_res_sq = ((Ax_local[:A.num_owned] - b_local[:A.num_owned]) ** 2).sum()
            
            # All-reduce residual (need to move to CPU for gloo/nccl compatibility)
            if args.device == 'cuda':
                # For NCCL, keep on GPU
                global_res_sq = local_res_sq.clone()
                dist.all_reduce(global_res_sq, op=dist.ReduceOp.SUM)
            else:
                global_res_sq = local_res_sq.clone()
                dist.all_reduce(global_res_sq, op=dist.ReduceOp.SUM)
            
            residual = global_res_sq.sqrt().item()
            
            if rank == 0:
                if peak_mem > 0:
                    mem_str = f", mem/GPU=[" + ",".join(f"{m:.2f}" for m in all_mems) + "]GB"
                else:
                    mem_str = ""
                print(f"DOF={actual_dof:>10,}, time={elapsed:.3f}s, res={residual:.2e}{mem_str}")
            
            results.append({
                'dof': actual_dof,
                'time': elapsed,
                'residual': residual,
                'memory_gb': peak_mem,
                'world_size': world_size,
                'device': args.device
            })
            
            # Clean up
            del A, val_dev, row_dev, col_dev, x_owned, b_owned
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            import traceback
            if rank == 0:
                print(f"DOF={actual_dof:>10,}: Error - {type(e).__name__}: {e}")
                traceback.print_exc()
            
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            break
    
    # Save results
    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = output_dir / f'results_{args.device}_p{world_size}.json'
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {result_file}")
    
    dist.destroy_process_group()


def plot_results(args):
    """Generate comparison plots - only P=4, with time, memory, and error."""
    import matplotlib.pyplot as plt
    
    output_dir = Path(args.output_dir)
    
    # Load only P=4 results
    cpu_file = output_dir / 'results_cpu_p4.json'
    cuda_file = output_dir / 'results_cuda_p4.json'
    
    results = {}
    if cpu_file.exists():
        with open(cpu_file) as f:
            results['cpu'] = json.load(f)
    if cuda_file.exists():
        with open(cuda_file) as f:
            results['cuda'] = json.load(f)
    
    if not results:
        print("No P=4 result files found.")
        return
    
    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    colors = {'cpu': '#2E86AB', 'cuda': '#E94F37'}
    labels = {'cpu': 'CPU (Gloo, 4 proc)', 'cuda': 'CUDA (NCCL, 4 GPU)'}
    
    # Panel 1: Time
    ax = axes[0]
    for device, data in results.items():
        dofs = [d['dof'] for d in data]
        times = [d['time'] for d in data]
        ax.loglog(dofs, times, 'o-', color=colors[device], label=labels[device],
                  linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1)
    
    ax.set_xlabel('DOF', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('Solve Time', fontsize=13, fontweight='bold')
    
    # Panel 2: Memory (GPU and estimated CPU RAM)
    ax = axes[1]
    
    # GPU memory
    if 'cuda' in results:
        data = results['cuda']
        dofs = [d['dof'] for d in data if d.get('memory_gb', 0) > 0]
        mems = [d['memory_gb'] for d in data if d.get('memory_gb', 0) > 0]
        
        if dofs:
            ax.loglog(dofs, mems, 's-', color=colors['cuda'], label='GPU Memory/card',
                     linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1)
    
    # Estimate CPU RAM: ~500 bytes/DOF for sparse matrix + vectors
    # For distributed: each process has ~n/P DOF + halo
    if 'cpu' in results:
        data = results['cpu']
        dofs = [d['dof'] for d in data]
        # Estimate: (5*nnz + 3*n) * 8 bytes / P, nnz ≈ 5*n for 2D Poisson
        # ≈ (25n + 3n) * 8 / 4 = 56n bytes/proc = 56 bytes/DOF per proc
        mems = [d * 56 / 4 / 1e9 for d in dofs]  # GB per process
        ax.loglog(dofs, mems, 'o--', color=colors['cpu'], label='CPU RAM/proc (est)',
                 linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=1)
    
    ax.set_xlabel('DOF', fontsize=12)
    ax.set_ylabel('Memory per Process (GB)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('Memory Usage', fontsize=13, fontweight='bold')
    
    # Panel 3: Residual (Error)
    ax = axes[2]
    markers = {'cuda': 's', 'cpu': 'o'}
    # Add slight offset to avoid overlapping
    offsets = {'cuda': 1.0, 'cpu': 1.02}
    for device, data in results.items():
        dofs = [d['dof'] * offsets[device] for d in data]  # slight x-offset
        residuals = [d['residual'] for d in data]
        ax.loglog(dofs, residuals, f'{markers[device]}-', color=colors[device], label=labels[device],
                  linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1)
    
    ax.set_xlabel('DOF', fontsize=12)
    ax.set_ylabel('Residual ||Ax - b||', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('Convergence (atol=1e-8)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf']:
        fig.savefig(output_dir / f'distributed_benchmark.{fmt}', dpi=150, bbox_inches='tight')
    
    print(f"Plots saved to {output_dir}")
    plt.close()


def main():
    args = get_args()
    
    if args.only_plot:
        plot_results(args)
        return
    
    if IN_DISTRIBUTED:
        benchmark_distributed(args)
    else:
        print("Run with torchrun:")
        print(f"  torchrun --standalone --nproc_per_node=4 {sys.argv[0]} --device cpu")
        print(f"  torchrun --standalone --nproc_per_node=4 {sys.argv[0]} --device cuda")


if __name__ == '__main__':
    main()
