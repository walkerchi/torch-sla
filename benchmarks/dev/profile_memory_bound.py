#!/usr/bin/env python
"""
Performance profiling script to analyze if CG/BiCGSTAB solvers are memory-bound.

This script performs:
1. Roofline model analysis
2. SpMV memory bandwidth measurement
3. Kernel-level profiling
4. Memory access pattern analysis

Key metrics:
- Arithmetic Intensity (AI) = FLOP / Bytes
- Achieved Bandwidth vs Peak Bandwidth
- Memory vs Compute time breakdown
"""

import torch
import time
import sys
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_sla as sla
from torch_sla.backends.pytorch_backend import CachedSparseMatrix, pcg_solve_optimized, jacobi_preconditioner

@dataclass
class RooflineMetrics:
    """Roofline model metrics."""
    flops_per_iter: int  # FLOPs per CG iteration
    bytes_per_iter: int  # Memory traffic per iteration
    arithmetic_intensity: float  # FLOP/Byte
    achieved_gflops: float
    achieved_bandwidth_gb: float
    peak_bandwidth_gb: float
    is_memory_bound: bool
    bound_percentage: float  # How close to the bound

def get_gpu_peak_bandwidth() -> float:
    """Estimate peak memory bandwidth in GB/s."""
    if not torch.cuda.is_available():
        return 100.0  # Assume ~100 GB/s for CPU
    
    gpu_name = torch.cuda.get_device_name(0).lower()
    
    # Known GPU bandwidths (approximate)
    bandwidth_map = {
        'a100': 2039,  # A100 80GB
        'h100': 3350,  # H100
        'v100': 900,   # V100
        '4090': 1008,  # RTX 4090
        '4080': 717,   # RTX 4080
        '3090': 936,   # RTX 3090
        '3080': 760,   # RTX 3080
        'a6000': 768,  # RTX A6000
        'a5000': 768,
        'a4000': 448,
    }
    
    for key, bw in bandwidth_map.items():
        if key in gpu_name:
            return bw
    
    # Default estimate: try to measure
    return estimate_peak_bandwidth()

def estimate_peak_bandwidth() -> float:
    """Measure peak memory bandwidth empirically."""
    if not torch.cuda.is_available():
        return 100.0
    
    # Use large array copy to estimate bandwidth
    n = 100_000_000  # 100M elements
    x = torch.randn(n, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    
    # Warmup
    for _ in range(5):
        y.copy_(x)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(100):
        y.copy_(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    bytes_transferred = n * 4 * 2 * 100  # read + write, 100 iterations
    bandwidth_gb = bytes_transferred / elapsed / 1e9
    
    del x, y
    torch.cuda.empty_cache()
    
    return bandwidth_gb

def measure_spmv_bandwidth(A: CachedSparseMatrix, n_iters: int = 100) -> Tuple[float, float]:
    """
    Measure SpMV bandwidth and FLOPS.
    
    Returns:
        achieved_bandwidth_gb: GB/s
        achieved_gflops: GFLOPS
    """
    n = A.n
    nnz = len(A.val)
    
    x = torch.randn(n, dtype=A.dtype, device=A.device)
    y = torch.empty_like(x)
    
    # Warmup
    for _ in range(10):
        y = A.matvec(x)
    torch.cuda.synchronize() if A.device.type == 'cuda' else None
    
    # Measure
    start = time.perf_counter()
    for _ in range(n_iters):
        y = A.matvec(x)
    torch.cuda.synchronize() if A.device.type == 'cuda' else None
    elapsed = time.perf_counter() - start
    
    time_per_spmv = elapsed / n_iters
    
    # Memory traffic for SpMV y = A @ x (CSR format):
    # Read: values (nnz * dtype_size), col_indices (nnz * 4), row_ptr (n+1)*4, x (n * dtype_size)
    # Write: y (n * dtype_size)
    # Plus irregular access to x (assume cache hit rate ~50% for well-structured matrix)
    dtype_size = 8 if A.dtype == torch.float64 else 4
    
    bytes_read = (nnz * dtype_size +  # values
                  nnz * 4 +            # col indices
                  (n + 1) * 4 +        # row ptr
                  n * dtype_size)      # x vector (assuming good cache)
    bytes_write = n * dtype_size       # y vector
    total_bytes = bytes_read + bytes_write
    
    # FLOPs: 2 * nnz (multiply-add per nonzero)
    flops = 2 * nnz
    
    achieved_bandwidth_gb = total_bytes * n_iters / elapsed / 1e9
    achieved_gflops = flops * n_iters / elapsed / 1e9
    
    return achieved_bandwidth_gb, achieved_gflops, time_per_spmv * 1000

def analyze_cg_iteration_breakdown(A: CachedSparseMatrix) -> Dict[str, float]:
    """
    Break down CG iteration into memory-bound vs compute-bound operations.
    
    CG iteration:
    1. SpMV: Ap = A @ p           - Memory bound (main cost)
    2. dot: pAp = p.Ap           - Memory bound
    3. axpy: x += alpha * p      - Memory bound  
    4. axpy: r -= alpha * Ap     - Memory bound
    5. precond: z = M^-1 r       - Memory bound (for Jacobi)
    6. dot: rz = r.z             - Memory bound
    7. axpy: p = z + beta * p    - Memory bound
    
    Returns: Dict with time breakdown
    """
    n = A.n
    dtype = A.dtype
    device = A.device
    
    # Allocate vectors
    p = torch.randn(n, dtype=dtype, device=device)
    x = torch.randn(n, dtype=dtype, device=device)
    r = torch.randn(n, dtype=dtype, device=device)
    z = torch.randn(n, dtype=dtype, device=device)
    Ap = torch.empty_like(p)
    
    D_inv = 1.0 / A.diagonal
    
    n_iters = 100
    
    # Warmup
    for _ in range(10):
        Ap = A.matvec(p)
        pAp = torch.dot(p, Ap)
        x.add_(p, alpha=0.5)
        r.add_(Ap, alpha=-0.5)
        z.copy_(D_inv * r)
        rz = torch.dot(r, z)
        p.mul_(0.5).add_(z)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    timings = {}
    
    # 1. SpMV
    start = time.perf_counter()
    for _ in range(n_iters):
        Ap = A.matvec(p)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    timings['spmv'] = (time.perf_counter() - start) / n_iters * 1000
    
    # 2. dot product (pAp)
    start = time.perf_counter()
    for _ in range(n_iters):
        pAp = torch.dot(p, Ap)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    timings['dot_pAp'] = (time.perf_counter() - start) / n_iters * 1000
    
    # 3. axpy (x += alpha * p)
    start = time.perf_counter()
    for _ in range(n_iters):
        x.add_(p, alpha=0.5)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    timings['axpy_x'] = (time.perf_counter() - start) / n_iters * 1000
    
    # 4. axpy (r -= alpha * Ap)
    start = time.perf_counter()
    for _ in range(n_iters):
        r.add_(Ap, alpha=-0.5)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    timings['axpy_r'] = (time.perf_counter() - start) / n_iters * 1000
    
    # 5. Jacobi preconditioner (z = D^-1 * r)
    start = time.perf_counter()
    for _ in range(n_iters):
        z.copy_(D_inv * r)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    timings['precond'] = (time.perf_counter() - start) / n_iters * 1000
    
    # 6. dot product (rz)
    start = time.perf_counter()
    for _ in range(n_iters):
        rz = torch.dot(r, z)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    timings['dot_rz'] = (time.perf_counter() - start) / n_iters * 1000
    
    # 7. direction update (p = z + beta * p)
    start = time.perf_counter()
    for _ in range(n_iters):
        p.mul_(0.5).add_(z)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    timings['direction'] = (time.perf_counter() - start) / n_iters * 1000
    
    # Total
    timings['total'] = sum(timings.values())
    timings['spmv_fraction'] = timings['spmv'] / timings['total'] * 100
    
    return timings

def create_poisson_2d(grid_n: int, device: str = 'cpu', dtype=torch.float64):
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

def compute_roofline_metrics(A: CachedSparseMatrix, peak_bandwidth_gb: float) -> RooflineMetrics:
    """
    Compute roofline model metrics for CG solver.
    """
    n = A.n
    nnz = len(A.val)
    dtype_size = 8 if A.dtype == torch.float64 else 4
    
    # FLOPs per CG iteration:
    # SpMV: 2*nnz
    # dot products: 2*2*n (two dot products)
    # axpy: 2*2*n (two axpy)
    # precond: n (element-wise multiply)
    # direction update: 2*n
    flops_per_iter = 2*nnz + 4*n + 4*n + n + 2*n
    
    # Memory traffic per CG iteration:
    # SpMV: values(nnz*ds), col_idx(nnz*4), row_ptr((n+1)*4), x(n*ds), y(n*ds)
    # dot products: 2*(2*n*ds)
    # axpy: 2*(3*n*ds) - read two, write one
    # precond: 3*n*ds
    # direction update: 3*n*ds
    bytes_per_iter = (nnz * dtype_size + nnz * 4 + (n+1) * 4 + 2*n*dtype_size +  # SpMV
                      4 * n * dtype_size +  # dot products
                      6 * n * dtype_size +  # axpy
                      3 * n * dtype_size +  # precond
                      3 * n * dtype_size)   # direction
    
    arithmetic_intensity = flops_per_iter / bytes_per_iter
    
    # Measure achieved performance
    bw, gflops, _ = measure_spmv_bandwidth(A)
    
    # Is memory bound?
    # Memory bound if: achieved_bandwidth / peak_bandwidth > achieved_gflops / peak_gflops
    # For GPUs, peak GFLOPS >> peak bandwidth * AI, so almost always memory bound
    is_memory_bound = bw > 0.3 * peak_bandwidth_gb  # Using 30% as threshold for "close to peak"
    bound_percentage = bw / peak_bandwidth_gb * 100
    
    return RooflineMetrics(
        flops_per_iter=flops_per_iter,
        bytes_per_iter=bytes_per_iter,
        arithmetic_intensity=arithmetic_intensity,
        achieved_gflops=gflops,
        achieved_bandwidth_gb=bw,
        peak_bandwidth_gb=peak_bandwidth_gb,
        is_memory_bound=is_memory_bound or arithmetic_intensity < 10,  # AI < 10 typically memory bound
        bound_percentage=bound_percentage
    )

def print_optimization_recommendations(metrics: RooflineMetrics, timings: Dict[str, float]):
    """Print optimization recommendations based on profiling results."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    
    if metrics.is_memory_bound:
        print("\n✅ CONFIRMED: Solver is MEMORY BOUND")
        print(f"   - Arithmetic Intensity: {metrics.arithmetic_intensity:.2f} FLOP/Byte (low)")
        print(f"   - Achieved Bandwidth: {metrics.achieved_bandwidth_gb:.1f} GB/s ({metrics.bound_percentage:.1f}% of peak)")
        print(f"   - Peak Bandwidth: {metrics.peak_bandwidth_gb:.1f} GB/s")
        
        print("\n📋 Memory-Bound Optimization Strategies:")
        print("-" * 60)
        
        print("""
1. REDUCE MEMORY TRAFFIC
   
   a) Use Lower Precision (float32 instead of float64)
      - 2x less memory traffic
      - Current code already supports mixed_precision mode
      - Use iterative refinement for high precision solution
      
   b) Matrix-Free Methods
      - Don't store matrix explicitly
      - Compute stencil operations on-the-fly
      - Reduces memory from O(nnz) to O(n)
      
   c) Sparse Matrix Compression
      - Use blocked formats (BSR) for structured sparsity
      - Exploit diagonal structure with DIA format
      - ELLPACK for uniform row lengths

2. IMPROVE MEMORY ACCESS PATTERNS

   a) Reorder Matrix (RCM ordering) ✓ Already implemented
      - Reduces bandwidth, improves cache locality
      - Enable with use_rcm=True
   
   b) Cache Blocking
      - Process matrix in cache-sized blocks
      - Reuse vector data across multiple rows
   
   c) NUMA-Aware Allocation
      - Pin memory to specific CPU sockets
      - Use cudaHostAlloc for faster CPU-GPU transfers

3. FUSE OPERATIONS TO REDUCE MEMORY PASSES

   a) Kernel Fusion
      - Fuse axpy + precond + dot into single kernel
      - Use custom CUDA kernels or torch.compile
      
   b) Pipelined CG ✓ Already implemented
      - Overlaps SpMV with vector operations
      - Hides memory latency
   
   c) Communication-Avoiding CG (CA-CG)
      - s-step CG variant
      - Reduces global synchronization
      - Compute multiple iterations worth of basis vectors

4. REDUCE SYNCHRONIZATION OVERHEAD

   a) Async Operations
      - Use CUDA streams for overlapping
      - Non-blocking memory transfers
   
   b) Reduce Convergence Checks ✓ Already implemented
      - check_interval parameter
      - Only sync every N iterations
   
   c) CUDA Graphs
      - Capture entire iteration loop
      - Eliminates kernel launch overhead

5. BATCHED SOLVING ✓ Already implemented

   a) Multiple RHS
      - batched_pcg_solve handles multiple RHS
      - Better GPU utilization via SpMM
      - Amortizes memory access overhead
""")

        # Specific recommendations based on timing breakdown
        spmv_frac = timings.get('spmv_fraction', 0)
        if spmv_frac > 70:
            print(f"\n⚡ SpMV dominates ({spmv_frac:.1f}% of iteration time)")
            print("   Focus optimizations on SpMV:")
            print("   - Use cuSPARSE with tuned algorithms")
            print("   - Consider matrix format: CSR → BSR for block structure")
            print("   - Try Merge-based SpMV for irregular matrices")
        
    else:
        print("\n❓ Solver may be COMPUTE BOUND or LATENCY BOUND")
        print("   Consider:")
        print("   - Optimizing preconditioner (more FLOPs, fewer iterations)")
        print("   - Reducing Python overhead with torch.compile")
        print("   - CUDA kernel optimizations")

def main():
    print("=" * 80)
    print("Memory Bound Analysis for CG/BiCGSTAB Solvers")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Get peak bandwidth
    print("\nMeasuring peak memory bandwidth...")
    peak_bw = get_gpu_peak_bandwidth()
    print(f"Estimated peak bandwidth: {peak_bw:.1f} GB/s")
    
    # Test sizes
    test_sizes = [100, 500, 1000]
    if device == 'cuda':
        test_sizes = [100, 500, 1000, 1414]
    
    for grid_n in test_sizes:
        n = grid_n * grid_n
        print(f"\n{'='*80}")
        print(f"Problem Size: {n:,} DOF (grid {grid_n}x{grid_n})")
        print(f"{'='*80}")
        
        # Create matrix
        for dtype in [torch.float64, torch.float32]:
            dtype_name = 'float64' if dtype == torch.float64 else 'float32'
            print(f"\n  [{dtype_name}]")
            
            val, row, col, shape = create_poisson_2d(grid_n, device, dtype)
            A = CachedSparseMatrix(val, row, col, shape)
            
            nnz = len(val)
            print(f"  Matrix: {n:,} x {n:,}, nnz = {nnz:,}, sparsity = {nnz/n/n*100:.2f}%")
            
            # Measure SpMV bandwidth
            bw, gflops, spmv_time = measure_spmv_bandwidth(A)
            print(f"  SpMV: {spmv_time:.3f} ms, {bw:.1f} GB/s, {gflops:.2f} GFLOPS")
            
            # Roofline metrics
            metrics = compute_roofline_metrics(A, peak_bw)
            print(f"  Arithmetic Intensity: {metrics.arithmetic_intensity:.2f} FLOP/Byte")
            print(f"  Bandwidth utilization: {metrics.bound_percentage:.1f}%")
            print(f"  Memory bound: {'YES' if metrics.is_memory_bound else 'NO'}")
            
            # Timing breakdown
            if n >= 10000:  # Only for larger problems
                timings = analyze_cg_iteration_breakdown(A)
                print(f"\n  CG Iteration Breakdown:")
                print(f"    SpMV:       {timings['spmv']:.4f} ms ({timings['spmv_fraction']:.1f}%)")
                print(f"    dot(p,Ap):  {timings['dot_pAp']:.4f} ms")
                print(f"    x += αp:    {timings['axpy_x']:.4f} ms")
                print(f"    r -= αAp:   {timings['axpy_r']:.4f} ms")
                print(f"    z = M⁻¹r:   {timings['precond']:.4f} ms")
                print(f"    dot(r,z):   {timings['dot_rz']:.4f} ms")
                print(f"    p update:   {timings['direction']:.4f} ms")
                print(f"    Total:      {timings['total']:.4f} ms")
            
            del A, val, row, col
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    
    # Print recommendations for largest problem
    print("\n")
    if device == 'cuda':
        val, row, col, shape = create_poisson_2d(1000, device, torch.float64)
        A = CachedSparseMatrix(val, row, col, shape)
        metrics = compute_roofline_metrics(A, peak_bw)
        timings = analyze_cg_iteration_breakdown(A)
        print_optimization_recommendations(metrics, timings)
        del A, val, row, col
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

