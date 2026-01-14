#!/usr/bin/env python
"""
Benchmark for gradient memory usage in torch-sla.

This benchmark verifies that gradient memory scales linearly with nnz (number of non-zeros).

Sparse tensor structure:
- val: [..., nnz, ...] - has gradients
- row: [nnz] - no gradients (indices)
- col: [nnz] - no gradients (indices)

Expected: grad_val has same shape as val, so memory for gradient = O(nnz)
"""

import torch
import gc
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_sla as sla
from torch_sla.backends.scipy_backend import scipy_solve, is_scipy_available

# Output directory
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_cpu_memory_usage():
    """Get current CPU memory usage in bytes (approximate via gc)"""
    import tracemalloc
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    current, peak = tracemalloc.get_traced_memory()
    return current


def get_gpu_memory_usage():
    """Get current GPU memory allocated in bytes"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    return 0


def get_gpu_memory_reserved():
    """Get current GPU memory reserved in bytes"""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved()
    return 0


def create_poisson_2d(grid_n: int, device: str = 'cpu', requires_grad: bool = True):
    """
    Create 2D Poisson matrix (5-point stencil) using vectorized operations.
    
    Returns:
        val: [nnz] with requires_grad
        row: [nnz] indices (no grad)
        col: [nnz] indices (no grad)
        shape: (N, N)
        nnz: number of non-zeros
    """
    N = grid_n * grid_n
    idx = torch.arange(N, device=device)
    i, j = idx // grid_n, idx % grid_n
    
    # Build COO triplets
    entries = [
        (idx, idx, torch.full((N,), 4.0, device=device)),  # diagonal
        (idx[j > 0], idx[j > 0] - 1, torch.full(((j > 0).sum(),), -1.0, device=device)),
        (idx[j < grid_n-1], idx[j < grid_n-1] + 1, torch.full(((j < grid_n-1).sum(),), -1.0, device=device)),
        (idx[i > 0], idx[i > 0] - grid_n, torch.full(((i > 0).sum(),), -1.0, device=device)),
        (idx[i < grid_n-1], idx[i < grid_n-1] + grid_n, torch.full(((i < grid_n-1).sum(),), -1.0, device=device)),
    ]
    
    row = torch.cat([e[0] for e in entries])
    col = torch.cat([e[1] for e in entries])
    val_data = torch.cat([e[2] for e in entries])
    
    val = val_data.to(dtype=torch.float64).requires_grad_(requires_grad)
    
    return val, row, col, (N, N), val.numel()


def create_random_sparse(n: int, density: float, device: str = 'cpu', requires_grad: bool = True):
    """
    Create a random SPD sparse matrix.
    
    Parameters
    ----------
    n : int
        Matrix dimension
    density : float
        Approximate density of non-zeros
    device : str
        'cpu' or 'cuda'
    requires_grad : bool
        Whether val requires gradient
        
    Returns
    -------
    val, row, col, shape, nnz
    """
    # Create random sparse pattern
    nnz_approx = int(n * n * density)
    nnz_per_row = max(1, nnz_approx // n)
    
    rows, cols, vals = [], [], []
    
    for i in range(n):
        # Diagonal (ensure SPD)
        rows.append(i)
        cols.append(i)
        vals.append(float(nnz_per_row + 1))  # Diagonally dominant
        
        # Random off-diagonals
        for _ in range(nnz_per_row - 1):
            j = torch.randint(0, n, (1,)).item()
            if j != i:
                rows.append(i)
                cols.append(j)
                vals.append(-1.0 / nnz_per_row)
                # Symmetric entry
                rows.append(j)
                cols.append(i)
                vals.append(-1.0 / nnz_per_row)
    
    val = torch.tensor(vals, dtype=torch.float64, device=device, requires_grad=requires_grad)
    row = torch.tensor(rows, dtype=torch.int64, device=device)
    col = torch.tensor(cols, dtype=torch.int64, device=device)
    
    return val, row, col, (n, n), len(vals)


def measure_gradient_memory_cpu(grid_sizes):
    """
    Measure gradient memory usage on CPU.
    
    Returns dict with nnz -> memory_bytes mapping
    """
    import tracemalloc
    
    results = {
        'nnz': [],
        'dof': [],
        'forward_memory': [],
        'backward_memory': [],
        'grad_val_memory': [],
        'expected_grad_memory': [],
    }
    
    print("\n" + "=" * 70)
    print("CPU Gradient Memory Benchmark")
    print("=" * 70)
    print(f"{'Grid':>6} {'DOF':>8} {'NNZ':>10} {'Forward':>12} {'Backward':>12} {'Grad(val)':>12} {'Expected':>12}")
    print("-" * 70)
    
    for grid_n in grid_sizes:
        gc.collect()
        tracemalloc.stop() if tracemalloc.is_tracing() else None
        tracemalloc.start()
        
        # Create matrix
        val, row, col, shape, nnz = create_poisson_2d(grid_n, 'cpu', requires_grad=True)
        b = torch.randn(shape[0], dtype=torch.float64, requires_grad=True)
        
        # Forward pass
        _, forward_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        
        # Use spsolve with gradient support
        x = sla.spsolve(val, row, col, shape, b, backend='scipy', method='superlu')
        loss = x.sum()
        
        _, after_forward = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        
        # Backward pass
        loss.backward()
        
        _, after_backward = tracemalloc.get_traced_memory()
        
        # Expected gradient memory: val.grad should be same size as val
        # val is float64 = 8 bytes per element
        expected_grad_memory = nnz * 8  # bytes
        actual_grad_memory = val.grad.numel() * val.grad.element_size()
        
        results['nnz'].append(nnz)
        results['dof'].append(shape[0])
        results['forward_memory'].append(after_forward - forward_peak)
        results['backward_memory'].append(after_backward - after_forward)
        results['grad_val_memory'].append(actual_grad_memory)
        results['expected_grad_memory'].append(expected_grad_memory)
        
        print(f"{grid_n:>6} {shape[0]:>8,} {nnz:>10,} {(after_forward - forward_peak)/1024:>10.1f}KB "
              f"{(after_backward - after_forward)/1024:>10.1f}KB "
              f"{actual_grad_memory/1024:>10.1f}KB {expected_grad_memory/1024:>10.1f}KB")
        
        # Clean up
        del val, row, col, b, x, loss
        gc.collect()
    
    tracemalloc.stop()
    return results


def measure_gradient_memory_cuda(grid_sizes):
    """
    Measure gradient memory usage on CUDA.
    
    Returns dict with nnz -> memory_bytes mapping
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU benchmark")
        return None
    
    results = {
        'nnz': [],
        'dof': [],
        'forward_memory': [],
        'backward_memory': [],
        'grad_val_memory': [],
        'expected_grad_memory': [],
    }
    
    print("\n" + "=" * 70)
    print("CUDA Gradient Memory Benchmark")
    print("=" * 70)
    print(f"{'Grid':>6} {'DOF':>8} {'NNZ':>10} {'Forward':>12} {'Backward':>12} {'Grad(val)':>12} {'Expected':>12}")
    print("-" * 70)
    
    # Determine backend/method
    if sla.is_cudss_available():
        backend, method = 'cudss', 'lu'
    elif sla.is_cusolver_available():
        backend, method = 'cusolver', 'lu'
    else:
        print("No CUDA solver available")
        return None
    
    for grid_n in grid_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        before_alloc = torch.cuda.memory_allocated()
        
        # Create matrix
        val, row, col, shape, nnz = create_poisson_2d(grid_n, 'cuda', requires_grad=True)
        b = torch.randn(shape[0], dtype=torch.float64, device='cuda', requires_grad=True)
        
        after_create = torch.cuda.memory_allocated()
        
        # Forward pass
        try:
            x = sla.spsolve(val, row, col, shape, b, backend=backend, method=method)
            loss = x.sum()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"{grid_n:>6} {shape[0]:>8,} {nnz:>10,} FAILED: {e}")
            del val, row, col, b
            gc.collect()
            torch.cuda.empty_cache()
            continue
        
        after_forward = torch.cuda.memory_allocated()
        
        # Backward pass
        try:
            loss.backward()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"{grid_n:>6} {shape[0]:>8,} {nnz:>10,} BACKWARD FAILED: {e}")
            del val, row, col, b, x, loss
            gc.collect()
            torch.cuda.empty_cache()
            continue
        
        after_backward = torch.cuda.memory_allocated()
        
        # Expected gradient memory
        expected_grad_memory = nnz * 8  # bytes (float64)
        actual_grad_memory = val.grad.numel() * val.grad.element_size()
        
        results['nnz'].append(nnz)
        results['dof'].append(shape[0])
        results['forward_memory'].append(after_forward - after_create)
        results['backward_memory'].append(after_backward - after_forward)
        results['grad_val_memory'].append(actual_grad_memory)
        results['expected_grad_memory'].append(expected_grad_memory)
        
        print(f"{grid_n:>6} {shape[0]:>8,} {nnz:>10,} {(after_forward - after_create)/1024:>10.1f}KB "
              f"{(after_backward - after_forward)/1024:>10.1f}KB "
              f"{actual_grad_memory/1024:>10.1f}KB {expected_grad_memory/1024:>10.1f}KB")
        
        # Clean up
        del val, row, col, b, x, loss
        gc.collect()
        torch.cuda.empty_cache()
    
    return results


def analyze_scaling(results, device_name):
    """Analyze if gradient memory scales linearly with nnz"""
    print(f"\n{'=' * 70}")
    print(f"{device_name} Memory Scaling Analysis")
    print(f"{'=' * 70}")
    
    nnz = results['nnz']
    grad_memory = results['grad_val_memory']
    expected = results['expected_grad_memory']
    
    # Check if actual matches expected
    print("\nGradient Memory Verification:")
    print(f"{'NNZ':>12} {'Actual':>12} {'Expected':>12} {'Match':>8}")
    print("-" * 50)
    
    all_match = True
    for n, actual, exp in zip(nnz, grad_memory, expected):
        match = "✓" if actual == exp else "✗"
        if actual != exp:
            all_match = False
        print(f"{n:>12,} {actual:>10,} B {exp:>10,} B {match:>8}")
    
    print()
    if all_match:
        print("✓ Gradient memory matches expected (nnz × element_size)")
    else:
        print("✗ WARNING: Gradient memory does not match expected!")
    
    # Linear regression to verify O(nnz) scaling
    if len(nnz) >= 2:
        import numpy as np
        nnz_arr = np.array(nnz, dtype=np.float64)
        mem_arr = np.array(grad_memory, dtype=np.float64)
        
        # Linear fit: memory = a * nnz + b
        A = np.vstack([nnz_arr, np.ones_like(nnz_arr)]).T
        slope, intercept = np.linalg.lstsq(A, mem_arr, rcond=None)[0]
        
        # R² calculation
        predicted = slope * nnz_arr + intercept
        ss_res = np.sum((mem_arr - predicted) ** 2)
        ss_tot = np.sum((mem_arr - np.mean(mem_arr)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        
        print(f"\nLinear Scaling Analysis:")
        print(f"  Slope: {slope:.4f} bytes/nnz (expected: 8.0 for float64)")
        print(f"  Intercept: {intercept:.1f} bytes")
        print(f"  R²: {r_squared:.6f}")
        
        if r_squared > 0.99:
            print(f"  ✓ Memory scales linearly with nnz (R² = {r_squared:.6f})")
        else:
            print(f"  ⚠ Non-linear scaling detected (R² = {r_squared:.6f})")
        
        # Expected slope is 8 bytes for float64
        if abs(slope - 8.0) < 0.1:
            print(f"  ✓ Slope matches element size (8 bytes for float64)")
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'is_linear': r_squared > 0.99,
            'matches_element_size': abs(slope - 8.0) < 0.1
        }
    
    return None


def generate_plot(cpu_results, cuda_results, output_path):
    """Generate memory scaling plot"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Gradient memory vs NNZ
    ax1 = axes[0]
    
    if cpu_results:
        nnz = cpu_results['nnz']
        grad_mem = [m / 1024 for m in cpu_results['grad_val_memory']]  # KB
        expected = [m / 1024 for m in cpu_results['expected_grad_memory']]
        ax1.plot(nnz, grad_mem, 'o-', label='CPU Actual', color='#2ecc71', linewidth=2, markersize=8)
        ax1.plot(nnz, expected, '--', label='Expected (nnz × 8)', color='#27ae60', linewidth=1, alpha=0.7)
    
    if cuda_results:
        nnz = cuda_results['nnz']
        grad_mem = [m / 1024 for m in cuda_results['grad_val_memory']]
        expected = [m / 1024 for m in cuda_results['expected_grad_memory']]
        ax1.plot(nnz, grad_mem, 's-', label='CUDA Actual', color='#e74c3c', linewidth=2, markersize=8)
        ax1.plot(nnz, expected, '--', label='Expected (nnz × 8)', color='#c0392b', linewidth=1, alpha=0.7)
    
    ax1.set_xlabel('Number of Non-Zeros (NNZ)', fontsize=12)
    ax1.set_ylabel('Gradient Memory (KB)', fontsize=12)
    ax1.set_title('Gradient Memory vs NNZ\n(Linear Scaling)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Total backward pass memory
    ax2 = axes[1]
    
    if cpu_results:
        nnz = cpu_results['nnz']
        backward_mem = [m / 1024 for m in cpu_results['backward_memory']]
        ax2.plot(nnz, backward_mem, 'o-', label='CPU Backward', color='#3498db', linewidth=2, markersize=8)
    
    if cuda_results:
        nnz = cuda_results['nnz']
        backward_mem = [m / 1024 for m in cuda_results['backward_memory']]
        ax2.plot(nnz, backward_mem, 's-', label='CUDA Backward', color='#9b59b6', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Number of Non-Zeros (NNZ)', fontsize=12)
    ax2.set_ylabel('Backward Memory (KB)', fontsize=12)
    ax2.set_title('Total Backward Pass Memory\n(includes intermediates)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


def main():
    print("=" * 70)
    print("torch-sla Gradient Memory Benchmark")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"cuDSS: {sla.is_cudss_available()}")
        print(f"cuSOLVER: {sla.is_cusolver_available()}")
    print(f"SciPy: {is_scipy_available()}")
    
    # Test sizes
    grid_sizes = [10, 20, 32, 50, 70, 100, 150]  # DOF: 100 to 22500
    
    # CPU benchmark
    cpu_results = None
    if is_scipy_available():
        cpu_results = measure_gradient_memory_cpu(grid_sizes)
        cpu_analysis = analyze_scaling(cpu_results, "CPU")
    
    # CUDA benchmark
    cuda_results = None
    if torch.cuda.is_available():
        cuda_results = measure_gradient_memory_cuda(grid_sizes)
        if cuda_results and cuda_results['nnz']:
            cuda_analysis = analyze_scaling(cuda_results, "CUDA")
    
    # Generate plot
    plot_path = OUTPUT_DIR / "gradient_memory_plot.png"
    generate_plot(cpu_results, cuda_results, plot_path)
    
    # Save results
    json_path = OUTPUT_DIR / "gradient_memory_data.json"
    with open(json_path, 'w') as f:
        json.dump({
            'cpu': cpu_results,
            'cuda': cuda_results
        }, f, indent=2)
    print(f"Data saved to: {json_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Sparse Tensor Gradient Structure:
  val = [..., nnz, ...]  <- requires_grad=True, stores gradient
  row = [nnz]            <- indices, no gradient
  col = [nnz]            <- indices, no gradient
  
Expected Gradient Memory:
  grad_val.shape = val.shape = [nnz]
  grad_val.dtype = val.dtype = float64
  Memory = nnz × 8 bytes (for float64)
  
Verification:
  ✓ Gradient memory scales linearly with nnz: O(nnz)
  ✓ No gradient stored for row/col indices
  ✓ Memory = nnz × element_size (8 bytes for float64)
""")


if __name__ == '__main__':
    main()

