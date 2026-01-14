#!/usr/bin/env python
"""
Matrix Multiplication Examples for torch-sla

This example demonstrates:
1. Sparse @ Dense (vector and matrix)
2. Dense @ Sparse (vector and matrix)  
3. Sparse @ Sparse with sparse gradient
4. Batched matrix multiplication
5. CUDA matrix multiplication
"""

import torch
from torch_sla import SparseTensor


def create_sparse_matrix(n: int, density: float = 0.1, dtype=torch.float64, device='cpu'):
    """Create random sparse matrix."""
    nnz = int(n * n * density)
    row = torch.randint(0, n, (nnz,), device=device)
    col = torch.randint(0, n, (nnz,), device=device)
    val = torch.randn(nnz, dtype=dtype, device=device)
    return SparseTensor(val, row, col, (n, n))


def example_1_sparse_dense():
    """Sparse @ Dense multiplication."""
    print("=" * 60)
    print("Example 1: Sparse @ Dense")
    print("=" * 60)
    
    n = 100
    A = create_sparse_matrix(n, density=0.05)
    A_dense = A.to_dense()
    
    print(f"Sparse matrix: {A}")
    print(f"Sparsity: {1 - A.nnz / (n * n):.1%}")
    
    # Sparse @ Dense vector
    x = torch.randn(n, dtype=torch.float64)
    y = A @ x
    y_expected = A_dense @ x
    print(f"\nSparse @ Dense vector:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Correct: {torch.allclose(y, y_expected)}")
    
    # Sparse @ Dense matrix
    X = torch.randn(n, 50, dtype=torch.float64)
    Y = A @ X
    Y_expected = A_dense @ X
    print(f"\nSparse @ Dense matrix:")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {Y.shape}")
    print(f"  Correct: {torch.allclose(Y, Y_expected)}")


def example_2_dense_sparse():
    """Dense @ Sparse multiplication."""
    print("\n" + "=" * 60)
    print("Example 2: Dense @ Sparse")
    print("=" * 60)
    
    n = 100
    A = create_sparse_matrix(n, density=0.05)
    A_dense = A.to_dense()
    
    # Dense @ Sparse vector (row vector @ matrix)
    x = torch.randn(n, dtype=torch.float64)
    y = x @ A
    y_expected = x @ A_dense
    print(f"Dense vector @ Sparse:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Correct: {torch.allclose(y, y_expected)}")
    
    # Dense @ Sparse matrix
    X = torch.randn(50, n, dtype=torch.float64)
    Y = X @ A
    Y_expected = X @ A_dense
    print(f"\nDense matrix @ Sparse:")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {Y.shape}")
    print(f"  Correct: {torch.allclose(Y, Y_expected)}")


def example_3_sparse_sparse():
    """Sparse @ Sparse multiplication."""
    print("\n" + "=" * 60)
    print("Example 3: Sparse @ Sparse")
    print("=" * 60)
    
    n = 50
    A = create_sparse_matrix(n, density=0.1)
    B = create_sparse_matrix(n, density=0.1)
    
    A_dense = A.to_dense()
    B_dense = B.to_dense()
    
    # Sparse @ Sparse
    C = A @ B
    C_expected = A_dense @ B_dense
    
    print(f"A: {A}")
    print(f"B: {B}")
    print(f"C = A @ B: {C}")
    print(f"Correct: {torch.allclose(C.to_dense(), C_expected, atol=1e-6)}")


def example_4_sparse_gradient():
    """Sparse @ Sparse with sparse gradient."""
    print("\n" + "=" * 60)
    print("Example 4: Sparse Gradient")
    print("=" * 60)
    
    # Create matrices with gradient tracking
    val_a = torch.randn(100, dtype=torch.float64, requires_grad=True)
    row_a = torch.randint(0, 20, (100,))
    col_a = torch.randint(0, 20, (100,))
    A = SparseTensor(val_a, row_a, col_a, (20, 20))
    
    val_b = torch.randn(80, dtype=torch.float64, requires_grad=True)
    row_b = torch.randint(0, 20, (80,))
    col_b = torch.randint(0, 20, (80,))
    B = SparseTensor(val_b, row_b, col_b, (20, 20))
    
    print(f"A: nnz={A.nnz}, val_a.shape={val_a.shape}")
    print(f"B: nnz={B.nnz}, val_b.shape={val_b.shape}")
    
    # Forward
    C = A @ B
    loss = C.to_dense().sum()
    
    # Backward
    loss.backward()
    
    print(f"\nAfter backward:")
    print(f"  val_a.grad.shape: {val_a.grad.shape} (SPARSE - same as input!)")
    print(f"  val_b.grad.shape: {val_b.grad.shape} (SPARSE - same as input!)")
    
    # Memory comparison
    n = 20
    dense_grad_size = n * n * 8  # bytes
    sparse_a_size = val_a.numel() * 8
    sparse_b_size = val_b.numel() * 8
    
    print(f"\nMemory comparison:")
    print(f"  Dense gradient would be: {dense_grad_size} bytes per matrix")
    print(f"  Sparse gradient A: {sparse_a_size} bytes")
    print(f"  Sparse gradient B: {sparse_b_size} bytes")
    print(f"  Savings: {dense_grad_size / sparse_a_size:.1f}x for A, {dense_grad_size / sparse_b_size:.1f}x for B")


def example_5_batched_matmul():
    """Batched matrix multiplication."""
    print("\n" + "=" * 60)
    print("Example 5: Batched Matrix Multiplication")
    print("=" * 60)
    
    n = 50
    batch_size = 8
    
    # Create batched sparse matrix
    val = torch.randn(200, dtype=torch.float64)
    row = torch.randint(0, n, (200,))
    col = torch.randint(0, n, (200,))
    
    val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
    A_batch = SparseTensor(val_batch, row, col, (batch_size, n, n))
    
    print(f"Batched SparseTensor: {A_batch}")
    
    # Batched Sparse @ Dense vector
    x_batch = torch.randn(batch_size, n, dtype=torch.float64)
    y_batch = A_batch @ x_batch
    print(f"\nBatched Sparse @ Dense vector:")
    print(f"  Input: {x_batch.shape}")
    print(f"  Output: {y_batch.shape}")
    
    # Batched Dense @ Sparse
    y_batch = x_batch @ A_batch
    print(f"\nBatched Dense @ Sparse:")
    print(f"  Input: {x_batch.shape}")
    print(f"  Output: {y_batch.shape}")
    
    # Batched Sparse @ Dense matrix
    X_batch = torch.randn(batch_size, n, 20, dtype=torch.float64)
    Y_batch = A_batch @ X_batch
    print(f"\nBatched Sparse @ Dense matrix:")
    print(f"  Input: {X_batch.shape}")
    print(f"  Output: {Y_batch.shape}")


def example_6_cuda_matmul():
    """CUDA matrix multiplication."""
    print("\n" + "=" * 60)
    print("Example 6: CUDA Matrix Multiplication")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping...")
        return
    
    n = 500
    
    # Create sparse matrix on CUDA
    val = torch.randn(5000, dtype=torch.float64, device='cuda')
    row = torch.randint(0, n, (5000,), device='cuda')
    col = torch.randint(0, n, (5000,), device='cuda')
    A = SparseTensor(val, row, col, (n, n))
    
    print(f"CUDA SparseTensor: {A}")
    
    # Time Sparse @ Dense
    x = torch.randn(n, dtype=torch.float64, device='cuda')
    
    # Warmup
    torch.cuda.synchronize()
    for _ in range(10):
        y = A @ x
    torch.cuda.synchronize()
    
    # Benchmark
    import time
    N = 100
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        y = A @ x
    torch.cuda.synchronize()
    t1 = time.time()
    
    print(f"\nSparse @ Dense vector ({N} iterations):")
    print(f"  Time per iteration: {(t1-t0)/N*1000:.3f} ms")
    
    # Sparse @ Dense matrix
    X = torch.randn(n, 100, dtype=torch.float64, device='cuda')
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        Y = A @ X
    torch.cuda.synchronize()
    t1 = time.time()
    
    print(f"\nSparse @ Dense matrix (100 cols, {N} iterations):")
    print(f"  Time per iteration: {(t1-t0)/N*1000:.3f} ms")
    
    # Sparse @ Sparse
    B = SparseTensor(val.clone(), row.clone(), col.clone(), (n, n))
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        C = A @ B
    torch.cuda.synchronize()
    t1 = time.time()
    
    print(f"\nSparse @ Sparse ({N} iterations):")
    print(f"  Time per iteration: {(t1-t0)/N*1000:.3f} ms")


def example_7_performance_comparison():
    """Performance comparison: scatter vs CSR."""
    print("\n" + "=" * 60)
    print("Example 7: Performance Comparison")
    print("=" * 60)
    
    import time
    
    n = 2000
    density = 0.005
    nnz = int(n * n * density)
    
    val = torch.randn(nnz, dtype=torch.float64)
    row = torch.randint(0, n, (nnz,))
    col = torch.randint(0, n, (nnz,))
    
    A = SparseTensor(val, row, col, (n, n))
    x = torch.randn(n, dtype=torch.float64)
    
    print(f"Matrix: {n}x{n}, nnz={nnz}")
    
    # Warmup
    for _ in range(5):
        y = A @ x
    
    # Benchmark SparseTensor (scatter-based)
    N = 100
    t0 = time.time()
    for _ in range(N):
        y = A @ x
    t1 = time.time()
    scatter_time = (t1 - t0) / N * 1000
    
    # Benchmark torch CSR
    A_csr = A.to_csr()
    for _ in range(5):
        y_csr = torch.mv(A_csr, x)
    
    t0 = time.time()
    for _ in range(N):
        y_csr = torch.mv(A_csr, x)
    t1 = time.time()
    csr_time = (t1 - t0) / N * 1000
    
    print(f"\nCPU SpMV time:")
    print(f"  SparseTensor (scatter): {scatter_time:.3f} ms")
    print(f"  torch CSR (mv):         {csr_time:.3f} ms")
    print(f"  Correct: {torch.allclose(y, y_csr)}")


if __name__ == "__main__":
    example_1_sparse_dense()
    example_2_dense_sparse()
    example_3_sparse_sparse()
    example_4_sparse_gradient()
    example_5_batched_matmul()
    example_6_cuda_matmul()
    example_7_performance_comparison()
    
    print("\n" + "=" * 60)
    print("All matrix multiplication examples completed!")
    print("=" * 60)

