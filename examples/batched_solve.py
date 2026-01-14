#!/usr/bin/env python
"""
Batched Solve Examples for torch-sla

This example demonstrates:
1. Batched SparseTensor with same layout
2. solve_batch for different values, same structure
3. SparseTensorList for different layouts
4. CUDA batched operations
"""

import torch
from torch_sla import SparseTensor, SparseTensorList


def create_tridiagonal(n: int, dtype=torch.float64, device='cpu'):
    """Create tridiagonal SPD matrix."""
    idx = torch.arange(n, device=device)
    
    # Diagonal entries: all n indices with value 4.0
    diag_row = idx
    diag_col = idx
    diag_val = torch.full((n,), 4.0, dtype=dtype, device=device)
    
    # Sub-diagonal entries: indices 1 to n-1 with value -1.0
    sub_row = idx[1:]
    sub_col = idx[:-1]
    sub_val = torch.full((n - 1,), -1.0, dtype=dtype, device=device)
    
    # Super-diagonal entries: indices 0 to n-2 with value -1.0
    sup_row = idx[:-1]
    sup_col = idx[1:]
    sup_val = torch.full((n - 1,), -1.0, dtype=dtype, device=device)
    
    # Concatenate all entries
    row = torch.cat([diag_row, sub_row, sup_row])
    col = torch.cat([diag_col, sub_col, sup_col])
    val = torch.cat([diag_val, sub_val, sup_val])
    
    return val, row, col, (n, n)


def example_1_batched_tensor():
    """Batched SparseTensor with same layout."""
    print("=" * 60)
    print("Example 1: Batched SparseTensor")
    print("=" * 60)
    
    n = 50
    val, row, col, shape = create_tridiagonal(n)
    
    # Create batch by repeating values with variations
    batch_size = 8
    val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
    
    # Vary the diagonal values slightly for each batch
    for i in range(batch_size):
        scale = 1.0 + 0.1 * i
        val_batch[i] = val * scale
    
    # Create batched SparseTensor
    A_batch = SparseTensor(
        val_batch, row, col, (batch_size, n, n)
    )
    print(f"Batched SparseTensor: {A_batch}")
    print(f"  Shape: {A_batch.shape}")
    print(f"  Batch shape: {A_batch.batch_shape}")
    print(f"  Sparse shape: {A_batch.sparse_shape}")
    print(f"  NNZ: {A_batch.nnz}")
    
    # Batched solve
    b_batch = torch.randn(batch_size, n, dtype=torch.float64)
    x_batch = A_batch.solve(b_batch)
    
    print(f"\nBatched solve:")
    print(f"  Input b shape: {b_batch.shape}")
    print(f"  Output x shape: {x_batch.shape}")
    
    # Verify each solution
    max_residual = 0
    for i in range(batch_size):
        A_i = SparseTensor(val_batch[i], row, col, (n, n))
        residual = (A_i @ x_batch[i] - b_batch[i]).norm() / b_batch[i].norm()
        max_residual = max(max_residual, residual.item())
    print(f"  Max relative residual: {max_residual:.2e}")


def example_2_solve_batch():
    """solve_batch for different values with same structure."""
    print("\n" + "=" * 60)
    print("Example 2: solve_batch (Same Structure, Different Values)")
    print("=" * 60)
    
    n = 50
    val, row, col, shape = create_tridiagonal(n)
    
    # Create template SparseTensor
    A = SparseTensor(val, row, col, shape)
    print(f"Template matrix: {A}")
    
    # Create batch of values (same structure, different values)
    batch_size = 16
    val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
    for i in range(batch_size):
        val_batch[i] = val * (1.0 + 0.05 * i)
    
    b_batch = torch.randn(batch_size, n, dtype=torch.float64)
    
    # Use solve_batch - efficient for same structure, different values
    x_batch = A.solve_batch(val_batch, b_batch)
    
    print(f"\nsolve_batch:")
    print(f"  Batch size: {batch_size}")
    print(f"  Values batch shape: {val_batch.shape}")
    print(f"  RHS batch shape: {b_batch.shape}")
    print(f"  Solution shape: {x_batch.shape}")
    
    # Verify
    max_residual = 0
    for i in range(batch_size):
        A_i = SparseTensor(val_batch[i], row, col, (n, n))
        residual = (A_i @ x_batch[i] - b_batch[i]).norm() / b_batch[i].norm()
        max_residual = max(max_residual, residual.item())
    print(f"  Max relative residual: {max_residual:.2e}")


def example_3_sparse_tensor_list():
    """SparseTensorList for different layouts."""
    print("\n" + "=" * 60)
    print("Example 3: SparseTensorList (Different Layouts)")
    print("=" * 60)
    
    # Create matrices with different sizes
    sizes = [20, 50, 100, 200]
    tensors = []
    b_list = []
    
    for n in sizes:
        val, row, col, shape = create_tridiagonal(n)
        A = SparseTensor(val, row, col, shape)
        tensors.append(A)
        b_list.append(torch.randn(n, dtype=torch.float64))
    
    # Create SparseTensorList
    matrices = SparseTensorList(tensors)
    print(f"SparseTensorList: {matrices}")
    print(f"  Length: {len(matrices)}")
    print(f"  Shapes: {matrices.shapes}")
    
    # Batch solve with different layouts
    x_list = matrices.solve(b_list)
    
    print(f"\nBatch solve results:")
    for i, (A, x, b) in enumerate(zip(matrices, x_list, b_list)):
        residual = (A @ x - b).norm() / b.norm()
        print(f"  Matrix {i} ({sizes[i]}x{sizes[i]}): residual = {residual:.2e}")
    
    # Other batch operations
    norms = matrices.norm('fro')
    print(f"\nFrobenius norms: {[f'{n:.2f}' for n in norms]}")
    
    # Property detection (auto-computed)
    is_sym_list = matrices.is_symmetric()
    is_pd_list = matrices.is_positive_definite()
    print(f"All symmetric: {all(r.item() for r in is_sym_list)}")
    print(f"All positive definite: {all(r.item() for r in is_pd_list)}")


def example_4_batched_eigenvalues():
    """Batched eigenvalue computation."""
    print("\n" + "=" * 60)
    print("Example 4: Batched Eigenvalues")
    print("=" * 60)
    
    n = 30
    val, row, col, shape = create_tridiagonal(n)
    
    batch_size = 4
    val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
    for i in range(batch_size):
        val_batch[i] = val * (1.0 + 0.2 * i)
    
    A_batch = SparseTensor(val_batch, row, col, (batch_size, n, n))
    
    # Batched eigenvalue computation
    k = 5
    eigenvalues, eigenvectors = A_batch.eigsh(k=k, which='LM')
    
    print(f"Batched eigsh:")
    print(f"  Eigenvalues shape: {eigenvalues.shape}")
    print(f"  Eigenvectors shape: {eigenvectors.shape}")
    print(f"\nLargest eigenvalues per batch:")
    for i in range(batch_size):
        print(f"  Batch {i}: {eigenvalues[i].tolist()}")
    
    # Batched SVD
    U, S, Vt = A_batch.svd(k=k)
    print(f"\nBatched SVD:")
    print(f"  U shape: {U.shape}")
    print(f"  S shape: {S.shape}")
    print(f"  Vt shape: {Vt.shape}")


def example_5_cuda_batched():
    """CUDA batched operations."""
    print("\n" + "=" * 60)
    print("Example 5: CUDA Batched Operations")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping...")
        return
    
    n = 100
    val, row, col, shape = create_tridiagonal(n, device='cuda')
    
    batch_size = 8
    val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
    
    A_batch = SparseTensor(
        val_batch, row, col, (batch_size, n, n)
    )
    print(f"CUDA Batched SparseTensor: {A_batch}")
    
    # Batched solve on CUDA
    b_batch = torch.randn(batch_size, n, dtype=torch.float64, device='cuda')
    x_batch = A_batch.solve(b_batch)
    
    print(f"\nCUDA batched solve:")
    print(f"  Device: {x_batch.device}")
    print(f"  Shape: {x_batch.shape}")
    
    # Batched matmul on CUDA
    y_batch = A_batch @ b_batch
    print(f"  Matmul device: {y_batch.device}")
    
    # Batched eigenvalues on CUDA (uses LOBPCG)
    eigenvalues, _ = A_batch.eigsh(k=5, which='LM')
    print(f"  Eigenvalues device: {eigenvalues.device}")
    print(f"  Largest eigenvalues (batch 0): {eigenvalues[0].tolist()}")


def example_6_multi_batch():
    """Multi-dimensional batch."""
    print("\n" + "=" * 60)
    print("Example 6: Multi-dimensional Batch")
    print("=" * 60)
    
    n = 20
    val, row, col, shape = create_tridiagonal(n)
    
    # Create 4D batched tensor [B1, B2, M, N]
    B1, B2 = 2, 3
    val_batch = val.unsqueeze(0).unsqueeze(0).expand(B1, B2, -1).clone()
    
    A_batch = SparseTensor(val_batch, row, col, (B1, B2, n, n))
    print(f"4D Batched SparseTensor: {A_batch}")
    print(f"  Shape: {A_batch.shape}")
    print(f"  Batch shape: {A_batch.batch_shape}")
    print(f"  Batch size: {A_batch.batch_size}")
    
    # Batched operations
    norms = A_batch.norm('fro')
    print(f"\nBatched norms shape: {norms.shape}")
    print(f"Norms:\n{norms}")


if __name__ == "__main__":
    example_1_batched_tensor()
    example_2_solve_batch()
    example_3_sparse_tensor_list()
    example_4_batched_eigenvalues()
    example_5_cuda_batched()
    example_6_multi_batch()
    
    print("\n" + "=" * 60)
    print("All batched examples completed successfully!")
    print("=" * 60)
