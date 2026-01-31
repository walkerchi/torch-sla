#!/usr/bin/env python
"""
Basic Usage Examples for torch-sla

This example demonstrates:
1. Creating SparseTensor from various sources
2. Sparsity pattern visualization with spy()
3. Property detection (symmetry, positive definiteness)
4. Matrix operations (multiplication, norms, eigenvalues)
5. Solve operations and gradients
"""

import torch
from torch_sla import SparseTensor, spsolve


# =============================================================================
# 1. Creation
# =============================================================================

def example_1_create_sparse():
    """Create SparseTensor from dense matrix (easier for small matrices)."""
    # Create a simple 3x3 SPD tridiagonal matrix
    dense = torch.tensor([[4.0, -1.0,  0.0],
                          [-1.0, 4.0, -1.0],
                          [ 0.0, -1.0, 4.0]], dtype=torch.float64)
    
    # Create SparseTensor from dense
    A = SparseTensor.from_dense(dense)
    print(f"Created: {A}")
    print(f"Dense form:\n{A.to_dense()}")
    
    return A


def example_2_from_dense():
    """Create SparseTensor from dense matrix."""
    # Create a random sparse matrix
    n = 100
    A_dense = torch.randn(n, n, dtype=torch.float64)
    A_dense = A_dense @ A_dense.T  # Make symmetric
    A_dense[A_dense.abs() < 1.0] = 0  # Sparsify
    A_dense += torch.eye(n) * n  # Make positive definite
    
    # Convert to SparseTensor
    A = SparseTensor.from_dense(A_dense)
    print(f"Created from dense: {A}")
    print(f"Sparsity: {1 - A.nnz / (n * n):.1%}")
    
    return A


# =============================================================================
# 2. Visualization
# =============================================================================

def example_3_spy_visualization():
    """Visualize sparsity patterns with spy()."""
    # Create a 2D Poisson matrix (5-point stencil)
    n = 20
    N = n * n
    idx = torch.arange(N)
    i, j = idx // n, idx % n
    
    # Build COO indices
    diag_val = torch.full((N,), 4.0, dtype=torch.float64)
    left_mask = j > 0
    right_mask = j < n - 1
    up_mask = i > 0
    down_mask = i < n - 1
    
    row = torch.cat([idx, idx[left_mask], idx[right_mask], idx[up_mask], idx[down_mask]])
    col = torch.cat([idx, idx[left_mask] - 1, idx[right_mask] + 1, idx[up_mask] - n, idx[down_mask] + n])
    val = torch.cat([
        diag_val,
        torch.full((left_mask.sum(),), -1.0, dtype=torch.float64),
        torch.full((right_mask.sum(),), -1.0, dtype=torch.float64),
        torch.full((up_mask.sum(),), -1.0, dtype=torch.float64),
        torch.full((down_mask.sum(),), -1.0, dtype=torch.float64),
    ])
    
    A = SparseTensor(val, row, col, (N, N))
    print(f"2D Poisson matrix: {A}")
    print(f"Sparsity: {1 - A.nnz / (N * N):.1%}")
    
    # Visualize sparsity pattern
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # 1. Basic spy plot
        A.spy(title=f'2D Poisson ({n}×{n} grid)', cmap='Greys',
              save_path='figures/spy_poisson.png')
        plt.close()
        
        # 2. Random sparse matrix for comparison
        nnz = 500
        B = SparseTensor(
            torch.randn(nnz, dtype=torch.float64),
            torch.randint(0, 100, (nnz,)),
            torch.randint(0, 100, (nnz,)),
            (100, 100)
        )
        B.spy(title='Random Sparse', cmap='viridis',
              save_path='figures/spy_random.png')
        plt.close()
        
        print("Saved spy plots to figures/")
        
    except ImportError:
        print("matplotlib not available, skipping visualization")
    
    return A


# =============================================================================
# 3. Property Detection
# =============================================================================

def example_4_property_detection():
    """Detect matrix properties."""
    # Create SPD matrix from dense
    dense_A = torch.tensor([[4.0, -1.0,  0.0],
                            [-1.0, 4.0, -1.0],
                            [ 0.0, -1.0, 4.0]], dtype=torch.float64)
    
    A = SparseTensor.from_dense(dense_A)
    
    # Property checks
    print(f"Matrix A:\n{A.to_dense()}")
    print(f"\nIs symmetric: {A.is_symmetric()}")
    print(f"Is positive definite (Gershgorin): {A.is_positive_definite('gershgorin')}")
    print(f"Is positive definite (Cholesky): {A.is_positive_definite('cholesky')}")
    
    # Non-symmetric matrix
    dense_B = torch.tensor([[4.0, -1.0],
                            [-2.0, 4.0]], dtype=torch.float64)
    
    B = SparseTensor.from_dense(dense_B)
    print(f"\nMatrix B (non-symmetric):\n{B.to_dense()}")
    print(f"Is symmetric: {B.is_symmetric()}")


# =============================================================================
# 4. Matrix Operations
# =============================================================================

def example_5_matrix_multiplication():
    """Matrix multiplication operations."""
    # Create sparse matrix from dense
    dense = torch.tensor([[4.0, -1.0,  0.0],
                          [-1.0, 4.0, -1.0],
                          [ 0.0, -1.0, 4.0]], dtype=torch.float64)
    A = SparseTensor.from_dense(dense)
    A_dense = A.to_dense()
    
    # Sparse @ Dense vector
    x = torch.randn(3, dtype=torch.float64)
    y = A @ x
    print(f"Sparse @ Dense vector: correct = {torch.allclose(y, A_dense @ x)}")
    
    # Sparse @ Dense matrix
    X = torch.randn(3, 5, dtype=torch.float64)
    Y = A @ X
    print(f"Sparse @ Dense matrix: correct = {torch.allclose(Y, A_dense @ X)}")
    
    # Dense @ Sparse
    y = x @ A
    print(f"Dense @ Sparse: correct = {torch.allclose(y, x @ A_dense)}")
    
    # Sparse @ Sparse
    B = SparseTensor.from_dense(dense.clone())
    C = A @ B
    print(f"Sparse @ Sparse: correct = {torch.allclose(C.to_dense(), A_dense @ A_dense)}")


def example_6_norms_and_eigenvalues():
    """Norms, eigenvalues, and SVD."""
    # Create tridiagonal matrix
    n = 30
    idx = torch.arange(n)
    
    diag_val = torch.full((n,), 4.0, dtype=torch.float64)
    off_val = torch.full((n - 1,), -1.0, dtype=torch.float64)
    
    row = torch.cat([idx, idx[1:], idx[:-1]])
    col = torch.cat([idx, idx[:-1], idx[1:]])
    val = torch.cat([diag_val, off_val, off_val])
    
    A = SparseTensor(val, row, col, (n, n))
    
    # Norms
    print(f"Frobenius norm: {A.norm('fro'):.4f}")
    
    # Eigenvalues
    eigenvalues, _ = A.eigsh(k=5, which='LM')
    print(f"Largest 5 eigenvalues: {eigenvalues.tolist()}")
    
    # SVD
    U, S, Vt = A.svd(k=5)
    print(f"Largest 5 singular values: {S.tolist()}")
    
    # Condition number
    cond = A.condition_number(ord=2)
    print(f"Condition number: {cond:.4f}")


# =============================================================================
# 5. Solve and Gradients
# =============================================================================

def example_7_basic_solve():
    """Basic sparse linear solve."""
    dense = torch.tensor([[4.0, -1.0,  0.0],
                          [-1.0, 4.0, -1.0],
                          [ 0.0, -1.0, 4.0]], dtype=torch.float64)
    
    A = SparseTensor.from_dense(dense)
    b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    
    # Solve Ax = b
    x = A.solve(b)
    print(f"Solution x: {x}")
    
    # Verify
    residual = A @ x - b
    print(f"Residual ||Ax - b||: {residual.norm():.2e}")


def example_8_gradient_through_solve():
    """Compute gradients through sparse solve."""
    # Create matrix with gradient tracking
    val = torch.tensor([4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0], 
                       dtype=torch.float64, requires_grad=True)
    row = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    col = torch.tensor([0, 1, 0, 1, 2, 1, 2])
    
    b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    
    # Solve
    x = spsolve(val, row, col, (3, 3), b)
    
    # Backpropagate
    loss = x.sum()
    loss.backward()
    
    print(f"Solution x: {x.tolist()}")
    print(f"∂L/∂val: {val.grad.tolist()}")
    print(f"∂L/∂b: {b.grad.tolist()}")


def example_9_sparse_gradient():
    """Sparse @ Sparse with sparse gradient."""
    # Create matrices with gradient tracking
    val_a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
    row_a = torch.tensor([0, 0, 1, 1])
    col_a = torch.tensor([0, 1, 0, 1])
    A = SparseTensor(val_a, row_a, col_a, (2, 2))
    
    val_b = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    row_b = torch.tensor([0, 1])
    col_b = torch.tensor([0, 1])
    B = SparseTensor(val_b, row_b, col_b, (2, 2))
    
    # Sparse @ Sparse
    C = A @ B
    loss = C.to_dense().sum()
    loss.backward()
    
    print(f"A @ B = \n{C.to_dense()}")
    print(f"\nGradients (SPARSE - same size as input values):")
    print(f"  val_a.grad: {val_a.grad.tolist()}")
    print(f"  val_b.grad: {val_b.grad.tolist()}")


# =============================================================================
# 6. Batched Operations
# =============================================================================

def example_10_batched_operations():
    """Batched operations."""
    n = 10
    idx = torch.arange(n)
    
    # Create tridiagonal matrix
    diag_val = torch.full((n,), 4.0, dtype=torch.float64)
    off_val = torch.full((n - 1,), -1.0, dtype=torch.float64)
    
    row = torch.cat([idx, idx[1:], idx[:-1]])
    col = torch.cat([idx, idx[:-1], idx[1:]])
    val = torch.cat([diag_val, off_val, off_val])
    
    # Create batched tensor [B, M, N]
    batch_size = 4
    val_batch = val.unsqueeze(0).expand(batch_size, -1).clone()
    for i in range(batch_size):
        val_batch[i] = val * (1 + 0.1 * i)
    
    A_batch = SparseTensor(val_batch, row, col, (batch_size, n, n))
    print(f"Batched SparseTensor: {A_batch}")
    
    # Batched solve
    b_batch = torch.randn(batch_size, n, dtype=torch.float64)
    x_batch = A_batch.solve(b_batch)
    print(f"Batched solve output shape: {x_batch.shape}")
    
    # Batched properties
    print(f"Batched is_symmetric: {A_batch.is_symmetric()}")
    print(f"Batched norms: {A_batch.norm('fro').tolist()}")


if __name__ == "__main__":
    import os
    os.makedirs('figures', exist_ok=True)
    
    print("=" * 60)
    print("1. CREATION")
    print("=" * 60)
    example_1_create_sparse()
    print()
    example_2_from_dense()
    
    print("\n" + "=" * 60)
    print("2. VISUALIZATION")
    print("=" * 60)
    example_3_spy_visualization()
    
    print("\n" + "=" * 60)
    print("3. PROPERTY DETECTION")
    print("=" * 60)
    example_4_property_detection()
    
    print("\n" + "=" * 60)
    print("4. MATRIX OPERATIONS")
    print("=" * 60)
    example_5_matrix_multiplication()
    print()
    example_6_norms_and_eigenvalues()
    
    print("\n" + "=" * 60)
    print("5. SOLVE AND GRADIENTS")
    print("=" * 60)
    example_7_basic_solve()
    print()
    example_8_gradient_through_solve()
    print()
    example_9_sparse_gradient()
    
    print("\n" + "=" * 60)
    print("6. BATCHED OPERATIONS")
    print("=" * 60)
    example_10_batched_operations()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
