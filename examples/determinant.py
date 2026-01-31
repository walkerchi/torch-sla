"""
Determinant Computation with Gradient Support
==============================================

This example demonstrates how to compute determinants of sparse matrices
with automatic differentiation support on both CPU and CUDA.

Features:
- Basic determinant computation
- Gradient computation via autograd
- CUDA support
- Batched matrices
- Distributed sparse matrices
- Integration with optimization
"""

import torch
from torch_sla import SparseTensor, DSparseTensor

print("=" * 70)
print("Determinant Computation Examples")
print("=" * 70)
print()

# ============================================================================
# Example 1: Basic Determinant
# ============================================================================
print("Example 1: Basic Determinant Computation")
print("-" * 70)

# Create a simple 3x3 tridiagonal matrix from dense
dense = torch.tensor([[4.0, -1.0,  0.0],
                      [-1.0, 4.0, -1.0],
                      [ 0.0, -1.0, 4.0]], dtype=torch.float64)

A = SparseTensor.from_dense(dense)
det = A.det()

print(f"Matrix: 3x3 tridiagonal")
print(f"  [[4, -1,  0],")
print(f"   [-1, 4, -1],")
print(f"   [ 0, -1, 4]]")
print(f"Determinant: {det.item():.6f}")
print()

# ============================================================================
# Example 2: Gradient Computation
# ============================================================================
print("Example 2: Gradient Computation")
print("-" * 70)

# Create matrix with gradient tracking
# Note: For gradient computation, create SparseTensor with requires_grad on val
val = torch.tensor([2.0, 1.0, 1.0, 3.0], dtype=torch.float64, requires_grad=True)
row = torch.tensor([0, 0, 1, 1])
col = torch.tensor([0, 1, 0, 1])

A = SparseTensor(val, row, col, (2, 2))
det = A.det()

print(f"Matrix: [[2, 1], [1, 3]]")
print(f"Determinant: {det.item():.6f}")

# Compute gradient: ∂det/∂A_ij = det(A) * (A^{-1})_ji
det.backward()

print(f"\nGradients (∂det/∂val):")
print(f"  ∂det/∂A[0,0] = {val.grad[0].item():.6f}")
print(f"  ∂det/∂A[0,1] = {val.grad[1].item():.6f}")
print(f"  ∂det/∂A[1,0] = {val.grad[2].item():.6f}")
print(f"  ∂det/∂A[1,1] = {val.grad[3].item():.6f}")
print()

# ============================================================================
# Example 3: CUDA Support (Performance Warning!)
# ============================================================================
if torch.cuda.is_available():
    print("Example 3: CUDA Support - Performance Comparison")
    print("-" * 70)
    
    dense_cuda = torch.tensor([[1.0, 2.0],
                               [3.0, 4.0]], dtype=torch.float64, device='cuda')
    
    A_cuda = SparseTensor.from_dense(dense_cuda)
    
    # Method 1: Direct CUDA (slow - uses dense conversion)
    import time
    torch.cuda.synchronize()
    start = time.time()
    det_cuda = A_cuda.det()
    torch.cuda.synchronize()
    time_cuda = (time.time() - start) * 1000
    
    # Method 2: CPU computation (fast - uses sparse LU)
    start = time.time()
    det_cpu = A_cuda.cpu().det()
    time_cpu = (time.time() - start) * 1000
    
    print(f"Matrix: [[1, 2], [3, 4]]")
    print(f"\nMethod 1 - Direct CUDA: {det_cuda.item():.6f}")
    print(f"  Time: {time_cuda:.3f} ms")
    print(f"  Note: Uses dense conversion (slow for sparse matrices)")
    
    print(f"\nMethod 2 - CPU for CUDA: {det_cpu.item():.6f}")
    print(f"  Time: {time_cpu:.3f} ms")
    print(f"  Note: Uses sparse LU (recommended)")
    
    print(f"\n⚠️  CPU is {time_cuda/time_cpu:.1f}x faster than CUDA for sparse matrices!")
    print(f"Recommendation: Use A_cuda.cpu().det() instead of A_cuda.det()")
    print()
else:
    print("Example 3: CUDA not available, skipping")
    print()

# ============================================================================
# Example 4: Batched Matrices
# ============================================================================
print("Example 4: Batched Matrices")
print("-" * 70)

# Create 3 different 2x2 matrices
val_batch = torch.tensor([
    [2.0, 0.0, 0.0, 3.0],  # [[2, 0], [0, 3]], det = 6
    [1.0, 0.5, 0.5, 1.0],  # [[1, 0.5], [0.5, 1]], det = 0.75
    [4.0, -1.0, -1.0, 4.0] # [[4, -1], [-1, 4]], det = 15
], dtype=torch.float64)
row = torch.tensor([0, 0, 1, 1])
col = torch.tensor([0, 1, 0, 1])

A_batch = SparseTensor(val_batch, row, col, (3, 2, 2))
det_batch = A_batch.det()

print(f"Batch size: 3")
print(f"Determinants: {det_batch.tolist()}")
print(f"Expected: [6.0, 0.75, 15.0]")
print()

# ============================================================================
# Example 5: Distributed Sparse Matrices
# ============================================================================
print("Example 5: Distributed Sparse Matrices")
print("-" * 70)

# Create distributed sparse tensor with 2 partitions
val = torch.tensor([4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0], dtype=torch.float64)
row = torch.tensor([0, 0, 1, 1, 1, 2, 2])
col = torch.tensor([0, 1, 0, 1, 2, 1, 2])

D = DSparseTensor(val, row, col, (3, 3), num_partitions=2, verbose=False)

# Compute determinant (automatically gathers all partitions)
det_dist = D.det()

print(f"Matrix: 3x3 tridiagonal (2 partitions)")
print(f"Determinant: {det_dist.item():.6f}")
print(f"Note: DSparseTensor.det() gathers all partitions to compute")
print()

# ============================================================================
# Example 6: Optimization with Determinant Constraint
# ============================================================================
print("Example 6: Optimization with Determinant Constraint")
print("-" * 70)

# Optimize matrix values to achieve target determinant
val = torch.tensor([1.0, 0.5, 0.5, 1.0], dtype=torch.float64, requires_grad=True)
row = torch.tensor([0, 0, 1, 1])
col = torch.tensor([0, 1, 0, 1])

target_det = torch.tensor(2.0, dtype=torch.float64)
optimizer = torch.optim.Adam([val], lr=0.1)

print("Optimizing matrix to achieve det(A) = 2.0")
print("Initial matrix values:", val.detach().tolist())

for i in range(50):
    optimizer.zero_grad()
    A = SparseTensor(val, row, col, (2, 2))
    det = A.det()
    loss = (det - target_det) ** 2
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 10 == 0:
        print(f"  Iteration {i+1}: det = {det.item():.6f}, loss = {loss.item():.6e}")

print(f"Final matrix values: {val.detach().tolist()}")
print(f"Final determinant: {A.det().item():.6f}")
print()

# ============================================================================
# Example 7: Numerical Stability
# ============================================================================
print("Example 7: Numerical Stability")
print("-" * 70)

# Test with different matrix sizes
for n in [2, 5, 10]:
    # Create diagonal matrix with known determinant
    val = torch.arange(1, n+1, dtype=torch.float64)
    row = torch.arange(n)
    col = torch.arange(n)
    
    A = SparseTensor(val, row, col, (n, n))
    det = A.det()
    
    # Expected determinant is the product of diagonal elements
    expected = torch.prod(val)
    error = abs(det.item() - expected.item())
    
    print(f"  {n}x{n} diagonal matrix:")
    print(f"    Computed:  {det.item():.6f}")
    print(f"    Expected:  {expected.item():.6f}")
    print(f"    Error:     {error:.2e}")

print()

# ============================================================================
# Example 8: Matrix Properties via Determinant
# ============================================================================
print("Example 8: Matrix Properties via Determinant")
print("-" * 70)

# Check if matrix is singular
def is_singular(A, tol=1e-10):
    det = A.det()
    return abs(det.item()) < tol

# Non-singular matrix
val1 = torch.tensor([2.0, 1.0, 1.0, 2.0], dtype=torch.float64)
A1 = SparseTensor(val1, torch.tensor([0,0,1,1]), torch.tensor([0,1,0,1]), (2,2))
print(f"Matrix 1: [[2, 1], [1, 2]]")
print(f"  det = {A1.det().item():.6f}")
print(f"  Singular: {is_singular(A1)}")

# Nearly singular matrix (to avoid LU decomposition failure)
val2 = torch.tensor([1.0, 2.0, 2.0, 4.001], dtype=torch.float64)
A2 = SparseTensor(val2, torch.tensor([0,0,1,1]), torch.tensor([0,1,0,1]), (2,2))
print(f"\nMatrix 2: [[1, 2], [2, 4.001]] (nearly singular)")
print(f"  det = {A2.det().item():.6f}")
print(f"  Singular: {is_singular(A2)}")
print(f"  Note: Exact singular matrices may fail LU decomposition")

print()

print("=" * 70)
print("All examples completed!")
print("=" * 70)

