"""
Example: Sparse Linear Solve with torch-sla

This example demonstrates how to use different backends for solving
sparse linear equations Ax = b.
"""

import torch
import torch_sla as sla


def create_spd_matrix(n: int, density: float = 0.3, device: str = 'cpu'):
    """Create a sparse symmetric positive definite matrix"""
    A = torch.rand(n, n, dtype=torch.float64, device=device)
    A = A @ A.T + torch.eye(n, dtype=torch.float64, device=device) * n
    A[A.abs() < (1 - density)] = 0
    return A.to_sparse_coo()


def example_cpu_solvers():
    """Example using CPU iterative solvers"""
    print("\n" + "=" * 60)
    print("CPU Iterative Solvers")
    print("=" * 60)

    n = 100
    A = create_spd_matrix(n, density=0.3, device='cpu')
    b = torch.randn(n, dtype=torch.float64)

    # Conjugate Gradient (for SPD matrices)
    x_cg = sla.spsolve(
        A.values(), A.indices()[0], A.indices()[1], A.shape, b,
        method='cg', atol=1e-10, maxiter=10000
    )

    # BiCGStab (for general matrices)
    x_bicg = sla.spsolve(
        A.values(), A.indices()[0], A.indices()[1], A.shape, b,
        method='bicgstab', atol=1e-10, maxiter=10000
    )

    # Verify solutions
    A_dense = A.to_dense()
    residual_cg = (torch.mv(A_dense, x_cg) - b).norm() / b.norm()
    residual_bicg = (torch.mv(A_dense, x_bicg) - b).norm() / b.norm()

    print(f"Matrix size: {n}x{n}, NNZ: {A._nnz()}")
    print(f"CG relative residual: {residual_cg:.2e}")
    print(f"BiCGStab relative residual: {residual_bicg:.2e}")


def example_cupy():
    """Example using CuPy solvers (direct and iterative on GPU)"""
    print("\n" + "=" * 60)
    print("CuPy GPU Solvers (Direct + Iterative)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping CuPy example")
        return

    if not sla.is_cupy_available():
        print("CuPy backend not available, skipping")
        return

    n = 100
    A = create_spd_matrix(n, density=0.3, device='cuda')
    b = torch.randn(n, dtype=torch.float64, device='cuda')

    # Direct solver (SuperLU on GPU)
    x_direct = sla.spsolve(
        A.values(), A.indices()[0], A.indices()[1], A.shape, b,
        backend='cupy', method='lu'
    )

    # CG iterative solver (for SPD matrices)
    x_cg = sla.spsolve(
        A.values(), A.indices()[0], A.indices()[1], A.shape, b,
        backend='cupy', method='cg', atol=1e-10
    )

    # GMRES iterative solver (for general matrices)
    x_gmres = sla.spsolve(
        A.values(), A.indices()[0], A.indices()[1], A.shape, b,
        backend='cupy', method='gmres', atol=1e-10
    )

    # Verify solutions
    A_dense = A.to_dense()
    residual_direct = (torch.mv(A_dense, x_direct) - b).norm() / b.norm()
    residual_cg = (torch.mv(A_dense, x_cg) - b).norm() / b.norm()
    residual_gmres = (torch.mv(A_dense, x_gmres) - b).norm() / b.norm()

    print(f"Matrix size: {n}x{n}, NNZ: {A._nnz()}")
    print(f"Direct (spsolve) relative residual: {residual_direct:.2e}")
    print(f"CG relative residual: {residual_cg:.2e}")
    print(f"GMRES relative residual: {residual_gmres:.2e}")

    # Also demonstrate float32 support
    A32 = create_spd_matrix(n, density=0.3, device='cuda').coalesce()
    A32 = torch.sparse_coo_tensor(A32.indices(), A32.values().float(), A32.shape).coalesce()
    b32 = torch.randn(n, dtype=torch.float32, device='cuda')
    x32 = sla.spsolve(
        A32.values(), A32.indices()[0], A32.indices()[1], A32.shape, b32,
        backend='cupy', method='lu'
    )
    residual32 = (torch.mv(A32.to_dense(), x32) - b32).norm() / b32.norm()
    print(f"Float32 spsolve relative residual: {residual32:.2e}")


def example_cudss():
    """Example using cuDSS direct solvers"""
    print("\n" + "=" * 60)
    print("cuDSS Direct Solvers")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping cuDSS example")
        return

    if not sla.is_cudss_available():
        print("cuDSS backend not available, skipping")
        return

    n = 100
    A = create_spd_matrix(n, density=0.3, device='cuda')
    b = torch.randn(n, dtype=torch.float64, device='cuda')

    # LU factorization
    x_lu = sla.spsolve(
        A.values(), A.indices()[0], A.indices()[1], A.shape, b,
        backend='cudss', method='lu'
    )

    # Cholesky factorization (for SPD matrices)
    x_chol = sla.spsolve(
        A.values(), A.indices()[0], A.indices()[1], A.shape, b,
        backend='cudss', method='cholesky'
    )

    # LDLT factorization (for symmetric matrices)
    x_ldlt = sla.spsolve(
        A.values(), A.indices()[0], A.indices()[1], A.shape, b,
        backend='cudss', method='ldlt'
    )

    # Verify solutions
    A_dense = A.to_dense()
    residual_lu = (torch.mv(A_dense, x_lu) - b).norm() / b.norm()
    residual_chol = (torch.mv(A_dense, x_chol) - b).norm() / b.norm()
    residual_ldlt = (torch.mv(A_dense, x_ldlt) - b).norm() / b.norm()

    print(f"Matrix size: {n}x{n}, NNZ: {A._nnz()}")
    print(f"LU relative residual: {residual_lu:.2e}")
    print(f"Cholesky relative residual: {residual_chol:.2e}")
    print(f"LDLT relative residual: {residual_ldlt:.2e}")


def example_gradient():
    """Example showing gradient computation"""
    print("\n" + "=" * 60)
    print("Gradient Computation Example")
    print("=" * 60)

    n = 50
    
    # Create sparse matrix with gradient tracking
    A_dense = torch.rand(n, n, dtype=torch.float64)
    A_dense = A_dense @ A_dense.T + torch.eye(n, dtype=torch.float64) * n
    A_dense[A_dense.abs() < 0.7] = 0
    A = A_dense.to_sparse_coo()

    val = A.values().clone().requires_grad_(True)
    b = torch.randn(n, dtype=torch.float64).requires_grad_(True)

    # Solve with gradient
    x = sla.spsolve(
        val, A.indices()[0], A.indices()[1], A.shape, b,
        method='bicgstab', atol=1e-10
    )

    # Compute loss and backpropagate
    loss = (x ** 2).sum()
    loss.backward()

    print(f"Matrix size: {n}x{n}, NNZ: {A._nnz()}")
    print(f"Solution norm: {x.norm():.4f}")
    print(f"Gradient w.r.t. values: shape={val.grad.shape}, norm={val.grad.norm():.4f}")
    print(f"Gradient w.r.t. b: shape={b.grad.shape}, norm={b.grad.norm():.4f}")


def example_convenience_functions():
    """Example using convenience functions with PyTorch sparse tensors"""
    print("\n" + "=" * 60)
    print("Convenience Functions")
    print("=" * 60)

    n = 50
    A = create_spd_matrix(n, density=0.3, device='cpu')
    b = torch.randn(n, dtype=torch.float64)

    # Using spsolve_coo with sparse COO tensor
    x1 = sla.spsolve_coo(A, b, method='bicgstab')

    # Using spsolve_csr with sparse CSR tensor
    A_csr = A.to_sparse_csr()
    x2 = sla.spsolve_csr(A_csr, b, method='bicgstab')

    print(f"Matrix size: {n}x{n}")
    print(f"Solution via spsolve_coo: norm={x1.norm():.4f}")
    print(f"Solution via spsolve_csr: norm={x2.norm():.4f}")
    print(f"Solutions match: {torch.allclose(x1, x2, rtol=1e-5)}")


if __name__ == '__main__':
    print("torch-sla Examples")
    print(f"Available backends: {sla.get_available_backends()}")
    print(f"CuPy available: {sla.is_cupy_available()}")
    print(f"cuDSS available: {sla.is_cudss_available()}")

    # Run examples
    example_cpu_solvers()
    example_cupy()
    example_cudss()
    example_gradient()
    example_convenience_functions()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
