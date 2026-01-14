#!/usr/bin/env python
"""
Nonlinear Solve Examples for torch-sla

This example demonstrates:
1. Basic nonlinear equation solving with Newton's method
2. Nonlinear PDE: heat equation with temperature-dependent conductivity
3. Multiple parameters with gradient computation
4. Different solvers: Newton, Picard, Anderson
5. Using adjoint gradients for optimization
"""

import torch
from torch_sla import SparseTensor, nonlinear_solve


def example_1_scalar_equation():
    """
    Solve a simple scalar nonlinear equation: u³ - u - θ = 0
    
    This demonstrates:
    - Basic nonlinear_solve usage
    - Adjoint gradient computation
    - Comparison with analytical gradient
    """
    print("=" * 60)
    print("Example 1: Scalar Nonlinear Equation")
    print("=" * 60)
    print("Solving: u³ - u - θ = 0")
    print()
    
    def residual(u, theta):
        return u**3 - u - theta
    
    # Parameter with gradient tracking
    theta = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    u0 = torch.tensor([1.5], dtype=torch.float64)
    
    # Solve
    u = nonlinear_solve(residual, u0, theta, method='newton', verbose=True)
    
    print(f"\nSolution: u = {u.item():.10f}")
    print(f"Residual: F(u) = {residual(u, theta).item():.2e}")
    
    # Compute gradient via adjoint method
    loss = u.sum()
    loss.backward()
    
    # Analytical gradient: du/dθ = 1 / (3u² - 1)
    analytical_grad = 1.0 / (3 * u.item()**2 - 1)
    
    print(f"\n∂u/∂θ (adjoint):    {theta.grad.item():.10f}")
    print(f"∂u/∂θ (analytical): {analytical_grad:.10f}")
    print(f"Error: {abs(theta.grad.item() - analytical_grad):.2e}")


def example_2_nonlinear_pde():
    """
    Solve nonlinear 1D heat equation with temperature-dependent conductivity.
    
    PDE: -d/dx(k(u) * du/dx) = f
    where k(u) = 1 + α*u (conductivity depends on temperature)
    
    Discretized: K @ u + α * u² = f
    
    This demonstrates:
    - SparseTensor.nonlinear_solve interface
    - Physical problem setup
    - Gradient sensitivity analysis
    """
    print("\n" + "=" * 60)
    print("Example 2: Nonlinear Heat Equation")
    print("=" * 60)
    print("PDE: -d/dx(k(u) * du/dx) = f, where k(u) = 1 + α*u")
    print()
    
    # Grid setup
    n = 50
    h = 1.0 / (n + 1)
    
    # Create 1D Laplacian matrix (tridiagonal)
    diag = 2.0 * torch.ones(n, dtype=torch.float64) / h**2
    off = -1.0 * torch.ones(n-1, dtype=torch.float64) / h**2
    
    row = torch.cat([torch.arange(n), torch.arange(n-1), torch.arange(1, n)])
    col = torch.cat([torch.arange(n), torch.arange(1, n), torch.arange(n-1)])
    val = torch.cat([diag, off, off])
    
    K = SparseTensor(val, row, col, (n, n))
    print(f"Stiffness matrix: {K}")
    
    # Parameters
    alpha = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    f = torch.ones(n, dtype=torch.float64, requires_grad=True)  # Uniform heat source
    
    # Nonlinear residual
    def residual(u, K, alpha, f):
        return K @ u + alpha * u**2 - f
    
    # Solve
    u0 = torch.zeros(n, dtype=torch.float64)
    u = K.nonlinear_solve(residual, u0, alpha, f, method='newton', verbose=True)
    
    # Results
    print(f"\nSolution range: [{u.min().item():.4f}, {u.max().item():.4f}]")
    print(f"Residual norm: {torch.norm(residual(u, K, alpha, f)).item():.2e}")
    
    # Sensitivity analysis
    total_temp = u.sum()
    total_temp.backward()
    
    print(f"\n--- Sensitivity Analysis ---")
    print(f"Total temperature: {total_temp.item():.4f}")
    print(f"∂(Σu)/∂α = {alpha.grad.item():.6f}")
    print(f"  → Increasing α (nonlinearity) {'decreases' if alpha.grad.item() < 0 else 'increases'} temperature")
    print(f"Average ∂(Σu)/∂f = {f.grad.mean().item():.6f}")
    print(f"  → Each unit increase in source adds ~{f.grad.mean().item():.4f} to total temp")


def example_3_multiple_parameters():
    """
    Solve with multiple parameters and verify all gradients.
    
    Equation: a*u² + b*u + c = 0 (quadratic)
    
    This demonstrates:
    - Multiple parameter gradients
    - Comparison with analytical gradients
    """
    print("\n" + "=" * 60)
    print("Example 3: Multiple Parameters")
    print("=" * 60)
    print("Solving: a*u² + b*u + c = 0")
    print()
    
    def residual(u, a, b, c):
        return a * u**2 + b * u + c
    
    # Parameters (will give u = 2.0 as solution)
    a = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    u0 = torch.tensor([2.5], dtype=torch.float64)  # Start near solution
    
    # Solve
    u = nonlinear_solve(residual, u0, a, b, c, method='newton', tol=1e-12)
    
    print(f"Solution: u = {u.item():.10f} (expected: 2.0)")
    print(f"Residual: {abs(residual(u, a, b, c).item()):.2e}")
    
    # Compute gradients
    loss = u.sum()
    loss.backward()
    
    # Analytical gradients via implicit differentiation
    # F(u, a, b, c) = au² + bu + c = 0
    # dF/du = 2au + b
    # ∂u/∂a = -u² / (2au + b)
    # ∂u/∂b = -u / (2au + b)
    # ∂u/∂c = -1 / (2au + b)
    u_val = u.item()
    denom = 2 * a.item() * u_val + b.item()
    
    analytical = {
        'a': -u_val**2 / denom,
        'b': -u_val / denom,
        'c': -1 / denom
    }
    
    print(f"\n--- Gradient Verification ---")
    print(f"{'Param':<6} {'Adjoint':<15} {'Analytical':<15} {'Error':<10}")
    print("-" * 50)
    for name, (grad, ana) in zip(['a', 'b', 'c'], 
                                   [(a.grad.item(), analytical['a']),
                                    (b.grad.item(), analytical['b']),
                                    (c.grad.item(), analytical['c'])]):
        print(f"{name:<6} {grad:<15.10f} {ana:<15.10f} {abs(grad-ana):<10.2e}")


def example_4_solver_comparison():
    """
    Compare different nonlinear solvers: Newton, Picard, Anderson.
    
    This demonstrates:
    - Different solver methods
    - Convergence characteristics
    """
    print("\n" + "=" * 60)
    print("Example 4: Solver Comparison")
    print("=" * 60)
    print("Solving: u - tanh(1.5*u + 0.5) = 0")
    print()
    
    def residual(u, alpha):
        return u - torch.tanh(alpha * u + 0.5)
    
    alpha = torch.tensor([1.5], dtype=torch.float64)
    u0 = torch.tensor([0.0], dtype=torch.float64)
    
    methods = ['newton', 'picard', 'anderson']
    results = {}
    
    for method in methods:
        print(f"\n--- {method.upper()} ---")
        u = nonlinear_solve(
            residual, u0, alpha, 
            method=method, 
            verbose=True, 
            max_iter=100,
            tol=1e-10
        )
        F = residual(u, alpha)
        results[method] = (u.item(), abs(F.item()))
        print(f"Solution: u = {u.item():.6f}, Residual: {abs(F.item()):.2e}")
    
    print(f"\n--- Summary ---")
    print(f"{'Method':<10} {'Solution':<15} {'Residual':<15}")
    print("-" * 40)
    for method, (sol, res) in results.items():
        print(f"{method:<10} {sol:<15.6f} {res:<15.2e}")


def example_5_optimization():
    """
    Use adjoint gradients for parameter optimization.
    
    Goal: Find parameter α such that the solution u matches a target.
    
    This demonstrates:
    - Integration with PyTorch optimizers
    - Inverse problem solving
    """
    print("\n" + "=" * 60)
    print("Example 5: Parameter Optimization")
    print("=" * 60)
    print("Goal: Find α such that u(α) ≈ target")
    print()
    
    # Setup: simple nonlinear equation u³ + α*u = b
    def residual(u, alpha, b):
        return u**3 + alpha * u - b
    
    # True parameter and target
    true_alpha = torch.tensor([2.0], dtype=torch.float64)
    b = torch.tensor([3.0], dtype=torch.float64)
    u0 = torch.tensor([1.0], dtype=torch.float64)
    
    target = nonlinear_solve(residual, u0, true_alpha, b, verbose=False)
    print(f"True α = {true_alpha.item():.4f}")
    print(f"Target u = {target.item():.6f}")
    
    # Optimization: find α from observations
    alpha_guess = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    
    print(f"\n--- Gradient Descent ---")
    print(f"Initial guess: α = {alpha_guess.item():.4f}")
    
    lr = 2.0  # Larger learning rate for faster convergence
    for epoch in range(30):
        # Forward solve
        u = nonlinear_solve(residual, u0, alpha_guess, b, verbose=False)
        
        # Loss: match target
        loss = (u - target)**2
        
        # Backward via adjoint
        loss.backward()
        
        # Manual gradient descent (to avoid optimizer state issues)
        with torch.no_grad():
            alpha_guess -= lr * alpha_guess.grad
            alpha_guess.grad.zero_()
        
        if epoch % 5 == 0 or epoch == 29:
            print(f"Epoch {epoch:3d}: α = {alpha_guess.item():.6f}, Loss = {loss.item():.6e}")
    
    print(f"\n--- Result ---")
    print(f"Recovered α = {alpha_guess.item():.6f}")
    print(f"True α      = {true_alpha.item():.6f}")
    print(f"Error       = {abs(alpha_guess.item() - true_alpha.item()):.6f}")


def example_6_sparse_nonlinear():
    """
    Solve a larger nonlinear system using SparseTensor.
    
    This demonstrates:
    - Scaling to larger problems
    - Practical FEM-like setup
    """
    print("\n" + "=" * 60)
    print("Example 6: Larger Sparse Nonlinear System")
    print("=" * 60)
    
    # 2D Poisson-like problem on a grid
    nx, ny = 20, 20
    n = nx * ny
    print(f"Grid: {nx} x {ny} = {n} DOF")
    
    # Create 2D Laplacian (5-point stencil)
    def idx(i, j):
        return i * ny + j
    
    rows, cols, vals = [], [], []
    
    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)
            # Diagonal
            rows.append(k)
            cols.append(k)
            vals.append(4.0)
            
            # Off-diagonals
            if i > 0:
                rows.append(k)
                cols.append(idx(i-1, j))
                vals.append(-1.0)
            if i < nx - 1:
                rows.append(k)
                cols.append(idx(i+1, j))
                vals.append(-1.0)
            if j > 0:
                rows.append(k)
                cols.append(idx(i, j-1))
                vals.append(-1.0)
            if j < ny - 1:
                rows.append(k)
                cols.append(idx(i, j+1))
                vals.append(-1.0)
    
    row = torch.tensor(rows)
    col = torch.tensor(cols)
    val = torch.tensor(vals, dtype=torch.float64)
    
    A = SparseTensor(val, row, col, (n, n))
    print(f"Matrix: {A}")
    print(f"Sparsity: {100 * (1 - A.nnz / n**2):.1f}%")
    
    # Nonlinear problem: A @ u + 0.1*u³ = f
    f = torch.randn(n, dtype=torch.float64, requires_grad=True)
    
    def residual(u, A, f):
        return A @ u + 0.1 * u**3 - f
    
    u0 = torch.zeros(n, dtype=torch.float64)
    
    print("\nSolving nonlinear system...")
    u = A.nonlinear_solve(residual, u0, f, method='newton', verbose=True)
    
    F = residual(u, A, f)
    print(f"\nResidual norm: {torch.norm(F).item():.2e}")
    print(f"Solution range: [{u.min().item():.4f}, {u.max().item():.4f}]")
    
    # Gradient
    loss = u.sum()
    loss.backward()
    print(f"||∂L/∂f||: {torch.norm(f.grad).item():.6f}")


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# torch-sla Nonlinear Solve Examples")
    print("#" * 60 + "\n")
    
    example_1_scalar_equation()
    example_2_nonlinear_pde()
    example_3_multiple_parameters()
    example_4_solver_comparison()
    example_5_optimization()
    example_6_sparse_nonlinear()
    
    print("\n" + "#" * 60)
    print("# All examples completed successfully!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()

