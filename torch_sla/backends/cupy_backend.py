"""
CuPy backend for torch-sla

Provides GPU-accelerated sparse linear solvers via CuPy's scipy-compatible interface.

Supported methods:
- Direct: spsolve (SuperLU on GPU), splu (cached LU factorization)
- Iterative: cg, cgs, gmres, minres, lsqr, lsmr

Features:
- Zero-copy data transfer between PyTorch and CuPy via DLPack
- Supports both float32 and float64
- Direct solvers natively support multi-RHS (2D b)
"""

import os
import torch
from typing import Tuple, Optional


def _ensure_cusolver_lib():
    """Ensure libcusolver.so.11 is discoverable for CuPy's direct solvers."""
    try:
        import site

        for base in site.getsitepackages():
            lib_path = os.path.join(base, "nvidia", "cusolver", "lib")
            if os.path.isdir(lib_path):
                ld_path = os.environ.get("LD_LIBRARY_PATH", "")
                if lib_path not in ld_path:
                    os.environ["LD_LIBRARY_PATH"] = lib_path + ":" + ld_path
                return
    except Exception:
        pass


# Set up library path before any cupy imports
_ensure_cusolver_lib()


def _torch_to_cupy(tensor: torch.Tensor):
    """Convert PyTorch CUDA tensor to CuPy array (zero-copy via DLPack)."""
    import cupy as cp

    return cp.from_dlpack(tensor.detach())


def _cupy_to_torch(array) -> torch.Tensor:
    """Convert CuPy array to PyTorch CUDA tensor (zero-copy via DLPack)."""
    return torch.from_dlpack(array)


def _build_csr(val, row, col, shape):
    """Build CuPy CSR matrix from COO components."""
    import cupy as cp
    from cupyx.scipy.sparse import coo_matrix

    cp_val = _torch_to_cupy(val)
    cp_row = _torch_to_cupy(row.int())
    cp_col = _torch_to_cupy(col.int())
    A_coo = coo_matrix((cp_val, (cp_row, cp_col)), shape=shape)
    return A_coo.tocsr()


def cupy_solve(
    val: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    shape: Tuple[int, int],
    b: torch.Tensor,
    method: str = "lu",
    atol: float = 1e-10,
    maxiter: int = 10000,
    tol: float = 1e-12,
) -> torch.Tensor:
    """
    Solve sparse linear system Ax = b using CuPy.

    Parameters
    ----------
    val : torch.Tensor
        [nnz] Non-zero values (CUDA tensor)
    row : torch.Tensor
        [nnz] Row indices
    col : torch.Tensor
        [nnz] Column indices
    shape : Tuple[int, int]
        (m, n) Matrix shape
    b : torch.Tensor
        [m] or [m, K] Right-hand side
    method : str
        Solver method: 'lu', 'cg', 'cgs', 'gmres', 'minres', 'lsqr', 'lsmr'
    atol : float
        Absolute tolerance for iterative solvers
    maxiter : int
        Maximum iterations for iterative solvers
    tol : float
        Tolerance for direct solvers (unused, kept for API compatibility)

    Returns
    -------
    torch.Tensor
        Solution tensor on the same device as input
    """
    import cupy as cp
    from cupyx.scipy.sparse import linalg as splinalg

    A_csr = _build_csr(val, row, col, shape)
    cp_b = _torch_to_cupy(b)

    if method in ("lu", "spsolve"):
        cp_x = splinalg.spsolve(A_csr, cp_b)
    elif method in ("cg", "cgs", "gmres", "minres", "lsqr", "lsmr"):
        solver_fn = getattr(splinalg, method)
        if cp_b.ndim == 2:
            # Iterative solvers only support 1D RHS; loop over columns
            cols = []
            for k in range(cp_b.shape[1]):
                x_k, info = solver_fn(A_csr, cp_b[:, k], atol=atol, maxiter=maxiter)
                if info != 0:
                    import warnings

                    warnings.warn(
                        f"CuPy {method} did not converge for column {k} (info={info})"
                    )
                cols.append(x_k)
            cp_x = cp.stack(cols, axis=1)
        else:
            cp_x, info = solver_fn(A_csr, cp_b, atol=atol, maxiter=maxiter)
            if info != 0:
                import warnings

                warnings.warn(f"CuPy {method} did not converge (info={info})")
    else:
        raise ValueError(
            f"Unknown CuPy method: {method}. "
            f"Available: lu, cg, cgs, gmres, minres, lsqr, lsmr"
        )

    return _cupy_to_torch(cp_x)
