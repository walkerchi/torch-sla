"""
nvmath-python backend for cuDSS (NVIDIA Direct Sparse Solver)

Supports solving linear systems with pure Python calls to nvmath.bindings.cudss.
Provides LU, Cholesky, and LDLT factorizations for sparse linear systems on CUDA.

Requirements:
    pip install nvmath-python[cu12]
"""

import torch
import nvmath.bindings.cudss as cudss

# cudaDataType constants
CUDA_R_32F = 0
CUDA_R_64F = 1
CUDA_R_32I = 10

# Map torch dtype to cudaDataType
_DTYPE_MAP = {
    torch.float32: CUDA_R_32F,
    torch.float64: CUDA_R_64F,
}

# Map matrix_type string to (MatrixType, MatrixViewType)
_MTYPE_MAP = {
    "general": (cudss.MatrixType.GENERAL.value, cudss.MatrixViewType.FULL.value),
    "symmetric": (cudss.MatrixType.SYMMETRIC.value, cudss.MatrixViewType.LOWER.value),
    "SYMMETRIC": (cudss.MatrixType.SYMMETRIC.value, cudss.MatrixViewType.LOWER.value),
    "spd": (cudss.MatrixType.SPD.value, cudss.MatrixViewType.LOWER.value),
    "SPD": (cudss.MatrixType.SPD.value, cudss.MatrixViewType.LOWER.value),
}


def nvmath_solve(val, row, col, shape, b, matrix_type="general"):
    """Solve sparse linear system Ax = b using cuDSS via nvmath-python.

    Parameters
    ----------
    val : torch.Tensor
        Non-zero values of the sparse matrix (CUDA).
    row : torch.Tensor
        Row indices (COO format, CUDA).
    col : torch.Tensor
        Column indices (COO format, CUDA).
    shape : tuple of (int, int)
        Matrix dimensions (m, n). Must be square.
    b : torch.Tensor
        Right-hand side vector [m] or matrix [m, nrhs] (CUDA).
    matrix_type : str
        One of "general", "symmetric", "spd".

    Returns
    -------
    torch.Tensor
        Solution x with same shape as b.
    """
    m, n = shape
    assert m == n, "Matrix must be square"
    assert val.is_cuda, "val must be on CUDA"
    assert b.is_cuda, "b must be on CUDA"

    # 1. COO → CSR via PyTorch (GPU-native, no CPU roundtrip)
    indices = torch.stack([row, col], dim=0)
    A_coo = torch.sparse_coo_tensor(indices, val, (m, n)).coalesce()
    A_csr = A_coo.to_sparse_csr()
    crow = A_csr.crow_indices().int()   # cuDSS requires int32
    ccol = A_csr.col_indices().int()
    cval = A_csr.values()
    nnz = cval.numel()

    # 2. Prepare b in column-major layout (cuDSS only supports COL_MAJOR)
    is_1d = b.dim() == 1
    b_2d = b.unsqueeze(1) if is_1d else b
    nrhs = b_2d.size(1)
    b_col = b_2d.t().contiguous()   # [nrhs, m] stores columns of b contiguously
    x_col = torch.zeros_like(b_col)

    # Resolve dtype
    value_type = _DTYPE_MAP[cval.dtype]
    mtype, mview = _MTYPE_MAP.get(matrix_type, _MTYPE_MAP["general"])

    # 3. cuDSS three-phase solve
    handle = cudss.create()
    try:
        cudss.set_stream(handle, torch.cuda.current_stream().cuda_stream)

        A_desc = cudss.matrix_create_csr(
            m, n, nnz,
            crow.data_ptr(), 0, ccol.data_ptr(), cval.data_ptr(),
            CUDA_R_32I, value_type,
            mtype, mview, cudss.IndexBase.ZERO.value
        )
        b_desc = cudss.matrix_create_dn(
            m, nrhs, m, b_col.data_ptr(), value_type, cudss.Layout.COL_MAJOR.value
        )
        x_desc = cudss.matrix_create_dn(
            m, nrhs, m, x_col.data_ptr(), value_type, cudss.Layout.COL_MAJOR.value
        )
        config = cudss.config_create()
        data = cudss.data_create(handle)

        cudss.execute(handle, cudss.Phase.ANALYSIS.value, config, data, A_desc, x_desc, b_desc)
        cudss.execute(handle, cudss.Phase.FACTORIZATION.value, config, data, A_desc, x_desc, b_desc)
        cudss.execute(handle, cudss.Phase.SOLVE.value, config, data, A_desc, x_desc, b_desc)

        torch.cuda.synchronize()

        # Cleanup
        cudss.data_destroy(handle, data)
        cudss.config_destroy(config)
        cudss.matrix_destroy(x_desc)
        cudss.matrix_destroy(b_desc)
        cudss.matrix_destroy(A_desc)
    finally:
        cudss.destroy(handle)

    # 4. Convert back to original shape
    x = x_col.t()  # [m, nrhs]
    return x.squeeze(1) if is_1d else x.contiguous()


class _NvmathCudssModule:
    """Drop-in replacement for the JIT-compiled C++ cudss_spsolve module.

    Exposes the same interface: solve(), lu(), cholesky(), ldlt().
    Each method accepts (indices, values, m, n, b, ...) matching the C++ signatures.
    """

    def solve(self, indices, values, m, n, b, matrix_type="general", reorder="default"):
        row, col = indices[0], indices[1]
        return nvmath_solve(values, row, col, (m, n), b, matrix_type)

    def lu(self, indices, values, m, n, b):
        row, col = indices[0], indices[1]
        return nvmath_solve(values, row, col, (m, n), b, "general")

    def cholesky(self, indices, values, m, n, b):
        row, col = indices[0], indices[1]
        return nvmath_solve(values, row, col, (m, n), b, "spd")

    def ldlt(self, indices, values, m, n, b):
        row, col = indices[0], indices[1]
        return nvmath_solve(values, row, col, (m, n), b, "symmetric")
