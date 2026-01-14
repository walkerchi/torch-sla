"""
Backend management for torch-sla

This module provides a unified interface for different sparse linear algebra backends:

Backends:
- 'scipy': SciPy backend (CPU only) - Uses SuperLU for direct solvers
- 'eigen': Eigen backend (CPU only) - Iterative solvers (CG, BiCGStab)
- 'pytorch': PyTorch-native (CPU & CUDA) - Iterative solvers with Jacobi preconditioning
- 'cusolver': NVIDIA cuSOLVER (CUDA only) - Direct solvers (not recommended, use cudss)
- 'cudss': NVIDIA cuDSS (CUDA only) - Direct solvers (LU, Cholesky, LDLT)

Methods (solver algorithms):
- 'superlu': SuperLU direct solver (scipy backend)
- 'umfpack': UMFPACK direct solver (scipy backend, requires scikit-umfpack)
- 'cg': Conjugate Gradient (iterative, for SPD matrices)
- 'bicgstab': BiCGStab (iterative, general matrices)
- 'gmres': GMRES (iterative, general matrices)
- 'qr': QR decomposition (direct)
- 'lu': LU decomposition (direct)
- 'cholesky': Cholesky decomposition (direct, SPD matrices)
- 'ldlt': LDLT decomposition (direct, symmetric matrices)

Recommended Backends (based on benchmarks):
==========================================

float64 (double precision):
---------------------------
- CPU, DOF < 100K: scipy+superlu (best balance of speed and accuracy)
- CPU, DOF >= 100K: scipy+superlu (still fast, machine precision ~1e-15)
- CUDA, DOF < 100K: cudss+cholesky (for SPD) or cudss+lu (general)
- CUDA, DOF >= 100K: cudss+cholesky (direct, ~1e-14 precision)
- CUDA, DOF > 2M: pytorch+cg (iterative, ~1e-6 precision, memory efficient)

float32 (single precision):
---------------------------
- CPU: scipy+superlu (precision ~1e-6)
- CUDA: cudss+cholesky/ldlt (precision ~1e-6)
- Note: cuSOLVER does NOT support float32!
- Note: Iterative methods may not converge well with float32

Usage:
    # Auto-select backend based on device and problem size
    x = spsolve(A, b)  # Uses scipy for CPU, cudss for CUDA (small), pytorch for CUDA (large)
    
    # Specify backend and method
    x = spsolve(A, b, backend='scipy', method='superlu')
    x = spsolve(A, b, backend='cudss', method='cholesky')
    x = spsolve(A, b, backend='pytorch', method='cg')  # GPU iterative
"""

from typing import Optional, List, Dict, Literal
import torch

# Type aliases
BackendType = Literal['scipy', 'eigen', 'pytorch', 'cusolver', 'cudss', 'auto']
MethodType = Literal[
    'auto',
    # Direct methods
    'superlu', 'umfpack', 'lu', 'qr', 'cholesky', 'ldlt',
    # Iterative methods
    'cg', 'bicgstab', 'gmres', 'lgmres', 'minres', 'qmr'
]

# Backend -> supported methods mapping
BACKEND_METHODS: Dict[str, List[str]] = {
    'scipy': ['superlu', 'umfpack', 'cg', 'bicgstab', 'gmres', 'lgmres', 'minres', 'qmr'],
    'eigen': ['cg', 'bicgstab'],
    'pytorch': ['cg', 'bicgstab'],  # PyTorch-native iterative with Jacobi preconditioning
    'cusolver': ['qr', 'cholesky', 'lu'],  # Note: does not support float32!
    'cudss': ['lu', 'cholesky', 'ldlt'],
}

# Default methods for each backend (based on benchmarks)
DEFAULT_METHODS: Dict[str, str] = {
    'scipy': 'superlu',      # Best for CPU: fast + machine precision
    'eigen': 'bicgstab',
    'pytorch': 'cg',         # Use CG for SPD (most common), with Jacobi preconditioning
    'cusolver': 'lu',        # Not recommended, use cudss instead
    'cudss': 'cholesky',     # Best for CUDA: fastest + high precision
}

# Threshold for switching from direct to iterative on CUDA (DOF)
# Based on benchmark: direct solvers (cudss) work well up to ~2M DOF
CUDA_ITERATIVE_THRESHOLD = 2_000_000

# Backend availability flags
_scipy_available: Optional[bool] = None
_cusolver_available: Optional[bool] = None
_cudss_available: Optional[bool] = None
_eigen_available: Optional[bool] = None


def _check_cuda() -> bool:
    """Check if CUDA is available"""
    return torch.cuda.is_available()


def is_scipy_available() -> bool:
    """Check if SciPy backend is available"""
    global _scipy_available
    if _scipy_available is None:
        try:
            import scipy.sparse.linalg
            _scipy_available = True
        except ImportError:
            _scipy_available = False
    return _scipy_available


def is_eigen_available() -> bool:
    """Check if Eigen backend (C++ extension) is available"""
    global _eigen_available
    if _eigen_available is None:
        try:
            _load_eigen_backend()
            _eigen_available = True
        except Exception:
            _eigen_available = False
    return _eigen_available


def is_pytorch_available() -> bool:
    """Check if PyTorch-native backend is available (always True)"""
    return True


def is_cusolver_available() -> bool:
    """Check if cuSOLVER backend is available"""
    global _cusolver_available
    if _cusolver_available is None:
        if not _check_cuda():
            _cusolver_available = False
        else:
            try:
                _load_cusolver_backend()
                _cusolver_available = True
            except Exception:
                _cusolver_available = False
    return _cusolver_available


def is_cudss_available() -> bool:
    """Check if cuDSS backend is available"""
    global _cudss_available
    if _cudss_available is None:
        if not _check_cuda():
            _cudss_available = False
        else:
            try:
                _load_cudss_backend()
                _cudss_available = True
            except Exception:
                _cudss_available = False
    return _cudss_available


def get_available_backends() -> List[str]:
    """Get list of available backends"""
    backends = []
    
    if is_scipy_available():
        backends.append('scipy')
    
    if is_eigen_available():
        backends.append('eigen')
    
    backends.append('pytorch')  # Always available
    
    if is_cusolver_available():
        backends.append('cusolver')
    
    if is_cudss_available():
        backends.append('cudss')
    
    return backends


def get_backend_methods(backend: str) -> List[str]:
    """Get list of methods supported by a backend"""
    return BACKEND_METHODS.get(backend, [])


def get_default_method(backend: str) -> str:
    """Get default method for a backend"""
    return DEFAULT_METHODS.get(backend, 'auto')


def select_backend(
    device: torch.device,
    n: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    prefer_direct: bool = True
) -> str:
    """
    Auto-select the best backend based on device, problem size, and dtype.
    
    Recommendations based on benchmark results:
    - CPU: scipy+superlu (all sizes, fast + machine precision)
    - CUDA (DOF < 2M): cudss+cholesky (fast + high precision)
    - CUDA (DOF >= 2M): pytorch+cg (memory efficient, ~1e-6 precision)
    
    Parameters
    ----------
    device : torch.device
        Target device (cpu or cuda)
    n : int, optional
        Problem size (DOF). If > CUDA_ITERATIVE_THRESHOLD, prefer iterative.
    dtype : torch.dtype, optional
        Data type. Note: cuSOLVER does not support float32!
    prefer_direct : bool
        If True, prefer direct solvers over iterative (when applicable)
        
    Returns
    -------
    str
        Backend name ('scipy', 'eigen', 'pytorch', 'cusolver', or 'cudss')
    """
    if device.type == 'cpu':
        # CPU: scipy is best (SuperLU: fast + machine precision)
        if is_scipy_available():
            return 'scipy'
        if is_eigen_available():
            return 'eigen'
        return 'pytorch'  # Fallback to PyTorch-native
    
    elif device.type == 'cuda':
        # Check dtype - cuSOLVER does NOT support float32!
        is_float32 = dtype == torch.float32 if dtype is not None else False
        
        # Large problem: use iterative (PyTorch-native on GPU with Jacobi preconditioning)
        if n is not None and n > CUDA_ITERATIVE_THRESHOLD:
            return 'pytorch'
        
        # Small/medium problem: prefer direct solvers
        if prefer_direct:
            # cuDSS is best for CUDA (supports both float32 and float64)
            if is_cudss_available():
                return 'cudss'
            # cuSOLVER only for float64
            if is_cusolver_available() and not is_float32:
                return 'cusolver'
        
        # Fallback to iterative
        return 'pytorch'
    
    else:
        raise ValueError(f"Unsupported device type: {device.type}")


def select_method(
    backend: str,
    is_symmetric: bool = False,
    is_spd: bool = False,
    prefer_direct: bool = True
) -> str:
    """
    Auto-select the best method for a given backend and matrix properties.
    
    Recommendations based on benchmark results:
    - scipy: superlu (direct, best precision) or cg (iterative, for SPD)
    - cudss: cholesky (SPD, fastest) > ldlt (symmetric) > lu (general)
    - pytorch: cg (SPD) or bicgstab (general), both with Jacobi preconditioning
    
    Parameters
    ----------
    backend : str
        Backend name
    is_symmetric : bool
        Whether the matrix is symmetric
    is_spd : bool
        Whether the matrix is symmetric positive definite
    prefer_direct : bool
        If True, prefer direct solvers
        
    Returns
    -------
    str
        Method name
    """
    methods = BACKEND_METHODS.get(backend, [])
    
    if backend == 'scipy':
        if prefer_direct:
            return 'superlu'  # Best: fast + machine precision
        elif is_spd:
            return 'cg'
        else:
            return 'bicgstab'
    
    elif backend == 'eigen':
        return 'cg' if is_spd else 'bicgstab'
    
    elif backend == 'pytorch':
        # Iterative with Jacobi preconditioning
        return 'cg' if is_spd else 'bicgstab'
    
    elif backend == 'cusolver':
        # Note: cuSOLVER does not support float32!
        if is_spd and 'cholesky' in methods:
            return 'cholesky'
        return 'lu' if 'lu' in methods else 'qr'
    
    elif backend == 'cudss':
        # Recommendation: cholesky > ldlt > lu (based on benchmarks)
        if is_spd and 'cholesky' in methods:
            return 'cholesky'  # Fastest for SPD
        elif is_symmetric and 'ldlt' in methods:
            return 'ldlt'
        return 'lu'
    
    return DEFAULT_METHODS.get(backend, methods[0] if methods else 'auto')


def _load_eigen_backend():
    """Load Eigen backend (C++ iterative solvers)"""
    global _eigen_module
    
    if _eigen_module is not None:
        return _eigen_module
    
    import os
    from torch.utils.cpp_extension import load
    
    try:
        _eigen_module = load(
            name="spsolve",
            sources=[os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "..", "csrc", "spsolve", "spsolve.cpp"
            ))],
            verbose=False
        )
        return _eigen_module
    except Exception as e:
        raise RuntimeError(f"Failed to load Eigen backend: {e}")


def _load_cusolver_backend():
    """Load cuSOLVER backend"""
    global _cusolver_module
    
    if _cusolver_module is not None:
        return _cusolver_module
    
    if not _check_cuda():
        raise RuntimeError("CUDA is not available")
    
    import os
    from torch.utils.cpp_extension import load
    
    try:
        _cusolver_module = load(
            name="cusolver_spsolve",
            sources=[os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "..", "csrc", "cusolver", "cusolver_spsolve.cu"
            ))],
            extra_ldflags=['-lcusolver', '-lcusparse'],
            verbose=False
        )
        return _cusolver_module
    except Exception as e:
        raise RuntimeError(f"Failed to load cuSOLVER backend: {e}")


def _find_cudss_paths():
    """Find cuDSS include and library paths from pip package"""
    import site
    import os
    
    # Search paths
    search_paths = site.getsitepackages() + [site.getusersitepackages()]
    
    for base in search_paths:
        # nvidia-cudss-cu12 installs to nvidia/cu12/
        include_path = os.path.join(base, 'nvidia', 'cu12', 'include')
        lib_path = os.path.join(base, 'nvidia', 'cu12', 'lib')
        
        if os.path.exists(os.path.join(include_path, 'cudss.h')):
            return include_path, lib_path
    
    return None, None


def _load_cudss_backend():
    """Load cuDSS backend"""
    global _cudss_module
    
    if _cudss_module is not None:
        return _cudss_module
    
    if not _check_cuda():
        raise RuntimeError("CUDA is not available")
    
    import os
    from torch.utils.cpp_extension import load
    
    # Find cuDSS paths
    cudss_include, cudss_lib = _find_cudss_paths()
    
    if cudss_include is None:
        raise RuntimeError("cuDSS not found. Install with: pip install nvidia-cudss-cu12")
    
    try:
        _cudss_module = load(
            name="cudss_spsolve",
            sources=[os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "..", "csrc", "cudss", "cudss_spsolve.cu"
            ))],
            extra_include_paths=[cudss_include],
            extra_ldflags=[f'-L{cudss_lib}', '-lcudss', '-lcusparse', f'-Wl,-rpath,{cudss_lib}'],
            verbose=False
        )
        return _cudss_module
    except Exception as e:
        raise RuntimeError(f"Failed to load cuDSS backend: {e}")


# Lazy-loaded modules
_eigen_module = None
_cusolver_module = None
_cudss_module = None


# Convenience functions for getting modules
def get_cpu_module():
    """Get Eigen backend module (legacy name for compatibility)"""
    return get_eigen_module()


def get_eigen_module():
    """Get Eigen backend module"""
    return _load_eigen_backend()


def get_cusolver_module():
    """Get cuSOLVER backend module"""
    return _load_cusolver_backend()


def get_cudss_module():
    """Get cuDSS backend module"""
    return _load_cudss_backend()
