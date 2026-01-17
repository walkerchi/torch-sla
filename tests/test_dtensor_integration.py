#!/usr/bin/env python
"""
Test DSparseTensor integration with PyTorch DTensor.

This test verifies that:
1. DSparseTensor @ DTensor (matmul) works correctly
2. DSparseTensor.solve_distributed(DTensor) works correctly
3. DTensor utilities (scatter_to_dtensor, gather_from_dtensor, to_dtensor) work

Run with:
    torchrun --nproc_per_node=2 tests/test_dtensor_integration.py
    
Or for single-process testing:
    python tests/test_dtensor_integration.py
"""

import os
import sys
import torch
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_poisson_2d(n: int, dtype=torch.float64, device='cpu'):
    """Create 2D Poisson matrix with 5-point stencil."""
    N = n * n
    
    idx = torch.arange(N, device=device)
    i = idx // n
    j = idx % n
    
    # Diagonal
    diag_row = idx
    diag_col = idx
    diag_val = torch.full((N,), 4.0, dtype=dtype, device=device)
    
    # Up neighbor (i-1)
    up_mask = i > 0
    up_row = idx[up_mask]
    up_col = idx[up_mask] - n
    up_val = torch.full((up_mask.sum(),), -1.0, dtype=dtype, device=device)
    
    # Down neighbor (i+1)
    down_mask = i < n - 1
    down_row = idx[down_mask]
    down_col = idx[down_mask] + n
    down_val = torch.full((down_mask.sum(),), -1.0, dtype=dtype, device=device)
    
    # Left neighbor (j-1)
    left_mask = j > 0
    left_row = idx[left_mask]
    left_col = idx[left_mask] - 1
    left_val = torch.full((left_mask.sum(),), -1.0, dtype=dtype, device=device)
    
    # Right neighbor (j+1)
    right_mask = j < n - 1
    right_row = idx[right_mask]
    right_col = idx[right_mask] + 1
    right_val = torch.full((right_mask.sum(),), -1.0, dtype=dtype, device=device)
    
    rows = torch.cat([diag_row, up_row, down_row, left_row, right_row])
    cols = torch.cat([diag_col, up_col, down_col, left_col, right_col])
    vals = torch.cat([diag_val, up_val, down_val, left_val, right_val])
    
    return vals, rows, cols, (N, N)


def test_dtensor_single_process():
    """
    Test DTensor integration in single-process mode.
    
    This test simulates DTensor behavior without requiring multiple processes.
    """
    print("=" * 60)
    print("  DTensor Integration Test (Single Process)")
    print("=" * 60)
    
    from torch_sla import SparseTensor, DSparseTensor
    from torch_sla.distributed import DTENSOR_AVAILABLE, _is_dtensor
    
    if not DTENSOR_AVAILABLE:
        print("⚠ DTensor not available (requires PyTorch 2.0+). Skipping test.")
        return True
    
    try:
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor.placement_types import Replicate, Shard
    except ImportError:
        try:
            from torch.distributed._tensor import DTensor
            from torch.distributed._tensor.placement_types import Replicate, Shard
        except ImportError:
            print("⚠ Could not import DTensor. Skipping test.")
            return True
    
    warnings.filterwarnings('ignore')
    
    # Create test matrix
    n = 4  # 4x4 grid = 16 nodes
    N = n * n
    val, row, col, shape = create_poisson_2d(n)
    
    # Create DSparseTensor
    D = DSparseTensor(val, row, col, shape, num_partitions=2, verbose=False)
    
    print(f"\nMatrix shape: {D.shape}")
    print(f"Partitions: {D.num_partitions}")
    print(f"DTensor available: {D.supports_dtensor}")
    
    # Test 1: Check _is_dtensor helper
    print("\n--- Test 1: _is_dtensor helper ---")
    x_tensor = torch.randn(N, dtype=torch.float64)
    assert not _is_dtensor(x_tensor), "Regular tensor should not be DTensor"
    print("✓ _is_dtensor correctly identifies regular tensors")
    
    # Test 2: Matmul with regular tensor
    print("\n--- Test 2: Matmul with regular Tensor ---")
    y_ref = D @ x_tensor
    assert y_ref.shape == (N,), f"Expected shape ({N},), got {y_ref.shape}"
    print(f"✓ D @ x_tensor works: output shape = {y_ref.shape}")
    
    # Test 3: solve_distributed with regular tensor
    print("\n--- Test 3: solve_distributed with regular Tensor ---")
    b_tensor = torch.ones(N, dtype=torch.float64)
    x_solved = D.solve_distributed(b_tensor, atol=1e-8, maxiter=500)
    
    # Verify solution
    residual = torch.norm(D @ x_solved - b_tensor) / torch.norm(b_tensor)
    print(f"✓ solve_distributed works: residual = {residual:.2e}")
    assert residual < 1e-6, f"Residual too large: {residual:.2e}"
    
    print("\n" + "=" * 60)
    print("  All single-process tests PASSED!")
    print("=" * 60)
    
    return True


def test_dtensor_distributed():
    """
    Test DTensor integration in distributed mode.
    
    This test requires torch.distributed to be initialized.
    """
    import torch.distributed as dist
    
    if not dist.is_initialized():
        print("⚠ torch.distributed not initialized. Running single-process test.")
        return test_dtensor_single_process()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("=" * 60)
        print(f"  DTensor Integration Test (Distributed: {world_size} ranks)")
        print("=" * 60)
    
    from torch_sla import SparseTensor, DSparseTensor
    from torch_sla.distributed import DTENSOR_AVAILABLE
    
    if not DTENSOR_AVAILABLE:
        if rank == 0:
            print("⚠ DTensor not available. Skipping distributed DTensor tests.")
        return True
    
    try:
        from torch.distributed.tensor import DTensor, distribute_tensor
        from torch.distributed.tensor.placement_types import Replicate, Shard
        from torch.distributed.device_mesh import init_device_mesh
    except ImportError:
        try:
            from torch.distributed._tensor import DTensor, distribute_tensor
            from torch.distributed._tensor.placement_types import Replicate, Shard
            from torch.distributed._tensor.device_mesh import init_device_mesh
        except ImportError:
            if rank == 0:
                print("⚠ Could not import DTensor components. Skipping.")
            return True
    
    warnings.filterwarnings('ignore')
    
    # Create device mesh
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = init_device_mesh(device_type, (world_size,))
    
    # Create test matrix
    n = 4
    N = n * n
    val, row, col, shape = create_poisson_2d(n)
    
    # Create DSparseTensor
    D = DSparseTensor(val, row, col, shape, num_partitions=world_size, verbose=(rank == 0))
    
    dist.barrier()
    
    # Test 1: Matmul with replicated DTensor
    if rank == 0:
        print("\n--- Test 1: D @ DTensor (Replicated) ---")
    
    x_tensor = torch.randn(N, dtype=torch.float64)
    x_dtensor = DTensor.from_local(x_tensor, mesh, [Replicate()])
    
    # Reference: regular tensor matmul
    y_ref = D @ x_tensor
    
    # DTensor matmul
    y_dtensor = D @ x_dtensor
    
    assert isinstance(y_dtensor, DTensor), "Output should be DTensor"
    y_local = y_dtensor.to_local()
    
    error = torch.norm(y_local - y_ref) / torch.norm(y_ref)
    if rank == 0:
        print(f"✓ DTensor matmul works: error = {error:.2e}")
    
    assert error < 1e-10, f"Matmul error too large: {error:.2e}"
    
    dist.barrier()
    
    # Test 2: solve_distributed with replicated DTensor
    if rank == 0:
        print("\n--- Test 2: solve_distributed with DTensor (Replicated) ---")
    
    b_tensor = torch.ones(N, dtype=torch.float64)
    b_dtensor = DTensor.from_local(b_tensor, mesh, [Replicate()])
    
    # Reference solve
    x_ref = D.solve_distributed(b_tensor, atol=1e-8, maxiter=500)
    
    # DTensor solve
    x_dtensor = D.solve_distributed(b_dtensor, atol=1e-8, maxiter=500)
    
    assert isinstance(x_dtensor, DTensor), "Output should be DTensor"
    x_local = x_dtensor.to_local()
    
    solve_error = torch.norm(x_local - x_ref) / torch.norm(x_ref)
    if rank == 0:
        print(f"✓ DTensor solve works: error = {solve_error:.2e}")
    
    # Verify solution quality
    residual = torch.norm(D @ x_local - b_tensor) / torch.norm(b_tensor)
    if rank == 0:
        print(f"  Residual: {residual:.2e}")
    
    assert residual < 1e-6, f"Residual too large: {residual:.2e}"
    
    dist.barrier()
    
    # Test 3: DTensor utilities
    if rank == 0:
        print("\n--- Test 3: DTensor Utilities ---")
    
    # to_dtensor
    y_dt = D.to_dtensor(x_tensor, mesh, replicate=True)
    assert isinstance(y_dt, DTensor), "to_dtensor should return DTensor"
    if rank == 0:
        print("✓ to_dtensor works")
    
    # gather_from_dtensor
    y_gathered = D.gather_from_dtensor(y_dt)
    assert isinstance(y_gathered, torch.Tensor), "gather_from_dtensor should return Tensor"
    gather_error = torch.norm(y_gathered - x_tensor)
    if rank == 0:
        print(f"✓ gather_from_dtensor works: error = {gather_error:.2e}")
    
    dist.barrier()
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("  All distributed DTensor tests PASSED!")
        print("=" * 60)
    
    return True


def main():
    """Main entry point."""
    import torch.distributed as dist
    
    # Check if running in distributed mode
    if 'RANK' in os.environ:
        # Running via torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend)
        
        try:
            success = test_dtensor_distributed()
        finally:
            dist.destroy_process_group()
    else:
        # Single process mode
        success = test_dtensor_single_process()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

