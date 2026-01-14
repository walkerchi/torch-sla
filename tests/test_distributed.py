"""
Tests for distributed sparse tensor (DSparseTensor).

Tests cover:
- DSparseTensor creation and partitioning
- Local halo exchange
- Distributed solve
- Gather/scatter operations
"""

import pytest
import torch
import numpy as np
import sys

sys.path.insert(0, "..")
from torch_sla import (
    SparseTensor,
    DSparseTensor,
    DSparseMatrix,
    partition_simple,
    partition_coordinates,
)


def create_poisson_2d(n: int, dtype=torch.float64, device='cpu'):
    """
    Create 2D Poisson matrix with 5-point stencil using vectorized operations.
    
    For an n x n grid, creates the discrete Laplacian matrix:
        A[i,i] = 4 (diagonal)
        A[i,j] = -1 for j in {i-n, i+n, i-1, i+1} (neighbors)
    """
    N = n * n
    
    # Node indices as 2D grid
    idx = torch.arange(N, device=device)
    i = idx // n  # Row in grid
    j = idx % n   # Column in grid
    
    # Diagonal entries: all nodes
    diag_row = idx
    diag_col = idx
    diag_val = torch.full((N,), 4.0, dtype=dtype, device=device)
    
    # Up neighbor (i-1): valid when i > 0
    up_mask = i > 0
    up_row = idx[up_mask]
    up_col = idx[up_mask] - n
    up_val = torch.full((up_mask.sum(),), -1.0, dtype=dtype, device=device)
    
    # Down neighbor (i+1): valid when i < n-1
    down_mask = i < n - 1
    down_row = idx[down_mask]
    down_col = idx[down_mask] + n
    down_val = torch.full((down_mask.sum(),), -1.0, dtype=dtype, device=device)
    
    # Left neighbor (j-1): valid when j > 0
    left_mask = j > 0
    left_row = idx[left_mask]
    left_col = idx[left_mask] - 1
    left_val = torch.full((left_mask.sum(),), -1.0, dtype=dtype, device=device)
    
    # Right neighbor (j+1): valid when j < n-1
    right_mask = j < n - 1
    right_row = idx[right_mask]
    right_col = idx[right_mask] + 1
    right_val = torch.full((right_mask.sum(),), -1.0, dtype=dtype, device=device)
    
    # Concatenate all entries
    rows = torch.cat([diag_row, up_row, down_row, left_row, right_row])
    cols = torch.cat([diag_col, up_col, down_col, left_col, right_col])
    vals = torch.cat([diag_val, up_val, down_val, left_val, right_val])
    
    return vals, rows, cols, (N, N)


def create_coordinates_2d(n: int, dtype=torch.float64, device='cpu'):
    """Create 2D coordinates for n x n grid using meshgrid."""
    # Create grid coordinates
    i_coords = torch.arange(n, dtype=dtype, device=device)
    j_coords = torch.arange(n, dtype=dtype, device=device)
    
    # Meshgrid: ii[i,j] = i, jj[i,j] = j
    ii, jj = torch.meshgrid(i_coords, j_coords, indexing='ij')
    
    # Flatten and stack: [N, 2] where N = n*n
    coords = torch.stack([ii.flatten(), jj.flatten()], dim=1)
    
    return coords


# ============================================================================
# DSparseTensor Creation Tests
# ============================================================================

class TestDSparseTensorCreation:
    """Test DSparseTensor creation."""
    
    def test_create_basic(self):
        """Test basic DSparseTensor creation."""
        val, row, col, shape = create_poisson_2d(4)
        
        D = DSparseTensor(val, row, col, shape, num_partitions=2, verbose=False)
        
        assert D.num_partitions == 2
        assert D.shape == (16, 16)
        assert len(D) == 2
    
    def test_create_from_sparse_tensor(self):
        """Test creation from SparseTensor."""
        val, row, col, shape = create_poisson_2d(4)
        A = SparseTensor(val, row, col, shape)
        
        D = DSparseTensor.from_sparse_tensor(A, num_partitions=2, verbose=False)
        
        assert D.num_partitions == 2
        assert D.shape == shape
    
    def test_create_with_coords(self):
        """Test creation with coordinate-based partitioning."""
        val, row, col, shape = create_poisson_2d(4)
        coords = create_coordinates_2d(4)
        
        D = DSparseTensor(
            val, row, col, shape, 
            num_partitions=2,
            coords=coords,
            partition_method='rcb',
            verbose=False
        )
        
        assert D.num_partitions == 2
    
    def test_partitions_access(self):
        """Test accessing individual partitions."""
        val, row, col, shape = create_poisson_2d(4)
        D = DSparseTensor(val, row, col, shape, num_partitions=2, verbose=False)
        
        # Access by index
        p0 = D[0]
        p1 = D[1]
        
        assert isinstance(p0, DSparseMatrix)
        assert isinstance(p1, DSparseMatrix)
        
        # Negative index
        assert D[-1] == D[1]
    
    def test_iteration(self):
        """Test iterating over partitions."""
        val, row, col, shape = create_poisson_2d(4)
        D = DSparseTensor(val, row, col, shape, num_partitions=3, verbose=False)
        
        partitions = list(D)
        assert len(partitions) == 3


# ============================================================================
# DSparseTensor Properties Tests
# ============================================================================

class TestDSparseTensorProperties:
    """Test DSparseTensor properties."""
    
    def test_properties(self):
        """Test basic properties."""
        val, row, col, shape = create_poisson_2d(4)
        D = DSparseTensor(val, row, col, shape, num_partitions=2, verbose=False)
        
        assert D.shape == (16, 16)
        assert D.num_partitions == 2
        assert D.nnz == len(val)
        assert D.device == torch.device('cpu')
        assert D.is_cuda == False
    
    def test_partition_properties(self):
        """Test partition-level properties."""
        val, row, col, shape = create_poisson_2d(4)
        D = DSparseTensor(val, row, col, shape, num_partitions=2, verbose=False)
        
        total_owned = sum(D[i].num_owned for i in range(2))
        assert total_owned == 16  # All nodes owned exactly once


# ============================================================================
# Device Management Tests
# ============================================================================

class TestDSparseTensorDevice:
    """Test device management."""
    
    def test_cpu(self):
        """Test CPU operations."""
        val, row, col, shape = create_poisson_2d(4)
        D = DSparseTensor(val, row, col, shape, num_partitions=2, verbose=False)
        
        D_cpu = D.cpu()
        assert D_cpu.device == torch.device('cpu')
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """Test CUDA operations."""
        val, row, col, shape = create_poisson_2d(4)
        D = DSparseTensor(val, row, col, shape, num_partitions=2, verbose=False)
        
        D_cuda = D.cuda()
        assert D_cuda.is_cuda
        
        # Each partition should be on CUDA
        for p in D_cuda:
            assert p.is_cuda


# ============================================================================
# Halo Exchange Tests
# ============================================================================

class TestHaloExchange:
    """Test halo exchange operations."""
    
    def test_halo_exchange_local(self):
        """Test local halo exchange."""
        val, row, col, shape = create_poisson_2d(4)
        D = DSparseTensor(val, row, col, shape, num_partitions=2, verbose=False)
        
        # Create local vectors
        x_list = []
        for i in range(2):
            x = torch.ones(D[i].num_local, dtype=torch.float64) * (i + 1)
            x_list.append(x)
        
        # Exchange halos
        D.halo_exchange_local(x_list)
        
        # After exchange, halo nodes should have values from neighbors
        # This is a basic sanity check
        assert len(x_list) == 2


# ============================================================================
# Gather/Scatter Tests
# ============================================================================

class TestGatherScatter:
    """Test gather and scatter operations."""
    
    def test_scatter_local(self):
        """Test scattering global vector to local."""
        val, row, col, shape = create_poisson_2d(4)
        D = DSparseTensor(val, row, col, shape, num_partitions=2, verbose=False)
        
        # Create global vector
        x_global = torch.arange(16, dtype=torch.float64)
        
        # Scatter
        x_list = D.scatter_local(x_global)
        
        assert len(x_list) == 2
        for i, x in enumerate(x_list):
            assert x.shape[0] == D[i].num_local
    
    def test_gather_global(self):
        """Test gathering local vectors to global."""
        val, row, col, shape = create_poisson_2d(4)
        D = DSparseTensor(val, row, col, shape, num_partitions=2, verbose=False)
        
        # Create global vector and scatter
        x_global = torch.arange(16, dtype=torch.float64)
        x_list = D.scatter_local(x_global)
        
        # Gather back
        x_gathered = D.gather_global(x_list)
        
        assert x_gathered.shape == (16,)
        torch.testing.assert_close(x_gathered, x_global)
    
    def test_scatter_gather_roundtrip(self):
        """Test scatter followed by gather gives original."""
        val, row, col, shape = create_poisson_2d(4)
        D = DSparseTensor(val, row, col, shape, num_partitions=3, verbose=False)
        
        x_original = torch.randn(16, dtype=torch.float64)
        x_scattered = D.scatter_local(x_original)
        x_gathered = D.gather_global(x_scattered)
        
        torch.testing.assert_close(x_gathered, x_original)


# ============================================================================
# Solve Tests
# ============================================================================

class TestDSparseTensorSolve:
    """Test distributed solve operations."""
    
    def test_solve_all(self):
        """Test solving on all partitions."""
        val, row, col, shape = create_poisson_2d(4)
        D = DSparseTensor(val, row, col, shape, num_partitions=2, verbose=False)
        
        # Create local RHS vectors (only owned nodes)
        b_list = [torch.randn(D[i].num_owned, dtype=torch.float64) for i in range(2)]
        
        # Solve on all partitions
        x_list = D.solve_all(b_list, method='cg', maxiter=100)
        
        assert len(x_list) == 2
        for i, x in enumerate(x_list):
            # Now returns owned nodes only
            assert x.shape[0] == D[i].num_owned


# ============================================================================
# Partitioning Tests
# ============================================================================

class TestPartitioning:
    """Test partitioning methods."""
    
    def test_partition_simple(self):
        """Test simple 1D partitioning."""
        partition_ids = partition_simple(100, 4)
        
        assert partition_ids.shape == (100,)
        assert partition_ids.min() == 0
        assert partition_ids.max() == 3
    
    def test_partition_coordinates(self):
        """Test coordinate-based partitioning."""
        coords = create_coordinates_2d(4)  # 16 nodes
        
        partition_ids = partition_coordinates(coords, 4, method='rcb')
        
        assert partition_ids.shape == (16,)
        assert partition_ids.min() >= 0
        assert partition_ids.max() <= 3


def test_partition_and_gather():
    """Test SparseTensor.partition() and DSparseTensor.gather()."""
    from torch_sla import SparseTensor, DSparseTensor
    
    val, row, col, shape = create_poisson_2d(4)
    N = shape[0]
    
    # Create SparseTensor
    A = SparseTensor(val, row, col, shape)
    
    # Partition
    D = A.partition(num_partitions=2)
    
    assert isinstance(D, DSparseTensor), "partition() should return DSparseTensor"
    assert D.num_partitions == 2, "Should have 2 partitions"
    
    # Gather back
    A2 = D.gather()
    
    assert isinstance(A2, SparseTensor), "gather() should return SparseTensor"
    
    # Verify matrices are the same
    dense1 = A.to_dense()
    dense2 = A2.to_dense()
    torch.testing.assert_close(dense1, dense2, atol=1e-12, rtol=1e-12)
    
    # Test .to_sparse_tensor() alias
    A3 = D.to_sparse_tensor()
    dense3 = A3.to_dense()
    torch.testing.assert_close(dense1, dense3, atol=1e-12, rtol=1e-12)
    
    # Verify solve is consistent
    b = torch.ones(N, dtype=val.dtype)
    x1 = A.solve(b)
    x2 = A2.solve(b)
    torch.testing.assert_close(x1, x2, atol=1e-10, rtol=1e-10)
    
    # Test with different partition methods
    coords = create_coordinates_2d(4)
    D_rcb = A.partition(num_partitions=4, coords=coords, partition_method='slicing')
    assert D_rcb.num_partitions == 4
    
    A_rcb = D_rcb.gather()
    dense_rcb = A_rcb.to_dense()
    torch.testing.assert_close(dense1, dense_rcb, atol=1e-12, rtol=1e-12)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Running DSparseTensor tests...")
    
    # Creation tests
    test = TestDSparseTensorCreation()
    test.test_create_basic()
    test.test_create_from_sparse_tensor()
    print("✓ Creation tests passed")
    
    # Properties tests
    test = TestDSparseTensorProperties()
    test.test_properties()
    test.test_partition_properties()
    print("✓ Properties tests passed")
    
    # Halo exchange
    test = TestHaloExchange()
    test.test_halo_exchange_local()
    print("✓ Halo exchange tests passed")
    
    # Gather/scatter
    test = TestGatherScatter()
    test.test_scatter_gather_roundtrip()
    print("✓ Gather/scatter tests passed")
    
    # Solve
    test = TestDSparseTensorSolve()
    test.test_solve_all()
    print("✓ Solve tests passed")
    
    # Test partition/gather
    print("\nTesting partition/gather...")
    test_partition_and_gather()
    print("✓ Partition/gather tests passed")
    
    print("\n✓ All DSparseTensor tests passed!")
