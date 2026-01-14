"""Tests for SparseTensor and DSparseTensor I/O functionality."""

import os
import sys
import tempfile
import shutil
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_sla import SparseTensor, DSparseTensor, DSparseMatrix
from torch_sla.io import (
    save_sparse, load_sparse, load_sparse_as_partition,
    save_distributed, load_partition, load_metadata, load_distributed_as_sparse,
    save_dsparse, load_dsparse,
)


def create_poisson_2d(n, dtype=torch.float64):
    """Create 2D Poisson matrix."""
    N = n * n
    idx = torch.arange(N)
    i, j = idx // n, idx % n
    entries = [
        (idx, idx, torch.full((N,), 4.0, dtype=dtype)),
        (idx[i > 0], idx[i > 0] - n, torch.full(((i > 0).sum(),), -1.0, dtype=dtype)),
        (idx[i < n-1], idx[i < n-1] + n, torch.full(((i < n-1).sum(),), -1.0, dtype=dtype)),
        (idx[j > 0], idx[j > 0] - 1, torch.full(((j > 0).sum(),), -1.0, dtype=dtype)),
        (idx[j < n-1], idx[j < n-1] + 1, torch.full(((j < n-1).sum(),), -1.0, dtype=dtype)),
    ]
    vals = torch.cat([e[2] for e in entries])
    rows = torch.cat([e[0] for e in entries])
    cols = torch.cat([e[1] for e in entries])
    return vals, rows, cols, (N, N)


class TestSparseTensorIO:
    """Test SparseTensor save/load."""
    
    def test_save_load_basic(self):
        """Test basic save and load."""
        val, row, col, shape = create_poisson_2d(4)
        A = SparseTensor(val, row, col, shape)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "matrix.safetensors")
            
            # Save
            A.save(path)
            assert os.path.exists(path)
            
            # Load
            B = SparseTensor.load(path)
            
            # Verify
            assert B.sparse_shape == A.sparse_shape
            assert B.nnz == A.nnz
            assert torch.allclose(B.values, A.values)
            assert torch.equal(B.row_indices, A.row_indices)
            assert torch.equal(B.col_indices, A.col_indices)
    
    def test_save_load_with_metadata(self):
        """Test save with custom metadata."""
        val = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        row = torch.tensor([0, 0, 1, 1])
        col = torch.tensor([0, 1, 0, 1])
        A = SparseTensor(val, row, col, (2, 2))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "matrix.safetensors")
            A.save(path, metadata={"description": "test matrix"})
            
            B = SparseTensor.load(path)
            assert B.sparse_shape == (2, 2)
            assert torch.allclose(B.values, A.values)
    
    def test_roundtrip_preserves_dtype(self):
        """Test that dtype is preserved."""
        for dtype in [torch.float32, torch.float64]:
            val = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
            row = torch.tensor([0, 1, 2])
            col = torch.tensor([0, 1, 2])
            A = SparseTensor(val, row, col, (3, 3))
            
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "matrix.safetensors")
                A.save(path)
                B = SparseTensor.load(path)
                
                assert B.dtype == A.dtype


class TestDistributedIO:
    """Test distributed save/load functionality."""
    
    def test_save_distributed_basic(self):
        """Test saving partitioned data."""
        val, row, col, shape = create_poisson_2d(4)
        A = SparseTensor(val, row, col, shape)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            A.save_distributed(tmpdir, num_partitions=2)
            
            # Check files exist
            assert os.path.exists(os.path.join(tmpdir, "metadata.json"))
            assert os.path.exists(os.path.join(tmpdir, "partition_0.safetensors"))
            assert os.path.exists(os.path.join(tmpdir, "partition_1.safetensors"))
            
            # Check metadata
            meta = load_metadata(tmpdir)
            assert meta["num_partitions"] == 2
            assert meta["shape"] == list(shape)
    
    def test_load_partition(self):
        """Test loading individual partitions."""
        val, row, col, shape = create_poisson_2d(4)
        A = SparseTensor(val, row, col, shape)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            A.save_distributed(tmpdir, num_partitions=2)
            
            # Load each partition
            p0 = load_partition(tmpdir, rank=0)
            p1 = load_partition(tmpdir, rank=1)
            
            assert isinstance(p0, DSparseMatrix)
            assert isinstance(p1, DSparseMatrix)
            assert p0.partition.partition_id == 0
            assert p1.partition.partition_id == 1
            assert p0.global_shape == shape
            assert p1.global_shape == shape
    
    def test_load_partition_classmethod(self):
        """Test loading partition via DSparseMatrix.load()."""
        val, row, col, shape = create_poisson_2d(4)
        A = SparseTensor(val, row, col, shape)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            A.save_distributed(tmpdir, num_partitions=2)
            
            p0 = DSparseMatrix.load(tmpdir, rank=0, world_size=2)
            assert p0.partition.partition_id == 0
    
    def test_distributed_coverage(self):
        """Test that all nodes are covered exactly once."""
        val, row, col, shape = create_poisson_2d(4)
        N = shape[0]
        A = SparseTensor(val, row, col, shape)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            A.save_distributed(tmpdir, num_partitions=3)
            
            # Load all partitions
            partitions = [load_partition(tmpdir, rank=i) for i in range(3)]
            
            # Check that owned nodes cover all global nodes exactly once
            all_owned = torch.cat([p.partition.owned_nodes for p in partitions])
            all_owned_sorted, _ = torch.sort(all_owned)
            expected = torch.arange(N)
            
            assert torch.equal(all_owned_sorted, expected), \
                f"Owned nodes don't cover all global nodes: {all_owned_sorted} vs {expected}"


class TestDSparseTensorIO:
    """Test DSparseTensor save/load."""
    
    def test_save_load_dsparse(self):
        """Test DSparseTensor save and load."""
        val, row, col, shape = create_poisson_2d(4)
        D = DSparseTensor(val, row, col, shape, num_partitions=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            D.save(tmpdir)
            
            # Check files
            assert os.path.exists(os.path.join(tmpdir, "metadata.json"))
            assert os.path.exists(os.path.join(tmpdir, "partition_0.safetensors"))
            assert os.path.exists(os.path.join(tmpdir, "partition_1.safetensors"))
            
            # Load
            D2 = DSparseTensor.load(tmpdir)
            
            # Verify
            assert D2.shape == D.shape
            assert D2.num_partitions == D.num_partitions
            assert D2.nnz == D.nnz
    
    def test_save_load_preserves_matvec(self):
        """Test that loaded DSparseTensor produces same matvec results."""
        val, row, col, shape = create_poisson_2d(4)
        D = DSparseTensor(val, row, col, shape, num_partitions=2)
        A = D.to_sparse_tensor()
        
        x = torch.randn(shape[1], dtype=val.dtype)
        y_ref = A @ x
        y_dist = D @ x
        
        with tempfile.TemporaryDirectory() as tmpdir:
            D.save(tmpdir)
            D2 = DSparseTensor.load(tmpdir)
            
            y_loaded = D2 @ x
            
            assert torch.allclose(y_loaded, y_ref, rtol=1e-10)


class TestRealDistributedIO:
    """Test I/O in a simulated distributed scenario."""
    
    def test_rank_loads_only_its_partition(self):
        """Simulate each rank loading only its partition."""
        val, row, col, shape = create_poisson_2d(6)
        A = SparseTensor(val, row, col, shape)
        world_size = 4
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # "Rank 0" prepares the data
            A.save_distributed(tmpdir, num_partitions=world_size)
            
            # Simulate each rank loading its own partition
            partitions = []
            for rank in range(world_size):
                p = DSparseMatrix.load(tmpdir, rank, world_size)
                partitions.append(p)
                
                # Each rank should only have its partition data
                assert p.partition.partition_id == rank
            
            # Verify consistency
            x = torch.randn(shape[1], dtype=val.dtype)
            y_ref = A @ x
            
            # Simulate distributed matvec
            y_parts = []
            for p in partitions:
                owned = p.partition.owned_nodes
                halo = p.partition.halo_nodes
                
                x_local = torch.zeros(p.num_local, dtype=val.dtype)
                x_local[:p.num_owned] = x[owned]
                if p.num_halo > 0:
                    x_local[p.num_owned:] = x[halo]
                
                y_local = p.matvec(x_local, exchange_halo=False)
                y_parts.append((owned, y_local[:p.num_owned]))
            
            # Gather
            y_result = torch.zeros_like(y_ref)
            for owned, y_owned in y_parts:
                y_result[owned] = y_owned
            
            assert torch.allclose(y_result, y_ref, rtol=1e-10)


class TestCrossFormatIO:
    """Test cross-format loading (SparseTensor <-> distributed)."""
    
    def test_sparse_to_distributed_partition(self):
        """Test: SparseTensor.save() -> load_sparse_as_partition() for each rank."""
        val, row, col, shape = create_poisson_2d(4)
        A = SparseTensor(val, row, col, shape)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "matrix.safetensors")
            A.save(path)
            
            # Simulate distributed load from single file
            world_size = 2
            partitions = []
            for rank in range(world_size):
                p = load_sparse_as_partition(path, rank, world_size)
                partitions.append(p)
            
            # Verify matvec
            x = torch.randn(shape[1], dtype=val.dtype)
            y_ref = A @ x
            
            y_parts = []
            for p in partitions:
                owned = p.partition.owned_nodes
                halo = p.partition.halo_nodes
                
                x_local = torch.zeros(p.num_local, dtype=val.dtype)
                x_local[:p.num_owned] = x[owned]
                if p.num_halo > 0:
                    x_local[p.num_owned:] = x[halo]
                
                y_local = p.matvec(x_local, exchange_halo=False)
                y_parts.append((owned, y_local[:p.num_owned]))
            
            y_result = torch.zeros_like(y_ref)
            for owned, y_owned in y_parts:
                y_result[owned] = y_owned
            
            assert torch.allclose(y_result, y_ref, rtol=1e-10)
    
    def test_distributed_to_sparse(self):
        """Test: save_distributed() -> load_distributed_as_sparse()."""
        val, row, col, shape = create_poisson_2d(4)
        A = SparseTensor(val, row, col, shape)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save as distributed
            A.save_distributed(tmpdir, num_partitions=3)
            
            # Load back as single SparseTensor
            B = load_distributed_as_sparse(tmpdir)
            
            # Verify shapes match
            assert B.sparse_shape == A.sparse_shape
            
            # Verify matvec produces same results
            x = torch.randn(shape[1], dtype=val.dtype)
            y_A = A @ x
            y_B = B @ x
            
            assert torch.allclose(y_A, y_B, rtol=1e-10)
    
    def test_dsparse_save_load_as_sparse(self):
        """Test: DSparseTensor.save() -> load_distributed_as_sparse()."""
        val, row, col, shape = create_poisson_2d(4)
        D = DSparseTensor(val, row, col, shape, num_partitions=2)
        A_ref = D.to_sparse_tensor()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            D.save(tmpdir)
            
            # Load as SparseTensor
            A = load_distributed_as_sparse(tmpdir)
            
            # Verify
            x = torch.randn(shape[1], dtype=val.dtype)
            y_ref = A_ref @ x
            y_loaded = A @ x
            
            assert torch.allclose(y_ref, y_loaded, rtol=1e-10)
    
    def test_roundtrip_sparse_distributed_sparse(self):
        """Test full roundtrip: SparseTensor -> distributed -> SparseTensor."""
        val, row, col, shape = create_poisson_2d(5)
        A = SparseTensor(val, row, col, shape)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save as distributed
            A.save_distributed(tmpdir, num_partitions=4)
            
            # Load back as SparseTensor
            B = load_distributed_as_sparse(tmpdir)
            
            # Verify solve produces same results
            b = torch.ones(shape[0], dtype=val.dtype)
            x_A = A.solve(b)
            x_B = B.solve(b)
            
            assert torch.allclose(x_A, x_B, rtol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

