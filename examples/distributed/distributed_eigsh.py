#!/usr/bin/env python
"""
Distributed Eigenvalue Computation Example

Usage:
    torchrun --standalone --nproc_per_node=4 distributed_eigsh.py
"""

import os
import torch
import torch.distributed as dist
from torch_sla.distributed import DSparseMatrix, partition_simple


def main():
    # Initialize distributed
    dist.init_process_group(backend='gloo')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("=" * 60)
        print("Distributed Eigenvalues: A @ v = λ v")
        print(f"  World size: {world_size}")
        print("=" * 60)
    
    # Problem size
    n = 100
    k = 5
    
    # Create tridiagonal SPD matrix
    idx = torch.arange(n)
    val = torch.cat([
        torch.full((n,), 4.0, dtype=torch.float64),
        torch.full((n-1,), -1.0, dtype=torch.float64),
        torch.full((n-1,), -1.0, dtype=torch.float64)
    ])
    row = torch.cat([idx, idx[1:], idx[:-1]])
    col = torch.cat([idx, idx[:-1], idx[1:]])
    
    # Each rank creates its local partition
    A = DSparseMatrix.from_global(
        val, row, col, (n, n),
        num_partitions=world_size,
        my_partition=rank,
        partition_ids=partition_simple(n, world_size),
        verbose=(rank == 0)
    )
    
    print(f"[Rank {rank}] {A.num_owned} owned, {A.num_halo} halo nodes")
    dist.barrier()
    
    # Distributed LOBPCG
    if rank == 0:
        print(f"\nComputing {k} largest eigenvalues with distributed LOBPCG...")
    
    eigenvalues, eigenvectors_owned = A.eigsh(k=k, which="LM", maxiter=200, verbose=True)
    
    if rank == 0:
        print(f"\nEigenvalues: {[f'{v:.4f}' for v in eigenvalues.tolist()]}")
    
    print(f"[Rank {rank}] Eigenvector shape: {eigenvectors_owned.shape}")
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("Distributed eigsh completed!")
        print("=" * 60)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
