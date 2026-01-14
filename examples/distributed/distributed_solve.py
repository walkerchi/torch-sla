#!/usr/bin/env python
"""
Distributed Linear Solve Example

Usage:
    torchrun --standalone --nproc_per_node=4 distributed_solve.py
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
        print("Distributed Solve: A @ x = b")
        print(f"  World size: {world_size}")
        print("=" * 60)
    
    # Problem size
    n = 100
    
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
    
    # Create local RHS (only for owned nodes)
    b_owned = torch.ones(A.num_owned, dtype=torch.float64)
    
    # Distributed CG solve
    if rank == 0:
        print("\nSolving with distributed CG...")
    
    x_owned = A.solve(b_owned, atol=1e-10, maxiter=1000, verbose=True)
    
    print(f"[Rank {rank}] ||x_owned|| = {x_owned.norm():.4f}")
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("Distributed solve completed!")
        print("=" * 60)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
