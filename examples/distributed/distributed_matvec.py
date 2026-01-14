#!/usr/bin/env python
"""
Distributed Matrix-Vector Multiplication Example

Usage:
    torchrun --standalone --nproc_per_node=4 distributed_matvec.py
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
        print("Distributed Matvec: y = A @ x")
        print(f"  World size: {world_size}")
        print("=" * 60)
    
    # Problem size
    n = 100
    
    # Create tridiagonal matrix (each rank creates its partition)
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
    
    # Create local vector x (owned part)
    x_owned = A.partition.owned_nodes.double()
    
    # Extend to local size and do matvec
    x_local = torch.zeros(A.num_local, dtype=torch.float64)
    x_local[:A.num_owned] = x_owned
    
    # Distributed matvec with automatic halo exchange
    y_local = A.matvec(x_local, exchange_halo=True)
    
    print(f"[Rank {rank}] y_owned = {y_local[:A.num_owned].tolist()[:3]}...")
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("Distributed matvec completed!")
        print("=" * 60)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
