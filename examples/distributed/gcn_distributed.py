#!/usr/bin/env python
"""
Distributed GCN Training using DSparseTensor

Demonstrates training a GCN on a single large graph partitioned across GPUs:
- DSparseTensor for distributed sparse matrix storage
- Halo exchange for neighbor feature aggregation
- Distributed gradient aggregation

This is for large-scale graph learning (social networks, web graphs, etc.)

Run with:
    torchrun --nproc_per_node=2 examples/distributed/gcn_distributed.py

Or single-process simulation:
    python examples/distributed/gcn_distributed.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# Large Graph Creation
# =============================================================================

def create_large_sbm_graph(
    num_nodes: int = 1000,
    num_communities: int = 4,
    p_intra: float = 0.05,
    p_inter: float = 0.001,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int], torch.Tensor]:
    """
    Create a large Stochastic Block Model graph.
    
    Returns:
        values, row_indices, col_indices, shape, labels
    """
    torch.manual_seed(seed)
    
    # Assign communities
    nodes_per_comm = num_nodes // num_communities
    labels = torch.arange(num_communities).repeat_interleave(nodes_per_comm)
    if len(labels) < num_nodes:
        labels = torch.cat([
            labels,
            torch.full((num_nodes - len(labels),), num_communities - 1)
        ])
    
    # Generate edges efficiently
    edges_src = []
    edges_dst = []
    
    # Sample edges in blocks to manage memory
    block_size = 1000
    for i in range(0, num_nodes, block_size):
        i_end = min(i + block_size, num_nodes)
        for j in range(i, num_nodes, block_size):
            j_end = min(j + block_size, num_nodes)
            
            # Generate random edges for this block
            i_idx = torch.arange(i, i_end).unsqueeze(1).expand(-1, j_end - j)
            j_idx = torch.arange(j, j_end).unsqueeze(0).expand(i_end - i, -1)
            
            # Only upper triangular (i < j)
            mask = i_idx < j_idx
            
            if mask.any():
                i_flat = i_idx[mask]
                j_flat = j_idx[mask]
                
                # Edge probability based on community
                same_comm = labels[i_flat] == labels[j_flat]
                probs = torch.where(same_comm, p_intra, p_inter)
                
                edge_mask = torch.rand_like(probs) < probs
                
                if edge_mask.any():
                    edges_src.append(i_flat[edge_mask])
                    edges_dst.append(j_flat[edge_mask])
    
    if edges_src:
        src = torch.cat(edges_src)
        dst = torch.cat(edges_dst)
    else:
        src = torch.tensor([], dtype=torch.long)
        dst = torch.tensor([], dtype=torch.long)
    
    # Undirected + self-loops
    row = torch.cat([src, dst, torch.arange(num_nodes)])
    col = torch.cat([dst, src, torch.arange(num_nodes)])
    val = torch.ones(len(row), dtype=torch.float32)
    
    return val, row, col, (num_nodes, num_nodes), labels


# =============================================================================
# Distributed GCN Layer
# =============================================================================

class DistributedGCNConv(nn.Module):
    """
    GCN layer for distributed training with DSparseTensor.
    
    Each rank holds a partition of the graph and uses halo exchange
    for neighbor aggregation at partition boundaries.
    """
    
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        adj_partition,  # DSparseMatrix
        deg_inv_sqrt: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with distributed adjacency.
        
        Args:
            x: [num_local, in_channels] local node features
            adj_partition: DSparseMatrix local partition
            deg_inv_sqrt: [num_local] precomputed D^{-1/2}
        
        Returns:
            [num_owned, out_channels] output for owned nodes only
        """
        # Transform features
        h = x @ self.weight  # [num_local, out_channels]
        
        # Scale by degree
        h = deg_inv_sqrt.unsqueeze(1) * h
        
        # Message passing with halo exchange
        out = adj_partition.matvec(h)  # Includes halo exchange
        
        # Scale output by degree (only owned nodes)
        num_owned = adj_partition.num_owned
        out = out[:num_owned] * deg_inv_sqrt[:num_owned].unsqueeze(1)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class DistributedGCN(nn.Module):
    """
    Multi-layer GCN for distributed training.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(DistributedGCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(DistributedGCNConv(hidden_channels, hidden_channels))
        self.convs.append(DistributedGCNConv(hidden_channels, out_channels))
    
    def forward(
        self,
        x: torch.Tensor,
        adj_partition,
        deg_inv_sqrt: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [num_local, in_channels] local features (owned + halo)
            adj_partition: DSparseMatrix
            deg_inv_sqrt: [num_local] degree normalization
        
        Returns:
            [num_owned, out_channels] predictions for owned nodes
        """
        num_owned = adj_partition.num_owned
        
        for i, conv in enumerate(self.convs[:-1]):
            x_out = conv(x, adj_partition, deg_inv_sqrt)
            x_out = F.relu(x_out)
            x_out = F.dropout(x_out, p=self.dropout, training=self.training)
            
            # Extend back to full local size for next layer
            x = torch.zeros(x.size(0), x_out.size(1), dtype=x.dtype, device=x.device)
            x[:num_owned] = x_out
            # Halo values will be filled by next matvec
        
        return self.convs[-1](x, adj_partition, deg_inv_sqrt)


# =============================================================================
# Training Utilities
# =============================================================================

def compute_local_degree(adj_partition) -> torch.Tensor:
    """Compute D^{-1/2} for local nodes."""
    # Sum rows for degree
    deg = adj_partition.matvec(
        torch.ones(adj_partition.num_local, device=adj_partition.device),
        exchange_halo=False
    )
    
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    return deg_inv_sqrt


def run_single_process_demo():
    """
    Demonstrate distributed GCN in single-process mode.
    
    Creates a partitioned graph and simulates distributed training.
    """
    from torch_sla import SparseTensor, DSparseTensor
    
    print("=" * 70)
    print("Distributed GCN Training (Single-Process Simulation)")
    print("=" * 70)
    
    # Create large graph
    num_nodes = 500
    num_classes = 4
    num_features = 32
    num_partitions = 2
    
    print(f"\nCreating graph with {num_nodes} nodes...")
    val, row, col, shape, labels = create_large_sbm_graph(
        num_nodes=num_nodes,
        num_communities=num_classes,
        p_intra=0.1,
        p_inter=0.01
    )
    
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {len(row)}")
    print(f"  Classes: {num_classes}")
    
    # Create features
    x = torch.randn(num_nodes, num_features)
    for c in range(num_classes):
        mask = (labels == c)
        x[mask, c * (num_features // num_classes):(c + 1) * (num_features // num_classes)] += 1.5
    
    # Create DSparseTensor
    print(f"\nPartitioning into {num_partitions} parts...")
    D = DSparseTensor(val, row, col, shape, num_partitions=num_partitions, verbose=True)
    
    # Train/test split
    perm = torch.randperm(num_nodes)
    n_train = int(0.6 * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    test_mask[perm[n_train:]] = True
    
    # Model
    model = DistributedGCN(
        in_channels=num_features,
        hidden_channels=64,
        out_channels=num_classes,
        num_layers=2
    )
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    
    # For single-process simulation, we use partition 0
    partition = D[0]
    owned_nodes = partition.partition.owned_nodes
    local_nodes = partition.partition.local_nodes
    
    # Prepare local data
    x_local = x[local_nodes]
    y_local = labels[owned_nodes]
    train_mask_local = train_mask[owned_nodes]
    test_mask_local = test_mask[owned_nodes]
    
    # Compute degree normalization
    deg_inv_sqrt = compute_local_degree(partition)
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    print("\nTraining (simulated on partition 0):")
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        
        out = model(x_local, partition, deg_inv_sqrt)
        loss = F.cross_entropy(out[train_mask_local], y_local[train_mask_local])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                out = model(x_local, partition, deg_inv_sqrt)
                pred = out.argmax(dim=1)
                
                train_acc = (pred[train_mask_local] == y_local[train_mask_local]).float().mean()
                test_acc = (pred[test_mask_local] == y_local[test_mask_local]).float().mean()
                
                print(f"  Epoch {epoch:3d}: Loss={loss:.4f}, Train={train_acc:.3f}, Test={test_acc:.3f}")
    
    print("\nNote: This is a single-partition simulation.")
    print("For true distributed training, run with torchrun.")


def run_distributed_training():
    """
    True distributed training with torch.distributed.
    
    Each rank holds one partition and communicates via halo exchange.
    """
    import torch.distributed as dist
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    from torch_sla import SparseTensor, DSparseTensor
    
    if rank == 0:
        print("=" * 70)
        print(f"Distributed GCN Training ({world_size} GPUs)")
        print("=" * 70)
    
    # Create graph (same on all ranks for simplicity)
    num_nodes = 500
    num_classes = 4
    num_features = 32
    
    val, row, col, shape, labels = create_large_sbm_graph(
        num_nodes=num_nodes,
        num_communities=num_classes
    )
    
    # Create features
    torch.manual_seed(42)
    x = torch.randn(num_nodes, num_features)
    for c in range(num_classes):
        mask = (labels == c)
        x[mask, c * (num_features // num_classes):(c + 1) * (num_features // num_classes)] += 1.5
    
    # Each rank creates its partition
    partition = DSparseTensor.from_global_distributed(
        val, row, col, shape,
        rank=rank, world_size=world_size,
        partition_method='simple'
    )
    
    if rank == 0:
        print(f"  Partition 0: {partition.num_owned} owned, {partition.num_halo} halo")
    
    dist.barrier()
    
    # Local data
    owned_nodes = partition.partition.owned_nodes
    local_nodes = partition.partition.local_nodes
    x_local = x[local_nodes]
    y_local = labels[owned_nodes]
    
    # Train/test split
    perm = torch.randperm(num_nodes)
    n_train = int(0.6 * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    test_mask[perm[n_train:]] = True
    
    train_mask_local = train_mask[owned_nodes]
    test_mask_local = test_mask[owned_nodes]
    
    # Model
    model = DistributedGCN(
        in_channels=num_features,
        hidden_channels=64,
        out_channels=num_classes
    )
    
    # Wrap for distributed
    if torch.cuda.is_available():
        model = model.cuda(rank)
        x_local = x_local.cuda(rank)
        y_local = y_local.cuda(rank)
        train_mask_local = train_mask_local.cuda(rank)
        test_mask_local = test_mask_local.cuda(rank)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    # Degree normalization
    deg_inv_sqrt = compute_local_degree(partition)
    if torch.cuda.is_available():
        deg_inv_sqrt = deg_inv_sqrt.cuda(rank)
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    if rank == 0:
        print("\nTraining:")
    
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        
        out = model.module(x_local, partition, deg_inv_sqrt)
        loss = F.cross_entropy(out[train_mask_local], y_local[train_mask_local])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                out = model.module(x_local, partition, deg_inv_sqrt)
                pred = out.argmax(dim=1)
                
                train_correct = (pred[train_mask_local] == y_local[train_mask_local]).sum()
                test_correct = (pred[test_mask_local] == y_local[test_mask_local]).sum()
                train_total = train_mask_local.sum()
                test_total = test_mask_local.sum()
                
                # Gather across ranks
                dist.all_reduce(train_correct)
                dist.all_reduce(test_correct)
                dist.all_reduce(train_total)
                dist.all_reduce(test_total)
                
                if rank == 0:
                    train_acc = train_correct.float() / train_total
                    test_acc = test_correct.float() / test_total
                    print(f"  Epoch {epoch:3d}: Loss={loss:.4f}, Train={train_acc:.3f}, Test={test_acc:.3f}")
    
    dist.barrier()
    if rank == 0:
        print("\nDistributed training completed!")


def main():
    """Main entry point."""
    import torch.distributed as dist
    
    if 'RANK' in os.environ:
        # Running with torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend)
        
        try:
            run_distributed_training()
        finally:
            dist.destroy_process_group()
    else:
        # Single process mode
        run_single_process_demo()


if __name__ == "__main__":
    main()










