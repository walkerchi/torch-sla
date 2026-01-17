#!/usr/bin/env python
"""
Distributed Batched GCN Training using DSparseTensorList

Demonstrates training on many graphs distributed across GPUs:
- DSparseTensorList for distributed graph collections
- Small graphs assigned whole to ranks (no edge cuts)
- Large graphs partitioned across ranks
- Efficient for molecular property prediction at scale

Run with:
    torchrun --nproc_per_node=2 examples/distributed/gcn_batched_distributed.py

Or single-process simulation:
    python examples/distributed/gcn_batched_distributed.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# Synthetic Molecule Dataset
# =============================================================================

def create_molecule_batch(
    num_molecules: int = 100,
    min_atoms: int = 5,
    max_atoms: int = 50,
    num_atom_features: int = 16,
    num_classes: int = 2,
    seed: int = 42
) -> Tuple["SparseTensorList", List[torch.Tensor], torch.Tensor]:
    """Create synthetic molecule dataset."""
    from torch_sla import SparseTensor, SparseTensorList
    
    torch.manual_seed(seed)
    
    adjacencies = []
    features_list = []
    labels = []
    
    for i in range(num_molecules):
        num_atoms = torch.randint(min_atoms, max_atoms + 1, (1,)).item()
        label = i % num_classes
        
        # Random edges
        edge_prob = 0.2 + 0.1 * label
        idx = torch.triu_indices(num_atoms, num_atoms, offset=1)
        edge_mask = torch.rand(idx.size(1)) < edge_prob
        
        src = idx[0][edge_mask]
        dst = idx[1][edge_mask]
        
        # Undirected + self-loops
        row = torch.cat([src, dst, torch.arange(num_atoms)])
        col = torch.cat([dst, src, torch.arange(num_atoms)])
        val = torch.ones(len(row))
        
        adj = SparseTensor(val, row, col, (num_atoms, num_atoms))
        features = torch.randn(num_atoms, num_atom_features)
        features[:, label] += 1.0
        
        adjacencies.append(adj)
        features_list.append(features)
        labels.append(label)
    
    return SparseTensorList(adjacencies), features_list, torch.tensor(labels)


# =============================================================================
# GCN Layer
# =============================================================================

class SimpleGCNConv(nn.Module):
    """Simple GCN convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj, deg_inv_sqrt: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h = x @ self.weight
        h = deg_inv_sqrt.unsqueeze(-1) * h
        
        if hasattr(adj, 'matvec'):
            # DSparseMatrix - need to handle multi-dimensional features
            # matvec only works for 1D, so process each column
            if h.dim() == 1:
                out = adj.matvec(h)
            else:
                # For 2D features, use matmul column by column
                out_cols = []
                for i in range(h.size(1)):
                    out_cols.append(adj.matvec(h[:, i]))
                out = torch.stack(out_cols, dim=1)
            
            num_owned = adj.num_owned
            out = out[:num_owned] * deg_inv_sqrt[:num_owned].unsqueeze(-1)
        else:
            # SparseTensor
            out = adj @ h
            out = deg_inv_sqrt.unsqueeze(-1) * out
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class GraphGCN(nn.Module):
    """GCN for graph classification."""
    
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
        self.convs.append(SimpleGCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SimpleGCNConv(hidden_channels, hidden_channels))
        self.convs.append(SimpleGCNConv(hidden_channels, hidden_channels))
        
        self.classifier = nn.Linear(hidden_channels, out_channels)
    
    def forward_single(self, x: torch.Tensor, adj, deg_inv_sqrt: torch.Tensor) -> torch.Tensor:
        """Forward on single graph, return graph embedding."""
        for conv in self.convs[:-1]:
            x = conv(x, adj, deg_inv_sqrt)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, adj, deg_inv_sqrt)
        
        # Global mean pooling
        return x.mean(dim=0)
    
    def forward_batch(
        self,
        x_list: List[torch.Tensor],
        adj_list,
        deg_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Forward on batch of graphs."""
        embeddings = []
        for x, adj, deg in zip(x_list, adj_list, deg_list):
            emb = self.forward_single(x, adj, deg)
            embeddings.append(emb)
        
        graph_embs = torch.stack(embeddings)
        return self.classifier(graph_embs)


# =============================================================================
# Distributed Training
# =============================================================================

def compute_degree_normalization(adj) -> torch.Tensor:
    """Compute D^{-1/2} for adjacency."""
    if hasattr(adj, 'matvec'):
        # DSparseMatrix
        deg = adj.matvec(
            torch.ones(adj.num_local, device=adj.device),
            exchange_halo=False
        )
    else:
        # SparseTensor
        deg = adj.sum(axis=1)
    
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    return deg_inv_sqrt


def run_single_process_demo():
    """Single-process demonstration with DSparseTensorList."""
    from torch_sla import SparseTensor, SparseTensorList, DSparseTensor
    from torch_sla.distributed import DSparseTensorList
    
    print("=" * 70)
    print("Distributed Batched GCN (Single-Process Simulation)")
    print("=" * 70)
    
    # Create dataset (small for quick demo)
    num_molecules = 10
    num_features = 8
    num_classes = 2
    num_partitions = 2
    
    print(f"\nCreating {num_molecules} molecules...")
    graphs, features, labels = create_molecule_batch(
        num_molecules=num_molecules,
        min_atoms=5,
        max_atoms=15,
        num_atom_features=num_features,
        num_classes=num_classes
    )
    
    print(f"  Graphs: {len(graphs)}")
    print(f"  Sizes: min={min(f.size(0) for f in features)}, max={max(f.size(0) for f in features)}")
    
    # Create DSparseTensorList
    print(f"\nPartitioning into {num_partitions} parts...")
    dstl = graphs.partition(
        num_partitions=num_partitions,
        threshold=20,
        verbose=True
    )
    
    print(f"\n  {dstl}")
    
    # Quick functional test - just verify matmul works
    print("\nTesting matmul on DSparseTensorList...")
    n_local = len(dstl)
    for i in range(min(3, n_local)):
        x = torch.randn(dstl[i].num_local)
        y = dstl[i].matvec(x)
        print(f"  Graph {i}: {x.shape} -> {y.shape}")
    
    print("\n--- Conversion Demo ---")
    gathered = dstl.gather()
    print(f"Gathered back: {gathered}")
    
    print("\nDemo completed!")


def run_distributed_training():
    """True distributed training with multiple GPUs."""
    import torch.distributed as dist
    from torch_sla import SparseTensor, SparseTensorList
    from torch_sla.distributed import DSparseTensorList
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("=" * 70)
        print(f"Distributed Batched GCN ({world_size} GPUs)")
        print("=" * 70)
    
    # Create dataset (same on all ranks)
    num_molecules = 100
    num_features = 16
    num_classes = 2
    
    graphs, features, labels = create_molecule_batch(
        num_molecules=num_molecules,
        min_atoms=5,
        max_atoms=30,
        num_atom_features=num_features,
        num_classes=num_classes
    )
    
    # Partition across ranks
    dstl = graphs.partition(
        num_partitions=world_size,
        threshold=50,
        verbose=(rank == 0)
    )
    
    # Each rank works on its local graphs
    n_local = len(dstl)
    
    if rank == 0:
        print(f"\n  Rank 0 has {n_local} local graphs")
    
    dist.barrier()
    
    # Prepare local data
    local_features = [features[dstl._graph_ids[i]] for i in range(n_local)]
    local_labels = labels[[dstl._graph_ids[i] for i in range(n_local)]]
    local_degs = [compute_degree_normalization(dstl[i]) for i in range(n_local)]
    
    # Train/test split (per rank)
    n_train = int(0.8 * n_local)
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, n_local))
    
    # Model
    model = GraphGCN(
        in_channels=num_features,
        hidden_channels=32,
        out_channels=num_classes
    )
    
    # Move to GPU if available
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if torch.cuda.is_available():
        local_features = [f.to(device) for f in local_features]
        local_labels = local_labels.to(device)
        local_degs = [d.to(device) for d in local_degs]
    
    # Distributed wrapper
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank] if torch.cuda.is_available() else None
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    if rank == 0:
        print("\nTraining:")
    
    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        
        if train_idx:
            train_x = [local_features[i] for i in train_idx]
            train_adj = [dstl[i] for i in train_idx]
            train_deg = [local_degs[i] for i in train_idx]
            train_y = local_labels[train_idx]
            
            out = model.module.forward_batch(train_x, train_adj, train_deg)
            loss = F.cross_entropy(out, train_y)
            loss.backward()
        
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Compute local accuracy
                local_correct = torch.tensor(0, device=device)
                local_total = torch.tensor(0, device=device)
                
                if test_idx:
                    test_x = [local_features[i] for i in test_idx]
                    test_adj = [dstl[i] for i in test_idx]
                    test_deg = [local_degs[i] for i in test_idx]
                    test_y = local_labels[test_idx]
                    
                    test_out = model.module.forward_batch(test_x, test_adj, test_deg)
                    test_pred = test_out.argmax(dim=1)
                    local_correct = (test_pred == test_y).sum()
                    local_total = torch.tensor(len(test_idx), device=device)
                
                # Aggregate across ranks
                dist.all_reduce(local_correct)
                dist.all_reduce(local_total)
                
                if rank == 0:
                    global_acc = local_correct.float() / local_total.clamp(min=1)
                    print(f"  Epoch {epoch:3d}: Global Test Acc={global_acc:.3f}")
    
    dist.barrier()
    if rank == 0:
        print("\nDistributed batched training completed!")


def main():
    """Main entry point."""
    import torch.distributed as dist
    
    if 'RANK' in os.environ:
        # Running with torchrun
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

