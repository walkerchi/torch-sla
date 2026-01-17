#!/usr/bin/env python
"""
Batched Graph Neural Network Training using SparseTensorList

Demonstrates graph-level classification on molecule-like graphs:
- SparseTensorList for handling multiple graphs with different sizes
- Block-diagonal batching for efficient message passing
- Graph-level readout and classification

This is the standard approach for molecular property prediction.

Example:
    python examples/gcn_batched.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_sla import SparseTensor, SparseTensorList


# =============================================================================
# Synthetic Molecule Dataset
# =============================================================================

def create_random_molecule(
    num_atoms: int,
    edge_prob: float = 0.3,
    num_atom_features: int = 16,
    dtype: torch.dtype = torch.float32,
    device: str = 'cpu'
) -> Tuple[SparseTensor, torch.Tensor]:
    """
    Create a random molecule-like graph.
    
    Returns:
        adj: SparseTensor adjacency matrix
        features: [num_atoms, num_atom_features] node features
    """
    # Generate random edges
    idx = torch.triu_indices(num_atoms, num_atoms, offset=1)
    edge_mask = torch.rand(idx.size(1)) < edge_prob
    
    src = idx[0][edge_mask]
    dst = idx[1][edge_mask]
    
    # Undirected edges
    row = torch.cat([src, dst, torch.arange(num_atoms)])  # Include self-loops
    col = torch.cat([dst, src, torch.arange(num_atoms)])
    val = torch.ones(len(row), dtype=dtype, device=device)
    
    adj = SparseTensor(val, row, col, (num_atoms, num_atoms))
    
    # Random node features
    features = torch.randn(num_atoms, num_atom_features, dtype=dtype, device=device)
    
    return adj, features


def create_molecule_dataset(
    num_molecules: int = 100,
    min_atoms: int = 5,
    max_atoms: int = 30,
    num_atom_features: int = 16,
    num_classes: int = 2,
    seed: int = 42
) -> Tuple[SparseTensorList, List[torch.Tensor], torch.Tensor]:
    """
    Create synthetic molecule classification dataset.
    
    Returns:
        graphs: SparseTensorList of adjacency matrices
        features: List of node feature tensors
        labels: [num_molecules] class labels
    """
    torch.manual_seed(seed)
    
    adjacencies = []
    features_list = []
    labels = []
    
    for i in range(num_molecules):
        num_atoms = torch.randint(min_atoms, max_atoms + 1, (1,)).item()
        
        # Class determines graph structure
        label = i % num_classes
        edge_prob = 0.2 + 0.1 * label  # Different connectivity per class
        
        adj, features = create_random_molecule(
            num_atoms, edge_prob=edge_prob, num_atom_features=num_atom_features
        )
        
        # Add class signal to features
        features[:, label] += 1.0
        
        adjacencies.append(adj)
        features_list.append(features)
        labels.append(label)
    
    return SparseTensorList(adjacencies), features_list, torch.tensor(labels)


# =============================================================================
# GCN Layer for Batched Graphs
# =============================================================================

class BatchedGCNConv(nn.Module):
    """
    GCN convolution that works with block-diagonal adjacency.
    
    Uses SparseTensor @ Dense for message passing.
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
    
    def forward(self, x: torch.Tensor, adj: SparseTensor) -> torch.Tensor:
        """
        Args:
            x: [total_nodes, in_channels] concatenated node features
            adj: Block-diagonal SparseTensor adjacency
        
        Returns:
            [total_nodes, out_channels] updated features
        """
        # Transform features
        h = x @ self.weight  # [total_nodes, out_channels]
        
        # Message passing via SparseTensor @ Dense
        out = adj @ h  # Block-diagonal preserves graph separation
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


def normalize_block_diagonal(adj: SparseTensor, node_counts: List[int]) -> SparseTensor:
    """
    Normalize block-diagonal adjacency: D^{-1/2} A D^{-1/2}.
    
    Args:
        adj: Block-diagonal SparseTensor
        node_counts: Number of nodes per graph (for batching info)
    
    Returns:
        Normalized SparseTensor
    """
    # Compute degree
    deg = adj.sum(axis=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    row, col = adj.row_indices, adj.col_indices
    scale = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    new_values = adj.values * scale
    
    return SparseTensor(new_values, row, col, adj.shape)


# =============================================================================
# Graph-Level Pooling
# =============================================================================

def global_mean_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """
    Mean pooling over nodes for each graph.
    
    Args:
        x: [total_nodes, features] node embeddings
        batch: [total_nodes] graph index for each node
        num_graphs: Number of graphs
    
    Returns:
        [num_graphs, features] graph embeddings
    """
    # Sum over nodes per graph
    out = torch.zeros(num_graphs, x.size(1), dtype=x.dtype, device=x.device)
    out.scatter_add_(0, batch.unsqueeze(1).expand(-1, x.size(1)), x)
    
    # Count nodes per graph
    counts = torch.zeros(num_graphs, dtype=x.dtype, device=x.device)
    counts.scatter_add_(0, batch, torch.ones(x.size(0), dtype=x.dtype, device=x.device))
    
    # Mean
    return out / counts.unsqueeze(1).clamp(min=1)


def global_add_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Sum pooling over nodes for each graph."""
    out = torch.zeros(num_graphs, x.size(1), dtype=x.dtype, device=x.device)
    out.scatter_add_(0, batch.unsqueeze(1).expand(-1, x.size(1)), x)
    return out


# =============================================================================
# Full Model
# =============================================================================

class BatchedGCN(nn.Module):
    """
    GCN for graph-level classification using SparseTensorList.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        pool: str = 'mean'
    ):
        super().__init__()
        self.dropout = dropout
        self.pool = pool
        
        self.convs = nn.ModuleList()
        self.convs.append(BatchedGCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(BatchedGCNConv(hidden_channels, hidden_channels))
        self.convs.append(BatchedGCNConv(hidden_channels, hidden_channels))
        
        self.classifier = nn.Linear(hidden_channels, out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        adj_norm: SparseTensor,
        batch: torch.Tensor,
        num_graphs: int
    ) -> torch.Tensor:
        """
        Args:
            x: [total_nodes, in_channels] concatenated features
            adj_norm: Normalized block-diagonal adjacency
            batch: [total_nodes] graph index for each node
            num_graphs: Number of graphs
        
        Returns:
            [num_graphs, out_channels] graph predictions
        """
        for conv in self.convs[:-1]:
            x = conv(x, adj_norm)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, adj_norm)
        
        # Pooling
        if self.pool == 'mean':
            x = global_mean_pool(x, batch, num_graphs)
        else:
            x = global_add_pool(x, batch, num_graphs)
        
        return self.classifier(x)


# =============================================================================
# Training
# =============================================================================

def prepare_batch(
    graphs: SparseTensorList,
    features: List[torch.Tensor],
    indices: List[int]
) -> Tuple[torch.Tensor, SparseTensor, torch.Tensor, int]:
    """
    Prepare a batch of graphs for the model.
    
    Returns:
        x: [total_nodes, features] concatenated node features
        adj: Block-diagonal normalized adjacency
        batch: [total_nodes] graph assignment
        num_graphs: Number of graphs in batch
    """
    batch_graphs = SparseTensorList([graphs[i] for i in indices])
    batch_features = [features[i] for i in indices]
    
    # Get block-diagonal adjacency
    adj = batch_graphs.to_block_diagonal()
    
    # Get node counts for batch tensor
    node_counts = [f.size(0) for f in batch_features]
    
    # Create batch tensor
    batch = torch.cat([
        torch.full((n,), i, dtype=torch.long)
        for i, n in enumerate(node_counts)
    ])
    
    # Concatenate features
    x = torch.cat(batch_features, dim=0)
    
    # Normalize adjacency
    adj_norm = normalize_block_diagonal(adj, node_counts)
    
    return x, adj_norm, batch, len(indices)


def train_epoch(
    model: nn.Module,
    graphs: SparseTensorList,
    features: List[torch.Tensor],
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 32
) -> float:
    """Train for one epoch."""
    model.train()
    
    n = len(graphs)
    perm = torch.randperm(n)
    total_loss = 0
    
    for i in range(0, n, batch_size):
        indices = perm[i:i + batch_size].tolist()
        x, adj, batch, num_graphs = prepare_batch(graphs, features, indices)
        batch_labels = labels[indices]
        
        optimizer.zero_grad()
        out = model(x, adj, batch, num_graphs)
        loss = F.cross_entropy(out, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(indices)
    
    return total_loss / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    graphs: SparseTensorList,
    features: List[torch.Tensor],
    labels: torch.Tensor,
    batch_size: int = 32
) -> float:
    """Evaluate accuracy."""
    model.eval()
    
    n = len(graphs)
    correct = 0
    
    for i in range(0, n, batch_size):
        indices = list(range(i, min(i + batch_size, n)))
        x, adj, batch, num_graphs = prepare_batch(graphs, features, indices)
        batch_labels = labels[indices]
        
        out = model(x, adj, batch, num_graphs)
        pred = out.argmax(dim=1)
        correct += (pred == batch_labels).sum().item()
    
    return correct / n


# =============================================================================
# Demo
# =============================================================================

def demo_sparse_tensor_list():
    """Demonstrate SparseTensorList functionality."""
    print("=" * 70)
    print("SparseTensorList for Batched Graphs")
    print("=" * 70)
    
    # Create sample graphs
    graphs, features, labels = create_molecule_dataset(
        num_molecules=10, min_atoms=5, max_atoms=15, num_classes=2
    )
    
    print(f"\n1. Dataset: {len(graphs)} graphs")
    print(f"   Sizes: {[f.size(0) for f in features]}")
    print(f"   Total nodes: {sum(f.size(0) for f in features)}")
    
    # SparseTensorList operations
    print(f"\n2. SparseTensorList properties:")
    print(f"   {graphs}")
    print(f"   Total nnz: {graphs.total_nnz}")
    print(f"   Block sizes: {graphs.block_sizes}")
    
    # Convert to block diagonal
    print(f"\n3. Block-diagonal conversion:")
    block_diag = graphs.to_block_diagonal()
    print(f"   Block diagonal: {block_diag}")
    
    # Matmul
    print(f"\n4. Batch matmul:")
    results = graphs @ features  # List of outputs
    print(f"   Output shapes: {[r.shape for r in results]}")
    
    # Recover from block diagonal
    print(f"\n5. Recover from block diagonal:")
    recovered = SparseTensorList.from_block_diagonal(
        block_diag, graphs.block_sizes
    )
    print(f"   Recovered: {recovered}")
    
    # Connected components
    print(f"\n6. Connected components:")
    print(f"   Block diagonal has isolated components: {block_diag.has_isolated_components()}")
    components = block_diag.to_connected_components()
    print(f"   Split into {len(components)} components")


def demo_gcn_training():
    """Train GCN on synthetic molecule dataset."""
    print("\n" + "=" * 70)
    print("GCN Training on Batched Molecule Graphs")
    print("=" * 70)
    
    # Create dataset
    num_molecules = 200
    num_features = 16
    num_classes = 2
    
    graphs, features, labels = create_molecule_dataset(
        num_molecules=num_molecules,
        min_atoms=8,
        max_atoms=25,
        num_atom_features=num_features,
        num_classes=num_classes
    )
    
    print(f"\nDataset:")
    print(f"  Molecules: {num_molecules}")
    print(f"  Classes: {num_classes}")
    print(f"  Node features: {num_features}")
    print(f"  Atom counts: min={min(f.size(0) for f in features)}, "
          f"max={max(f.size(0) for f in features)}")
    
    # Split
    n_train = int(0.8 * num_molecules)
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, num_molecules))
    
    train_graphs = SparseTensorList([graphs[i] for i in train_idx])
    train_features = [features[i] for i in train_idx]
    train_labels = labels[train_idx]
    
    test_graphs = SparseTensorList([graphs[i] for i in test_idx])
    test_features = [features[i] for i in test_idx]
    test_labels = labels[test_idx]
    
    # Model
    model = BatchedGCN(
        in_channels=num_features,
        hidden_channels=32,
        out_channels=num_classes,
        num_layers=3,
        dropout=0.3
    )
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    print("\nTraining:")
    for epoch in range(1, 51):
        loss = train_epoch(
            model, train_graphs, train_features, train_labels,
            optimizer, batch_size=32
        )
        
        if epoch % 10 == 0:
            train_acc = evaluate(model, train_graphs, train_features, train_labels)
            test_acc = evaluate(model, test_graphs, test_features, test_labels)
            print(f"  Epoch {epoch:3d}: Loss={loss:.4f}, Train={train_acc:.3f}, Test={test_acc:.3f}")
    
    # Final evaluation
    final_test_acc = evaluate(model, test_graphs, test_features, test_labels)
    print(f"\nFinal test accuracy: {final_test_acc:.4f}")


if __name__ == "__main__":
    demo_sparse_tensor_list()
    demo_gcn_training()
    
    print("\n" + "=" * 70)
    print("Batched GCN demo completed!")
    print("=" * 70)

