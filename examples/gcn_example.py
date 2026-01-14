#!/usr/bin/env python
"""
Graph Convolutional Network (GCN) Example using torch-sla SparseTensor

Demonstrates full use of SparseTensor API for graph neural networks:
- SparseTensor @ Dense for message passing
- Element-wise operations (*, +, -, clamp, abs, etc.)
- Reductions (sum, mean, max, min)
- Property detection (is_symmetric, is_positive_definite)
- solve() for graph regularization

GCN layer: H' = σ(A_norm @ H @ W + b)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_sla import SparseTensor


# =============================================================================
# Graph Construction using SparseTensor
# =============================================================================

def create_sbm_graph(
    num_nodes: int = 300,
    num_communities: int = 3,
    p_intra: float = 0.1,
    p_inter: float = 0.01,
    seed: int = 42
) -> Tuple[SparseTensor, torch.Tensor]:
    """
    Create Stochastic Block Model graph as SparseTensor.
    
    Returns:
        adj: SparseTensor adjacency matrix (with self-loops)
        labels: [num_nodes] community labels
    """
    torch.manual_seed(seed)
    
    # Assign community labels
    nodes_per_comm = num_nodes // num_communities
    labels = torch.arange(num_communities, dtype=torch.long).repeat_interleave(nodes_per_comm)
    if len(labels) < num_nodes:
        labels = torch.cat([labels, torch.full((num_nodes - len(labels),), num_communities - 1, dtype=torch.long)])
    
    # Generate edges based on community structure using vectorized operations
    # Create upper triangular indices (i < j pairs)
    idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
    i_idx, j_idx = idx[0], idx[1]
    
    # Determine probability for each edge based on community membership
    same_community = labels[i_idx] == labels[j_idx]
    probs = torch.where(same_community, p_intra, p_inter)
    
    # Sample edges
    edge_mask = torch.rand(probs.shape) < probs
    src_upper = i_idx[edge_mask]
    dst_upper = j_idx[edge_mask]
    
    # Create undirected edges (both directions)
    src_edges = torch.cat([src_upper, dst_upper])
    dst_edges = torch.cat([dst_upper, src_upper])
    
    # Add self-loops
    self_loops = torch.arange(num_nodes, dtype=torch.long)
    row = torch.cat([src_edges, self_loops])
    col = torch.cat([dst_edges, self_loops])
    val = torch.ones(row.shape[0], dtype=torch.float32)
    
    adj = SparseTensor(val, row, col, (num_nodes, num_nodes))
    return adj, labels


def normalize_adjacency(adj: SparseTensor, mode: str = 'sym') -> SparseTensor:
    """
    Normalize adjacency matrix using SparseTensor operations.
    
    Args:
        adj: SparseTensor adjacency matrix
        mode: 'sym' for D^{-1/2}AD^{-1/2}, 'row' for D^{-1}A
    
    Returns:
        Normalized SparseTensor
    """
    N = adj.sparse_shape[0]
    row, col = adj.row_indices, adj.col_indices
    
    # Compute degree using SparseTensor.sum(axis=1) -> row sums
    # But we need to handle this manually since sum over sparse dim returns dense
    deg = adj.sum(axis=1)  # [N] - sum over columns for each row
    
    if mode == 'sym':
        # D^{-1/2} A D^{-1/2}
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Scale values: val * d_i^{-1/2} * d_j^{-1/2}
        scale = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        new_values = adj.values * scale
    else:
        # D^{-1} A
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        new_values = adj.values * deg_inv[row]
    
    return SparseTensor(new_values, row, col, adj.shape)


def compute_laplacian(adj: SparseTensor, normalize: bool = True) -> SparseTensor:
    """
    Compute graph Laplacian: L = I - A_norm (normalized) or L = D - A.
    
    Uses SparseTensor arithmetic operations.
    """
    N = adj.sparse_shape[0]
    
    if normalize:
        # L = I - D^{-1/2}AD^{-1/2}
        A_norm = normalize_adjacency(adj, mode='sym')
        
        # Create identity SparseTensor
        I = SparseTensor(
            torch.ones(N, dtype=adj.dtype),
            torch.arange(N),
            torch.arange(N),
            (N, N)
        )
        
        # L = I - A_norm: combine indices
        L_row = torch.cat([I.row_indices, A_norm.row_indices])
        L_col = torch.cat([I.col_indices, A_norm.col_indices])
        L_val = torch.cat([I.values, -A_norm.values])
        
        return SparseTensor(L_val, L_row, L_col, (N, N))
    else:
        # L = D - A
        deg = adj.sum(axis=1)
        
        # Diagonal entries
        diag_row = torch.arange(N)
        diag_val = deg
        
        # Off-diagonal: -A (excluding self-loops)
        mask = adj.row_indices != adj.col_indices
        off_row = adj.row_indices[mask]
        off_col = adj.col_indices[mask]
        off_val = -adj.values[mask]
        
        L_row = torch.cat([diag_row, off_row])
        L_col = torch.cat([diag_row, off_col])
        L_val = torch.cat([diag_val, off_val])
        
        return SparseTensor(L_val, L_row, L_col, (N, N))


# =============================================================================
# GCN Layer using SparseTensor.__matmul__
# =============================================================================

class GCNConv(nn.Module):
    """
    Graph Convolution using SparseTensor @ operator.
    
    H' = A_norm @ H @ W + b
    
    Uses SparseTensor's native __matmul__ for message passing.
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
    
    def forward(self, x: torch.Tensor, adj_norm: SparseTensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_channels] node features
            adj_norm: Normalized adjacency SparseTensor
        
        Returns:
            [N, out_channels] updated features
        """
        # Transform: H @ W
        h = x @ self.weight  # [N, out_channels]
        
        # Message passing using SparseTensor @ Dense
        # SparseTensor.__matmul__ handles this natively
        out = adj_norm @ h  # Uses SparseTensor._spmv_coo internally
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class GATConv(nn.Module):
    """
    Graph Attention using SparseTensor for edge indexing.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        self.weight = nn.Parameter(torch.empty(in_channels, heads * out_channels))
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        
        out_dim = heads * out_channels if concat else out_channels
        self.bias = nn.Parameter(torch.empty(out_dim)) if bias else None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: SparseTensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_channels]
            adj: SparseTensor adjacency (edge structure)
        """
        N = x.size(0)
        H, C = self.heads, self.out_channels
        
        # Get edge indices from SparseTensor
        row, col = adj.row_indices, adj.col_indices
        
        # Linear transformation
        x = (x @ self.weight).view(N, H, C)
        
        # Attention scores
        alpha_src = (x * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        alpha = F.leaky_relu(alpha_src[row] + alpha_dst[col], 0.2)  # [E, H]
        
        # Sparse softmax
        alpha_max = torch.zeros(N, H, device=x.device, dtype=x.dtype)
        alpha_max.scatter_reduce_(0, row.unsqueeze(1).expand(-1, H), alpha, reduce='amax')
        alpha = (alpha - alpha_max[row]).exp()
        
        alpha_sum = torch.zeros(N, H, device=x.device, dtype=x.dtype)
        alpha_sum.scatter_add_(0, row.unsqueeze(1).expand(-1, H), alpha)
        alpha = alpha / (alpha_sum[row] + 1e-8)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weighted aggregation
        out = torch.zeros(N, H, C, device=x.device, dtype=x.dtype)
        msg = alpha.unsqueeze(-1) * x[col]
        out.scatter_add_(0, row.view(-1, 1, 1).expand(-1, H, C), msg)
        
        out = out.view(N, H * C) if self.concat else out.mean(dim=1)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


# =============================================================================
# Models
# =============================================================================

class GCN(nn.Module):
    """Multi-layer GCN."""
    
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
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self._cached_adj_norm = None
    
    def forward(self, x: torch.Tensor, adj: SparseTensor) -> torch.Tensor:
        # Cache normalized adjacency
        if self._cached_adj_norm is None:
            self._cached_adj_norm = normalize_adjacency(adj, mode='sym')
        
        for conv in self.convs[:-1]:
            x = conv(x, self._cached_adj_norm)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return self.convs[-1](x, self._cached_adj_norm)


class GAT(nn.Module):
    """Multi-layer GAT."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.6
    ):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
    
    def forward(self, x: torch.Tensor, adj: SparseTensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, adj)


# =============================================================================
# Training Utilities
# =============================================================================

def create_dataset(num_nodes: int = 300, num_features: int = 32, num_classes: int = 3, seed: int = 42):
    """Create synthetic node classification dataset."""
    torch.manual_seed(seed)
    
    adj, labels = create_sbm_graph(num_nodes, num_classes, seed=seed)
    
    # Features with class signal
    x = torch.randn(num_nodes, num_features)
    for c in range(num_classes):
        mask = (labels == c)
        x[mask, c * (num_features // num_classes):(c + 1) * (num_features // num_classes)] += 1.5
    
    # Split
    perm = torch.randperm(num_nodes)
    n_train, n_val = int(0.6 * num_nodes), int(0.2 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True
    
    return x, adj, labels, train_mask, val_mask, test_mask


def train(model, x, adj, y, masks, epochs=200, lr=0.01, wd=5e-4):
    """Train and evaluate model."""
    train_mask, val_mask, test_mask = masks
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    best_val, best_test = 0, 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, adj)
        F.cross_entropy(out[train_mask], y[train_mask]).backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(x, adj).argmax(dim=1)
                train_acc = (pred[train_mask] == y[train_mask]).float().mean().item()
                val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
                test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()
                
                if val_acc > best_val:
                    best_val, best_test = val_acc, test_acc
                
                print(f"Epoch {epoch:3d}: Train={train_acc:.3f} Val={val_acc:.3f} Test={test_acc:.3f}")
    
    return best_val, best_test


# =============================================================================
# Demonstrations
# =============================================================================

def demo_sparse_tensor_api():
    """Demonstrate SparseTensor API for graph learning."""
    print("=" * 70)
    print("SparseTensor API for Graph Learning")
    print("=" * 70)
    
    # Create graph as SparseTensor
    adj, labels = create_sbm_graph(100, 3)
    print(f"\n1. Graph as SparseTensor: {adj}")
    
    # Property detection
    print(f"\n2. Properties:")
    print(f"   adj.is_symmetric() = {adj.is_symmetric().item()}")
    print(f"   adj.nnz = {adj.nnz}")
    
    # Element-wise operations
    print(f"\n3. Element-wise operations:")
    adj_scaled = adj * 0.5
    print(f"   adj * 0.5 → values: [{adj_scaled.values.min():.2f}, {adj_scaled.values.max():.2f}]")
    
    adj_normalized = normalize_adjacency(adj)
    print(f"   Normalized → values: [{adj_normalized.values.min():.3f}, {adj_normalized.values.max():.3f}]")
    
    adj_clipped = adj_normalized.clamp(min=0.01, max=0.5)
    print(f"   Clipped → values: [{adj_clipped.values.min():.3f}, {adj_clipped.values.max():.3f}]")
    
    # Reductions
    print(f"\n4. Reductions:")
    print(f"   adj.sum() = {adj.sum().item():.2f}")
    print(f"   adj.mean() = {adj.mean().item():.4f}")
    print(f"   adj.max() = {adj.max().item():.2f}")
    
    # Degree computation via sum
    degrees = adj.sum(axis=1)
    print(f"   Degrees (adj.sum(axis=1)): mean={degrees.mean():.2f}, min={degrees.min():.0f}, max={degrees.max():.0f}")
    
    # SparseTensor @ Dense
    print(f"\n5. Message Passing (SparseTensor @ Dense):")
    x = torch.randn(100, 16)
    h = adj_normalized @ x
    print(f"   adj_norm @ x: {x.shape} → {h.shape}")
    
    # Multiple features (matrix)
    X = torch.randn(100, 32)
    H = adj_normalized @ X
    print(f"   adj_norm @ X: {X.shape} → {H.shape}")
    
    # Laplacian
    print(f"\n6. Graph Laplacian:")
    L = compute_laplacian(adj, normalize=True)
    print(f"   L = I - A_norm: {L}")
    
    # Graph smoothing with solve
    print(f"\n7. Graph Smoothing (SparseTensor.solve):")
    # Create regularized system: (I + αL)
    alpha = 0.1
    I = SparseTensor(
        torch.ones(100, dtype=torch.float64),
        torch.arange(100), torch.arange(100), (100, 100)
    )
    L_double = SparseTensor(L.values.double(), L.row_indices, L.col_indices, L.shape)
    
    # Combine: I + αL
    A_row = torch.cat([I.row_indices, L_double.row_indices])
    A_col = torch.cat([I.col_indices, L_double.col_indices])
    A_val = torch.cat([I.values, alpha * L_double.values])
    A = SparseTensor(A_val, A_row, A_col, (100, 100))
    
    # Solve
    b = torch.randn(100, dtype=torch.float64)
    x_smooth = A.solve(b)
    residual = (A @ x_smooth - b).norm() / b.norm()
    print(f"   Solve (I + {alpha}L)x = b: residual = {residual:.2e}")


def demo_gcn():
    """Train GCN using SparseTensor."""
    print("\n" + "=" * 70)
    print("GCN Training with SparseTensor")
    print("=" * 70)
    
    x, adj, y, train_mask, val_mask, test_mask = create_dataset(300, 32, 3)
    
    print(f"\nDataset:")
    print(f"   Adjacency: {adj}")
    print(f"   Features: {x.shape}")
    print(f"   Classes: {y.max().item() + 1}")
    
    model = GCN(x.size(1), 64, y.max().item() + 1, num_layers=2)
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} params")
    
    print("\nTraining:")
    best_val, best_test = train(model, x, adj, y, (train_mask, val_mask, test_mask))
    print(f"\nBest: Val={best_val:.4f}, Test={best_test:.4f}")


def demo_gat():
    """Train GAT using SparseTensor."""
    print("\n" + "=" * 70)
    print("GAT Training with SparseTensor")
    print("=" * 70)
    
    x, adj, y, train_mask, val_mask, test_mask = create_dataset(300, 32, 3)
    
    model = GAT(x.size(1), 8, y.max().item() + 1, heads=8)
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")
    
    print("\nTraining:")
    best_val, best_test = train(model, x, adj, y, (train_mask, val_mask, test_mask), lr=0.005)
    print(f"\nBest: Val={best_val:.4f}, Test={best_test:.4f}")


def demo_gradient_flow():
    """Verify gradients flow through SparseTensor operations."""
    print("\n" + "=" * 70)
    print("Gradient Flow Through SparseTensor")
    print("=" * 70)
    
    adj, _ = create_sbm_graph(50, 3)
    adj_norm = normalize_adjacency(adj)
    
    # Features with gradients
    x = torch.randn(50, 16, requires_grad=True)
    
    # Forward: adj @ x
    h = adj_norm @ x
    loss = h.sum()
    loss.backward()
    
    print(f"\n   Input: {x.shape}")
    print(f"   Output: {h.shape}")
    print(f"   Gradient: {x.grad.shape}, norm={x.grad.norm():.4f}")
    print("   ✓ Gradients flow correctly!")
    
    # Element-wise grad
    val = adj.values.clone().requires_grad_(True)
    adj_grad = SparseTensor(val, adj.row_indices, adj.col_indices, adj.shape)
    
    scaled = adj_grad * 2
    loss = scaled.sum()
    loss.backward()
    
    print(f"\n   SparseTensor.values gradient: {val.grad.shape}, sum={val.grad.sum():.1f}")
    print("   ✓ Element-wise gradients work!")


if __name__ == "__main__":
    demo_sparse_tensor_api()
    demo_gradient_flow()
    demo_gcn()
    demo_gat()
    
    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)
