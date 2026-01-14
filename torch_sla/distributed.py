"""
Distributed Sparse Matrix for large-scale CFD/FEM computations.

Provides domain decomposition with halo exchange, following the standard
approach used in Ansys, OpenFOAM, and other industrial CFD/FEM solvers.

Key Features:
- Graph-based partitioning (METIS or simple geometric methods)
- Halo/ghost node exchange for parallel computations
- Support for both CPU and CUDA devices
- Same API as SparseTensor for easy migration

Example
-------
>>> from torch_sla import DSparseMatrix
>>> 
>>> # Create from global matrix
>>> A_global = SparseTensor(val, row, col, shape)
>>> A_dist = DSparseMatrix.from_global(A_global, num_partitions=4)
>>> 
>>> # Distributed solve
>>> x_dist = A_dist.solve(b_dist)
>>> 
>>> # Halo exchange for iterative methods
>>> A_dist.halo_exchange(local_x)
"""

import os
import torch
from typing import Tuple, List, Dict, Optional, Union, Literal
from dataclasses import dataclass
import warnings

from .backends import (
    is_scipy_available,
    is_eigen_available,
    is_cusolver_available,
    is_cudss_available,
    select_backend,
    select_method,
    BackendType,
    MethodType,
)

try:
    import torch.distributed as dist
    DIST_AVAILABLE = True
except ImportError:
    DIST_AVAILABLE = False


@dataclass
class Partition:
    """Represents a single partition/subdomain"""
    partition_id: int
    local_nodes: torch.Tensor      # Global indices of local nodes
    owned_nodes: torch.Tensor      # Nodes owned by this partition (not halo)
    halo_nodes: torch.Tensor       # Ghost/halo nodes from neighbors
    neighbor_partitions: List[int] # Neighboring partition IDs
    send_indices: Dict[int, torch.Tensor]  # Nodes to send to each neighbor
    recv_indices: Dict[int, torch.Tensor]  # Where to place received data
    global_to_local: torch.Tensor  # Mapping from global to local indices
    local_to_global: torch.Tensor  # Mapping from local to global indices
    

def partition_graph_metis(
    row: torch.Tensor,
    col: torch.Tensor,
    num_nodes: int,
    num_parts: int
) -> torch.Tensor:
    """
    Partition graph using METIS (if available) or fallback to simple method.
    
    Returns
    -------
    partition_ids : torch.Tensor
        Partition ID for each node [num_nodes]
    """
    try:
        import pymetis
        # Build adjacency list
        adjacency = [[] for _ in range(num_nodes)]
        row_cpu = row.cpu().numpy()
        col_cpu = col.cpu().numpy()
        
        for r, c in zip(row_cpu, col_cpu):
            if r != c:  # Skip diagonal
                adjacency[r].append(c)
        
        # Run METIS
        _, membership = pymetis.part_graph(num_parts, adjacency=adjacency)
        return torch.tensor(membership, dtype=torch.int64)
    
    except ImportError:
        warnings.warn("pymetis not available, using simple geometric partitioning")
        return partition_simple(num_nodes, num_parts)


def partition_simple(num_nodes: int, num_parts: int) -> torch.Tensor:
    """Simple 1D partitioning (fallback when METIS not available)"""
    nodes_per_part = (num_nodes + num_parts - 1) // num_parts
    partition_ids = torch.zeros(num_nodes, dtype=torch.int64)
    
    for i in range(num_nodes):
        partition_ids[i] = min(i // nodes_per_part, num_parts - 1)
    
    return partition_ids


def partition_coordinates(
    coords: torch.Tensor,
    num_parts: int,
    method: str = 'rcb'
) -> torch.Tensor:
    """
    Partition based on node coordinates using Recursive Coordinate Bisection (RCB).
    
    This is common in CFD/FEM for mesh partitioning.
    
    Parameters
    ----------
    coords : torch.Tensor
        Node coordinates [num_nodes, dim]
    num_parts : int
        Number of partitions (should be power of 2 for RCB)
    method : str
        'rcb': Recursive Coordinate Bisection
        'slicing': Simple slicing along longest axis
        
    Returns
    -------
    partition_ids : torch.Tensor
        Partition ID for each node
    """
    num_nodes = coords.size(0)
    partition_ids = torch.zeros(num_nodes, dtype=torch.int64)
    
    if method == 'rcb':
        _rcb_partition(coords, partition_ids, torch.arange(num_nodes), 0, num_parts)
    else:  # slicing
        # Find longest axis
        ranges = coords.max(0).values - coords.min(0).values
        axis = ranges.argmax().item()
        
        # Sort by that axis
        sorted_idx = coords[:, axis].argsort()
        nodes_per_part = (num_nodes + num_parts - 1) // num_parts
        
        for i, idx in enumerate(sorted_idx):
            partition_ids[idx] = min(i // nodes_per_part, num_parts - 1)
    
    return partition_ids


def _rcb_partition(
    coords: torch.Tensor,
    partition_ids: torch.Tensor,
    node_indices: torch.Tensor,
    part_offset: int,
    num_parts: int
):
    """Recursive Coordinate Bisection helper"""
    if num_parts == 1 or len(node_indices) == 0:
        partition_ids[node_indices] = part_offset
        return
    
    # Find longest axis
    local_coords = coords[node_indices]
    ranges = local_coords.max(0).values - local_coords.min(0).values
    axis = ranges.argmax().item()
    
    # Find median
    axis_vals = local_coords[:, axis]
    median = axis_vals.median()
    
    # Split
    left_mask = axis_vals <= median
    right_mask = ~left_mask
    
    left_nodes = node_indices[left_mask]
    right_nodes = node_indices[right_mask]
    
    # Handle uneven splits
    left_parts = num_parts // 2
    right_parts = num_parts - left_parts
    
    _rcb_partition(coords, partition_ids, left_nodes, part_offset, left_parts)
    _rcb_partition(coords, partition_ids, right_nodes, part_offset + left_parts, right_parts)


def find_halo_nodes(
    row: torch.Tensor,
    col: torch.Tensor,
    partition_ids: torch.Tensor,
    partition_id: int
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Find halo/ghost nodes for a partition.
    
    Halo nodes are nodes owned by other partitions but connected to this partition's nodes.
    
    Returns
    -------
    halo_nodes : torch.Tensor
        Global indices of halo nodes
    send_map : Dict[int, torch.Tensor]
        For each neighbor, which of our owned nodes to send
    """
    owned_mask = partition_ids == partition_id
    owned_nodes = owned_mask.nonzero().squeeze(-1)
    owned_set = set(owned_nodes.tolist())
    
    # Find edges crossing partition boundary
    halo_set = set()
    neighbor_nodes = {}  # neighbor_id -> set of nodes to send
    
    row_cpu = row.cpu()
    col_cpu = col.cpu()
    
    for r, c in zip(row_cpu.tolist(), col_cpu.tolist()):
        r_owned = r in owned_set
        c_owned = c in owned_set
        
        if r_owned and not c_owned:
            # c is a halo node
            halo_set.add(c)
            neighbor_id = partition_ids[c].item()
            if neighbor_id not in neighbor_nodes:
                neighbor_nodes[neighbor_id] = set()
            neighbor_nodes[neighbor_id].add(r)  # We need to send r to this neighbor
        
        if c_owned and not r_owned:
            # r is a halo node
            halo_set.add(r)
            neighbor_id = partition_ids[r].item()
            if neighbor_id not in neighbor_nodes:
                neighbor_nodes[neighbor_id] = set()
            neighbor_nodes[neighbor_id].add(c)
    
    halo_nodes = torch.tensor(sorted(halo_set), dtype=torch.int64)
    send_map = {k: torch.tensor(sorted(v), dtype=torch.int64) for k, v in neighbor_nodes.items()}
    
    return halo_nodes, send_map


class DSparseMatrix:
    """
    Distributed Sparse Matrix with halo exchange support.
    
    Designed for large-scale CFD/FEM computations following industrial
    practices from Ansys, OpenFOAM, etc.
    
    The matrix is partitioned across multiple processes/GPUs, with automatic
    halo (ghost) node management for parallel iterative solvers.
    
    Supports both CPU and CUDA devices.
    
    Attributes
    ----------
    partition : Partition
        Local partition information
    local_values : torch.Tensor
        Non-zero values for local portion of matrix
    local_row : torch.Tensor
        Local row indices
    local_col : torch.Tensor
        Local column indices
    local_shape : Tuple[int, int]
        Shape of local matrix (including halo)
    global_shape : Tuple[int, int]
        Shape of global matrix
    device : torch.device
        Device where the matrix data resides (cpu or cuda)
    
    Example
    -------
    >>> # Create distributed matrix on CPU
    >>> A = DSparseMatrix.from_global(val, row, col, shape, num_parts=4, my_part=0, device='cpu')
    >>> 
    >>> # Create distributed matrix on CUDA
    >>> A_cuda = DSparseMatrix.from_global(val, row, col, shape, num_parts=4, my_part=0, device='cuda')
    >>> 
    >>> # Distributed matrix-vector product with halo exchange
    >>> y = A.matvec(x)  # Automatically handles halo exchange
    >>> 
    >>> # Explicit halo exchange
    >>> A.halo_exchange(x)  # Update halo values in x
    """
    
    def __init__(
        self,
        partition: Partition,
        local_values: torch.Tensor,
        local_row: torch.Tensor,
        local_col: torch.Tensor,
        local_shape: Tuple[int, int],
        global_shape: Tuple[int, int],
        num_partitions: int,
        device: Union[str, torch.device] = 'cpu',
        verbose: bool = True
    ):
        # Convert device to torch.device
        if isinstance(device, str):
            device = torch.device(device)
        
        self.partition = partition
        self.local_values = local_values.to(device)
        self.local_row = local_row.to(device)
        self.local_col = local_col.to(device)
        self.local_shape = local_shape
        self.global_shape = global_shape
        self.num_partitions = num_partitions
        self.device = device
        self._verbose = verbose
        
        # Move partition tensors to device
        self._partition_to_device()
        
        # For display
        if verbose:
            self._print_partition_info()
    
    def _partition_to_device(self):
        """Move partition tensors to the target device"""
        # Note: We keep some partition info on CPU for indexing
        # Only move what's needed for computation
        pass
    
    def _print_partition_info(self):
        """Print partition info for user awareness"""
        owned = len(self.partition.owned_nodes)
        halo = len(self.partition.halo_nodes)
        total = self.local_shape[0]
        neighbors = len(self.partition.neighbor_partitions)
        
        print(f"[Partition {self.partition.partition_id}/{self.num_partitions}] "
              f"Nodes: {owned} owned + {halo} halo = {total} local | "
              f"Neighbors: {neighbors} | "
              f"Global: {self.global_shape[0]}x{self.global_shape[1]} | "
              f"Device: {self.device}")
    
    def to(self, device: Union[str, torch.device]) -> "DSparseMatrix":
        """
        Move the distributed matrix to a different device.
        
        Parameters
        ----------
        device : str or torch.device
            Target device ('cpu', 'cuda', 'cuda:0', etc.)
            
        Returns
        -------
        DSparseMatrix
            New distributed matrix on the target device
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        return DSparseMatrix(
            partition=self.partition,
            local_values=self.local_values.to(device),
            local_row=self.local_row.to(device),
            local_col=self.local_col.to(device),
            local_shape=self.local_shape,
            global_shape=self.global_shape,
            num_partitions=self.num_partitions,
            device=device,
            verbose=False  # Don't print again when moving
        )
    
    def cuda(self, device: Optional[int] = None) -> "DSparseMatrix":
        """Move to CUDA device"""
        if device is not None:
            return self.to(f'cuda:{device}')
        return self.to('cuda')
    
    def cpu(self) -> "DSparseMatrix":
        """Move to CPU"""
        return self.to('cpu')
    
    @property
    def is_cuda(self) -> bool:
        """Check if matrix is on CUDA"""
        return self.device.type == 'cuda'
    
    @classmethod
    def from_global(
        cls,
        values: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
        shape: Tuple[int, int],
        num_partitions: int,
        my_partition: int,
        partition_ids: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        device: Union[str, torch.device] = 'cpu',
        verbose: bool = True
    ) -> "DSparseMatrix":
        """
        Create distributed matrix from global COO data.
        
        Parameters
        ----------
        values, row, col : torch.Tensor
            Global COO sparse matrix data
        shape : Tuple[int, int]
            Global matrix shape
        num_partitions : int
            Number of partitions
        my_partition : int
            This process's partition ID (0 to num_partitions-1)
        partition_ids : torch.Tensor, optional
            Pre-computed partition assignments. If None, computed automatically.
        coords : torch.Tensor, optional
            Node coordinates for geometric partitioning [num_nodes, dim]
        device : str or torch.device
            Device for local data ('cpu', 'cuda', 'cuda:0', etc.)
        verbose : bool
            Whether to print partition info
            
        Returns
        -------
        DSparseMatrix
            Local portion of the distributed matrix
        """
        num_nodes = shape[0]
        
        # Compute partitioning if not provided
        if partition_ids is None:
            if coords is not None:
                partition_ids = partition_coordinates(coords, num_partitions)
            else:
                partition_ids = partition_graph_metis(row, col, num_nodes, num_partitions)
        
        # Find owned and halo nodes
        owned_mask = partition_ids == my_partition
        owned_nodes = owned_mask.nonzero().squeeze(-1)
        halo_nodes, send_map = find_halo_nodes(row, col, partition_ids, my_partition)
        
        # All local nodes (owned + halo)
        local_nodes = torch.cat([owned_nodes, halo_nodes])
        num_local = len(local_nodes)
        
        # Build global-to-local mapping
        global_to_local = torch.full((num_nodes,), -1, dtype=torch.int64)
        for local_idx, global_idx in enumerate(local_nodes):
            global_to_local[global_idx] = local_idx
        
        # Extract local matrix entries
        row_cpu = row.cpu()
        col_cpu = col.cpu()
        val_cpu = values.cpu()
        
        local_rows = []
        local_cols = []
        local_vals = []
        
        for r, c, v in zip(row_cpu.tolist(), col_cpu.tolist(), val_cpu.tolist()):
            local_r = global_to_local[r].item()
            local_c = global_to_local[c].item()
            
            if local_r >= 0 and local_c >= 0:
                local_rows.append(local_r)
                local_cols.append(local_c)
                local_vals.append(v)
        
        local_row = torch.tensor(local_rows, dtype=torch.int64)
        local_col = torch.tensor(local_cols, dtype=torch.int64)
        local_values = torch.tensor(local_vals, dtype=values.dtype)
        
        # Build recv_indices (where to place received halo data)
        recv_indices = {}
        halo_offset = len(owned_nodes)
        halo_list = halo_nodes.tolist()
        
        for neighbor_id in send_map.keys():
            neighbor_owned = (partition_ids == neighbor_id).nonzero().squeeze(-1).tolist()
            recv_idx = []
            for node in neighbor_owned:
                if node in halo_list:
                    recv_idx.append(halo_offset + halo_list.index(node))
            recv_indices[neighbor_id] = torch.tensor(recv_idx, dtype=torch.int64)
        
        # Convert send_map from global node IDs to local indices
        # send_map currently contains global node IDs, but halo_exchange needs local indices
        send_indices_local = {}
        for neighbor_id, global_nodes in send_map.items():
            local_idx = global_to_local[global_nodes]
            send_indices_local[neighbor_id] = local_idx
        
        partition = Partition(
            partition_id=my_partition,
            local_nodes=local_nodes,
            owned_nodes=owned_nodes,
            halo_nodes=halo_nodes,
            neighbor_partitions=list(send_map.keys()),
            send_indices=send_indices_local,  # Use local indices instead of global
            recv_indices=recv_indices,
            global_to_local=global_to_local,
            local_to_global=local_nodes.clone()
        )
        
        return cls(
            partition=partition,
            local_values=local_values,
            local_row=local_row,
            local_col=local_col,
            local_shape=(num_local, num_local),
            global_shape=shape,
            num_partitions=num_partitions,
            device=device,
            verbose=verbose
        )
    
    @property
    def num_owned(self) -> int:
        """Number of owned (non-halo) nodes"""
        return len(self.partition.owned_nodes)
    
    @property
    def num_halo(self) -> int:
        """Number of halo/ghost nodes"""
        return len(self.partition.halo_nodes)
    
    @property
    def num_local(self) -> int:
        """Total local nodes (owned + halo)"""
        return self.local_shape[0]
    
    @property
    def nnz(self) -> int:
        """Number of non-zeros in local matrix"""
        return len(self.local_values)
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of matrix values"""
        return self.local_values.dtype
    
    def halo_exchange(
        self,
        x: torch.Tensor,
        async_op: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Exchange halo/ghost values with neighbors.
        
        This is the core operation for parallel iterative methods.
        Updates the halo portion of x with values from neighboring partitions.
        
        Parameters
        ----------
        x : torch.Tensor
            Local vector [num_local] with owned values filled in.
            Halo values will be updated.
        async_op : bool
            If True, return immediately and return a future.
            
        Returns
        -------
        x : torch.Tensor
            Vector with updated halo values (same tensor, modified in-place)
            
        Example
        -------
        >>> # During iterative solve
        >>> for iteration in range(max_iter):
        >>>     # Compute local update
        >>>     x_new = local_gauss_seidel_step(A_local, x, b)
        >>>     
        >>>     # Exchange boundary values
        >>>     A.halo_exchange(x_new)
        >>>     
        >>>     # Check convergence using owned nodes only
        >>>     residual = compute_residual(A_local, x_new, b)
        """
        if not DIST_AVAILABLE or not dist.is_initialized():
            # Single-process fallback: just return (no exchange needed)
            return x
        
        # Prepare send buffers
        send_buffers = {}
        recv_buffers = {}
        
        for neighbor_id in self.partition.neighbor_partitions:
            send_idx = self.partition.send_indices[neighbor_id].to(self.device)
            send_buffers[neighbor_id] = x[send_idx].contiguous()
            
            recv_idx = self.partition.recv_indices[neighbor_id]
            recv_buffers[neighbor_id] = torch.empty(len(recv_idx), dtype=x.dtype, device=self.device)
        
        # Use send/recv for p2p communication
        # Note: For NCCL, we use synchronous send/recv
        backend = dist.get_backend() if dist.is_initialized() else 'gloo'
        
        if backend == 'nccl':
            # NCCL: use synchronous send/recv pairs
            for neighbor_id in sorted(self.partition.neighbor_partitions):
                if self.partition.partition_id < neighbor_id:
                    # Lower rank sends first, then receives
                    dist.send(send_buffers[neighbor_id], dst=neighbor_id)
                    dist.recv(recv_buffers[neighbor_id], src=neighbor_id)
                else:
                    # Higher rank receives first, then sends
                    dist.recv(recv_buffers[neighbor_id], src=neighbor_id)
                    dist.send(send_buffers[neighbor_id], dst=neighbor_id)
        else:
            # Gloo: use non-blocking isend/irecv
            requests = []
            for neighbor_id in self.partition.neighbor_partitions:
                req = dist.isend(send_buffers[neighbor_id], dst=neighbor_id)
                requests.append(req)
                req = dist.irecv(recv_buffers[neighbor_id], src=neighbor_id)
                requests.append(req)
            
            if async_op:
                return requests
            
            for req in requests:
                req.wait()
        
        # Update halo values
        for neighbor_id in self.partition.neighbor_partitions:
            recv_idx = self.partition.recv_indices[neighbor_id].to(self.device)
            x[recv_idx] = recv_buffers[neighbor_id]
        
        return x
    
    def halo_exchange_local(
        self,
        x_list: List[torch.Tensor]
    ) -> None:
        """
        Local halo exchange for single-process multi-partition simulation.
        
        Useful for testing/debugging without actual distributed setup.
        
        Parameters
        ----------
        x_list : List[torch.Tensor]
            List of local vectors, one per partition
        """
        if not hasattr(self, '_all_partitions'):
            return
        
        # Build mapping from global to local for each partition
        for part_id in range(len(x_list)):
            partition = self._all_partitions[part_id]
            x = x_list[part_id]
            
            # For each halo node, find which neighbor owns it and get the value
            halo_offset = len(partition.owned_nodes)
            
            for halo_idx, global_node in enumerate(partition.halo_nodes.tolist()):
                local_halo_idx = halo_offset + halo_idx
                
                # Find which partition owns this node
                for neighbor_id in partition.neighbor_partitions:
                    neighbor_partition = self._all_partitions[neighbor_id]
                    neighbor_g2l = neighbor_partition.global_to_local
                    
                    if global_node < len(neighbor_g2l):
                        local_idx_in_neighbor = neighbor_g2l[global_node].item()
                        if local_idx_in_neighbor >= 0 and local_idx_in_neighbor < len(neighbor_partition.owned_nodes):
                            # This neighbor owns the node
                            x[local_halo_idx] = x_list[neighbor_id][local_idx_in_neighbor]
                            break
    
    def matvec(self, x: torch.Tensor, exchange_halo: bool = True) -> torch.Tensor:
        """
        Local matrix-vector product y = A_local @ x.
        
        Parameters
        ----------
        x : torch.Tensor
            Local vector [num_local]
        exchange_halo : bool
            If True, perform halo exchange before multiplication
            
        Returns
        -------
        y : torch.Tensor
            Result vector [num_local]
        """
        if exchange_halo:
            self.halo_exchange(x)
        
        # Sparse matvec
        A_local = torch.sparse_coo_tensor(
            torch.stack([self.local_row, self.local_col]),
            self.local_values,
            self.local_shape
        ).to(self.device)
        
        return torch.mv(A_local.to_sparse_csr(), x)
    
    def solve(
        self,
        b: torch.Tensor,
        method: str = 'cg',
        atol: float = 1e-10,
        maxiter: int = 1000,
        verbose: bool = False,
        distributed: bool = True
    ) -> torch.Tensor:
        """
        Solve linear system Ax = b.
        
        Parameters
        ----------
        b : torch.Tensor
            Right-hand side. Shape [num_owned] for owned nodes only.
        method : str
            Solver method: 'cg' (default), 'jacobi', 'gauss_seidel'
        atol : float
            Absolute tolerance for convergence
        maxiter : int
            Maximum iterations
        verbose : bool
            Print convergence info (rank 0 only for distributed)
        distributed : bool, default=True
            If True (default): Solve the GLOBAL system using distributed
            algorithms with all_reduce for global dot products.
            If False: Solve only the LOCAL subdomain problem (useful as
            preconditioner in domain decomposition methods).
            
        Returns
        -------
        x : torch.Tensor
            Solution for owned nodes, shape [num_owned]
            
        Examples
        --------
        >>> # Distributed solve (default) - all ranks cooperate
        >>> x = local_matrix.solve(b_owned)
        
        >>> # Local subdomain solve - no global communication
        >>> x = local_matrix.solve(b_owned, distributed=False)
        """
        if distributed:
            return self._solve_distributed_cg(b, atol, maxiter, verbose)
        else:
            return self._solve_local(b, method, atol, maxiter, verbose)
    
    def _solve_local(
        self,
        b: torch.Tensor,
        method: str,
        atol: float,
        maxiter: int,
        verbose: bool
    ) -> torch.Tensor:
        """Local subdomain solve (no global communication)."""
        # Handle b size
        if b.shape[0] == self.num_owned:
            b_full = torch.zeros(self.num_local, dtype=b.dtype, device=self.device)
            b_full[:self.num_owned] = b
            b = b_full
        elif b.shape[0] != self.num_local:
            raise ValueError(f"b must have size num_owned={self.num_owned} or num_local={self.num_local}")
        
        x = torch.zeros(self.num_local, dtype=b.dtype, device=self.device)
        
        if method == 'jacobi':
            x = self._solve_jacobi(x, b, atol, maxiter, verbose)
        elif method == 'gauss_seidel':
            x = self._solve_gauss_seidel(x, b, atol, maxiter, verbose)
        else:  # CG
            x = self._solve_cg(x, b, atol, maxiter, verbose)
        
        return x[:self.num_owned]
    
    def _solve_cg(self, x, b, atol, maxiter, verbose):
        """
        Local CG solver for subdomain problems.
        
        This solves only the local subdomain problem without global reductions.
        Useful as a preconditioner or subdomain solver in domain decomposition.
        """
        r = b - self.matvec(x)
        p = r.clone()
        rs_old = torch.dot(r[:self.num_owned], r[:self.num_owned])
        
        for i in range(maxiter):
            Ap = self.matvec(p)
            pAp = torch.dot(p[:self.num_owned], Ap[:self.num_owned])
            
            if pAp.abs() < 1e-30:
                break
                
            alpha = rs_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            
            rs_new = torch.dot(r[:self.num_owned], r[:self.num_owned])
            
            if verbose and i % 100 == 0:
                print(f"  CG iter {i}: residual = {rs_new.sqrt():.2e}")
            
            if rs_new.sqrt() < atol:
                if verbose:
                    print(f"  CG converged at iter {i}")
                break
            
            if rs_old.abs() < 1e-30:
                break
                
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        
        return x
    
    def _solve_jacobi(self, x, b, atol, maxiter, verbose):
        """Jacobi iteration with halo exchange"""
        # Extract diagonal
        diag_mask = self.local_row == self.local_col
        diag_indices = self.local_row[diag_mask]
        diag_values = self.local_values[diag_mask]
        
        D_inv = torch.zeros(self.num_local, dtype=b.dtype, device=self.device)
        D_inv[diag_indices] = 1.0 / diag_values
        
        for i in range(maxiter):
            x_new = D_inv * (b - self.matvec(x) + D_inv.reciprocal() * x)
            
            diff = (x_new[:self.num_owned] - x[:self.num_owned]).norm()
            x = x_new
            
            if verbose and i % 100 == 0:
                print(f"  Jacobi iter {i}: diff = {diff:.2e}")
            
            if diff < atol:
                if verbose:
                    print(f"  Jacobi converged at iter {i}")
                break
        
        return x
    
    def _solve_gauss_seidel(self, x, b, atol, maxiter, verbose):
        """Gauss-Seidel iteration with halo exchange"""
        # Build CSR for efficient row access
        A_csr = torch.sparse_coo_tensor(
            torch.stack([self.local_row, self.local_col]),
            self.local_values,
            self.local_shape
        ).to_sparse_csr()
        
        crow = A_csr.crow_indices()
        col = A_csr.col_indices()
        val = A_csr.values()
        
        for iteration in range(maxiter):
            x_old = x.clone()
            
            # Exchange halo before sweep
            self.halo_exchange(x)
            
            # Forward sweep on owned nodes only
            for i in range(self.num_owned):
                row_start = crow[i].item()
                row_end = crow[i + 1].item()
                
                sigma = 0.0
                diag = 1.0
                for j in range(row_start, row_end):
                    c = col[j].item()
                    v = val[j].item()
                    if c == i:
                        diag = v
                    else:
                        sigma += v * x[c].item()
                
                x[i] = (b[i].item() - sigma) / diag
            
            diff = (x[:self.num_owned] - x_old[:self.num_owned]).norm()
            
            if verbose and iteration % 100 == 0:
                print(f"  GS iter {iteration}: diff = {diff:.2e}")
            
            if diff < atol:
                if verbose:
                    print(f"  GS converged at iter {iteration}")
                break
        
        return x
    
    def _solve_distributed_cg(
        self,
        b_owned: torch.Tensor,
        atol: float,
        maxiter: int,
        verbose: bool
    ) -> torch.Tensor:
        """
        Distributed Conjugate Gradient solver.
        
        The key differences from local CG:
        1. Halo exchange before each matvec
        2. Global all_reduce for dot products
        """
        num_owned = self.num_owned
        num_local = self.num_local
        dtype = b_owned.dtype
        device = self.device
        rank = self.partition.partition_id
        
        # Initialize x_local = 0 (owned + halo)
        x_local = torch.zeros(num_local, dtype=dtype, device=device)
        
        # Extend b to local size (halo part is 0)
        b_local = torch.zeros(num_local, dtype=dtype, device=device)
        b_local[:num_owned] = b_owned
        
        # r = b - A @ x (local, no halo exchange needed for x=0)
        r_local = b_local.clone()
        p_local = r_local.clone()
        
        # rs_old = r^T @ r (global reduction, only sum owned nodes)
        rs_local = torch.dot(r_local[:num_owned], r_local[:num_owned])
        rs_old = self._global_reduce_sum(rs_local)
        
        for i in range(maxiter):
            # Halo exchange for p before matvec
            self.halo_exchange(p_local)
            
            # Ap = A @ p (local matvec)
            Ap_local = self.matvec(p_local, exchange_halo=False)
            
            # pAp = p^T @ A @ p (global reduction)
            pAp_local = torch.dot(p_local[:num_owned], Ap_local[:num_owned])
            pAp = self._global_reduce_sum(pAp_local)
            
            if pAp.abs() < 1e-30:
                break
            
            alpha = rs_old / pAp
            
            # Update x and r (local)
            x_local = x_local + alpha * p_local
            r_local = r_local - alpha * Ap_local
            
            # rs_new = r^T @ r (global reduction)
            rs_local = torch.dot(r_local[:num_owned], r_local[:num_owned])
            rs_new = self._global_reduce_sum(rs_local)
            
            residual = rs_new.sqrt()
            
            if verbose and rank == 0 and i % 50 == 0:
                print(f"  Distributed CG iter {i}: residual = {residual:.2e}")
            
            if residual < atol:
                if verbose and rank == 0:
                    print(f"  Distributed CG converged at iter {i}, residual = {residual:.2e}")
                break
            
            beta = rs_new / rs_old
            p_local = r_local + beta * p_local
            rs_old = rs_new
        
        # Return only owned part
        return x_local[:num_owned]
    
    def _global_reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        """Perform global all_reduce sum."""
        if not DIST_AVAILABLE or not dist.is_initialized():
            return value
        
        # Ensure tensor is on the correct device for the backend
        backend = dist.get_backend()
        if backend == 'nccl' and not value.is_cuda:
            # NCCL requires CUDA tensors
            value = value.to(self.device)
        
        result = value.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result
    
    def eigsh(
        self,
        k: int = 6,
        which: str = "LM",
        maxiter: int = 200,
        tol: float = 1e-8,
        verbose: bool = False,
        distributed: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute k eigenvalues of symmetric matrix.
        
        Parameters
        ----------
        k : int
            Number of eigenvalues to compute
        which : str
            Which eigenvalues: "LM" (largest magnitude), "SM" (smallest magnitude)
        maxiter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        verbose : bool
            Print convergence info (rank 0 only)
        distributed : bool, default=True
            If True (default): Use distributed LOBPCG with global reductions.
            If False: Gather to single SparseTensor and compute locally
            (not recommended for large matrices).
            
        Returns
        -------
        eigenvalues : torch.Tensor
            k eigenvalues, shape [k]
        eigenvectors_owned : torch.Tensor
            Eigenvectors for owned nodes only, shape [num_owned, k]
        """
        if not distributed:
            # Gather to single node (not recommended)
            import warnings
            warnings.warn("distributed=False gathers entire matrix to one node. "
                         "Use distributed=True for large-scale problems.")
            st = self.to_sparse_tensor()
            eigenvalues, eigenvectors = st.eigsh(k=k, which=which)
            # Extract local portion
            owned_nodes = self.partition.owned_nodes
            return eigenvalues, eigenvectors[owned_nodes]
        n = self.global_shape[0]
        num_owned = self.num_owned
        rank = self.partition.partition_id
        dtype = self.local_values.dtype
        device = self.device
        
        # Initialize random subspace
        torch.manual_seed(42 + rank)  # Different per rank for diversity
        m = min(2 * k, n)
        
        # Each rank has its local portion of X
        X_owned = torch.randn(num_owned, m, dtype=dtype, device=device)
        
        # Orthogonalize globally
        X_owned = self._global_orthogonalize(X_owned)
        
        eigenvalues_prev = None
        
        for iteration in range(maxiter):
            # Distributed matvec: AX
            AX_owned = self._global_matvec_batch(X_owned)
            
            # Rayleigh-Ritz: H = X^T @ AX (global reduction)
            # Local contribution
            H_local = X_owned.T @ AX_owned
            H = self._global_reduce_sum(H_local)
            
            # Solve small eigenvalue problem (same on all ranks)
            eigenvalues, eigenvectors = torch.linalg.eigh(H)
            
            # Sort eigenvalues
            if which == "LM":
                idx_sort = eigenvalues.abs().argsort(descending=True)
            else:
                idx_sort = eigenvalues.abs().argsort()
            eigenvalues = eigenvalues[idx_sort]
            eigenvectors = eigenvectors[:, idx_sort]
            
            # Update X = X @ V (local)
            X_owned = X_owned @ eigenvectors
            
            # Check convergence
            if eigenvalues_prev is not None:
                diff = (eigenvalues[:k] - eigenvalues_prev[:k]).abs()
                if (diff < tol * eigenvalues[:k].abs().clamp(min=1e-10)).all():
                    if verbose and rank == 0:
                        print(f"  Distributed LOBPCG converged at iteration {iteration}")
                    break
            eigenvalues_prev = eigenvalues.clone()
            
            if verbose and rank == 0 and iteration % 20 == 0:
                print(f"  Distributed LOBPCG iter {iteration}: λ_0 = {eigenvalues[0]:.6f}")
            
            # Expand subspace with residual
            if iteration < maxiter - 1:
                AX_new = self._global_matvec_batch(X_owned)
                residual = AX_new - X_owned * eigenvalues.unsqueeze(0)
                
                # Combine and orthogonalize
                combined = torch.cat([X_owned[:, :k], residual[:, :k]], dim=1)
                X_owned = self._global_orthogonalize(combined)
                
                # Ensure correct size
                if X_owned.size(1) < m:
                    extra = torch.randn(num_owned, m - X_owned.size(1), dtype=dtype, device=device)
                    X_owned = torch.cat([X_owned, extra], dim=1)
                    X_owned = self._global_orthogonalize(X_owned)
        
        return eigenvalues[:k], X_owned[:, :k]
    
    def _global_matvec_batch(self, X_owned: torch.Tensor) -> torch.Tensor:
        """
        Distributed matvec for a batch of vectors.
        
        Each rank computes A @ X for its local portion.
        """
        num_owned = self.num_owned
        num_local = self.num_local
        m = X_owned.size(1)
        dtype = X_owned.dtype
        device = self.device
        
        # Extend to local size (owned + halo)
        X_local = torch.zeros(num_local, m, dtype=dtype, device=device)
        X_local[:num_owned] = X_owned
        
        # Gather global X for halo (simplified - in production use p2p)
        X_global = self._gather_all_vectors(X_owned)
        
        # Fill halo from global
        halo_nodes = self.partition.halo_nodes
        if len(halo_nodes) > 0:
            X_local[num_owned:] = X_global[halo_nodes]
        
        # Local matvec for each column
        Y_local = torch.zeros(num_local, m, dtype=dtype, device=device)
        for j in range(m):
            Y_local[:, j] = self.matvec(X_local[:, j], exchange_halo=False)
        
        return Y_local[:num_owned]
    
    def _gather_all_vectors(self, X_owned: torch.Tensor) -> torch.Tensor:
        """Gather vectors from all ranks to build global vector."""
        n = self.global_shape[0]
        m = X_owned.size(1)
        dtype = X_owned.dtype
        device = self.device
        
        X_global = torch.zeros(n, m, dtype=dtype, device=device)
        owned_nodes = self.partition.owned_nodes
        X_global[owned_nodes] = X_owned
        
        # All-reduce to combine
        self._global_reduce_sum_inplace(X_global)
        
        return X_global
    
    def _global_reduce_sum_inplace(self, tensor: torch.Tensor) -> None:
        """In-place global all_reduce sum."""
        if DIST_AVAILABLE and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    def _global_orthogonalize(self, X_owned: torch.Tensor) -> torch.Tensor:
        """
        Globally orthogonalize a distributed matrix using TSQR.
        
        Simplified version: gather, QR, scatter.
        Production version would use TSQR for better scalability.
        """
        # Gather global X
        X_global = self._gather_all_vectors(X_owned)
        
        # QR on global (same result on all ranks)
        Q, _ = torch.linalg.qr(X_global)
        
        # Extract local portion
        owned_nodes = self.partition.owned_nodes
        return Q[owned_nodes]
    
    def gather_global(self, x_local: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Gather local vectors to global vector (on rank 0).
        
        Parameters
        ----------
        x_local : torch.Tensor
            Local vector [num_owned]
            
        Returns
        -------
        x_global : torch.Tensor or None
            Global vector on rank 0, None on other ranks
        """
        if not DIST_AVAILABLE or not dist.is_initialized():
            # Single process: just expand to global
            x_global = torch.zeros(self.global_shape[0], dtype=x_local.dtype, device=x_local.device)
            x_global[self.partition.owned_nodes] = x_local[:self.num_owned]
            return x_global
        
        # Distributed gather
        owned_vals = x_local[:self.num_owned]
        
        # Gather sizes
        local_size = torch.tensor([self.num_owned], device=self.device)
        sizes = [torch.zeros(1, dtype=torch.int64, device=self.device) for _ in range(self.num_partitions)]
        dist.all_gather(sizes, local_size)
        
        # Gather values
        if dist.get_rank() == 0:
            x_global = torch.zeros(self.global_shape[0], dtype=x_local.dtype, device=self.device)
            gathered = [torch.zeros(s.item(), dtype=x_local.dtype, device=self.device) for s in sizes]
            dist.gather(owned_vals, gather_list=gathered, dst=0)
            
            # Place in global vector (need owned_nodes from all partitions)
            # This requires additional communication of owned_nodes
            return x_global
        else:
            dist.gather(owned_vals, dst=0)
            return None
    
    def __repr__(self) -> str:
        return (f"DSparseMatrix(partition={self.partition.partition_id}/{self.num_partitions}, "
                f"local={self.num_local} ({self.num_owned}+{self.num_halo}), "
                f"global={self.global_shape}, nnz={self.nnz}, device={self.device})")
    
    # =========================================================================
    # Persistence (I/O)
    # =========================================================================
    
    @classmethod
    def load(
        cls,
        directory: Union[str, "os.PathLike"],
        rank: int,
        world_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu"
    ) -> "DSparseMatrix":
        """
        Load a partition from disk for the given rank.
        
        Each rank should call this with its own rank to load only its partition.
        
        Parameters
        ----------
        directory : str or PathLike
            Directory containing partitioned data.
        rank : int
            Rank of this process.
        world_size : int, optional
            Total number of processes (must match num_partitions).
        device : str or torch.device
            Device to load tensors to.
        
        Returns
        -------
        DSparseMatrix
            The partition for this rank.
        
        Example
        -------
        >>> rank = dist.get_rank()
        >>> world_size = dist.get_world_size()
        >>> partition = DSparseMatrix.load("matrix_dist", rank, world_size, "cuda")
        """
        from .io import load_partition
        return load_partition(directory, rank, world_size, device)


def create_distributed_matrices(
    values: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    shape: Tuple[int, int],
    num_partitions: int,
    coords: Optional[torch.Tensor] = None,
    device: Union[str, torch.device] = 'cpu'
) -> List[DSparseMatrix]:
    """
    Create all distributed matrix partitions for local simulation.
    
    .. deprecated::
        Use DSparseTensor instead for a more Pythonic interface.
    
    Useful for testing/debugging without actual distributed setup.
    
    Parameters
    ----------
    values, row, col : torch.Tensor
        Global COO sparse matrix data
    shape : Tuple[int, int]
        Global matrix shape
    num_partitions : int
        Number of partitions
    coords : torch.Tensor, optional
        Node coordinates for geometric partitioning
    device : str or torch.device
        Device for all partitions ('cpu', 'cuda', 'cuda:0', etc.)
    
    Returns
    -------
    List[DSparseMatrix]
        List of DSparseMatrix, one per partition
    """
    warnings.warn(
        "create_distributed_matrices is deprecated. Use DSparseTensor instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    matrices = []
    
    # Compute partition IDs once
    if coords is not None:
        partition_ids = partition_coordinates(coords, num_partitions)
    else:
        partition_ids = partition_graph_metis(row, col, shape[0], num_partitions)
    
    for i in range(num_partitions):
        mat = DSparseMatrix.from_global(
            values, row, col, shape, num_partitions, i,
            partition_ids=partition_ids, device=device
        )
        matrices.append(mat)
    
    # Store reference to all partitions for local halo exchange
    for mat in matrices:
        mat._all_partitions = [m.partition for m in matrices]
    
    return matrices


class DSparseTensor:
    """
    Distributed Sparse Tensor with automatic partitioning and halo exchange.
    
    A Pythonic wrapper that provides a unified interface for distributed
    sparse matrix operations. Supports indexing to access individual partitions.
    
    Parameters
    ----------
    values : torch.Tensor
        Non-zero values [nnz]
    row_indices : torch.Tensor
        Row indices [nnz]
    col_indices : torch.Tensor
        Column indices [nnz]
    shape : Tuple[int, int]
        Matrix shape (m, n)
    num_partitions : int
        Number of partitions to create
    coords : torch.Tensor, optional
        Node coordinates for geometric partitioning [num_nodes, dim]
    partition_method : str
        Partitioning method: 'metis', 'rcb', 'slicing', 'simple'
    device : str or torch.device
        Device for the matrix data
    verbose : bool
        Whether to print partition info
    
    Example
    -------
    >>> import torch
    >>> from torch_sla import DSparseTensor
    >>> 
    >>> # Create distributed tensor with 4 partitions
    >>> A = DSparseTensor(val, row, col, shape, num_partitions=4)
    >>> 
    >>> # Access individual partitions
    >>> A0 = A[0]  # First partition
    >>> A1 = A[1]  # Second partition
    >>> 
    >>> # Iterate over partitions
    >>> for partition in A:
    >>>     x = partition.solve(b_local)
    >>> 
    >>> # Properties
    >>> print(A.num_partitions)  # 4
    >>> print(A.shape)           # Global shape
    >>> print(len(A))            # 4
    >>> 
    >>> # Move to CUDA
    >>> A_cuda = A.cuda()
    >>> 
    >>> # Local halo exchange (for testing)
    >>> x_list = [torch.zeros(A[i].num_local) for i in range(4)]
    >>> A.halo_exchange_local(x_list)
    """
    
    def __init__(
        self,
        values: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        shape: Tuple[int, int],
        num_partitions: int,
        coords: Optional[torch.Tensor] = None,
        partition_method: str = 'auto',
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True
    ):
        self._values = values
        self._row_indices = row_indices
        self._col_indices = col_indices
        self._shape = shape
        self._num_partitions = num_partitions
        self._coords = coords
        self._partition_method = partition_method
        self._verbose = verbose
        
        # Infer device from input tensor if not explicitly specified
        if device is None:
            device = values.device
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        
        # Compute partition IDs
        # NOTE: In distributed mode, this should be computed on rank 0 and broadcast
        # to ensure consistency. See _compute_partitions_distributed() for distributed-safe version.
        self._partition_ids = self._compute_partitions(partition_method, coords)
        
        # Create all partitions
        self._partitions: List[DSparseMatrix] = []
        self._create_partitions()
    
    def _compute_partitions(
        self, 
        method: str, 
        coords: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute partition assignments for each node."""
        if method == 'auto':
            if coords is not None:
                method = 'rcb'
            else:
                method = 'metis'
        
        if method == 'metis':
            return partition_graph_metis(
                self._row_indices, self._col_indices, 
                self._shape[0], self._num_partitions
            )
        elif method in ['rcb', 'slicing']:
            if coords is None:
                raise ValueError(f"Partition method '{method}' requires coords")
            return partition_coordinates(coords, self._num_partitions, method=method)
        elif method == 'simple':
            return partition_simple(self._shape[0], self._num_partitions)
        else:
            raise ValueError(f"Unknown partition method: {method}")
    
    def _create_partitions(self):
        """Create all partition matrices."""
        for i in range(self._num_partitions):
            mat = DSparseMatrix.from_global(
                self._values, self._row_indices, self._col_indices,
                self._shape, self._num_partitions, i,
                partition_ids=self._partition_ids,
                device=self._device,
                verbose=self._verbose
            )
            self._partitions.append(mat)
        
        # Store reference to all partitions for local halo exchange
        for mat in self._partitions:
            mat._all_partitions = [m.partition for m in self._partitions]
    
    @classmethod
    def from_sparse_tensor(
        cls,
        sparse_tensor: "SparseTensor",
        num_partitions: int,
        coords: Optional[torch.Tensor] = None,
        partition_method: str = 'auto',
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True
    ) -> "DSparseTensor":
        """
        Create DSparseTensor from a SparseTensor.
        
        Parameters
        ----------
        sparse_tensor : SparseTensor
            Input sparse tensor (must be 2D, not batched)
        num_partitions : int
            Number of partitions
        coords : torch.Tensor, optional
            Node coordinates for geometric partitioning
        partition_method : str
            Partitioning method
        device : str or torch.device, optional
            Target device (defaults to sparse_tensor's device)
        verbose : bool
            Whether to print partition info
            
        Returns
        -------
        DSparseTensor
            Distributed sparse tensor
        """
        # Avoid circular import
        from .sparse_tensor import SparseTensor
        
        if sparse_tensor.is_batched:
            raise ValueError("DSparseTensor does not support batched SparseTensor. "
                           "Use a 2D SparseTensor.")
        
        if device is None:
            device = sparse_tensor.device
        
        # Use sparse_shape for the matrix dimensions
        sparse_shape = sparse_tensor.sparse_shape
        
        return cls(
            sparse_tensor.values,
            sparse_tensor.row_indices,
            sparse_tensor.col_indices,
            sparse_shape,
            num_partitions=num_partitions,
            coords=coords,
            partition_method=partition_method,
            device=device,
            verbose=verbose
        )
    
    @classmethod
    def from_torch_sparse(
        cls,
        A: torch.Tensor,
        num_partitions: int,
        **kwargs
    ) -> "DSparseTensor":
        """Create DSparseTensor from PyTorch sparse tensor."""
        if A.layout == torch.sparse_csr:
            A = A.to_sparse_coo()
        
        indices = A._indices()
        values = A._values()
        
        return cls(
            values, indices[0], indices[1], tuple(A.shape),
            num_partitions=num_partitions, **kwargs
        )
    
    @classmethod
    def from_global_distributed(
        cls,
        values: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        shape: Tuple[int, int],
        rank: int,
        world_size: int,
        coords: Optional[torch.Tensor] = None,
        partition_method: str = 'auto',
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True
    ) -> "DSparseMatrix":
        """
        Create local partition in a distributed-safe manner.
        
        This method ensures that all ranks compute the same partition assignment
        by having rank 0 compute the partition IDs and broadcasting to all ranks.
        
        Parameters
        ----------
        values : torch.Tensor
            Global non-zero values [nnz]
        row_indices : torch.Tensor
            Global row indices [nnz]
        col_indices : torch.Tensor
            Global column indices [nnz]
        shape : Tuple[int, int]
            Global matrix shape (M, N)
        rank : int
            Current process rank
        world_size : int
            Total number of processes
        coords : torch.Tensor, optional
            Node coordinates for geometric partitioning [num_nodes, dim]
        partition_method : str
            Partitioning method: 'metis', 'rcb', 'slicing', 'simple'
        device : str or torch.device, optional
            Target device
        verbose : bool
            Whether to print partition info
            
        Returns
        -------
        DSparseMatrix
            Local partition matrix for this rank
            
        Example
        -------
        >>> import torch.distributed as dist
        >>> 
        >>> # In each process:
        >>> rank = dist.get_rank()
        >>> world_size = dist.get_world_size()
        >>> 
        >>> local_matrix = DSparseTensor.from_global_distributed(
        ...     val, row, col, shape, 
        ...     rank=rank, world_size=world_size
        ... )
        """
        import torch.distributed as dist
        
        if device is None:
            device = values.device
        if isinstance(device, str):
            device = torch.device(device)
        
        # Compute partition IDs on rank 0 and broadcast
        if rank == 0:
            # Create temporary DSparseTensor to compute partitions
            # Use 'simple' method if METIS might be non-deterministic
            if partition_method == 'auto':
                if coords is not None:
                    actual_method = 'rcb'
                else:
                    # Use simple partitioning by default in distributed mode
                    # to ensure determinism across ranks
                    actual_method = 'simple'
            else:
                actual_method = partition_method
            
            num_nodes = shape[0]
            if actual_method == 'simple':
                partition_ids = partition_simple(num_nodes, world_size)
            elif actual_method == 'metis':
                partition_ids = partition_graph_metis(
                    row_indices, col_indices, num_nodes, world_size
                )
            elif actual_method in ['rcb', 'slicing']:
                if coords is None:
                    raise ValueError(f"Method '{actual_method}' requires coords")
                partition_ids = partition_coordinates(coords, world_size, method=actual_method)
            else:
                raise ValueError(f"Unknown method: {actual_method}")
            
            partition_ids = partition_ids.to(device)
        else:
            # Create empty tensor to receive broadcast
            partition_ids = torch.zeros(shape[0], dtype=torch.int64, device=device)
        
        # Broadcast partition IDs from rank 0 to all ranks
        dist.broadcast(partition_ids, src=0)
        
        # Now create local partition using the consistent partition IDs
        local_matrix = DSparseMatrix.from_global(
            values, row_indices, col_indices, shape,
            world_size, rank,
            partition_ids=partition_ids,
            device=device,
            verbose=verbose and rank == 0  # Only print on rank 0
        )
        
        return local_matrix
    
    @classmethod
    def from_device_mesh(
        cls,
        values: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        shape: Tuple[int, int],
        device_mesh: "DeviceMesh",
        coords: Optional[torch.Tensor] = None,
        partition_method: str = 'simple',
        placement: str = 'shard_rows',
        verbose: bool = False
    ) -> "DSparseMatrix":
        """
        Create local partition using PyTorch DeviceMesh.
        
        This is the recommended method for distributed training with PyTorch's
        DTensor ecosystem. Each rank receives only its local partition.
        
        Parameters
        ----------
        values : torch.Tensor
            Global non-zero values [nnz] (same on all ranks)
        row_indices : torch.Tensor
            Global row indices [nnz]
        col_indices : torch.Tensor
            Global column indices [nnz]
        shape : Tuple[int, int]
            Global matrix shape (M, N)
        device_mesh : DeviceMesh
            PyTorch DeviceMesh specifying device topology
        coords : torch.Tensor, optional
            Node coordinates for geometric partitioning
        partition_method : str
            Partitioning method: 'metis', 'rcb', 'simple'
            Default is 'simple' for determinism in distributed setting
        placement : str
            How to distribute: 'shard_rows', 'shard_cols', 'replicate'
        verbose : bool
            Whether to print partition info
            
        Returns
        -------
        DSparseMatrix
            Local partition for this rank
            
        Example
        -------
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> from torch_sla import DSparseTensor
        >>> 
        >>> # Initialize 4-GPU device mesh
        >>> mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp",))
        >>> 
        >>> # Create distributed sparse tensor (each rank gets its partition)
        >>> local_matrix = DSparseTensor.from_device_mesh(
        ...     val, row, col, shape,
        ...     device_mesh=mesh,
        ...     partition_method='simple'
        ... )
        >>> 
        >>> # Local operations
        >>> y_local = local_matrix.matvec(x_local)
        >>> x_local = local_matrix.solve(b_local)
        """
        try:
            from torch.distributed.device_mesh import DeviceMesh
        except ImportError:
            raise ImportError("DeviceMesh requires PyTorch 2.0+. "
                            "Use from_global_distributed() instead.")
        
        if not DIST_AVAILABLE or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized. "
                             "Call dist.init_process_group() first.")
        
        # Get rank info from device mesh
        rank = device_mesh.get_local_rank()
        world_size = device_mesh.size()
        device_type = device_mesh.device_type
        
        # Determine target device
        if device_type == "cuda":
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device(device_type)
        
        # Use the distributed-safe factory method
        return cls.from_global_distributed(
            values, row_indices, col_indices, shape,
            rank=rank, world_size=world_size,
            coords=coords,
            partition_method=partition_method,
            device=device,
            verbose=verbose
        )
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Global matrix shape."""
        return self._shape
    
    @property
    def num_partitions(self) -> int:
        """Number of partitions."""
        return self._num_partitions
    
    @property
    def device(self) -> torch.device:
        """Device of the matrix data."""
        return self._device
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of matrix values."""
        return self._values.dtype
    
    @property
    def nnz(self) -> int:
        """Total number of non-zeros."""
        return self._values.size(0)
    
    @property
    def partition_ids(self) -> torch.Tensor:
        """Partition assignment for each node."""
        return self._partition_ids
    
    @property
    def is_cuda(self) -> bool:
        """Check if matrix is on CUDA."""
        return self._device.type == 'cuda'
    
    # =========================================================================
    # Indexing and Iteration
    # =========================================================================
    
    def __len__(self) -> int:
        """Number of partitions."""
        return self._num_partitions
    
    def __getitem__(self, idx: int) -> DSparseMatrix:
        """Get a specific partition."""
        if idx < 0:
            idx = self._num_partitions + idx
        if idx < 0 or idx >= self._num_partitions:
            raise IndexError(f"Partition index {idx} out of range [0, {self._num_partitions})")
        return self._partitions[idx]
    
    def __iter__(self):
        """Iterate over partitions."""
        return iter(self._partitions)
    
    # =========================================================================
    # Device Management
    # =========================================================================
    
    def to(self, device: Union[str, torch.device]) -> "DSparseTensor":
        """
        Move all partitions to a different device.
        
        Parameters
        ----------
        device : str or torch.device
            Target device
            
        Returns
        -------
        DSparseTensor
            New distributed tensor on target device
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        new_tensor = DSparseTensor.__new__(DSparseTensor)
        new_tensor._values = self._values.to(device)
        new_tensor._row_indices = self._row_indices.to(device)
        new_tensor._col_indices = self._col_indices.to(device)
        new_tensor._shape = self._shape
        new_tensor._num_partitions = self._num_partitions
        new_tensor._coords = self._coords
        new_tensor._partition_method = self._partition_method
        new_tensor._verbose = False  # Don't print again
        new_tensor._device = device
        new_tensor._partition_ids = self._partition_ids
        
        # Move partitions
        new_tensor._partitions = [p.to(device) for p in self._partitions]
        
        # Update references
        for mat in new_tensor._partitions:
            mat._all_partitions = [m.partition for m in new_tensor._partitions]
        
        return new_tensor
    
    def cuda(self, device: Optional[int] = None) -> "DSparseTensor":
        """Move to CUDA device."""
        if device is not None:
            return self.to(f'cuda:{device}')
        return self.to('cuda')
    
    def cpu(self) -> "DSparseTensor":
        """Move to CPU."""
        return self.to('cpu')
    
    # =========================================================================
    # Distributed Operations
    # =========================================================================
    
    def halo_exchange_local(self, x_list: List[torch.Tensor]) -> None:
        """
        Local halo exchange for single-process simulation.
        
        Exchanges halo values between all partitions locally.
        Useful for testing without actual distributed setup.
        
        Parameters
        ----------
        x_list : List[torch.Tensor]
            List of local vectors, one per partition. Each vector is
            modified in-place to update halo values.
        """
        if len(x_list) != self._num_partitions:
            raise ValueError(f"Expected {self._num_partitions} vectors, got {len(x_list)}")
        
        for part_id in range(self._num_partitions):
            partition = self._partitions[part_id].partition
            x = x_list[part_id]
            
            halo_offset = len(partition.owned_nodes)
            
            for halo_idx, global_node in enumerate(partition.halo_nodes.tolist()):
                local_halo_idx = halo_offset + halo_idx
                
                for neighbor_id in partition.neighbor_partitions:
                    neighbor_partition = self._partitions[neighbor_id].partition
                    neighbor_g2l = neighbor_partition.global_to_local
                    
                    if global_node < len(neighbor_g2l):
                        local_idx_in_neighbor = neighbor_g2l[global_node].item()
                        if local_idx_in_neighbor >= 0 and local_idx_in_neighbor < len(neighbor_partition.owned_nodes):
                            x[local_halo_idx] = x_list[neighbor_id][local_idx_in_neighbor]
                            break
    
    def matvec_all(
        self,
        x_list: List[torch.Tensor],
        exchange_halo: bool = True
    ) -> List[torch.Tensor]:
        """
        Matrix-vector multiply on all partitions.
        
        Performs y = A @ x for each partition, with optional halo exchange.
        
        Parameters
        ----------
        x_list : List[torch.Tensor]
            List of local vectors, one per partition. Each vector should have
            size = num_owned + num_halo for that partition.
        exchange_halo : bool
            Whether to perform halo exchange before multiplication.
            Default True.
            
        Returns
        -------
        List[torch.Tensor]
            List of result vectors, one per partition. Each result has
            size = num_owned (only owned nodes have valid results).
            
        Example
        -------
        >>> D = SparseTensor(val, row, col, shape).partition(4)
        >>> x_local = D.scatter_local(x_global)
        >>> y_local = D.matvec_all(x_local)
        >>> y_global = D.gather_global(y_local)
        """
        return [self._partitions[i].matvec(x_list[i], exchange_halo=exchange_halo)
                for i in range(self._num_partitions)]
    
    def solve_all(
        self,
        b_list: List[torch.Tensor],
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Solve on all partitions (subdomain solves).
        
        NOTE: This performs LOCAL subdomain solves, NOT a global distributed solve.
        Each partition solves its own local system independently.
        For a true distributed solve, use `solve_distributed()`.
        
        Parameters
        ----------
        b_list : List[torch.Tensor]
            List of local RHS vectors, one per partition
        **kwargs
            Additional arguments passed to each partition's solve method
            
        Returns
        -------
        List[torch.Tensor]
            List of solution vectors, one per partition
        """
        return [self._partitions[i].solve(b_list[i], **kwargs) 
                for i in range(self._num_partitions)]
    
    def solve_distributed(
        self,
        b_global: torch.Tensor,
        method: str = 'cg',
        atol: float = 1e-10,
        maxiter: int = 1000,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Distributed solve: find x such that A @ x = b using all partitions.
        
        This performs a TRUE distributed solve where all partitions collaborate
        to solve the global system. Uses distributed CG with global reductions.
        
        Parameters
        ----------
        b_global : torch.Tensor
            Global RHS vector [N]
        method : str
            Solver method: 'cg' (Conjugate Gradient)
        atol : float
            Absolute tolerance for convergence
        maxiter : int
            Maximum iterations
        verbose : bool
            Print convergence info
            
        Returns
        -------
        torch.Tensor
            Global solution vector [N]
            
        Example
        -------
        >>> D = A.partition(num_partitions=4)
        >>> x = D.solve_distributed(b)  # Distributed CG solve
        >>> residual = torch.norm(A @ x - b)
        """
        N = self._shape[0]
        dtype = b_global.dtype
        device = self._device
        
        # Initialize x = 0
        x_global = torch.zeros(N, dtype=dtype, device=device)
        
        # Scatter b to local
        b_local = self.scatter_local(b_global)
        
        # Distributed CG
        if method == 'cg':
            x_global = self._distributed_cg(x_global, b_global, atol, maxiter, verbose)
        else:
            raise ValueError(f"Unknown method: {method}. Supported: 'cg'")
        
        return x_global
    
    def _distributed_cg(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        atol: float,
        maxiter: int,
        verbose: bool
    ) -> torch.Tensor:
        """
        Distributed Conjugate Gradient.
        
        All partitions work together, with global reductions for inner products.
        """
        N = self._shape[0]
        dtype = b.dtype
        device = self._device
        
        # r = b - A @ x
        Ax = self @ x  # Uses __matmul__ which does scatter -> matvec_all -> gather
        r = b - Ax
        
        # p = r
        p = r.clone()
        
        # rs_old = r^T @ r (global)
        rs_old = torch.dot(r, r)
        
        for i in range(maxiter):
            # Ap = A @ p
            Ap = self @ p
            
            # pAp = p^T @ A @ p (global)
            pAp = torch.dot(p, Ap)
            
            if pAp.abs() < 1e-30:
                if verbose:
                    print(f"  Distributed CG: pAp too small at iter {i}")
                break
            
            # alpha = rs_old / pAp
            alpha = rs_old / pAp
            
            # x = x + alpha * p
            x = x + alpha * p
            
            # r = r - alpha * Ap
            r = r - alpha * Ap
            
            # rs_new = r^T @ r (global)
            rs_new = torch.dot(r, r)
            
            residual = rs_new.sqrt()
            
            if verbose and i % 100 == 0:
                print(f"  Distributed CG iter {i}: residual = {residual:.2e}")
            
            if residual < atol:
                if verbose:
                    print(f"  Distributed CG converged at iter {i}, residual = {residual:.2e}")
                break
            
            if rs_old.abs() < 1e-30:
                break
            
            # beta = rs_new / rs_old
            beta = rs_new / rs_old
            
            # p = r + beta * p
            p = r + beta * p
            
            rs_old = rs_new
        
        return x
    
    def gather_global(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Gather local vectors to global vector.
        
        Parameters
        ----------
        x_list : List[torch.Tensor]
            List of local vectors, one per partition
            
        Returns
        -------
        torch.Tensor
            Global vector
        """
        x_global = torch.zeros(self._shape[0], dtype=x_list[0].dtype, device=self._device)
        
        for i in range(self._num_partitions):
            partition = self._partitions[i].partition
            owned_nodes = partition.owned_nodes
            num_owned = len(owned_nodes)
            x_global[owned_nodes] = x_list[i][:num_owned].to(self._device)
        
        return x_global
    
    def scatter_local(self, x_global: torch.Tensor) -> List[torch.Tensor]:
        """
        Scatter global vector to local vectors.
        
        Parameters
        ----------
        x_global : torch.Tensor
            Global vector
            
        Returns
        -------
        List[torch.Tensor]
            List of local vectors (with halo values filled)
        """
        x_list = []
        
        for i in range(self._num_partitions):
            partition = self._partitions[i].partition
            local_nodes = partition.local_nodes
            x_local = x_global[local_nodes].to(self._partitions[i].device)
            x_list.append(x_local)
        
        return x_list
    
    def to_sparse_tensor(self) -> "SparseTensor":
        """
        Gather all partitions into a single SparseTensor.
        
        This creates a global SparseTensor from the distributed data.
        Useful for verification, debugging, or when you need to perform
        operations that require the full matrix.
        
        Returns
        -------
        SparseTensor
            Global sparse tensor containing all data
            
        Example
        -------
        >>> D = DSparseTensor(val, row, col, shape, num_partitions=4)
        >>> A = D.to_sparse_tensor()  # Gather to global SparseTensor
        >>> x = A.solve(b)  # Solve on the full matrix
        """
        from .sparse_tensor import SparseTensor
        
        # Return the original global data as SparseTensor
        return SparseTensor(
            self._values.to(self._device),
            self._row_indices.to(self._device),
            self._col_indices.to(self._device),
            self._shape
        )
    
    # Alias for convenience
    gather = to_sparse_tensor
    
    # =========================================================================
    # Distributed Algorithms (True Distributed, No Gather)
    # =========================================================================
    
    def _global_matvec_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Global matrix-vector multiplication that preserves gradients.
        
        Uses the original COO data to maintain gradient flow.
        For true distributed MPI execution, use _distributed_matvec instead.
        
        This method is used for gradient-enabled operations like eigsh, solve.
        """
        # Use original global COO data for gradient support
        # y[i] = sum_j A[i,j] * x[j]
        # y = scatter_add(values * x[col], row)
        y = torch.zeros(self._shape[0], dtype=x.dtype, device=x.device)
        vals = self._values.to(x.device)
        rows = self._row_indices.to(x.device)
        cols = self._col_indices.to(x.device)
        
        # y[row] += values * x[col]
        contributions = vals * x[cols]
        y = y.scatter_add(0, rows, contributions)
        return y
    
    def _distributed_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Distributed matrix-vector multiplication with gradient support.
        
        For single-node simulation with gradient support, uses _global_matvec_with_grad.
        For true distributed MPI execution, uses scatter -> local matvec -> gather.
        """
        # Check if we need gradients
        if self._values.requires_grad or x.requires_grad:
            # Use global matvec that preserves gradients
            return self._global_matvec_with_grad(x)
        
        # Otherwise use true distributed pattern
        x_local = self.scatter_local(x)
        y_local = self.matvec_all(x_local)
        return self.gather_global(y_local)
    
    def _distributed_lobpcg(
        self,
        k: int,
        largest: bool = True,
        maxiter: int = 1000,
        tol: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Distributed LOBPCG eigenvalue solver.
        
        Uses distributed matvec with global QR and Rayleigh-Ritz.
        No data gather required - only needs global reductions.
        """
        N = self._shape[0]
        dtype = self._values.dtype
        device = self._device
        
        # Initialize random subspace (global vectors)
        m = min(2 * k, N)
        X = torch.randn(N, m, dtype=dtype, device=device)
        
        # Global QR decomposition
        X, _ = torch.linalg.qr(X)
        
        eigenvalues_prev = None
        
        for iteration in range(maxiter):
            # Distributed matvec: AX = D @ X (column by column or batched)
            AX = torch.zeros_like(X)
            for j in range(X.shape[1]):
                AX[:, j] = self._distributed_matvec(X[:, j])
            
            # Rayleigh-Ritz: project onto subspace
            # H = X^T @ AX (global reduction)
            H = X.T @ AX
            
            # Solve small eigenvalue problem
            eigenvalues, eigenvectors = torch.linalg.eigh(H)
            
            # Sort eigenvalues
            if largest:
                idx = eigenvalues.argsort(descending=True)
            else:
                idx = eigenvalues.argsort()
            
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Update X = X @ V
            X = X @ eigenvectors
            
            # Check convergence
            if eigenvalues_prev is not None:
                diff = (eigenvalues[:k] - eigenvalues_prev[:k]).abs()
                if (diff < tol * eigenvalues[:k].abs().clamp(min=1e-10)).all():
                    break
            eigenvalues_prev = eigenvalues.clone()
            
            # Expand subspace with residual
            if iteration < maxiter - 1:
                # Compute residual: R = AX - X @ diag(eigenvalues)
                AX_new = torch.zeros_like(X)
                for j in range(X.shape[1]):
                    AX_new[:, j] = self._distributed_matvec(X[:, j])
                
                residual = AX_new - X * eigenvalues.unsqueeze(0)
                
                # Orthogonalize and expand
                combined = torch.cat([X[:, :k], residual[:, :k]], dim=1)
                X, _ = torch.linalg.qr(combined)
                
                # Pad if needed
                if X.size(1) < m:
                    extra = torch.randn(N, m - X.size(1), dtype=dtype, device=device)
                    X = torch.cat([X, extra], dim=1)
                    X, _ = torch.linalg.qr(X)
        
        return eigenvalues[:k], X[:, :k]
    
    def eigsh(
        self,
        k: int = 6,
        which: str = "LM",
        sigma: Optional[float] = None,
        return_eigenvectors: bool = True,
        maxiter: int = 1000,
        tol: float = 1e-8
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute k eigenvalues for symmetric matrices using distributed LOBPCG.
        
        This is a TRUE distributed algorithm - no data gather required.
        Uses distributed matvec with global QR decomposition.
        
        Parameters
        ----------
        k : int, optional
            Number of eigenvalues to compute. Default: 6.
        which : {"LM", "SM", "LA", "SA"}, optional
            Which eigenvalues to find:
            - "LM"/"LA": Largest (default)
            - "SM"/"SA": Smallest
        sigma : float, optional
            Find eigenvalues near sigma (not yet supported).
        return_eigenvectors : bool, optional
            Whether to return eigenvectors. Default: True.
        maxiter : int, optional
            Maximum LOBPCG iterations. Default: 1000.
        tol : float, optional
            Convergence tolerance. Default: 1e-8.
            
        Returns
        -------
        eigenvalues : torch.Tensor
            Shape [k].
        eigenvectors : torch.Tensor or None
            Shape [N, k] if return_eigenvectors is True.
        
        Notes
        -----
        **Distributed Algorithm:**
        
        - Uses distributed LOBPCG (Locally Optimal Block PCG)
        - Only requires distributed matvec + global reductions
        - Memory: O(N * k) per node for eigenvectors
        - Communication: O(k^2) per iteration for Rayleigh-Ritz
        
        **Gradient Support:**
        
        - Gradients flow through the distributed matvec operations
        - O(iterations) graph nodes (not O(1) like adjoint)
        """
        if sigma is not None:
            warnings.warn("sigma (shift-invert) not yet supported for distributed eigsh. Ignoring.")
        
        largest = which in ('LM', 'LA')
        eigenvalues, eigenvectors = self._distributed_lobpcg(k, largest=largest, maxiter=maxiter, tol=tol)
        
        if return_eigenvectors:
            return eigenvalues, eigenvectors
        return eigenvalues, None
    
    def eigs(
        self,
        k: int = 6,
        which: str = "LM",
        sigma: Optional[float] = None,
        return_eigenvectors: bool = True,
        maxiter: int = 1000,
        tol: float = 1e-8
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute k eigenvalues using distributed LOBPCG.
        
        For symmetric matrices, equivalent to eigsh().
        For non-symmetric, currently falls back to eigsh() (symmetric assumption).
        
        Parameters
        ----------
        k : int, optional
            Number of eigenvalues to compute. Default: 6.
        which : str, optional
            Which eigenvalues to find.
        sigma : float, optional
            Find eigenvalues near sigma.
        return_eigenvectors : bool, optional
            Whether to return eigenvectors. Default: True.
        maxiter : int, optional
            Maximum iterations. Default: 1000.
        tol : float, optional
            Convergence tolerance. Default: 1e-8.
            
        Returns
        -------
        eigenvalues : torch.Tensor
            Shape [k].
        eigenvectors : torch.Tensor or None
            Shape [N, k] if return_eigenvectors is True.
        """
        # For now, use eigsh (assumes symmetric)
        # TODO: Implement Arnoldi for non-symmetric
        return self.eigsh(k=k, which=which, sigma=sigma, 
                         return_eigenvectors=return_eigenvectors,
                         maxiter=maxiter, tol=tol)
    
    def svd(
        self, 
        k: int = 6,
        maxiter: int = 1000,
        tol: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute truncated SVD using distributed power iteration.
        
        Uses A^T @ A for eigenvalues, then recovers U from A @ V.
        
        Parameters
        ----------
        k : int, optional
            Number of singular values to compute. Default: 6.
        maxiter : int, optional
            Maximum iterations. Default: 1000.
        tol : float, optional
            Convergence tolerance. Default: 1e-8.
            
        Returns
        -------
        U : torch.Tensor
            Left singular vectors. Shape [M, k].
        S : torch.Tensor
            Singular values. Shape [k].
        Vt : torch.Tensor
            Right singular vectors. Shape [k, N].
        
        Notes
        -----
        **Distributed Algorithm:**
        
        - Computes eigenvalues of A^T @ A using distributed LOBPCG
        - No data gather required
        """
        M, N = self._shape
        dtype = self._values.dtype
        device = self._device
        
        # For SVD, we need A^T @ A which requires transpose
        # Create A^T as a DSparseTensor
        A_T = self.T()
        
        # Power iteration for A^T @ A
        # Initialize random vectors
        V = torch.randn(N, k, dtype=dtype, device=device)
        V, _ = torch.linalg.qr(V)
        
        for iteration in range(maxiter):
            # AV = A @ V
            AV = torch.zeros(M, k, dtype=dtype, device=device)
            for j in range(k):
                AV[:, j] = self._distributed_matvec(V[:, j])
            
            # AtAV = A^T @ (A @ V)
            AtAV = torch.zeros(N, k, dtype=dtype, device=device)
            for j in range(k):
                AtAV[:, j] = A_T._distributed_matvec(AV[:, j])
            
            # QR decomposition
            V_new, R = torch.linalg.qr(AtAV)
            
            # Check convergence
            diff = (V_new - V).norm()
            V = V_new
            
            if diff < tol:
                break
        
        # Compute singular values and U
        # AV = A @ V, then normalize to get U
        AV = torch.zeros(M, k, dtype=dtype, device=device)
        for j in range(k):
            AV[:, j] = self._distributed_matvec(V[:, j])
        
        # S = ||AV[:, j]||
        S = AV.norm(dim=0)
        
        # U = AV / S
        U = AV / S.unsqueeze(0).clamp(min=1e-10)
        
        return U, S, V.T
    
    def norm(self, ord: Literal['fro', 1, 2] = 'fro') -> torch.Tensor:
        """
        Compute matrix norm (distributed).
        
        For Frobenius norm, computed locally and aggregated.
        For spectral norm, uses distributed SVD.
        
        Parameters
        ----------
        ord : {'fro', 1, 2}
            Type of norm:
            - 'fro': Frobenius norm (distributed sum)
            - 1: Maximum column sum
            - 2: Spectral norm (largest singular value via distributed SVD)
            
        Returns
        -------
        torch.Tensor
            Scalar tensor containing the norm value.
        """
        if ord == 'fro':
            # Frobenius norm: sqrt(sum(values^2))
            # This is truly distributed - each partition has its own values
            return torch.sqrt((self._values ** 2).sum())
        elif ord == 2:
            # Spectral norm: largest singular value
            _, S, _ = self.svd(k=1, maxiter=100)
            return S[0]
        elif ord == 1:
            # Maximum column sum - need to gather
            warnings.warn("1-norm requires data gather. Using to_sparse_tensor().")
            return self.to_sparse_tensor().norm(ord=1)
        else:
            raise ValueError(f"Unknown norm order: {ord}")
    
    def condition_number(self, ord: int = 2) -> torch.Tensor:
        """
        Estimate condition number using distributed SVD.
        
        Parameters
        ----------
        ord : int, optional
            Norm order. Default: 2 (spectral).
            
        Returns
        -------
        torch.Tensor
            Condition number estimate (σ_max / σ_min).
        """
        if ord == 2:
            # Need largest and smallest singular values
            # Compute k=6 singular values
            _, S, _ = self.svd(k=6, maxiter=200)
            return S[0] / S[-1].clamp(min=1e-10)
        else:
            warnings.warn(f"ord={ord} requires data gather. Using to_sparse_tensor().")
            return self.to_sparse_tensor().condition_number(ord=ord)
    
    def T(self) -> "DSparseTensor":
        """
        Transpose the distributed sparse tensor.
        
        Returns a new DSparseTensor with swapped row/column indices.
        
        Returns
        -------
        DSparseTensor
            Transposed matrix.
        """
        # Swap row and column indices
        return DSparseTensor(
            self._values,
            self._col_indices,  # swap
            self._row_indices,  # swap
            (self._shape[1], self._shape[0]),
            num_partitions=self._num_partitions,
            coords=self._coords,
            partition_method=self._partition_method,
            device=self._device,
            verbose=False
        )
    
    # =========================================================================
    # Methods that require data gather (with warnings)
    # =========================================================================
    
    def to_dense(self) -> torch.Tensor:
        """
        Convert to dense tensor.
        
        WARNING: This gathers all data to a single node.
        Only use for small matrices or debugging.
        
        Returns
        -------
        torch.Tensor
            Dense matrix of shape (M, N).
        """
        warnings.warn("to_dense() gathers all data to a single node. "
                     "Only use for debugging or small matrices.")
        return self.to_sparse_tensor().to_dense()
    
    def is_symmetric(self, atol: float = 1e-8, rtol: float = 1e-5) -> torch.Tensor:
        """
        Check if matrix is symmetric.
        
        Can be done distributedly by comparing values with transpose.
        
        Parameters
        ----------
        atol : float
            Absolute tolerance for symmetry check.
        rtol : float
            Relative tolerance for symmetry check.
            
        Returns
        -------
        torch.Tensor
            Boolean scalar tensor.
        """
        # This can be done without gather by checking local values
        # For now, use simple implementation
        return self.to_sparse_tensor().is_symmetric(atol=atol, rtol=rtol)
    
    def is_positive_definite(self) -> torch.Tensor:
        """
        Check if matrix is positive definite.
        
        Uses distributed eigenvalue computation.
        
        Returns
        -------
        torch.Tensor
            Boolean scalar tensor.
        """
        # Check smallest eigenvalue > 0
        eigenvalues, _ = self.eigsh(k=1, which='SA', return_eigenvectors=False, maxiter=200)
        return eigenvalues[0] > 0
    
    def lu(self):
        """
        Compute LU decomposition.
        
        WARNING: LU is inherently not distributed-friendly.
        This gathers data to a single node.
        
        For distributed solves, use solve_distributed() with iterative methods.
        
        Returns
        -------
        LUFactorization
            Factorization object with solve() method.
        """
        warnings.warn("LU decomposition is not distributed. "
                     "Use solve_distributed() for distributed solves.")
        return self.to_sparse_tensor().lu()
    
    def spy(self, **kwargs):
        """
        Visualize sparsity pattern.
        
        Gathers data for visualization.
        
        Parameters
        ----------
        **kwargs
            Arguments passed to SparseTensor.spy().
        """
        return self.to_sparse_tensor().spy(**kwargs)
    
    def nonlinear_solve(
        self,
        residual_fn,
        u0: torch.Tensor,
        *params,
        method: str = 'newton',
        tol: float = 1e-6,
        atol: float = 1e-10,
        max_iter: int = 50,
        line_search: bool = True,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Solve nonlinear equation F(u, D, *params) = 0 using distributed Newton-Krylov.
        
        Uses Jacobian-free Newton-Krylov with distributed CG for linear solves.
        
        Parameters
        ----------
        residual_fn : callable
            Function F(u, D, *params) -> residual tensor.
            D is this DSparseTensor.
        u0 : torch.Tensor
            Initial guess (global vector).
        *params : torch.Tensor
            Additional parameters.
        method : str
            'newton': Newton-Krylov with distributed CG
            'picard': Fixed-point iteration
        tol : float
            Relative tolerance.
        atol : float
            Absolute tolerance.
        max_iter : int
            Maximum outer iterations.
        line_search : bool
            Use Armijo line search.
        verbose : bool
            Print convergence info.
            
        Returns
        -------
        torch.Tensor
            Solution u such that F(u, D, *params) ≈ 0.
        
        Notes
        -----
        **Distributed Algorithm:**
        
        - Uses Jacobian-free Newton-Krylov (JFNK)
        - Linear solves use distributed CG
        - Jacobian-vector products computed via finite differences
        """
        u = u0.clone()
        N = u.shape[0]
        dtype = u.dtype
        device = u.device
        
        for outer_iter in range(max_iter):
            # Compute residual
            F = residual_fn(u, self, *params)
            F_norm = F.norm()
            
            if verbose:
                print(f"  Newton iter {outer_iter}: ||F|| = {F_norm:.2e}")
            
            if F_norm < atol:
                if verbose:
                    print(f"  Converged (atol) at iteration {outer_iter}")
                break
            
            if outer_iter > 0 and F_norm < tol * F_norm_init:
                if verbose:
                    print(f"  Converged (rtol) at iteration {outer_iter}")
                break
            
            if outer_iter == 0:
                F_norm_init = F_norm
            
            if method == 'picard':
                # Simple fixed-point: u = u - F (assuming F = Au - b form)
                u = u - F
            else:
                # Newton-Krylov: solve J @ du = -F using CG with Jacobian-vector products
                # J @ v ≈ (F(u + eps*v) - F(u)) / eps
                eps = 1e-7 * max(u.norm(), 1.0)
                
                def matvec(v):
                    """Jacobian-vector product via finite differences."""
                    F_plus = residual_fn(u + eps * v, self, *params)
                    return (F_plus - F) / eps
                
                # Distributed CG for J @ du = -F
                du = torch.zeros_like(u)
                r = -F - matvec(du)  # r = -F - J @ 0 = -F
                p = r.clone()
                rs_old = torch.dot(r, r)
                
                for cg_iter in range(min(100, N)):
                    Jp = matvec(p)
                    pJp = torch.dot(p, Jp)
                    
                    if pJp.abs() < 1e-30:
                        break
                    
                    alpha = rs_old / pJp
                    du = du + alpha * p
                    r = r - alpha * Jp
                    rs_new = torch.dot(r, r)
                    
                    if rs_new.sqrt() < 1e-10:
                        break
                    
                    beta = rs_new / rs_old
                    p = r + beta * p
                    rs_old = rs_new
                
                # Line search
                if line_search:
                    alpha = 1.0
                    F_new_norm = residual_fn(u + alpha * du, self, *params).norm()
                    while F_new_norm > F_norm and alpha > 1e-8:
                        alpha *= 0.5
                        F_new_norm = residual_fn(u + alpha * du, self, *params).norm()
                    u = u + alpha * du
                else:
                    u = u + du
        
        return u
    
    # =========================================================================
    # Matrix Operations
    # =========================================================================
    
    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Distributed matrix-vector multiplication: y = D @ x
        
        Automatically handles scatter, distributed matvec, and gather.
        Supports gradient computation when values have requires_grad=True.
        
        Parameters
        ----------
        x : torch.Tensor
            Global vector of shape (N,) where N = shape[1]
            
        Returns
        -------
        torch.Tensor
            Global result vector of shape (M,) where M = shape[0]
            
        Example
        -------
        >>> D = A.partition(num_partitions=4)
        >>> y = D @ x  # Equivalent to A @ x
        
        Notes
        -----
        **Gradient Support:**
        
        For single-node simulation with gradient support, uses global COO matvec.
        For true MPI distributed execution without gradients, uses partition-based matvec.
        """
        return self._distributed_matvec(x)
    
    # =========================================================================
    # Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        return (f"DSparseTensor(shape={self._shape}, num_partitions={self._num_partitions}, "
                f"nnz={self.nnz}, device={self._device})")
    
    # =========================================================================
    # Persistence (I/O)
    # =========================================================================
    
    def save(
        self,
        directory: Union[str, "os.PathLike"],
        verbose: bool = False
    ) -> None:
        """
        Save DSparseTensor to disk.
        
        Creates a directory with metadata and per-partition files.
        
        Parameters
        ----------
        directory : str or PathLike
            Output directory.
        verbose : bool
            Print progress.
        
        Example
        -------
        >>> D = A.partition(num_partitions=4)
        >>> D.save("matrix_dist")
        """
        from .io import save_dsparse
        save_dsparse(self, directory, verbose)
    
    @classmethod
    def load(
        cls,
        directory: Union[str, "os.PathLike"],
        device: Union[str, torch.device] = "cpu"
    ) -> "DSparseTensor":
        """
        Load a complete DSparseTensor from disk.
        
        Parameters
        ----------
        directory : str or PathLike
            Directory containing saved data.
        device : str or torch.device
            Device to load to.
        
        Returns
        -------
        DSparseTensor
            The loaded distributed sparse tensor.
        
        Example
        -------
        >>> D = DSparseTensor.load("matrix_dist", device="cuda")
        """
        from .io import load_dsparse
        return load_dsparse(directory, device)
