# Distributed Sparse Tensor Examples

True distributed sparse linear algebra with `DSparseMatrix`.

## Quick Start

```bash
# Run all examples with 4 processes
./launch.sh all 4

# Or individual examples
torchrun --standalone --nproc_per_node=4 distributed_matvec.py
torchrun --standalone --nproc_per_node=4 distributed_solve.py
torchrun --standalone --nproc_per_node=4 distributed_eigsh.py
```

## Examples

| File | Operation | API |
|------|-----------|-----|
| `distributed_matvec.py` | y = A @ x | `A.matvec(x)` |
| `distributed_solve.py` | Ax = b | `A.solve(b)` |
| `distributed_eigsh.py` | Av = λv | `A.eigsh(k)` |

## API

```python
from torch_sla.distributed import DSparseMatrix, partition_simple

# Each rank creates its local partition
A = DSparseMatrix.from_global(
    values, row, col, (n, n),
    num_partitions=world_size,
    my_partition=rank,
    partition_ids=partition_simple(n, world_size)
)

# Distributed operations (default: distributed=True)
y = A.matvec(x, exchange_halo=True)  # Matvec with halo exchange
x = A.solve(b)                        # Distributed CG
λ, V = A.eigsh(k=5)                   # Distributed LOBPCG

# Local subdomain solve (no global communication)
x = A.solve(b, distributed=False)
```

## Architecture

```
Global Matrix A (n×n) partitioned across P ranks:

Rank 0: [owned_0 | halo_0]  ← neighbors: [1]
Rank 1: [owned_1 | halo_1]  ← neighbors: [0, 2]
Rank 2: [owned_2 | halo_2]  ← neighbors: [1, 3]
Rank 3: [owned_3 | halo_3]  ← neighbors: [2]
```

Each rank only stores its partition. Communication:
- **Halo exchange**: Point-to-point with neighbors
- **Global reductions**: `all_reduce` for dot products
