# CCXN

`CCXN` is a convolutional Cell Complex Neural Network. It runs on a cell or
simplicial complex, learning from signals defined on nodes, edges, and faces.
It follows the simplified CCXN design (Hajij et al., 2020), taking inspiration
from TopoModelX's implementation.

Each layer does two operations:

1. **Node-to-node** convolution using the rank-0 upper adjacency `A_up_0`.
2. **Edge-to-face** convolution using the transposed incidence matrix `B2^T`.

Attention is excluded in this version; it is a planned extension once the base
implementation is validated.

## Basic usage

```python
import torch
import numpy as np
from scipy.sparse import csr_matrix

from pytspl.simplicial_complex import SimplicialComplex
from pytspl.scnn.cell.ccxn import CCXN

# a small complex: a triangle (0,1,2) and an extra edge (2,3)
sc = SimplicialComplex(
    nodes=[0, 1, 2, 3],
    edges=[(0, 1), (0, 2), (1, 2), (2, 3)],
    triangles=[[0, 1, 2]],
)

# build the two operators as sparse torch tensors
def to_sparse(mat):
    mat = csr_matrix(mat).tocoo()
    idx = torch.tensor(np.vstack((mat.row, mat.col)), dtype=torch.long)
    val = torch.tensor(mat.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=mat.shape).coalesce()

B1 = np.abs(sc.B1)
A = (B1 @ B1.T)
np.fill_diagonal(A, 0.0)
adjacency_0 = to_sparse((A > 0).astype(float))  # A_up_0
incidence_2_t = to_sparse(sc.B2.T)  # B2^T

# input features
x_0 = torch.randn(len(sc.nodes), 3)
x_1 = torch.randn(len(sc.edges), 3)

model = CCXN(
    in_ch_0=3,  # node feature size
    in_ch_1=3,  # edge feature size
    in_ch_2=5,  # face feature size
    hidden=16,  # hidden feature size
    num_classes=4,  # whole-complex output size
    n_layers=2,
)

pred = model(x_0, x_1, adjacency_0, incidence_2_t)  # whole-complex prediction
x0, x1, x2 = model(x_0, x_1, adjacency_0, incidence_2_t, readout=False)  # per-cell signals
```

## The layer

`CCXNLayer` performs one round of the two operations:

- **Nodes** are updated by combining a neighbor contribution (mixing each node
  with its adjacency neighbors) and a self contribution (the node's own
  features), followed by a nonlinear activation.
- **Edges** are resized by a linear map into the working space. In this
  simplified version edges are resized rather than convolved.
- **Faces** accumulate the contribution gathered from their bordering edges
  (via `B2^T`), added to their existing features, followed by an activation.
  Faces thread through the layers rather than being recreated each layer.

## The model

`CCXN` stacks several `CCXNLayer`s. The first layer maps the input feature sizes
into a hidden space; later layers work entirely in that hidden space. The model
runs in two modes, chosen by the `readout` argument:

- `readout=True` (default): nodes are mean-pooled into one vector and passed
  through a linear layer to produce a single whole-complex prediction.
- `readout=False`: the