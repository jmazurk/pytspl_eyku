# Cell2Vec

Cell2Vec learns vector embeddings for the cells of a simplicial or cell
complex. Each cell of a chosen rank (nodes, edges, or polygons) is mapped to a
dense vector such that cells appearing in similar structural contexts receive
similar vectors.

The API follows the design of TopoEmbedX's Cell2Vec, using PyTSPL's
simplicial and cell complex representations. The biased random walk follows
Node2Vec.

The method is as follows:

1. Build a neighborhood graph over the cells of the chosen rank.
2. Generate second-order biased random walks over that graph.
3. Train a Word2Vec model on the walks.
4. Return one vector per cell.

## Basic usage

```python
from pytspl import load_dataset
from pytspl.embeddings.cell2vec import Cell2Vec

sc, coordinates, flow = load_dataset("paper", only_sc=True)

model = Cell2Vec(
    walk_number=10,
    walk_length=80,
    p=1.0,
    q=1.0,
    dimensions=128,
    window_size=5,
    epochs=5,
    workers=1,
    seed=42,
)
model.fit(sc, rank=1, neighborhood_type="adj")

embeddings = model.get_embedding()  # numpy array (num_edges, 128)
embeddings_by_cell = model.get_embedding(as_dict=True)  # {edge: vector}
```

## Constructor parameters

| Parameter | Meaning                                                    |
|-----------|------------------------------------------------------------|
| `walk_number` | Number of random walks started from each cell.             |
| `walk_length` | Number of steps in each walk.                              |
| `p` | Return parameter. Low `p` biases the walk toward backtracking. |
| `q` | In-out parameter. Low `q` biases the walk outward (exploration). |
| `dimensions` | Length of each output embedding vector.                    |
| `window_size` | Skip-gram context window size.                             |
| `epochs` | Number of Word2Vec training epochs.                        |
| `workers` | Worker threads. Use `workers=1` for reproducible results.  |
| `seed` | Random seed for walks and Word2Vec.                        |

Setting `p = q = 1` recovers an unbiased (first-order) random walk.

### fit(topology, rank=1, neighborhood_type="adj")

Trains the model on a complex.

- `topology`: a `SimplicialComplex` or `CellComplex`.
- `rank`: which cells to embed — `0` nodes, `1` edges, `2` polygons.
- `neighborhood_type`: how two cells are connected in the walk graph:
  - `"adj"` (adjacency): two cells are neighbors if they share a lower-dimensional cell.
  - `"coadj"` (coadjacency): two cells are neighbors if they share a higher-dimensional cell.

The adjacency/coadjacency distinction only affects rank-1 cells (edges). Ranks 0 and 2 have a single natural neighbor relation.

### get_embedding(as_dict=False)

Returns the learned embeddings.

- `as_dict=False`: a numpy array of shape `(num_cells, dimensions)`. Row `i` corresponds to the `i`-th cell.
- `as_dict=True`: a dictionary mapping each original cell (e.g. the edge `(0, 1)`) to its vector.

### get_params()

Returns the hyperparameters the model was configured with as a dictionary.

## Notes and limitations

- **Supported ranks.** Embedding is supported for ranks 0, 1, and 2, matching
  the current `CellComplex` dimension limit. Extending to higher ranks requires
  higher-dimensional complex support.
- **Isolated cells.** A cell with no neighbors under the chosen
  `neighborhood_type` cannot be walked from. Such cells are detected explicitly
  and assigned a zero vector. The output always has one row of length
  `dimensions` per cell.
- **Reproducibility.** With `workers=1` and a fixed `seed`, walks and embeddings
  are reproducible across runs.
- **Coadjacency sparsity.** On complexes with few faces, the coadjacency graph
  can be sparse, producing many isolated cells. This is expected and is not an error.