import numpy as np
from pytspl.cell_complex import CellComplex
from scipy.sparse import csr_matrix, issparse
from gensim.models import Word2Vec

class Cell2Vec:
    def __init__(
            self,
            walk_number = 10,
            walk_length = 80,
            p=1.0,
            q=1.0,
            dimensions=128,
            window_size=5,
            epochs=5,
            workers=1,
            seed=42,
    ):
        self.walk_number = walk_number
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.dimensions = dimensions
        self.window_size = window_size
        self.epochs = epochs
        self.workers = workers
        self.seed = seed

        self._embeddings = None
        self._index_to_cell = None


    def fit(self, topology, rank=1, neighborhood_type="adj"):
        neighbors = _build_neighbors(topology, k=rank, neighborhood_type=neighborhood_type)
        N = len(neighbors)

        isolated = [i for i in range(N) if len(neighbors[i]) == 0]

        self._index_to_cell = _get_cells(topology, rank)

        if self.seed is not None:
            np.random.seed(self.seed)

        walks = []

        for _ in range(self.walk_number):
            for start in range(N):
                walks.append(
                    _walk_pq(start, self.walk_length, neighbors, self.p, self.q)
                )

        walks_str = [[str(c) for c in w] for w in walks]
        model = Word2Vec(
            sentences=walks_str,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=self.workers,
            epochs=self.epochs,
            seed=self.seed,
        )

        emb = np.zeros((N, self.dimensions))
        for i in range(N):
            key = str(i)
            if key in model.wv:
                emb[i] = model.wv[key]

        for i in isolated:
            emb[i] = np.zeros(self.dimensions)

        self._embedding = emb

        return self


    def get_embedding(self, as_dict=False):
        if self._embedding is None:
            raise ValueError("Call fit() before get_embedding().")
        if as_dict:
            return {
                self._index_to_cell[i]: self._embedding[i]
                for i in range(len(self._index_to_cell))
            }
        return self._embedding


    def get_params(self):
        return {
            "walk_number": self.walk_number,
            "walk_length": self.walk_length,
            "p": self.p,
            "q": self.q,
            "dimensions": self.dimensions,
            "window_size": self.window_size,
            "epochs": self.epochs,
            "workers": self.workers,
            "seed": self.seed,
        }

def _walk_pq(start, walk_length, neighbors, p, q):
    rw = [start]
    prev = None
    current = start
    for _ in range(walk_length):
        nxt = _pick_next(prev, current, neighbors, p, q)
        rw.append(nxt)
        prev = current
        current = nxt
    return rw


def _pick_next(prev, current, neighbors, p, q):
    neigh = list(neighbors[current])

    if len(neigh) == 0:
        return current

    if prev is None:
        return np.random.choice(neigh)

    weights = []

    for n in neigh:
        if n == prev:
            weights.append(1.0 / p)
        elif n in neighbors[prev]:
            weights.append(1.0)
        else:
            weights.append(1.0 / q)

    weights = np.array(weights)
    weights = weights / weights.sum()
    return np.random.choice(neigh, p=weights)


def _build_neighbors(
            cc: CellComplex,
            k: int,
            multicount: bool = False,
            neighborhood_type : str = "adj",
):

    if k < 0 or k > 2:
        raise ValueError(
            "k must be in {0, 1, 2} for this CellComplex implementation.")

    if k == 0:
        n = len(cc.nodes)
        if n == 0:
            return {}

        A = np.zeros((n, n), dtype=float)
        for (u, v) in cc.edges:
            A[u, v] += 1.0
            A[v, u] += 1.0

        if not multicount:
            A = (A > 0).astype(float)

        np.fill_diagonal(A, 0.0)
        A = csr_matrix(A)

    elif k == 1:
        m = len(cc.edges)
        if m == 0:
            return {}


        B1 = np.abs(cc.B1)
        B2 = np.abs(cc.B2)


        A_down = B1.T @ B1  # shape (m, m)

        if B2.size > 0:
            A_up = B2 @ B2.T
        else:
            A_up = np.zeros((m, m))

        if neighborhood_type == "adj":
            A = A_down
        elif neighborhood_type == "coadj":
            A = A_up
        else:
            raise ValueError("neighborhood_type must be 'adj' or 'coadj'")

        if issparse(A):
            A = A.tolil()
            A.setdiag(0.0)
            A = A.tocsr()
        else:
            np.fill_diagonal(A, 0.0)

        if not multicount:
            # Binarize if we don't want multiplicity
            if issparse(A):
                A = (A > 0).astype(float).tocsr()
            else:
                A = (A > 0).astype(float)

        A = csr_matrix(A)

    else:
        if hasattr(cc, "polygons"):
            two_cells = cc.polygons
        else:
            two_cells = cc.triangles

        p = len(two_cells)
        if p == 0:
            return {}

        B2 = np.abs(cc.B2)  # shape: (num_edges, num_polygons)

        # lower adjacency for polygons: polygons sharing an edge
        A_down = B2.T @ B2

        # no upper adjacency (no 3-cells)
        A = A_down

        # Remove diagonal
        if issparse(A):
            A = A.tolil()
            A.setdiag(0.0)
            A = A.tocsr()
        else:
            np.fill_diagonal(A, 0.0)

        if not multicount:
            if issparse(A):
                A = (A > 0).astype(float).tocsr()
            else:
                A = (A > 0).astype(float)

        A = csr_matrix(A)

    A = A.tocoo()
    neighbors = {i: set() for i in range(A.shape[0])}
    for i, j in zip(A.row, A.col):
        neighbors[i].add(j)

    return neighbors




def _get_cells(topology, rank):
    if rank == 0:
        return list(topology.nodes)
    elif rank == 1:
        return list(topology.edges)
    elif rank == 2:
        if hasattr(topology, "polygons"):
            return list(topology.polygons)
        else:
            return list(topology.triangles)
    else:
        raise ValueError("rank must be 0, 1, or 2")

