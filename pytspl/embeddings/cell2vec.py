import numpy as np
from pytspl.cell_complex import CellComplex
from scipy.sparse import csr_matrix, issparse
from gensim.models import Word2Vec

class Cell2Vec:
    """Cell2Vec embedding class for simplicial or cell complexes.

    Generates biased random walks over a neighborhood graph and trains a Word2Vec model on them.
    """

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
        """Initialize a Cell2Vec object.
        Args:
            walk_number (int, optional): Number of walks started from each cell. Defaults to 10.
            walk_length (int, optional): Steps per walk. Defaults to 80.
            p (float, optional): Return parameter. Defaults to 1.0 for unbiased walk.
            q (float, optional): In-out parameter. Defaults to 1.0 for unbiased walk.
            dimensions (int, optional): Embedding vector length. Defaults to 128.
            window_size (int, optional): Context window size. Defaults to 5.
            epochs (int, optional): Word2Vec training epochs. Defaults to 5.
            workers (int, optional): Worker threads. Defaults to 1 for reproducibility.
            seed (int, optional): Random seed for walks and Word2Vec. Defaults to 42.
        """
        self.walk_number = walk_number
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.dimensions = dimensions
        self.window_size = window_size
        self.epochs = epochs
        self.workers = workers
        self.seed = seed

        self._embedding = None
        self._index_to_cell = None


    def fit(self, topology, rank=1, neighborhood_type="adj"):
        """Learn embeddings for the cells of a complex.

        Args:
            topology: Simplicial or Cell complex.
            rank (int, optional): Rank of embeddings. Defaults to 1.
            neighborhood_type (str, optional): Neighborhood type. Defaults to "adj".

        Returns:
            Cell2Vec: self.

        """
        neighbors = _build_neighbors(topology, k=rank, neighborhood_type=neighborhood_type)
        N = len(neighbors)

        isolated = [i for i in range(N) if len(neighbors[i]) == 0] # cells with no neighbours

        self._index_to_cell = _get_cells(topology, rank) # index to cell mapping

        if self.seed is not None:
            np.random.seed(self.seed)

        walks = []

        # create walks
        for _ in range(self.walk_number):
            for start in range(N):
                walks.append(
                    _walk_pq(start, self.walk_length, neighbors, self.p, self.q)
                )

        # train Word2Vec on the walks
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

        # put resulting vectors into embedding matrix
        emb = np.zeros((N, self.dimensions))
        for i in range(N):
            key = str(i)
            if key in model.wv:
                emb[i] = model.wv[key]

        for i in isolated:
            emb[i] = np.zeros(self.dimensions) # isolated cells will always have zero vector

        self._embedding = emb

        return self


    def get_embedding(self, as_dict=False):
        """Return the learned embeddings.

            Args:
                as_dict (bool, optional): If True, return a dict mapping each original cell.
                If False, return a numpy array where row i is the i-th cell's vector.Defaults to False.

            Returns:
                np.ndarray or dict: Embeddings as an array of shape (n_cells, dimensions), or as a {cell: vector} dict.

            Raises:
                ValueError: If called before fit().
        """
        if self._embedding is None:
            raise ValueError("Call fit() before get_embedding().")
        if as_dict:
            return {
                self._index_to_cell[i]: self._embedding[i]
                for i in range(len(self._index_to_cell))
            }
        return self._embedding


    def get_params(self):
        """Return the hyperparameters the model was configured with.

        Returns:
            dict: The constructor hyperparameters keyed by name.
        """
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
    """Run a biased random walk.

    Args:
        start (int): The cell to start the walk from.
        walk_length (int): The number of steps in walk.
        neighbors (dict): Maps each cell to its set of neighbours.
        p (float): Return parameter for unbiased walk, controls probability of going back.
        q (float): In-out parameter for unbiased walk, controls probability of exploring.

    Returns:
        list: The sequence of visited cells. Its length is (walk_length + 1).
    """
    rw = [start]
    prev = None # no previous cell on step 1.
    current = start
    for _ in range(walk_length):
        nxt = _pick_next(prev, current, neighbors, p, q)
        rw.append(nxt)
        prev = current
        current = nxt
    return rw


def _pick_next(prev, current, neighbors, p, q):
    """Pick the next cell in a biased random walk.

    Args:
        prev (int or None): Cell visited one step ago, None on the first step.
        current (int): The current cell on the walk.
        neighbors (dict): Maps each cell to its set of neighbour cells.
        p (float): Return parameter, controls probability of going back.
        q (float): In-out parameter, controls probability of exploring

    Returns:
        int: The next cell to visit.

    """
    neigh = list(neighbors[current])

    if len(neigh) == 0: # isolated cell, stay at current cell
        return current

    if prev is None: # first step has no bias
        return np.random.choice(neigh)

    weights = []

    for n in neigh:
        if n == prev:
            weights.append(1.0 / p) # going back
        elif n in neighbors[prev]:
            weights.append(1.0) # staying close
        else:
            weights.append(1.0 / q) # exploring

    weights = np.array(weights)
    weights = weights / weights.sum() # normalize to probabilities
    return np.random.choice(neigh, p=weights)


def _build_neighbors(
            cc: CellComplex,
            k: int,
            neighborhood_type : str = "adj",
):
    """
    Build the neighborhood dict for the k cells of the complex.
    Adjacency construction adapted from k-simplex2vec's _k_cell_adjacency method.
    Args:
        cc (CellComplex): Simplicial or cell complex.
        k (int): Rank of the cells to connect. 0 for nodes, 1 for edges, 2 for faces.
        neighborhood_type (str, optional): Neighborhood type. "adj" is for share a lower cell. "coadj" is for share a higher cell. Only affects k=1. Defaults to "adj".

    Returns:
        dict: Maps each cell index to a set of neighbour indices.
    """

    if k < 0 or k > 2:
        raise ValueError(
            "k must be in {0, 1, 2}.")

    if k == 0:
        n = len(cc.nodes)
        if n == 0:
            return {} # no nodes means no neighbours

        # nodes are neighbours if an edge connects them
        A = np.zeros((n, n), dtype=float)
        for (u, v) in cc.edges:
            A[u, v] += 1.0
            A[v, u] += 1.0

        A = (A > 0).astype(float) # 1 if neighbours, else 0

        np.fill_diagonal(A, 0.0) # no self connections
        A = csr_matrix(A)

    elif k == 1:
        m = len(cc.edges)
        if m == 0:
            return {} # no edges means no neighbours


        B1 = np.abs(cc.B1) # unsigned node-edge incidence
        B2 = np.abs(cc.B2) # unsigned edge-face incidence


        A_down = B1.T @ B1  # shared node count of edges

        if B2.size > 0:
            A_up = B2 @ B2.T # shared face count of edges
        else:
            A_up = np.zeros((m, m)) # no faces

        if neighborhood_type == "adj":
            A = A_down # adjacency, share a node
        elif neighborhood_type == "coadj":
            A = A_up # coadjacency, share a face
        else:
            raise ValueError("neighborhood_type must be 'adj' or 'coadj'")

        np.fill_diagonal(A, 0.0) # remove self connections
        A = (A > 0).astype(float) # 1 if neighbours, else 0

        A = csr_matrix(A)

    else:
        if hasattr(cc, "polygons"):
            two_cells = cc.polygons
        else:
            two_cells = cc.triangles

        p = len(two_cells)
        if p == 0:
            return {} # no faces means no neighbours

        B2 = np.abs(cc.B2)

        A_down = B2.T @ B2 # lower adjacency, faces sharing edges

        A = A_down # no upper adjacency

        np.fill_diagonal(A, 0.0) # Remove self connections

        A = (A > 0).astype(float) # 1 if neighbours, else 0

        A = csr_matrix(A)

    # convert the adjacency matrix into a neighbor dict for the walk
    A = A.tocoo() # coordinate format
    neighbors = {i: set() for i in range(A.shape[0])}
    for i, j in zip(A.row, A.col):
        neighbors[i].add(j)

    return neighbors




def _get_cells(topology, rank):
    """Provides the mapping from embedding row index back to the original cell, so results can be keyed by the actual cell.

    Args:
        topology (int): Simplicial or cell complex.
        rank (int): Rank of the cells to return, where nodes 0, edges 1, polygons 2.

    Returns:
        list: the cells of the given rank in index order.

    Raises:
        ValueError: If rank is not 0, 1, or 2.
    """

    if rank == 0:
        return list(topology.nodes)
    elif rank == 1:
        return list(topology.edges)
    elif rank == 2:
        # naming in cell complexes is polygons, in simplicial complexes its triangles.
        if hasattr(topology, "polygons"):
            return list(topology.polygons)
        else:
            return list(topology.triangles)
    else:
        raise ValueError("rank must be 0, 1, or 2")

