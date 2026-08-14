import numpy as np
import pytest

from pytspl.simplicial_complex import SimplicialComplex
from pytspl.cell_complex import CellComplex
from pytspl.embeddings.cell2vec import (
    Cell2Vec,
    _build_neighbors,
    _pick_next
)


# tiny hand-built complex to use in the tests

@pytest.fixture
def tiny_sc():
    """A triangle (0,1,2) plus a connected separate edge (2,3)."""
    return SimplicialComplex(
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (0, 2), (1, 2), (2, 3)],
        triangles=[[0, 1, 2]],
    )


# adjacency vs coadjacency tests

def test_adj_neighbors_exact(tiny_sc):
    nbrs = _build_neighbors(tiny_sc, k=1, neighborhood_type="adj")
    assert nbrs[0] == {1, 2}
    assert nbrs[1] == {0, 2, 3}
    assert nbrs[2] == {0, 1, 3}
    assert nbrs[3] == {1, 2}


def test_coadj_neighbors_exact(tiny_sc):
    nbrs = _build_neighbors(tiny_sc, k=1, neighborhood_type="coadj")
    assert nbrs[0] == {1, 2}
    assert nbrs[1] == {0, 2}
    assert nbrs[2] == {0, 1}
    assert nbrs[3] == set()

# embedding shape tests for ranks 0,1,2

@pytest.mark.parametrize("rank", [0, 1, 2])
def test_embedding_shape(tiny_sc, rank):
    model = Cell2Vec(walk_number=5, walk_length=10, dimensions=8, seed=42)
    model.fit(tiny_sc, rank=rank)
    emb = model.get_embedding()

    expected_rows = tiny_sc.shape[rank]
    assert emb.shape == (expected_rows, 8)

# handling isolated cells

@pytest.fixture
def isolated_sc():
    """Two disconnected edges: (0,1) and (2,3).
    Under adjacency, neither edge shares a node with the other,
    so both are isolated at rank 1.
    """
    return SimplicialComplex(
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (2, 3)],
    )


def test_isolated_cells_get_zeros(isolated_sc):
    model = Cell2Vec(walk_number=5, walk_length=10, dimensions=8, seed=42)
    model.fit(isolated_sc, rank=1, neighborhood_type="adj")
    emb = model.get_embedding()

    # both edges are isolated so both rows should be all zeros
    assert np.allclose(emb[0], 0.0)
    assert np.allclose(emb[1], 0.0)

# reproducle with fixed seed

def test_reproducible_with_seed(tiny_sc):
    m1 = Cell2Vec(walk_number=5, walk_length=10, dimensions=8, seed=42, workers=1)
    m1.fit(tiny_sc, rank=1)
    e1 = m1.get_embedding()

    m2 = Cell2Vec(walk_number=5, walk_length=10, dimensions=8, seed=42, workers=1)
    m2.fit(tiny_sc, rank=1)
    e2 = m2.get_embedding()

    assert np.allclose(e1, e2)

# doesnt mutate original complex

def test_fit_does_not_modify_complex(tiny_sc):
    edges_before = list(tiny_sc.edges)
    nodes_before = list(tiny_sc.nodes)
    b1_before = tiny_sc.B1.copy()

    model = Cell2Vec(walk_number=5, walk_length=10, dimensions=8, seed=42)
    model.fit(tiny_sc, rank=1)

    assert list(tiny_sc.edges) == edges_before
    assert list(tiny_sc.nodes) == nodes_before
    assert np.allclose(tiny_sc.B1, b1_before)

# p = q = 1 is unbiased

def test_pq_equal_one_is_unbiased():
    neighbors = {
        0: {1},
        1: {0, 2, 3},
        2: {1},
        3: {1},
    }
    np.random.seed(0)
    counts = {0: 0, 2: 0, 3: 0}
    trials = 6000
    for _ in range(trials):
        nxt = _pick_next(prev=0, current=1, neighbors=neighbors, p=1.0, q=1.0)
        counts[nxt] += 1

    for c in counts.values():
        assert abs(c / trials - 1/3) < 0.05

# accepts cell complex too

@pytest.fixture
def tiny_cc():
    return CellComplex(
        nodes=[0, 1, 2, 3, 4],
        edges=[(0, 1), (1, 2), (2, 3), (0, 3), (3, 4)],
        polygons=[[0, 1, 2, 3]],
    )

def test_cell_complex_accepted(tiny_cc):
    model = Cell2Vec(walk_number=5, walk_length=10, dimensions=8, seed=42)
    model.fit(tiny_cc, rank=1)
    emb = model.get_embedding()

    assert emb.shape == (len(tiny_cc.edges), 8)

def test_stable_cell_to_row_indexing(tiny_sc):
    model = Cell2Vec(walk_number=5, walk_length=10, dimensions=8, seed=42)
    model.fit(tiny_sc, rank=1)

    arr = model.get_embedding(as_dict=False)
    dct = model.get_embedding(as_dict=True)

    edges = list(tiny_sc.edges)

    assert len(dct) == len(edges)

    for i, edge in enumerate(edges):
        assert edge in dct
        assert np.allclose(dct[edge], arr[i])

def _cosine_sim_matrix(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb / norms
    return unit @ unit.T


def _nearest_neighbors(emb, k=2):
    sim = _cosine_sim_matrix(emb)
    np.fill_diagonal(sim, -np.inf)
    return {i: set(np.argsort(sim[i])[-k:]) for i in range(emb.shape[0])}

def test_structure_survives_different_seeds():
    from pytspl import load_dataset
    sc, _, _ = load_dataset("paper", only_sc=True)

    m1 = Cell2Vec(walk_number=20, walk_length=40, dimensions=16, seed=1, workers=1)
    m1.fit(sc, rank=1)
    e1 = m1.get_embedding()

    m2 = Cell2Vec(walk_number=20, walk_length=40, dimensions=16, seed=999, workers=1)
    m2.fit(sc, rank=1)
    e2 = m2.get_embedding()

    s1 = _cosine_sim_matrix(e1)
    s2 = _cosine_sim_matrix(e2)

    mean_diff = np.abs(s1 - s2).mean()
    assert mean_diff < 0.25

    nn1 = _nearest_neighbors(e1, k=3)
    nn2 = _nearest_neighbors(e2, k=3)
    overlaps = [len(nn1[i] & nn2[i]) / 3 for i in nn1]
    mean_overlap = np.mean(overlaps)
    assert mean_overlap > 0.4