import torch
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pytspl.simplicial_complex import SimplicialComplex
from pytspl.scnn.cell.ccxn_layer import CCXNLayer, CCXNConv


# shared helpers

def _to_sparse(mat):
    mat = csr_matrix(mat).tocoo()
    idx = torch.tensor(np.vstack((mat.row, mat.col)), dtype=torch.long)
    val = torch.tensor(mat.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=mat.shape).coalesce()


@pytest.fixture
def tiny_setup():
    """Triangle (0,1,2) + dangling edge (2,3), with operators and features."""
    sc = SimplicialComplex(
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (0, 2), (1, 2), (2, 3)],
        triangles=[[0, 1, 2]],
    )
    n_nodes, n_edges, n_faces = len(sc.nodes), len(sc.edges), len(sc.triangles)

    # A_up_0 (node adjacency, no self-loops)
    B1 = np.abs(sc.B1)
    A = B1 @ B1.T
    np.fill_diagonal(A, 0.0)
    A = (A > 0).astype(float)
    adjacency_0 = _to_sparse(A)

    # B2^T (edges -> faces)
    incidence_2_t = _to_sparse(sc.B2.T)

    return {
        "adjacency_0": adjacency_0,
        "incidence_2_t": incidence_2_t,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_faces": n_faces,
    }


# output shapes

def test_layer_output_shapes(tiny_setup):
    s = tiny_setup
    in_ch_0, in_ch_1, in_ch_2 = 3, 3, 5
    out_ch_0, out_ch_1 = 8, 8

    layer = CCXNLayer(in_ch_0, in_ch_1, in_ch_2, out_ch_0, out_ch_1)

    x_0 = torch.randn(s["n_nodes"], in_ch_0)
    x_1 = torch.randn(s["n_edges"], in_ch_1)
    x_2 = torch.zeros(s["n_faces"], in_ch_2)

    x_0_new, x_1_new, x_2_new = layer(
        x_0, x_1, x_2, s["adjacency_0"], s["incidence_2_t"]
    )

    assert x_0_new.shape == (s["n_nodes"], out_ch_0)   # nodes -> out_ch_0
    assert x_1_new.shape == (s["n_edges"], out_ch_1)   # edges -> out_ch_1
    assert x_2_new.shape == (s["n_faces"], in_ch_2)    # faces -> in_ch_2

def test_layer_backward(tiny_setup):
    s = tiny_setup
    layer = CCXNLayer(3, 3, 5, 8, 8)

    x_0 = torch.randn(s["n_nodes"], 3)
    x_1 = torch.randn(s["n_edges"], 3)
    x_2 = torch.zeros(s["n_faces"], 5)

    x_0_new, x_1_new, x_2_new = layer(
        x_0, x_1, x_2, s["adjacency_0"], s["incidence_2_t"]
    )

    # make a scalar loss from all three outputs, then backprop
    loss = x_0_new.sum() + x_1_new.sum() + x_2_new.sum()
    loss.backward()

    # every learnable parameter should have received a gradient
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"


def test_layer_deterministic(tiny_setup):
    s = tiny_setup

    x_0 = torch.randn(s["n_nodes"], 3)
    x_1 = torch.randn(s["n_edges"], 3)
    x_2 = torch.zeros(s["n_faces"], 5)

    # build layer 1 with a fixed seed
    torch.manual_seed(42)
    layer1 = CCXNLayer(3, 3, 5, 8, 8)
    out1 = layer1(x_0, x_1, x_2, s["adjacency_0"], s["incidence_2_t"])

    # build layer 2 with the SAME seed
    torch.manual_seed(42)
    layer2 = CCXNLayer(3, 3, 5, 8, 8)
    out2 = layer2(x_0, x_1, x_2, s["adjacency_0"], s["incidence_2_t"])

    # same seed + same input -> identical outputs
    for a, b in zip(out1, out2):
        assert torch.allclose(a, b)


def test_layer_no_faces():
    # complex with nodes and edges but NO faces
    sc = SimplicialComplex(
        nodes=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        # no triangles
    )
    n_nodes, n_edges = len(sc.nodes), len(sc.edges)
    n_faces = 0

    B1 = np.abs(sc.B1)
    A = B1 @ B1.T
    np.fill_diagonal(A, 0.0)
    A = (A > 0).astype(float)
    adjacency_0 = _to_sparse(A)

    # B2 is empty -> B2^T has shape (0, n_edges)
    B2_t = sc.B2.T
    incidence_2_t = _to_sparse(B2_t)

    layer = CCXNLayer(3, 3, 5, 8, 8)
    x_0 = torch.randn(n_nodes, 3)
    x_1 = torch.randn(n_edges, 3)
    x_2 = torch.zeros(n_faces, 5)

    x_0_new, x_1_new, x_2_new = layer(
        x_0, x_1, x_2, adjacency_0, incidence_2_t
    )

    # nodes and edges still update normally
    assert x_0_new.shape == (n_nodes, 8)
    assert x_1_new.shape == (n_edges, 8)
    # faces: zero of them -> empty face signal
    assert x_2_new.shape == (0, 5)


def test_layer_raises_on_bad_dims(tiny_setup):
    s = tiny_setup
    layer = CCXNLayer(3, 3, 5, 8, 8)   # expects 3 node feats, 3 edge feats

    x_2 = torch.zeros(s["n_faces"], 5)

    # wrong node feature size: 4 instead of 3
    x_0_bad = torch.randn(s["n_nodes"], 4)
    x_1_ok = torch.randn(s["n_edges"], 3)
    with pytest.raises(ValueError):
        layer(x_0_bad, x_1_ok, x_2, s["adjacency_0"], s["incidence_2_t"])

    # wrong edge feature size: 7 instead of 3
    x_0_ok = torch.randn(s["n_nodes"], 3)
    x_1_bad = torch.randn(s["n_edges"], 7)
    with pytest.raises(ValueError):
        layer(x_0_ok, x_1_bad, x_2, s["adjacency_0"], s["incidence_2_t"])