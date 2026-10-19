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
    """Triangle (0,1,2) with extra edge (2,3)"""
    sc = SimplicialComplex(
        nodes=[0, 1, 2, 3],
        edges=[(0, 1), (0, 2), (1, 2), (2, 3)],
        triangles=[[0, 1, 2]],
    )
    n_nodes, n_edges, n_faces = len(sc.nodes), len(sc.edges), len(sc.triangles)

    B1 = np.abs(sc.B1)
    A = B1 @ B1.T
    np.fill_diagonal(A, 0.0)
    A = (A > 0).astype(float)
    adjacency_0 = _to_sparse(A)

    incidence_2_t = _to_sparse(sc.B2.T)
    down_laplacian_1 = _to_sparse(sc.B1.T @ sc.B1)
    up_laplacian_1 = _to_sparse(sc.B2 @ sc.B2.T)

    return {
        "adjacency_0": adjacency_0,
        "incidence_2_t": incidence_2_t,
        "down_laplacian_1": down_laplacian_1,
        "up_laplacian_1": up_laplacian_1,
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
        x_0, x_1, x_2, s["adjacency_0"], s["incidence_2_t"],
        s["down_laplacian_1"], s["up_laplacian_1"]
    )

    assert x_0_new.shape == (s["n_nodes"], out_ch_0)
    assert x_1_new.shape == (s["n_edges"], out_ch_1)
    assert x_2_new.shape == (s["n_faces"], in_ch_2)

def test_layer_backward(tiny_setup):
    s = tiny_setup
    layer = CCXNLayer(3, 3, 5, 8, 8)

    x_0 = torch.randn(s["n_nodes"], 3)
    x_1 = torch.randn(s["n_edges"], 3)
    x_2 = torch.zeros(s["n_faces"], 5)

    x_0_new, x_1_new, x_2_new = layer(
        x_0, x_1, x_2, s["adjacency_0"], s["incidence_2_t"],
        s["down_laplacian_1"], s["up_laplacian_1"]
    )

    # make a scalar loss from all three outputs then backprop
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
    out1 = layer1(x_0, x_1, x_2, s["adjacency_0"], s["incidence_2_t"],
                  s["down_laplacian_1"], s["up_laplacian_1"])

    # build layer 2 with the same seed
    torch.manual_seed(42)
    layer2 = CCXNLayer(3, 3, 5, 8, 8)
    out2 = layer2(x_0, x_1, x_2, s["adjacency_0"], s["incidence_2_t"],
                  s["down_laplacian_1"], s["up_laplacian_1"])

    # same seed + same input =? identical outputs
    for a, b in zip(out1, out2):
        assert torch.allclose(a, b)


def test_layer_no_faces():
    # complex with nodes and edges but no faces
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

    # B2 is empty so B2^T has shape (0, n_edges)
    B2_t = sc.B2.T
    incidence_2_t = _to_sparse(B2_t)
    down_laplacian_1 = _to_sparse(sc.B1.T @ sc.B1)
    up_laplacian_1 = _to_sparse(sc.B2 @ sc.B2.T)

    layer = CCXNLayer(3, 3, 5, 8, 8)
    x_0 = torch.randn(n_nodes, 3)
    x_1 = torch.randn(n_edges, 3)
    x_2 = torch.zeros(n_faces, 5)

    x_0_new, x_1_new, x_2_new = layer(
        x_0, x_1, x_2, adjacency_0, incidence_2_t,
        down_laplacian_1, up_laplacian_1
    )

    # nodes and edges still update normally
    assert x_0_new.shape == (n_nodes, 8)
    assert x_1_new.shape == (n_edges, 8)
    # should get empty face signal
    assert x_2_new.shape == (0, 5)


def test_layer_raises_on_bad_dims(tiny_setup):
    s = tiny_setup
    layer = CCXNLayer(3, 3, 5, 8, 8)

    x_2 = torch.zeros(s["n_faces"], 5)

    x_0_bad = torch.randn(s["n_nodes"], 4)
    x_1_ok = torch.randn(s["n_edges"], 3)
    with pytest.raises(ValueError):
        layer(x_0_bad, x_1_ok, x_2, s["adjacency_0"], s["incidence_2_t"],
              s["down_laplacian_1"], s["up_laplacian_1"])

    x_0_ok = torch.randn(s["n_nodes"], 3)
    x_1_bad = torch.randn(s["n_edges"], 7)
    with pytest.raises(ValueError):
        layer(x_0_ok, x_1_bad, x_2, s["adjacency_0"], s["incidence_2_t"],
              s["down_laplacian_1"], s["up_laplacian_1"])


def test_edges_use_neighbours(tiny_setup):
    s = tiny_setup
    torch.manual_seed(0)
    layer = CCXNLayer(3, 3, 5, 8, 8)

    x_0 = torch.randn(s["n_nodes"], 3)
    x_1 = torch.randn(s["n_edges"], 3)
    x_2 = torch.zeros(s["n_faces"], 5)

    _, out_a, _ = layer(x_0, x_1, x_2, s["adjacency_0"], s["incidence_2_t"],
                        s["down_laplacian_1"], s["up_laplacian_1"])

    x_1_changed = x_1.clone()
    x_1_changed[3] += 10.0
    _, out_b, _ = layer(x_0, x_1_changed, x_2, s["adjacency_0"], s["incidence_2_t"],
                        s["down_laplacian_1"], s["up_laplacian_1"])

    assert not torch.allclose(out_a[2], out_b[2])


def test_edges_reduce_to_resize(tiny_setup):
    s = tiny_setup
    layer = CCXNLayer(3, 3, 5, 8, 8)
    with torch.no_grad():
        layer.edge_down.linear.weight.zero_()
        layer.edge_up.linear.weight.zero_()

    x_0 = torch.randn(s["n_nodes"], 3)
    x_1 = torch.randn(s["n_edges"], 3)
    x_2 = torch.zeros(s["n_faces"], 5)

    _, x_1_new, _ = layer(x_0, x_1, x_2, s["adjacency_0"], s["incidence_2_t"],
                          s["down_laplacian_1"], s["up_laplacian_1"])

    assert torch.allclose(x_1_new, torch.relu(layer.edge_self(x_1)))