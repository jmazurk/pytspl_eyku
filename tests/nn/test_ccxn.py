import torch
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pytspl.simplicial_complex import SimplicialComplex
from pytspl.scnn.cell.ccxn import CCXN


def _to_sparse(mat):
    mat = csr_matrix(mat).tocoo()
    idx = torch.tensor(np.vstack((mat.row, mat.col)), dtype=torch.long)
    val = torch.tensor(mat.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=mat.shape).coalesce()


@pytest.fixture
def tiny_setup():
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


# whole-complex readout shape
def test_model_readout_shape(tiny_setup):
    s = tiny_setup
    model = CCXN(in_ch_0=3, in_ch_1=3, in_ch_2=5, hidden=16,
                 num_classes=4, n_layers=2)

    x_0 = torch.randn(s["n_nodes"], 3)
    x_1 = torch.randn(s["n_edges"], 3)

    pred = model(x_0, x_1, s["adjacency_0"], s["incidence_2_t"],
                 s["down_laplacian_1"], s["up_laplacian_1"], readout=True)

    assert pred.shape == (4,)   # one prediction, num_classes=4


# per-cell mode shapes
def test_model_per_cell_shapes(tiny_setup):
    s = tiny_setup
    model = CCXN(in_ch_0=3, in_ch_1=3, in_ch_2=5, hidden=16,
                 num_classes=4, n_layers=2)

    x_0 = torch.randn(s["n_nodes"], 3)
    x_1 = torch.randn(s["n_edges"], 3)

    x_0_out, x_1_out, x_2_out = model(
        x_0, x_1, s["adjacency_0"], s["incidence_2_t"],
        s["down_laplacian_1"], s["up_laplacian_1"], readout=False
    )

    assert x_0_out.shape == (s["n_nodes"], 16)   # nodes at hidden size
    assert x_1_out.shape == (s["n_edges"], 16)   # edges at hidden size
    assert x_2_out.shape == (s["n_faces"], 16)   # faces at hidden size


def test_model_backward(tiny_setup):
    s = tiny_setup
    torch.manual_seed(0)
    model = CCXN(3, 3, 5, hidden=16, num_classes=4, n_layers=2)

    x_0 = torch.randn(s["n_nodes"], 3)
    x_1 = torch.randn(s["n_edges"], 3)

    x_0_out, x_1_out, x_2_out = model(x_0, x_1, s["adjacency_0"], s["incidence_2_t"],
                                      s["down_laplacian_1"], s["up_laplacian_1"], readout=False)

    loss = x_0_out.sum() + x_1_out.sum() + x_2_out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if "readout_linear" in name:
            continue
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"