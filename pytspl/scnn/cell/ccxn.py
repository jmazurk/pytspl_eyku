import torch
import torch.nn as nn

from pytspl.scnn.cell.ccxn_layer import CCXNLayer


class CCXN(nn.Module):
    """Multi-layer CCXN model."""

    def __init__(self, in_ch_0, in_ch_1, in_ch_2, hidden, num_classes, n_layers):
        """
            Args:
                in_ch_0 (int): Input node feature size.
                in_ch_1 (int): Input edge feature size.
                in_ch_2 (int): Input face feature size.
                hidden (int): Hidden feature size used after the first layer.
                num_classes (int): Output size of the whole-complex prediction.
                n_layers (int): Number of stacked CCXN layers.
        """
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # first layer takes the input sizes and later layers work in hidden
            node_in = in_ch_0 if i == 0 else hidden
            edge_in = in_ch_1 if i == 0 else hidden
            self.layers.append(
                CCXNLayer(
                    in_ch_0=node_in,
                    in_ch_1=edge_in,
                    in_ch_2=hidden,
                    out_ch_0=hidden,
                    out_ch_1=hidden,
                )
            )

        self.readout_linear = nn.Linear(hidden, num_classes)

    def forward(self, x_0, x_1, adjacency_0, incidence_2_t, x_2=None, readout=True):
        """
            Args:
                x_0 (Tensor): Node features, (num_nodes, in_ch_0).
                x_1 (Tensor): Edge features, (num_edges, in_ch_1).
                adjacency_0 (Tensor): Sparse node adjacency A_up_0.
                incidence_2_t (Tensor): Sparse transposed edge-to-face incidence B2^T.
                x_2 (Tensor, optional): Face features. Defaults to zeros.
                readout (bool, optional): If True, return one pooled whole-complex prediction,
                if False, return the per-cell signals. Default is True.

                Returns:
                    Tensor or tuple: (num_classes,) prediction if readout, else the
                    per-cell signals (x_0, x_1, x_2).
                """
        n_faces = incidence_2_t.shape[0]
        if x_2 is None:
            x_2 = torch.zeros(n_faces, self.layers[0].edge_to_face.linear.out_features)

        for layer in self.layers:
            x_0, x_1, x_2 = layer(x_0, x_1, x_2, adjacency_0, incidence_2_t)

        if not readout:
            return x_0, x_1, x_2

        pooled = x_0.mean(dim=0)
        prediction = self.readout_linear(pooled)
        return prediction