import torch
import torch.nn as nn

from pytspl.scnn.cell.ccxn_layer import CCXNLayer


class CCXN(nn.Module):
    def __init__(self, in_ch_0, in_ch_1, in_ch_2, hidden, num_classes, n_layers):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
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