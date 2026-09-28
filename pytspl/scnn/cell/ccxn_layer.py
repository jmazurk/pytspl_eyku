import torch
import torch.nn as nn



class CCXNLayer(nn.Module):

    def __init__(self,in_ch_0, in_ch_1, in_ch_2, out_ch_0, out_ch_1):
        super().__init__()
        self.node_to_node = CCXNConv(in_ch_0, out_ch_0)
        self.node_self = nn.Linear(in_ch_0, out_ch_0, bias=False)
        self.edge_resize = nn.Linear(in_ch_1, out_ch_1, bias=False)
        self.edge_to_face = CCXNConv(out_ch_1, in_ch_2)
        self.activation = nn.ReLU()

    def forward(self, x_0, x_1, x_2, adjacency_0, incidence_2_t):
        if x_0.shape[1] != self.node_to_node.linear.in_features:
            raise ValueError(
                f"x_0 has {x_0.shape[1]} features, "
                f"expected {self.node_to_node.linear.in_features}"
            )
        if x_1.shape[1] != self.edge_resize.in_features:
            raise ValueError(
                f"x_1 has {x_1.shape[1]} features, "
                f"expected {self.edge_to_face.linear.in_features}"
            )

        x_0_new = self.node_to_node(x_0, adjacency_0) + self.node_self(x_0)
        x_0_new = self.activation(x_0_new)

        x_1_new = self.edge_resize(x_1)
        x_1_new = self.activation(x_1_new)

        x_2_new = x_2 + self.edge_to_face(x_1_new, incidence_2_t)
        x_2_new = self.activation(x_2_new)

        return x_0_new, x_1_new, x_2_new




class CCXNConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, operator):
        x = self.linear(x)
        x = torch.sparse.mm(operator, x)
        return x


