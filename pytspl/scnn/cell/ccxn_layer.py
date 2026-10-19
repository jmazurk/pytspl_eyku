import torch
import torch.nn as nn



class CCXNLayer(nn.Module):

    def __init__(self,in_ch_0, in_ch_1, in_ch_2, out_ch_0, out_ch_1):
        super().__init__()
        self.node_to_node = CCXNConv(in_ch_0, out_ch_0)
        self.node_self = nn.Linear(in_ch_0, out_ch_0, bias=False)
        self.edge_down = CCXNConv(in_ch_1, out_ch_1)
        self.edge_up = CCXNConv(in_ch_1, out_ch_1)
        self.edge_self = nn.Linear(in_ch_1, out_ch_1, bias=False)
        self.edge_to_face = CCXNConv(out_ch_1, in_ch_2)
        self.activation = nn.ReLU()

    def forward(self, x_0, x_1, x_2, adjacency_0, incidence_2_t, down_laplacian_1, up_laplacian_1):
        """
        Args:
            x_0 (Tensor): Node features, (num_nodes, in_ch_0).
            x_1 (Tensor): Edge features, (num_edges, in_ch_1).
            x_2 (Tensor): Face features, (num_faces, in_ch_2).
            adjacency_0 (Tensor): Sparse node adjacency A_up_0.
            incidence_2_t (Tensor): Sparse transposed edge to face incidence B2^T.
            down_laplacian_1 (Tensor): Sparse lower edge Laplacian B1^T B1.
            up_laplacian_1 (Tensor): Sparse upper edge Laplacian B2 B2^T.

        Returns:
            tuple: Updated (x_0, x_1, x_2).

        Raises:
            ValueError: If node or edge feature sizes don't match the layer.
        """
        if x_0.shape[1] != self.node_to_node.linear.in_features:
            raise ValueError(
                f"x_0 has {x_0.shape[1]} features, "
                f"expected {self.node_to_node.linear.in_features}"
            )
        if x_1.shape[1] != self.edge_self.in_features:
            raise ValueError(
                f"x_1 has {x_1.shape[1]} features, "
                f"expected {self.edge_self.in_features}"
            )

        # nodes mix with neighbours and add own features
        x_0_new = self.node_to_node(x_0, adjacency_0) + self.node_self(x_0)
        x_0_new = self.activation(x_0_new)

        # edges mix with lower and upper neighbours and add own features
        x_1_new = (self.edge_down(x_1, down_laplacian_1) + self.edge_up(x_1, up_laplacian_1) + self.edge_self(x_1))
        x_1_new = self.activation(x_1_new)

        # faces add contribution from bordering edges
        x_2_new = x_2 + self.edge_to_face(x_1_new, incidence_2_t)
        x_2_new = self.activation(x_2_new)

        return x_0_new, x_1_new, x_2_new


class CCXNConv(nn.Module):
    """
    Applies a learnable linear map to each cell's features, then multiplies by
    a sparse operator.
    """

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Input feature size per cell.
            out_channels (int): Output feature size per cell.
        """
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, operator):
        """
        Args:
            x (Tensor): Cell features, shape (num_cells_in, in_channels).
            operator (Tensor): Sparse operator, shape (num_cells_out, num_cells_in).

        Returns:
            Tensor: shape (num_cells_out, out_channels).
        """
        x = self.linear(x) # mix cells' features
        x = torch.sparse.mm(operator, x) # spread across connected cells
        return x


