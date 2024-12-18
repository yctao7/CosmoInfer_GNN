"""
Define the Graph Neural Network architecture.

"""

import torch
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import (LayerNorm, MetaLayer, global_add_pool, 
                                global_max_pool, global_mean_pool)
from torch_scatter import (scatter_add, scatter_max, scatter_mean)

from .constants import device

from torch_geometric.nn import GCNConv, GATConv, GINConv

# Model for updating edge attritbutes
class EdgeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, residuals=True, norm=False):
        super().__init__()

        self.residuals = residuals
        self.norm = norm

        layers = [Linear(node_in*2 + edge_in, hid_channels),
                  ReLU(),
                  Linear(hid_channels, edge_out)]
        if self.norm:  layers.append(LayerNorm(edge_out))

        self.edge_mlp = Sequential(*layers)


    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        out = torch.cat([src, dest, edge_attr], dim=1)
        #out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.edge_mlp(out)
        if self.residuals:
            out = out + edge_attr
        return out

# Model for updating node attritbutes
class NodeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, residuals=True, norm=False):
        super().__init__()

        self.residuals = residuals
        self.norm = norm

        layers = [Linear(node_in + 3*edge_out + 1, hid_channels),
                  ReLU(),
                  Linear(hid_channels, node_out)]
        if self.norm:  layers.append(LayerNorm(node_out))

        self.node_mlp = Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        row, col = edge_index
        out = edge_attr

        # Multipooling layer
        out1 = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out2 = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        out3 = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out1, out2, out3, u[batch]], dim=1)

        out = self.node_mlp(out)
        if self.residuals:
            out = out + x
        return out

# First edge model for updating edge attritbutes when no initial node features are provided
class EdgeModelIn(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, norm=False):
        super().__init__()

        self.norm = norm

        layers = [Linear(edge_in, hid_channels),
                  ReLU(),
                  Linear(hid_channels, edge_out)]
        if self.norm:  layers.append(LayerNorm(edge_out))

        self.edge_mlp = Sequential(*layers)


    def forward(self, src, dest, edge_attr, u, batch):

        out = self.edge_mlp(edge_attr)

        return out

# First node model for updating node attritbutes when no initial node features are provided
class NodeModelIn(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, norm=False):
        super().__init__()

        self.norm = norm

        layers = [Linear(3*edge_out + 1, hid_channels),
                  ReLU(),
                  Linear(hid_channels, node_out)]
        if self.norm:  layers.append(LayerNorm(node_out))

        self.node_mlp = Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, u, batch):

        row, col = edge_index
        out = edge_attr

        # Multipooling layer
        out1 = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out2 = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        out3 = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([out1, out2, out3, u[batch]], dim=1)

        out = self.node_mlp(out)

        return out

# Graph Neural Network architecture, based on the Graph Network (arXiv:1806.01261)
# Employing the MetaLayer implementation in Pytorch-Geometric
class GNN(torch.nn.Module):
    def __init__(self, node_features, n_layers, hidden_channels, linkradius, dim_out, only_positions, residuals=True):
        super().__init__()

        self.n_layers = n_layers
        self.link_r = linkradius
        self.dim_out = dim_out
        self.only_positions = only_positions

        # Number of input node features (0 if only_positions is used)
        node_in = node_features
        # Input edge features: |p_i-p_j|, p_i*p_j, p_i*(p_i-p_j)
        edge_in = 3
        node_out = hidden_channels
        edge_out = hidden_channels
        hid_channels = hidden_channels

        layers = []

        # Encoder graph block
        # If use only positions, node features are created from the aggregation of edge attritbutes of neighbors
        if self.only_positions:
            inlayer = MetaLayer(node_model=NodeModelIn(node_in, node_out, edge_in, edge_out, hid_channels),
                                edge_model=EdgeModelIn(node_in, node_out, edge_in, edge_out, hid_channels))

        else:
            inlayer = MetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=False),
                                edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=False))

        layers.append(inlayer)

        # Change input node and edge feature sizes
        node_in = node_out
        edge_in = edge_out

        # Hidden graph blocks
        for i in range(n_layers-1):

            lay = MetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=residuals),
                            edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=residuals))
            layers.append(lay)

        self.layers = ModuleList(layers)

        # Save encoding dimension 
        self.encoding_dim = 3*node_out+1

        # Final aggregation layer
        self.outlayer = Sequential(Linear(self.encoding_dim, hid_channels),
                            ReLU(),
                            Linear(hid_channels, hid_channels),
                            ReLU(),
                            Linear(hid_channels, hid_channels),
                            ReLU(),
                            Linear(hid_channels, self.dim_out))

    def forward(self, data):

        h, edge_index, edge_attr, u = data.x, data.edge_index, data.edge_attr, data.u

        # Message passing layers
        for layer in self.layers:
            h, edge_attr, _ = layer(h, edge_index, edge_attr, u, data.batch)

        # Multipooling layer
        addpool = global_add_pool(h, data.batch)
        meanpool = global_mean_pool(h, data.batch)
        maxpool = global_max_pool(h, data.batch)

        encoding = torch.cat([addpool,meanpool,maxpool,u], dim=1)

        # Final linear layer
        out = self.outlayer(encoding)

        return out, encoding
    
    def compute_encoding(self, data):

        h, edge_index, edge_attr, u = data.x, data.edge_index, data.edge_attr, data.u

        # Message passing layers
        for layer in self.layers:
            h, edge_attr, _ = layer(h, edge_index, edge_attr, u, data.batch)

        # Multipooling layer
        addpool = global_add_pool(h, data.batch)
        meanpool = global_mean_pool(h, data.batch)
        maxpool = global_max_pool(h, data.batch)

        encoding = torch.cat([addpool,meanpool,maxpool,u], dim=1)

        return encoding
    
# Replace MetaLayer blocks with GCNConv layers for message passing
class GCN(torch.nn.Module):
    def __init__(self, node_features, n_layers, hidden_channels, dim_out):
        super().__init__()
        self.n_layers = n_layers

        # First layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_channels))

        # Hidden layers
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Save the encoding dimension for downstream tasks
        self.encoding_dim = hidden_channels * 3  # 3 for add, mean, max pooling

        # Final aggregation layer
        self.outlayer = torch.nn.Sequential(
            torch.nn.Linear(self.encoding_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, dim_out)
        )

    def forward(self, data):
        # Compute encoding
        encoding = self.compute_encoding(data)

        # Final linear layer
        out = self.outlayer(encoding)
        return out, encoding
    
    def compute_encoding(self, data):
        """
        Computes the encoding of the graph.
        This function aggregates node embeddings with add, mean, and max pooling.
        """
        h, edge_index = data.x, data.edge_index

        # Apply GCN layers
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.nn.ReLU()(h)

        # Pooling layer
        addpool = global_add_pool(h, data.batch)
        meanpool = global_mean_pool(h, data.batch)
        maxpool = global_max_pool(h, data.batch)

        # Concatenate pooled features
        encoding = torch.cat([addpool, meanpool, maxpool], dim=1)
        # print(f"Encoding shape: {encoding.shape}")  # Debugging shape
        return encoding

class GAT(torch.nn.Module):
    def __init__(self, node_features, n_layers, hidden_channels, dim_out, heads=8):
        """
        GAT implementation for graph-level tasks with pooling and MLP layers.

        Args:
            node_features (int): Number of input node features.
            n_layers (int): Number of GATConv layers.
            hidden_channels (int): Hidden dimension for node embeddings.
            dim_out (int): Output dimension of the network.
            heads (int): Number of attention heads for GATConv.
        """
        super().__init__()
        self.n_layers = n_layers
        
        # Define encoding_dim for graph-level tasks (excluding global feature)
        self.encoding_dim = 3 * hidden_channels * heads

        # First GAT layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(node_features, hidden_channels // heads, heads=heads, concat=True))

        # Hidden GAT layers
        for _ in range(n_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, concat=True))

        # Final aggregation layer
        self.outlayer = Sequential(
            Linear(3 * hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, dim_out)
        )

    def forward(self, data):
        """
        Forward pass for GAT.

        Args:
            data: PyTorch Geometric data object containing:
                - x: Node features.
                - edge_index: Graph connectivity in COO format.
                - batch: Batch assignment for nodes.

        Returns:
            - out: Output predictions.
            - encoding: Graph-level encoding (pooled features).
        """
        h, edge_index = data.x, data.edge_index

        # Apply GAT layers
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.nn.ReLU()(h)  # Activation after each GAT layer

        # Pooling layer (graph-level encoding)
        addpool = global_add_pool(h, data.batch)
        meanpool = global_mean_pool(h, data.batch)
        maxpool = global_max_pool(h, data.batch)

        # Combine pooled features
        encoding = torch.cat([addpool, meanpool, maxpool], dim=1)

        # Final linear layer for prediction
        out = self.outlayer(encoding)
        return out, encoding

    def compute_encoding(self, data):
        """
        Compute only the graph encoding (without final prediction layer).

        Args:
            data: PyTorch Geometric data object containing:
                - x: Node features.
                - edge_index: Graph connectivity in COO format.
                - batch: Batch assignment for nodes.

        Returns:
            - encoding: Graph-level encoding (pooled features).
        """
        h, edge_index = data.x, data.edge_index

        # Apply GAT layers
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.nn.ReLU()(h)

        # Pooling layer (graph-level encoding)
        addpool = global_add_pool(h, data.batch)
        meanpool = global_mean_pool(h, data.batch)
        maxpool = global_max_pool(h, data.batch)

        # Combine pooled features
        encoding = torch.cat([addpool, meanpool, maxpool], dim=1)

        return encoding

class GIN(torch.nn.Module):
    def __init__(self, node_features, n_layers, hidden_channels, dim_out):
        """
        GIN implementation for graph-level tasks with pooling and MLP layers.
        
        Args:
            node_features (int): Number of input node features.
            n_layers (int): Number of GINConv layers.
            hidden_channels (int): Hidden dimension for node embeddings.
            dim_out (int): Output dimension of the network.
        """
        super().__init__()
        
        self.n_layers = n_layers
        
        # Define encoding_dim for graph-level tasks
        self.encoding_dim = 3 * hidden_channels

        # Define the MLP for GINConv
        def build_mlp(in_channels, out_channels):
            return Sequential(
                Linear(in_channels, out_channels),
                ReLU(),
                Linear(out_channels, out_channels)
            )

        # First GIN layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(build_mlp(node_features, hidden_channels)))

        # Hidden GIN layers
        for _ in range(n_layers - 1):
            self.convs.append(GINConv(build_mlp(hidden_channels, hidden_channels)))

        # Final aggregation layer
        self.outlayer = Sequential(
            Linear(3 * hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, dim_out)
        )

    def forward(self, data):
        """
        Forward pass for GIN.
        
        Args:
            data: PyTorch Geometric data object containing:
                - x: Node features.
                - edge_index: Graph connectivity in COO format.
                - batch: Batch assignment for nodes.
        
        Returns:
            - out: Output predictions.
            - encoding: Graph-level encoding (pooled features).
        """
        h, edge_index = data.x, data.edge_index

        # Apply GIN layers
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.nn.ReLU()(h)  # Activation after each GIN layer

        # Pooling layer (graph-level encoding)
        addpool = global_add_pool(h, data.batch)
        meanpool = global_mean_pool(h, data.batch)
        maxpool = global_max_pool(h, data.batch)

        # Combine pooled features
        encoding = torch.cat([addpool, meanpool, maxpool], dim=1)

        # Final linear layer for prediction
        out = self.outlayer(encoding)
        return out, encoding

    def compute_encoding(self, data):
        """
        Compute only the graph encoding (without final prediction layer).
        
        Args:
            data: PyTorch Geometric data object containing:
                - x: Node features.
                - edge_index: Graph connectivity in COO format.
                - batch: Batch assignment for nodes.
        
        Returns:
            - encoding: Graph-level encoding (pooled features).
        """
        h, edge_index = data.x, data.edge_index

        # Apply GIN layers
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.nn.ReLU()(h)

        # Pooling layer (graph-level encoding)
        addpool = global_add_pool(h, data.batch)
        meanpool = global_mean_pool(h, data.batch)
        maxpool = global_max_pool(h, data.batch)

        # Combine pooled features
        encoding = torch.cat([addpool, meanpool, maxpool], dim=1)

        return encoding

def define_model(hparams, dim_in, dim_out):
    """Generate the GNN for the given dataset and hparams.
    Definition of model takes as argument datasets as GNN structure heavily relies on underlying graph structure.

    Args:
        hparams (HyperParameters): Hyperparameters.
        datasets (dict): Datasets, key is the name of the simulation suite.

    Returns:
        model.
    """

    # Initialize model
    ### GIN: node_features, n_layers, hidden_channels, dim_out
    
    ### GAT: node_features, n_layers, hidden_channels, dim_out, heads=8 (use default number of heads)
    if hparams.model_select == 'GNN':
        model = GNN(node_features=dim_in,
                    n_layers=hparams.n_layers,
                    hidden_channels=hparams.hidden_channels,
                    linkradius=hparams.r_link,
                    dim_out=dim_out,
                    only_positions=hparams.only_positions)
    elif hparams.model_select == 'GCN':
        model = GCN(node_features=dim_in,
                    n_layers=hparams.n_layers, 
                    hidden_channels=hparams.hidden_channels,
                    dim_out=dim_out)
    elif hparams.model_select == 'GIN':
        model = GIN(node_features=dim_in,
                    n_layers=hparams.n_layers,
                    hidden_channels=hparams.hidden_channels,
                    dim_out=dim_out)
    elif hparams.model_select == 'GAT':
        model = GAT(node_features=dim_in,
                    n_layers=hparams.n_layers,
                    hidden_channels=hparams.hidden_channels,
                    dim_out=dim_out)
    
    model.to(device)

    return model