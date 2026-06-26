import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import Linear

class GATv2ResNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=4, heads=4, dropout=0.2):
        
        super(GATv2ResNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Input Projection Layer
        # Projects the sparse input features (mostly 0s) into a richer hidden space.
        # This also standardizes the dimension size so we can add residual connections later.
        self.lin_in = Linear(in_channels, hidden_channels)

        # GATv2 Layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # concat=False ensures the output dimension stays at hidden_channels 
            # (averages the heads instead of concatenating them), which is required for the residual operation.
            self.convs.append(
                GATv2Conv(
                    hidden_channels, 
                    hidden_channels, 
                    heads=heads, 
                    concat=False, 
                    dropout=dropout, # This drops out specific attention edges during training to prevent overfitting!
                    edge_dim=1 
                )
            )

        # Output Projection Layer
        self.lin_out = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Safely grab and reshape the weight for GAT edge_attr
        edge_attr = getattr(data, 'weight', None)
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        # Initial projection
        x = self.lin_in(x)
        x = F.elu(x)

        # GATv2 Message Passing with Residual (Skip) Connections
        for conv in self.convs:
            x_residual = x  # Store the state of the nodes BEFORE the message passing

            # Apply node feature dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # GATv2 Convolution (Passing edge_attr alongside edge_index)
            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
                
            x = F.elu(x)
            
            # The Residual Connection
            x = x + x_residual 

        # Final prediction mapping
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_out(x)

        return x