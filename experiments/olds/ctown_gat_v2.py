# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import Linear

class CtownGATv2ResNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=5, heads=4, dropout=0.2):
        
        super(CtownGATv2ResNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Input Projection Layer
        self.lin_in = Linear(in_channels, hidden_channels)

        # GATv2 Layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_channels, 
                    hidden_channels, 
                    heads=heads, 
                    concat=False, 
                    dropout=dropout
                )
            )

        # Output Projection Layer
        self.lin_out = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Initial projection
        x = self.lin_in(x)
        x = F.elu(x)

        # GATv2 Message Passing with Residual (Skip) Connections
        for conv in self.convs:
            x_residual = x  

            # Apply node feature dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # GATv2 Convolution
            x = conv(x, edge_index)
            x = F.elu(x)
            
            # Residual Connection to prevent over-smoothing in deep networks
            x = x + x_residual 

        # Final prediction mapping
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_out(x)

        return x