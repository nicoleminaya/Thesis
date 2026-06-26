# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import Linear

class GATv2Net(torch.nn.Module): 
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=4, heads=4, dropout=0.2):
        super(GATv2Net, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout


        self.lin_in = Linear(in_channels, hidden_channels)

        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):


            self.convs.append(
                GATv2Conv(
                    hidden_channels, 
                    hidden_channels, 
                    heads=heads, 
                    concat=False, 
                    dropout=dropout,
                    edge_dim=1 
                )
            )


        self.lin_out = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        

        edge_attr = getattr(data, 'weight', None)
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)


        x = self.lin_in(x)
        x = F.elu(x)


        for conv in self.convs:
            x_residual = x  


            x = F.dropout(x, p=self.dropout, training=self.training)
            

            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
                
            x = F.elu(x)


            x = x + x_residual 


        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_out(x)

        return x