# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATNet(torch.nn.Module):
    def __init__(self, num_features, num_outputs):
        super(GATNet, self).__init__()
        # Using 4 heads, each outputting 24 features -> 96 total features
        self.conv1 = GATConv(num_features, 24, heads=4, concat=True)
        self.conv2 = GATConv(96, 24, heads=4, concat=True)
        self.conv3 = GATConv(96, 24, heads=4, concat=True)
        self.conv4 = GATConv(96, num_outputs, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Safely grab and reshape the weight for GAT edge_attr
        edge_attr = getattr(data, 'weight', None)
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        # ELU is the standard activation function for GATs
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.elu(self.conv3(x, edge_index, edge_attr))
        x = self.conv4(x, edge_index, edge_attr)
        
        return x