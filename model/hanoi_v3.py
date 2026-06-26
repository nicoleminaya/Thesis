# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class ChebNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChebNet, self).__init__()

        self.conv1 = ChebConv(in_channels, 14, K=8) #  ChebConv(in_channels, 16, K=9
        self.conv2 = ChebConv(14, 20, K=8) #ChebConv(16, 18, K=12) 
        self.conv3 = ChebConv(20, 27, K=8) #ChebConv(18, 25, K=10) 
        self.conv4 = ChebConv(27, out_channels, K=1, bias=False) #ChebConv(25, out_channels, K=1, bias=False)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.weight

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = self.conv4(x, edge_index, edge_weight)
        
        return x

