# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        # Using 4 heads, each outputting 8 features -> 32 total features
        self.conv1 = GATConv(in_channels, 8, heads=4)
        self.conv2 = GATConv(32, 8, heads=4)
        self.conv3 = GATConv(32, 8, heads=4)
        # Final layer uses 1 head and doesn't concatenate to match the required out_channels
        self.conv4 = GATConv(32, out_channels, heads=1, concat=False, bias=False)

    def forward(self, data):
        # GAT learns its own attention weights dynamically from node features.
        x, edge_index = data.x, data.edge_index
        
        x = F.silu(self.conv1(x, edge_index))
        x = F.silu(self.conv2(x, edge_index))
        x = F.silu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        
        return torch.sigmoid(x)


# RESULTS FROM HYPEROPT
#=========================================
#Optimization Complete! Best Parameters Found:
#  Hidden Channels: 16
#  Heads:           2
#  Learning Rate:   0.00067
#  Weight Decay:    1e-05
#  Dropout:         0.0
#=========================================


# SECOND OPTION STILL NOT TESTED
#=========================================
#Optimization Complete! Best Parameters Found:
#  Hidden Channels: 8
#  Heads:           8
#  Learning Rate:   0.00870
#  Weight Decay:    0.0001
#  Dropout:         0.1
#=========================================