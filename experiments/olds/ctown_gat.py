# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        # Using 4 heads, each outputting 16 features -> 64 total features
        self.conv1 = GATConv(in_channels, 16, heads=4)
        self.conv2 = GATConv(64, 16, heads=4)
        self.conv3 = GATConv(64, 8, heads=4) # 8 * 4 = 32 features
        self.conv4 = GATConv(32, out_channels, heads=1, concat=False, bias=False) # not concatenated the head, average instead

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.silu(self.conv1(x, edge_index))
        x = F.silu(self.conv2(x, edge_index))
        x = F.silu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        
        return torch.sigmoid(x)