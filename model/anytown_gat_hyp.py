import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=16, heads=2, dropout=0.0):
        super(GATNet, self).__init__()
        self.dropout = dropout
        
        # Layer 1: output size will be (hidden_channels * heads)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Layer 2
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        # Layer 3
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        # Layer 4 (Final layer, 1 head, no concatenation)
        self.conv4 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, bias=False, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.silu(self.conv1(x, edge_index))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.silu(self.conv2(x, edge_index))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.silu(self.conv3(x, edge_index))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index)
        
        return torch.sigmoid(x)