# -*- coding: utf-8 -*-
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader

def build_dataloader(G, set_x, set_y, batch_size, shuffle):
    data    = []
    master_graph    = from_networkx(G)
    
    # Safely extract all possible edge attributes
    edge_index = master_graph.edge_index
    weight = getattr(master_graph, 'weight', None)
    edge_attr = getattr(master_graph, 'edge_attr', None)

    for x, y in zip(set_x, set_y):
        # Create a new Data object directly for speed
        graph = master_graph.__class__() 
        graph.edge_index = edge_index
        
        # Attach weight and edge_attr if they exist in the master graph
        if weight is not None:
            graph.weight = weight
        if edge_attr is not None:
            graph.edge_attr = edge_attr
            
        graph.x = torch.Tensor(x)
        graph.y = torch.Tensor(y)
        data.append(graph)
        
    loader  = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader