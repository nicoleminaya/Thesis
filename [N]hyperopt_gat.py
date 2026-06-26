import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from epynet import Network
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from utils.graph_utils import get_nx_graph
from utils.DataReader import DataReader
from utils.dataloader import build_dataloader
from utils.SensorInstaller import SensorInstaller
from model.anytown_gat_dynamic import GATNet  # Import GAT dynamic model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- ----- ----- ----- ----- -----
# Configuration
# ----- ----- ----- ----- ----- -----
WDS_NAME = 'anytown'
DB_NAME = 'doe_pumpfed_1'
OBSRAT = 0.4 # SET A MIDDLE OBSERVATION RATIO TO AVOID LATER OVERFITTING
EPOCHS = 60 # Keep this low for hyperparameter search to save time

pathToRoot = os.path.dirname(os.path.realpath(__file__))
pathToDB = os.path.join(pathToRoot, 'data', f'db_{WDS_NAME}_{DB_NAME}')
pathToWDS = os.path.join(pathToRoot, 'water_networks', f'{WDS_NAME}.inp')

# Setup graph and data ONCE globally to speed up search
wds = Network(pathToWDS)
G = get_nx_graph(wds, mode='binary')
num_nodes = len(wds.junctions)

sensor_shop = SensorInstaller(wds, include_pumps_as_master=True) #include fixed master nodes
sensor_shop.deploy_by_xrandom(OBSRAT, seed=1) 
signal_mask = sensor_shop.signal_mask()

print("Loading Data for Hyperparameter Search...")
# Pass the explicit signal mask to DataReader
reader = DataReader(
    pathToDB, 
    n_junc=num_nodes, 
    signal_mask=signal_mask,
    node_order=np.array(list(G.nodes))-1
)

# Training Data
trn_x, _, _ = reader.read_data('trn', 'junc_heads', 'standardize', cover=True)
trn_y, _, _ = reader.read_data('trn', 'junc_heads', 'normalize', cover=False)
trn_ldr = build_dataloader(G, trn_x, trn_y, batch_size=64, shuffle=True)

# Validation Data (We use validation loss to score the hyperparameters)
vld_x, _, _ = reader.read_data('vld', 'junc_heads', 'standardize', cover=True)
vld_y, _, _ = reader.read_data('vld', 'junc_heads', 'normalize', cover=False)
vld_ldr = build_dataloader(G, vld_x, vld_y, batch_size=200, shuffle=False)

in_channels = trn_x.shape[-1]
out_channels = trn_y.shape[-1]

# ----- ----- ----- ----- ----- -----
# Objective Function (Trains one model and returns its score)
# ----- ----- ----- ----- ----- -----
def objective(params):
    print(f"\nTesting Config: hidden={params['hidden']}, heads={params['heads']}, lr={params['lr']:.4f}, dropout={params['dropout']}")
    
    # Initialize model with current hyperparameters
    model = GATNet(
        in_channels=in_channels, 
        out_channels=out_channels, 
        hidden_channels=int(params['hidden']), 
        heads=int(params['heads']),
        dropout=params['dropout']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = torch.nn.MSELoss()

    # Quick Training Loop
    for epoch in range(EPOCHS):
        model.train()
        for batch in trn_ldr:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
    # Evaluation on Validation Set
    model.eval()
    vld_loss = 0.0
    with torch.no_grad():
        for batch in vld_ldr:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            vld_loss += loss.item() * batch.num_graphs
            
    avg_vld_loss = vld_loss / len(vld_x)
    print(f" -> Validation Loss: {avg_vld_loss:.6f}")
    
    # Hyperopt tries to MINIMIZE the return value (loss)
    return {'loss': avg_vld_loss, 'status': STATUS_OK}

# ----- ----- ----- ----- ----- -----
# Search Space Definition
# ----- ----- ----- ----- ----- -----
space = {
    'hidden': hp.choice('hidden', [8, 16, 32, 64]),          # Number of features per head
    'heads': hp.choice('heads', [2, 4, 8]),                  # Number of attention heads
    'lr': hp.loguniform('lr', np.log(0.0005), np.log(0.01)), # Learning rate (between 0.0005 and 0.01)
    'weight_decay': hp.choice('weight_decay', [1e-5, 1e-4, 5e-4]),
    'dropout': hp.choice('dropout', [0.0, 0.1, 0.2, 0.3])    # Dropout to prevent overfitting
}

if __name__ == "__main__":
    print("\n--- Starting GAT Hyperparameter Search (OBSRAT 0.2) ---")
    trials = Trials()
    
    # Run 30 different combinations
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest, # Tree-structured Parzen Estimator (Bayesian optimization)
        max_evals=30,     
        trials=trials
    )
    
    print("\n=========================================")
    print("Optimization Complete! Best Parameters Found:")
    
    # Decode hp.choice indices back to actual values
    hidden_choices = [8, 16, 32, 64]
    heads_choices = [2, 4, 8]
    wd_choices = [1e-5, 1e-4, 5e-4]
    do_choices = [0.0, 0.1, 0.2, 0.3]
    
    print(f"  Hidden Channels: {hidden_choices[best['hidden']]}")
    print(f"  Heads:           {heads_choices[best['heads']]}")
    print(f"  Learning Rate:   {best['lr']:.5f}")
    print(f"  Weight Decay:    {wd_choices[best['weight_decay']]}")
    print(f"  Dropout:         {do_choices[best['dropout']]}")
    print("=========================================")