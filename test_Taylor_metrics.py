# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import dask.array as da
import networkx as nx
import torch
import csv
from epynet import Network

# Import utilities from the repository
from utils.graph_utils import get_nx_graph
from utils.DataReader import DataReader
from utils.SensorInstaller import SensorInstaller
from utils.Metrics import Metrics
from utils.MeanPredictor import MeanPredictor
from utils.baselines import interpolated_regularization
from utils.dataloader import build_dataloader

# ----- ----- ----- ----- ----- -----
# Command line arguments
# ----- ----- ----- ----- ----- -----
parser  = argparse.ArgumentParser()
parser.add_argument('--wds',
                    default = 'anytown',
                    type    = str,
                    help    = "Water distribution system."
                    )
parser.add_argument('--tag',
                    default = 'def',
                    type    = str,
                    help    = "Customer tag"
                    )
parser.add_argument('--deploy',
                    default = 'xrandom',
                    choices = ['master', 'dist', 'hydrodist', 'hds', 'hdvar', 'random', 'xrandom', 'gena'],
                    type    = str,
                    help    = "Method of sensor deployment.")
parser.add_argument('--obsrat',
                    default = 0.05,
                    type    = float,
                    help    = "Observation ratio."
                    )
parser.add_argument('--batch',
                    default = 64,
                    type    = float,
                    help    = "Batch size."
                    )
parser.add_argument('--adj',
                    default = 'binary',
                    choices = ['binary', 'weighted', 'logarithmic', 'pruned'],
                    type    = str,
                    help    = "Type of adjacency matrix.")
parser.add_argument('--runs',
                    default = 20,
                    type    = int,
                    help    = "Total experiments."
                    )
parser.add_argument('--gnn',    #TO INCLUDE GATs
                    default = 'cheb1',
                    choices = ['cheb1', 'cheb2', 'cheb3', 'gat', 'gat_hyp','gat2','orig'],
                    type    = str,
                    help    = "GNN architecture to use.")
args= parser.parse_args()

# --- MODIFICACIÓN: No cargar arquitectura si es el modelo 'orig' ---
if args.gnn == 'orig':
    Net = None
elif args.wds == 'anytown':
    if args.gnn == 'gat':
        from model.anytown_gat import GATNet as Net
    elif args.gnn == 'cheb1':
        from model.anytown import ChebNet as Net
    elif args.gnn == 'cheb2':
        from model.anytown_v2 import ChebNet as Net
    elif args.gnn == 'cheb3':
            from model.anytown_v3 import ChebNet as Net
    elif args.gnn == 'gat2':
        from model.anytown_gat_v2 import GATv2ResNet as Net
    elif args.gnn == 'gat_hyp':
        from model.anytown_gat_hyp import GATNet as Net
    else:
        from model.anytown import ChebNet as Net
elif args.wds == 'hanoi':
    if args.gnn == 'gat':
        from model.hanoi_gat import GATNet as Net
    elif args.gnn == 'gat2':
        from model.hanoi_gat_v2 import GATv2Net as Net
    elif args.gnn == 'cheb2':
        from model.hanoi_v2 import ChebNet as Net
    elif args.gnn == 'cheb3':
        from model.hanoi_v3 import ChebNet as Net
    else:
        from model.hanoi import ChebNet as Net
elif args.wds == 'bwsn':
    if args.gnn == 'cheb2':
        from model.bwsn_v2 import ChebNet as Net
    elif args.gnn == 'cheb3':
            from model.bwsn_v3 import ChebNet as Net
    elif args.gnn == 'gat':
            from model.bwsn_gat import GATNet as Net
    elif args.gnn == 'gat_hyp':
            from model.bwsn_gat_hyp import GATNet as Net
    elif args.gnn == 'gat2':
        from model.bwsn_gat_v2 import GATv2Net as Net
    else:
        from model.bwsn_v2 import ChebNet as Net
else:
    print('Water distribution system is unknown.\n')
    raise


# --- Configuration ---
WDS_NAME = args.wds #'anytown'
DB_NAME = 'doe_pumpfed_1'
GNN = args.gnn
#BUDGET = 1           # 5% Sensors
TAG = args.tag
OBSRAT = args.obsrat
# MODIFICACIÓN: Si es el modelo original (referencia), solo ejecutar 1 vez.
RUNS = 1 if args.gnn == 'orig' else args.runs           
ADJ = args.adj
BATCH_SIZE = args.batch
DEPLOY = args.deploy
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
base_dir = os.path.dirname(os.path.realpath(__file__))
results_file = os.path.join(base_dir, 'experiments', 'Taylor_metrics.csv')

def compute_metrics(p, p_hat):
    # Calculate Covariance and Standard Deviation for Taylor Diagram
    # p = Real Pressure, p_hat = Predicted Pressure
    msec = da.multiply(p - p.mean(), p_hat - p_hat.mean()).mean() # Covariance
    sigma = da.sqrt(da.square(p_hat - p_hat.mean()).mean()) #Standard deviation
    return msec.compute(), sigma.compute()

def run_evaluation():
    print(f" Starting Full Evaluation for {WDS_NAME} ({RUNS} runs)...")
    
    # Load Water Network & Graph ONCE
    inp_path = os.path.join('water_networks', f'{WDS_NAME}.inp')
    wds = Network(inp_path)
    G = get_nx_graph(wds, mode='binary')
    # Laplacian is needed for Interpolation baseline
    L = nx.linalg.laplacianmatrix.laplacian_matrix(G).todense()
    num_nodes = len(wds.junctions)

    for run_id in range(1, RUNS + 1):
        print(f"\n--- Processing Run #{run_id} ---")
        
        # 1. Define Paths for this specific run
        run_stamp = f"{WDS_NAME}-{DEPLOY}-{OBSRAT}-{ADJ}-{GNN}-{TAG}-{run_id}"
        model_path = os.path.join(base_dir, 'experiments', 'models', f"{run_stamp}.pt")
        sensor_path = os.path.join(base_dir, 'experiments', 'models', f"{run_stamp}_sensor_nodes.csv")
        
        # 2. Load the EXACT Sensors used in training
        # --- MODIFICACIÓN: Si es 'orig', no necesitamos el archivo de sensores
        if not os.path.exists(sensor_path):
            if args.gnn == 'orig':
                pass # Continuar sin sensores, solo queremos la referencia
            else:
                print(f" Warning: Sensor file not found for {run_stamp}. Skipping.")
                continue
            
        sensor_shop = SensorInstaller(wds)
        if os.path.exists(sensor_path):
            sensor_nodes = np.loadtxt(sensor_path, dtype=np.int32)
            sensor_shop.set_sensor_nodes(sensor_nodes)
        
        # 3. Load Data (Test Set) with this sensor mask
        db_path = os.path.join(base_dir, 'data', f'db_{WDS_NAME}_{DB_NAME}')
        reader = DataReader(
            db_path, 
            n_junc=num_nodes, 
            signal_mask=sensor_shop.signal_mask(),
            node_order=np.array(list(G.nodes))-1
        )
        
        # X is standardized (Inputs), Y is Normalized (Targets)
        tst_x, bias_std, scale_std = reader.read_data('tst', 'junc_heads', 'standardize', cover=True)
        tst_y, bias_nrm, scale_nrm = reader.read_data('tst', 'junc_heads', 'normalize', cover=False)
        
        tst_ldr = build_dataloader(G, tst_x, tst_y, BATCH_SIZE, shuffle=False)
        metrics_nrm = Metrics(bias_nrm, scale_nrm, DEVICE)
        num_graphs = len(tst_x)

        # Get Ground Truth (Unscaled)
        p_real = []
        for batch in tst_ldr:
            batch = batch.to(DEVICE)
            p_real.append(metrics_nrm._rescale(batch.y).reshape(-1, num_nodes).detach().cpu().numpy())
        p_real = da.array(np.concatenate(p_real))

        # --- MODIFICACIÓN: Calcular predicciones según el modelo
        if args.gnn == 'orig':
            # Para la referencia 'orig', la predicción es exactamente el ground truth
            p_gcn = p_real
        else:
            # --- MODEL EVALUATION (ChebNet, GAT, etc) ---
            model = Net(tst_x.shape[-1], tst_y.shape[-1]).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            
            p_gcn = []
            with torch.no_grad():
                for batch in tst_ldr:
                    batch = batch.to(DEVICE)
                    out = model(batch)
                    p_gcn.append(metrics_nrm._rescale(out).reshape(-1, num_nodes).detach().cpu().numpy())
            p_gcn = da.array(np.concatenate(p_gcn))
        

        msec, sigma = compute_metrics(p_real, p_gcn)
        with open(results_file, 'a+', newline='') as f:
            writer = csv.writer(f)
            # Differntiate metrics for the models
            writer.writerow([f"{run_stamp}-{args.gnn}", msec, sigma]) 
        print(f"   {args.gnn} Evaluated")

    print(f"\n All evaluations complete! Results saved to {results_file}")

if __name__ == "__main__":
    run_evaluation()
