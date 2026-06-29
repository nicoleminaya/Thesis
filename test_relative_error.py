# -*- coding: utf-8 -*-
import os
import argparse
from csv import writer
import numpy as np
import dask.array as da
import pandas as pd
import torch
from epynet import Network

from utils.graph_utils import get_nx_graph
from utils.DataReader import DataReader
from utils.Metrics import Metrics
from utils.dataloader import build_dataloader
from utils.SensorInstaller import SensorInstaller

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- ----- ----- ----- ----- -----
# Command line arguments
# ----- ----- ----- ----- ----- -----
parser  = argparse.ArgumentParser()
parser.add_argument('--wds', default='anytown', type=str, help="Water distribution system.")
parser.add_argument('--model', default='cheb1', type=str, help="Water distribution system.")
parser.add_argument('--batch', default=80, type=int, help="Batch size.")
parser.add_argument('--deploy', default='random', type=str, help="How to setup the transducers (e.g., xrandom, random, dist).")
parser.add_argument('--obsrat', default=0.05, type=float, help="Observation ratio to evaluate.")
parser.add_argument('--runs', default=22, type=int, help="Number of runs to evaluate.")
parser.add_argument('--tag', default='def', type=str, help="Custom tag.")
parser.add_argument('--db', default='doe_pumpfed_1', type=str, help="DB.")
parser.add_argument('--adj', default='binary', type=str, help="Adjacency matrix type.")
args    = parser.parse_args()

# ----- ----- ----- ----- ----- -----
# Paths
# ----- ----- ----- ----- ----- -----
wds_name    = args.wds
pathToRoot  = os.path.dirname(os.path.realpath(__file__))
pathToExps  = os.path.join(pathToRoot, 'experiments')
pathToDB    = os.path.join(pathToRoot, 'data', 'db_' + wds_name +'_'+ args.db)
pathToWDS   = os.path.join('water_networks', wds_name+'.inp')
pathToResults   =  os.path.join(
    pathToRoot, 'experiments', f'relative_error-{args.wds}-{args.deploy}-{args.obsrat}.csv'
)

# ----- ----- ----- ----- ----- -----
# Functions
# ----- ----- ----- ----- ----- -----
def restore_real_nodal_p(dta_ldr, num_nodes, num_graphs, metrics):
    nodal_pressures = np.empty((num_graphs, num_nodes))
    end_idx = 0
    for batch in dta_ldr:
        batch = batch.to(device)
        p   = metrics._rescale(batch.y).reshape(-1, num_nodes).detach().cpu().numpy()
        nodal_pressures[end_idx:end_idx+batch.num_graphs, :]    = p
        end_idx += batch.num_graphs
    return da.array(nodal_pressures)

def predict_nodal_p_gcn(dta_ldr, num_nodes, num_graphs, metrics, model_path, Net, tst_x, tst_y):
    model = Net(np.shape(tst_x)[-1], np.shape(tst_y)[-1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    nodal_pressures = np.empty((num_graphs, num_nodes))
    end_idx = 0
    for batch in dta_ldr:
        batch = batch.to(device)
        p   = model(batch)
        p   = metrics._rescale(p).reshape(-1, num_nodes).detach().cpu().numpy()
        nodal_pressures[end_idx:end_idx+batch.num_graphs, :]    = p
        end_idx += batch.num_graphs
    return da.array(nodal_pressures)

def load_model():
    if args.wds == 'anytown':
        #from model.anytown import ChebNet as Net #FOR CHEB1
        #from model.anytown_v2 import ChebNet as Net #FOR CHEB2
        #from model.anytown_gat import GATNet as Net #FOR GAT
        from model.anytown_gat_v2 import GATv2ResNet as Net #FOR GATv2
    elif args.wds == 'ctown':
        from model.ctown import ChebNet as Net
    elif args.wds == 'richmond':
        from model.richmond import ChebNet as Net
    else:
        raise ValueError('Water distribution system is unknown.')
    return Net

def compute_metrics(p, p_hat):
    diff    = da.subtract(p, p_hat)
    rel_diff= da.divide(diff, p)
    return rel_diff

# ----- ----- ----- ----- ----- -----
# Loading datasets
# ----- ----- ----- ----- ----- -----
wds = Network(pathToWDS)
G   = get_nx_graph(wds, mode=args.adj)

run_ids = np.arange(args.runs) + 1
df_list = []



for run_id in run_ids:
    # Build the run stamp exactly matching your trained models
    run_stamp   = f"{wds_name}-random-0.05-{args.adj}-{args.model}-{args.tag}_{run_id}-1"
    pathToModel = os.path.join(pathToExps, 'models', run_stamp+'.pt')
    pathToSens  = os.path.join(pathToExps, 'models', run_stamp+'_sensor_nodes.csv')
    
    if not os.path.exists(pathToModel) or not os.path.exists(pathToSens):
        print(f"Skipping {run_id}. run... Model or sensor file not found: {run_stamp}")
        continue
        
    print(f"Processing {run_id}. run ({run_stamp})...")

    # 1. Load the EXACT Sensors used in training
    sensor_shop = SensorInstaller(wds, include_pumps_as_master=True)
    sensor_nodes = np.loadtxt(pathToSens, dtype=np.int32)
    sensor_shop.set_sensor_nodes(sensor_nodes)

    # 2. Pass the exact signal_mask to the DataReader
    reader  = DataReader(
        pathToDB, 
        n_junc=len(wds.junctions.uid), 
        signal_mask=sensor_shop.signal_mask(),
        node_order=np.array(list(G.nodes))-1
    )
    
    tst_x, _, _ = reader.read_data(dataset='tst', varname='junc_heads', rescale='standardize', cover=True)
    tst_y, _, _ = reader.read_data(dataset='tst', varname='junc_heads', rescale='normalize', cover=False)
    _, bias_y, scale_y  = reader.read_data(dataset='trn', varname='junc_heads', rescale='normalize', cover=False)
    
    tst_ldr = build_dataloader(G, tst_x, tst_y, args.batch, shuffle=False)
    metrics = Metrics(bias_y, scale_y, device)
    
    num_nodes   = len(wds.junctions)
    num_graphs  = len(tst_x)

    Net = load_model()
    p       = restore_real_nodal_p(tst_ldr, num_nodes, num_graphs, metrics)
    p_hat   = predict_nodal_p_gcn(tst_ldr, num_nodes, num_graphs, metrics, pathToModel, Net, tst_x, tst_y)
    
    rel_err = compute_metrics(p, p_hat)
    df  = pd.DataFrame(rel_err.compute()).abs()
    
    # Calculate the mean relative error for each node specifically for this run
    mean_errors = df.mean(axis=0).to_dict()
    mean_errors['runid'] = run_id
    df_list.append(mean_errors)



# Combine and save results
if df_list:
    final_df = pd.DataFrame(df_list)
    # Reorder columns to put runid first
    cols = ['runid'] + [col for col in final_df.columns if col != 'runid']
    final_df = final_df[cols]
    final_df.to_csv(pathToResults, index=False)
    print(f"\nSaved relative errors for all nodes to:\n{pathToResults}")
else:
    print("\nNo models were successfully evaluated. Check your arguments!")