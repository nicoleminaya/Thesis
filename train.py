# -*- coding: utf-8 -*-
import argparse
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.nn import ChebConv
from epynet import Network

from utils.graph_utils import get_nx_graph, get_sensitivity_matrix
from utils.DataReader import DataReader
from utils.SensorInstaller import SensorInstaller
from utils.Metrics import Metrics
from utils.EarlyStopping import EarlyStopping
from utils.dataloader import build_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----- ----- ----- ----- ----- -----
# Command line arguments
# ----- ----- ----- ----- ----- -----
parser  = argparse.ArgumentParser()
parser.add_argument('--wds',
                    default = 'anytown',
                    type    = str,
                    help    = "Water distribution system.")
parser.add_argument('--db',
                    default = 'doe_pumpfed_1',
                    type    = str,
                    help    = "DB.")
parser.add_argument('--obsrat',
                    default = 0.05,
                    type    = float,
                    help    = "Observation ratio.")
parser.add_argument('--adj',
                    default = 'binary',
                    choices = ['binary', 'weighted', 'logarithmic', 'pruned'],
                    type    = str,
                    help    = "Type of adjacency matrix.")
parser.add_argument('--deploy',
                    default = 'random',
                    choices = ['master', 'dist', 'hydrodist', 'hds', 'hdvar', 'random', 'xrandom', 'gena'],
                    type    = str,
                    help    = "Method of sensor deployment.")
parser.add_argument('--epoch',
                    default = 1,
                    type    = int,
                    help    = "Number of epochs.")
parser.add_argument('--idx',
                    default = None,
                    type    = int,
                    help    = "Dev function.")
parser.add_argument('--batch',
                    default = '40',
                    type    = int,
                    help    = "Batch size.")
parser.add_argument('--lr',
                    default = 0.0003,
                    type    = float,
                    help    = "Learning rate.")
parser.add_argument('--decay',
                    default = 0.000006,
                    type    = float,
                    help    = "Weight decay.")
parser.add_argument('--tag',
                    default = 'def',
                    type    = str,
                    help    = "Custom tag.")
parser.add_argument('--deterministic',
                    action  = "store_true",
                    help    = "Setting random seed for sensor placement.")
parser.add_argument('--gnn',    #TO INCLUDE GATs
                    default = 'cheb1',
                    choices = ['cheb1', 'cheb2', 'cheb3', 'gat','gat_hyp' ,'gat2'],
                    type    = str,
                    help    = "GNN architecture to use.")
args    = parser.parse_args()

# ----- ----- ----- ----- ----- -----
# Paths
# ----- ----- ----- ----- ----- -----
wds_name    = args.wds
pathToRoot  = os.path.dirname(os.path.realpath(__file__))
pathToDB    = os.path.join(pathToRoot, 'data', 'db_' + wds_name +'_'+ args.db) # Data from generatedata.py output
pathToExps  = os.path.join(pathToRoot, 'experiments')
pathToLogs  = os.path.join(pathToExps, 'logs')
run_id  = 1
logs    = [f for f in glob.glob(os.path.join(pathToLogs, '*.csv'))]         # See if there was previous created logs for wds_deploy_budget_weightMatrix_tag
#run_stamp   = wds_name+'-'+args.deploy+'-'+str(args.obsrat)+'-'+args.adj+'-'+args.tag+'-'
run_stamp   = wds_name+'-'+args.deploy+'-'+str(args.obsrat)+'-'+args.adj+'-'+args.gnn+'-'+args.tag+'-'
while os.path.join(pathToLogs, run_stamp + str(run_id)+'.csv') in logs:
    run_id  += 1
run_stamp   = run_stamp + str(run_id)
pathToLog   = os.path.join(pathToLogs, run_stamp+'.csv')
pathToModel = os.path.join(pathToExps, 'models', run_stamp+'.pt')
pathToMeta  = os.path.join(pathToExps, 'models', run_stamp+'_meta.csv') # Parameter in the training call/corresponding log in expriments/logs
pathToSens  = os.path.join(pathToExps, 'models', run_stamp+'_sensor_nodes.csv') # Used sensor locations
pathToWDS   = os.path.join('water_networks', wds_name+'.inp')


# Customer arrangement
if args.wds == 'anytown':
    CUSTOM_ARRANGEMENTS = {
    0.05: [4],                
    0.1: [4, 8],        
    0.2: [4, 8, 18, 10],
    0.4: [5, 8, 9, 11, 12, 19, 13, 2],
    0.8: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]
    }
elif args.wds == 'hanoi':
    CUSTOM_ARRANGEMENTS = {
    0.05: [14],                
    0.1: [2, 11, 24],        
    0.2: [9, 12, 18, 20, 24, 27],
    0.4: [2, 5, 8, 11, 12, 14, 16, 20, 22, 25, 28, 30],
    0.8: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 20, 22, 23, 24, 25, 27, 28, 29, 31]
    }
elif args.wds == 'bwsn':
    CUSTOM_ARRANGEMENTS = {
    0.05: [1, 23, 38, 59, 89, 124],                
    0.1: [10, 18, 23, 40, 55, 62, 78, 92, 111, 120, 122, 125],        
    0.2: [7, 13, 15, 19, 20, 26, 31, 34, 38, 41, 46, 53, 54, 62, 66, 71, 75, 80, 93, 99, 112, 116, 120, 122, 124],
    0.4: [5, 7, 8, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 24, 28, 31, 33, 34, 37, 38, 39, 41, 42, 43, 44, 46, 52, 54, 55, 62, 64, 69, 71, 72, 75, 79, 82, 83, 88, 92, 98, 101, 103, 110, 112, 114, 116, 120, 122, 124],
    0.8: [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 104, 110, 112, 115, 116, 118, 120, 122, 125]
    }
else:
    CUSTOM_ARRANGEMENTS = {
    0.05: [4],                
    0.10: [4, 8],        
    0.20: [4, 8, 18, 10],
    0.40: [5, 8, 9, 11, 12, 19, 13, 2],
    0.80: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]
    }



# ----- ----- ----- ----- ----- -----
# Saving hyperparams
# ----- ----- ----- ----- ----- -----
hyperparams = {
        'db': args.db,
        'deploy': args.deploy,
        #'budget': args.budget,
        'obsrat': args.obsrat,
        'adj': args.adj,
        'epoch': args.epoch,
        'batch': args.batch,
        'lr': args.lr,
        }
hyperparams = pd.Series(hyperparams)
hyperparams.to_csv(pathToMeta, header=False)

# ----- ----- ----- ----- ----- -----
# Functions
# ----- ----- ----- ----- ----- -----
def train_one_epoch(): # PyTorch training loop
    model.train()
    total_loss  = 0
    for batch in trn_ldr:
        batch   = batch.to(device)
        optimizer.zero_grad()
        out     = model(batch)
        loss    = F.mse_loss(out, batch.y) # average MSE error for all #batch scenes pack
        loss.backward() # gradients calculation and update of internal weights 
        optimizer.step()
        total_loss  += loss.item() * batch.num_graphs # 
    return total_loss / len(trn_ldr.dataset) #once all scenes were checkout, extracts the average error

def eval_metrics(dataloader): # Measures how the model is doing
    model.eval()
    n   = len(dataloader.dataset)
    tot_loss        = 0
    tot_rel_err     = 0
    tot_rel_err_obs = 0
    tot_rel_err_hid = 0
    for batch in dataloader:
        batch   = batch.to(device)
        out     = model(batch)
        loss    = F.mse_loss(out, batch.y)
        rel_err = metrics.rel_err(out, batch.y)
        rel_err_obs = metrics.rel_err(
            out,
            batch.y,
            batch.x[:, -1].type(torch.bool) #Mask used to separate observed nodes
            )
        rel_err_hid = metrics.rel_err(
            out,
            batch.y,
            ~batch.x[:, -1].type(torch.bool)
            )
        tot_loss        += loss.item() * batch.num_graphs
        tot_rel_err     += rel_err.item() * batch.num_graphs
        tot_rel_err_obs += rel_err_obs.item() * batch.num_graphs
        tot_rel_err_hid += rel_err_hid.item() * batch.num_graphs
    loss        = tot_loss / n
    rel_err     = tot_rel_err / n
    rel_err_obs = tot_rel_err_obs / n
    rel_err_hid = tot_rel_err_hid / n
    return loss, rel_err, rel_err_obs, rel_err_hid

# ----- ----- ----- ----- ----- -----
# Loading trn and vld datasets
# ----- ----- ----- ----- ----- -----
wds = Network(pathToWDS)                                                # EPINET use
G   = get_nx_graph(wds, mode=args.adj)                                  # Converts EPANET network .inp file into a graph

if args.deterministic:
    seeds   = [1, 8, 5266, 739, 88867]
    seed    = seeds[run_id % len(seeds)]
else:
    seed    = None

sensor_budget   = int(len(wds.junctions) * args.obsrat)
#sensor_budget   = args.budget
print('Deploying {} sensors...\n'.format(sensor_budget))

sensor_shop = SensorInstaller(wds, include_pumps_as_master=True)        # Where to put sensors based on --deploy arg


#######
if args.deploy == 'master':
    sensor_shop.set_sensor_nodes(sensor_shop.master_nodes)
elif args.deploy == 'gena':

    # 1. Initialize the installer and grab the fixed Master Nodes
    fixed_master_nodes = set(sensor_shop.master_nodes)

    # 2. Fetch your specific arrangement for the current ratio
    if args.obsrat in CUSTOM_ARRANGEMENTS:
        # Convert your custom list to a set (handling strings vs ints if necessary)
        custom_nodes = set([int(n) for n in CUSTOM_ARRANGEMENTS[args.obsrat]]) 
        # Note: Remove `str(n)` if your nodes are strict integers in the wds object
    else:
        print(f" Warning: No custom arrangement found for ratio {args.obsrat}!")
        print(" Falling back to master nodes only.")
        custom_nodes = set()

    # 3. Combine your custom nodes with the required master nodes
    all_sensors = fixed_master_nodes.union(custom_nodes)

    # 4. Lock them into the installer
    sensor_shop.set_sensor_nodes(all_sensors)

    print(f" Total sensors deployed ({len(all_sensors)}): {sensor_shop.sensor_nodes}")

elif args.deploy == 'dist':
    sensor_shop.deploy_by_shortest_path(
            sensor_budget   = sensor_budget,
            weight_by       = 'length',
            sensor_nodes    = sensor_shop.master_nodes
            )
elif args.deploy == 'hydrodist':
    sensor_shop.deploy_by_shortest_path(
            sensor_budget   = sensor_budget,
            weight_by       = 'iweight',
            sensor_nodes    = sensor_shop.master_nodes
            )
elif args.deploy == 'hds':
    print('Calculating nodal sensitivity to demand change...\n')
    ptb = np.max(wds.junctions.basedemand) / 100
    S   = get_sensitivity_matrix(wds, ptb)
    sensor_shop.deploy_by_shortest_path_with_sensitivity(
            sensor_budget   = sensor_budget,
            node_weights_arr= np.sum(np.abs(S), axis=0),
            weight_by       = 'iweight',
            sensor_nodes    = sensor_shop.master_nodes
            )
elif args.deploy == 'hdvar':
    print('Calculating nodal head variation...\n')
    reader  = DataReader(
                pathToDB,
                n_junc  = len(wds.junctions),
                node_order  = np.array(list(G.nodes))-1
                )
    heads, _, _ = reader.read_data(
        dataset = 'trn',
        varname = 'junc_heads',
        rescale = None,
        cover   = False
        )
    sensor_shop.deploy_by_shortest_path_with_sensitivity(
            sensor_budget   = sensor_budget,
            node_weights_arr= heads.std(axis=0).T[0],
            weight_by       = 'iweight',
            sensor_nodes    = sensor_shop.master_nodes
            )
    del reader, heads
elif args.deploy == 'random':
    sensor_shop.deploy_by_random(
            sensor_budget   = len(sensor_shop.master_nodes)+sensor_budget,
            seed            = seed
            )
elif args.deploy == 'xrandom':
    sensor_shop.deploy_by_xrandom(
            sensor_budget   = sensor_budget,
            seed            = seed,
            sensor_nodes    = sensor_shop.master_nodes
            )
else:
    print('Sensor deployment technique is unknown.\n')
    raise

#if args.idx:
#    sensor_shop.set_sensor_nodes([args.idx])

if args.idx:                                                                            #add these lines to also consider master nodes
    # Grab the default master nodes (tanks/pumps) and add our specific node to them
    combined_nodes = sensor_shop.master_nodes.copy()
    combined_nodes.add(args.idx)
    sensor_shop.set_sensor_nodes(combined_nodes)

#np.savetxt(pathToSens, np.array(list(sensor_shop.sensor_nodes)), fmt='%d')
np.savetxt(pathToSens, np.array(list(sensor_shop.sensor_nodes), dtype=int), fmt='%d')

reader  = DataReader(
            pathToDB,
            n_junc  = len(wds.junctions),
            signal_mask = sensor_shop.signal_mask(),
            node_order  = np.array(list(G.nodes))-1
            )

trn_x, _, _ = reader.read_data(                     # Unknown nodes to 0
    dataset = 'trn',
    varname = 'junc_heads',
    rescale = 'standardize',
    cover   = True
    )
trn_y, bias_y, scale_y  = reader.read_data(         # Real data?
    dataset = 'trn',
    varname = 'junc_heads',
    rescale = 'normalize',
    cover   = False
    )
vld_x, _, _ = reader.read_data(
    dataset = 'vld',
    varname = 'junc_heads',
    rescale = 'standardize',
    cover   = True
    )
vld_y, _, _ = reader.read_data(
    dataset = 'vld',
    varname = 'junc_heads',
    rescale = 'normalize',
    cover   = False
    )

    #SELECT Corresponding gnn MODEL (CHEBNET O GAT)
if args.gnn == 'gatres':
    # Since GATRes is generic and doesn't rely on fixed polynomial sizes, 
    # the same model file for all WDS topologies!
    from model.gatres import GATResNet as Net
else:
    if args.wds == 'anytown':
        if args.gnn == 'gat':
            from model.anytown_gat import GATNet as Net # Comparativa final (20 runs)
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
    elif args.wds == 'ctown':
        if args.gnn == 'gat':
            from model.ctown_gat import GATNet as Net
        elif args.gnn == 'cheb2':
            from model.ctown_v2 import ChebNet as Net
        elif args.gnn == 'cheb3':
            from model.ctown_v3 import ChebNet as Net
        elif args.gnn == 'gat2':
            from model.ctown_gat_v2 import CtownGATv2ResNet as Net
        else:
            from model.ctown import ChebNet as Net
    elif args.wds == 'richmond':
        from model.richmond import ChebNet as Net
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
    elif args.wds == 'hanoi':
        if args.gnn == 'cheb2':
            from model.hanoi_v2 import ChebNet as Net
        elif args.gnn == 'cheb3':
            from model.hanoi_v3 import ChebNet as Net
        elif args.gnn == 'gat':
            from model.hanoi_gat import GATNet as Net
        elif args.gnn == 'gat2':
            from model.hanoi_gat_v2 import GATv2Net as Net
        else:
            from model.hanoi import ChebNet as Net
    else:
        print('Water distribution system is unknown.\n')
        raise


model = Net(np.shape(trn_x)[-1], np.shape(trn_y)[-1]).to(device)

if args.gnn == 'gatres':
    # GATRes uses a dynamic number of blocks, so we apply the optimizer to all parameters
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.decay,
        eps=1e-7
    )
else:
    # NEW OPTIMIZER CODE so it can work for any model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    # Original logic for ChebNet and standard GAT
    #optimizer = torch.optim.Adam([
    #    dict(params=model.conv1.parameters(), weight_decay=args.decay),
    #    dict(params=model.conv2.parameters(), weight_decay=args.decay),
    #    dict(params=model.conv3.parameters(), weight_decay=args.decay),
    #    dict(params=model.conv4.parameters(), weight_decay=0)
    #    ],
    #    lr  = args.lr,
    #    eps = 1e-7
    #)

# ----- ----- ----- ----- ----- -----
# Training
# ----- ----- ----- ----- ----- -----
trn_ldr = build_dataloader(G, trn_x, trn_y, args.batch, shuffle=True)
vld_ldr = build_dataloader(G, vld_x, vld_y, args.batch, shuffle=False)
metrics = Metrics(bias_y, scale_y, device)
estop   = EarlyStopping(min_delta=.00001, patience=20)
results = pd.DataFrame(columns=[
    'trn_loss', 'vld_loss', 'vld_rel_err', 'vld_rel_err_o', 'vld_rel_err_h'
    ])
header  = ''.join(['{:^15}'.format(colname) for colname in results.columns])
header  = '{:^5}'.format('epoch') + header
best_vld_loss   = np.inf
for epoch in range(0, args.epoch):
    trn_loss    = train_one_epoch()
    vld_loss, vld_rel_err, vld_rel_err_obs, vld_rel_err_hid = eval_metrics(vld_ldr)
    new_results = pd.Series({
        'trn_loss'      : trn_loss,
        'vld_loss'      : vld_loss,
        'vld_rel_err'   : vld_rel_err,
        'vld_rel_err_o' : vld_rel_err_obs,
        'vld_rel_err_h' : vld_rel_err_hid
        })
    results = pd.concat([results, pd.DataFrame([new_results])], ignore_index=True) #results = results.append(new_results, ignore_index=True)
    if epoch % 20 == 0:
        print(header)
    values  = ''.join(['{:^15.6f}'.format(value) for value in new_results.values])
    print('{:^5}'.format(epoch) + values)
    if vld_loss < best_vld_loss:
        best_vld_loss   = vld_loss
        torch.save(model.state_dict(), pathToModel)
    if estop.step(torch.tensor(vld_loss)):
        print('Early stopping...')
        break
results.to_csv(pathToLog)

# ----- ----- ----- ----- ----- -----
# Testing
# ----- ----- ----- ----- ----- -----
if best_vld_loss is not np.inf:
    print('Testing...\n')
    del trn_ldr, vld_ldr, trn_x, trn_y, vld_x, vld_y
    tst_x, _, _ = reader.read_data(
        dataset = 'tst',
        varname = 'junc_heads',
        rescale = 'standardize',
        cover   = True
        )
    tst_y, _, _ = reader.read_data(
        dataset = 'tst',
        varname = 'junc_heads',
        rescale = 'normalize',
        cover   = False
        )
    tst_ldr = build_dataloader(G, tst_x, tst_y, args.batch, shuffle=False)
    model.load_state_dict(torch.load(pathToModel))
    model.eval()
    tst_loss, tst_rel_err, tst_rel_err_obs, tst_rel_err_hid = eval_metrics(tst_ldr)
    results = pd.Series({
        'tst_loss'      : tst_loss,                 # 
        'tst_rel_err'   : tst_rel_err,              # Global error
        'tst_rel_err_o' : tst_rel_err_obs,          # Error in observed nodes, should be ~ to 0
        'tst_rel_err_h' : tst_rel_err_hid           # Error in the non-observed nodes, HIGH important
        })
    results.to_csv(pathToLog[:-4]+'_tst.csv')
