# -*- coding: utf-8 -*-
import os
import argparse
import datetime
import time
import pytz
from tqdm import tqdm
import yaml
import zarr
import numpy as np
import dask.array as da
import ray 
import pyDOE2 as doe
from epynet import Network

parser  = argparse.ArgumentParser()
parser.add_argument('--params', default='db_hanoi', type=str)
parser.add_argument('--nproc', default=4, type=int)
parser.add_argument('--batch', default=50, type=int)
args    = parser.parse_args()

pathToRoot      = os.path.dirname(os.path.realpath(__file__))
pathToExps      = os.path.join(pathToRoot, 'experiments')
pathToParam     = os.path.join(pathToExps, 'hyperparams', 'db', args.params+'.yaml') 
with open(pathToParam, 'r') as fin:
    params  = yaml.load(fin, Loader=yaml.Loader)
pathToNetwork   = os.path.join(pathToRoot, 'water_networks', params['wds']+'.inp') 
pathToDB        = os.path.join(pathToRoot, 'data', args.params)     

class SequenceGenerator():
    def __init__(self, store, n_scenes, feat_dict, chunks=None):
        self.store  = store
        self.chunks = chunks
        self.n_scenes   = n_scenes
        self.n_features = feat_dict['juncs'] + 1 # Nodes + 1 for total demand variation

    def design_experiments(self, algo):
        if algo == 'doe':
            design  = doe.lhs(self.n_features, samples=self.n_scenes)
            design  = da.from_array(design, chunks=self.chunks)
            da.to_zarr(design, url=self.store, component='raw_design', compute=True)

    def transform_scenes(self):
        n_junc  = feat_dict['juncs']
        raw_design  = da.from_zarr(url=self.store, component='raw_design')
        
        junc_demands = da.multiply(orig_dmds, da.add(da.multiply(raw_design[:, :n_junc], dmd_hi - dmd_lo), dmd_lo))
        tot_dmds  = da.sum(junc_demands, axis=1, keepdims=True)
        target_tot_dmds = da.multiply(orig_tot_dmd, da.add(da.multiply(raw_design[:, n_junc], tot_dmd_hi-tot_dmd_lo), tot_dmd_lo)).reshape((self.n_scenes, 1))
        
        junc_demands = da.multiply(junc_demands, da.divide(target_tot_dmds, tot_dmds))
        da.to_zarr(junc_demands.astype(np.float32).rechunk(self.chunks), url=self.store, component='junc_demands', compute=True)

@ray.remote
class simulator():
    def __init__(self):
        self.wds = Network(pathToNetwork)
        self.junc_heads = np.empty(shape=(n_batch, n_junc), dtype=np.float32)

    def evaluate_batch(self, scene_ids, boundaries):
        for idx, scene_id in enumerate(scene_ids):
            for junc_id, junc in enumerate(self.wds.junctions):
                junc.basedemand = boundaries[0][scene_id, junc_id]
            self.wds.solve()
            self.junc_heads[idx,:]  = self.wds.junctions.head.values
        return [scene_ids, self.junc_heads]

def print_store_stats(store):
    for key in root.keys():
        arr = da.from_zarr(url=store, component=key)
        print(key)
        print('max: {:.2f}'.format(arr.max().compute()))
        print('min: {:.2f}'.format(arr.min().compute()))
        print('avg: {:.2f}'.format(arr.mean().compute()))
        print('std: {:.2f}'.format(arr.std().compute()))
        print('')

def chunk_computation(boundaries):
    junc_heads  = np.empty_like(boundaries[0], dtype=np.float32)
    boundary_id = ray.put(boundaries)
    workers = [simulator.remote() for i in range(n_proc)]
    results = {}

    scene_id_batches = []
    new_batch = []
    for idx in range(junc_heads.shape[0]):
        if (idx % n_batch) == 0 and idx != 0:
            scene_id_batches.append(new_batch)
            new_batch = []
        new_batch.append(idx)
    if new_batch:
        scene_id_batches.append(new_batch)
    
    progressbar = tqdm(total=len(scene_id_batches))
    for worker in workers:
        if scene_id_batches:
            results[worker.evaluate_batch.remote(scene_id_batches.pop(), boundary_id)] = worker
    
    ready_ids, _ = ray.wait(list(results))
    while ready_ids:
        ready_id= ready_ids[0]
        result  = ray.get(ready_id)
        worker  = results.pop(ready_id)
        if scene_id_batches:
            results[worker.evaluate_batch.remote(scene_id_batches.pop(), boundary_id)] = worker
    
        for idx in range(len(result[0])):
            junc_heads[result[0][idx], :]    = result[1][idx,:]
    
        ready_ids, _    = ray.wait(list(results))
        progressbar.update(1)
    progressbar.close()
    return junc_heads

# ----- Initialization -----
wds         = Network(pathToNetwork)
dmd_lo      = params['demand']['nodalLo'] 
dmd_hi      = params['demand']['nodalHi'] 
tot_dmd_lo  = params['demand']['totalLo'] 
tot_dmd_hi  = params['demand']['totalHi'] 

n_scenes    = params['nScenes']
feat_dict   = { 'juncs' : len(wds.junctions.uid) }
n_proc  = args.nproc
n_batch = args.batch
orig_dmds = np.array(wds.junctions.basedemand, dtype=np.float32).reshape(1, -1)
orig_tot_dmd    = np.sum(wds.junctions.basedemand)

store   = zarr.DirectoryStore(pathToDB)
root    = zarr.group(store=store, overwrite=True, synchronizer=zarr.ThreadSynchronizer())
now     = datetime.datetime.now(pytz.UTC)
root.attrs['creation_date']   = str(now)
root.attrs['gmt_timestap']    = int(now.timestamp())
root.attrs['description']     = 'Hanoi purely gravity-fed design'

scene_generator = SequenceGenerator(store, n_scenes, feat_dict, chunks=(params['chunks']['height'], params['chunks']['width']))

print('Writing unscaled random experiment design to data store... ', end="", flush=True)
scene_generator.design_experiments(params['genAlgo'])
print('OK')

print('Splitting and scaling raw experiment design... ', end="", flush=True)
scene_generator.transform_scenes()
del root['raw_design']
print('OK')
print_store_stats(store)

# ----- Scene evaluation -----
junc_demands_store  = da.from_zarr(url=store, component='junc_demands')
junc_heads_store    = root.empty('junc_heads', shape=junc_demands_store.shape, chunks=(params['chunks']['height'], params['chunks']['width']), dtype='f4')

n_junc  = feat_dict['juncs']
n_experiment= junc_demands_store.shape[0]
chunk_len   = root['junc_demands'].chunks[0]
n_full_batch= n_experiment // chunk_len

ray.init()
time.sleep(2)

for batch_id in range(n_full_batch):
    beg_idx = batch_id*chunk_len
    end_idx = beg_idx + chunk_len
    boundaries  = [np.array(junc_demands_store[beg_idx:end_idx, :])]
    junc_heads_store[beg_idx:end_idx, :] = chunk_computation(boundaries)
    
if n_experiment % chunk_len:
    beg_idx = end_idx
    boundaries  = [np.array(junc_demands_store[beg_idx:, :])]
    junc_heads_store[beg_idx:, :] = chunk_computation(boundaries)

ray.shutdown()

head_treshold   = 0
junc_heads  = da.from_zarr(url=store, component='junc_heads')
min_heads   = junc_heads.min(axis=1).compute()
idx_ok      = np.where(min_heads > head_treshold)[0]

if len(idx_ok) < len(min_heads):
    for key in root.keys():
        arr = da.from_zarr(root[key])
        arr.to_zarr(url=store, component=key+'-tmp', overwrite=True, compute=True)
        arr = da.from_zarr(root[key+'-tmp'])
        arr = arr[idx_ok, :].rechunk(scene_generator.chunks).to_zarr(url=store, component=key, overwrite=True, compute=True)
        del root[key+'-tmp']

print('-----')
print_store_stats(store)

# ----- Splitting -----
vld_split   = params['data']['vldSplit']
tst_split   = params['data']['tstSplit']
idx_trn = int(np.floor(len(idx_ok) * (1-tst_split-vld_split)))
idx_vld = int(np.floor(len(idx_ok) * (1-tst_split)))

unsplit_keys= list(root.keys())
root_trn    = zarr.hierarchy.group(store=store, overwrite=True, synchronizer=zarr.ThreadSynchronizer(), path='trn')
root_vld    = zarr.hierarchy.group(store=store, overwrite=True, synchronizer=zarr.ThreadSynchronizer(), path='vld')
root_tst    = zarr.hierarchy.group(store=store, overwrite=True, synchronizer=zarr.ThreadSynchronizer(), path='tst')

for key in unsplit_keys:
    arr = da.from_zarr(root[key])
    arr_avg = da.mean(arr[:idx_trn, :]).compute()
    arr_std = da.std(arr[:idx_trn, :]).compute()
    arr_min = da.min(arr[:idx_trn, :]).compute()
    arr_max = da.max(arr[:idx_trn, :]).compute()
    arr_range   = arr_max - arr_min

    arr[:idx_trn, :].to_zarr(url=store, component='trn/'+key, overwrite=True, compute=True)
    arr[idx_trn:idx_vld, :].rechunk(scene_generator.chunks).to_zarr(url=store, component='vld/'+key, overwrite=True, compute=True)
    arr[idx_vld:, :].rechunk(scene_generator.chunks).to_zarr(url=store, component='tst/'+key, overwrite=True, compute=True)

    root_trn[key].attrs['avg'] = float(arr_avg)
    root_trn[key].attrs['std'] = float(arr_std)
    root_trn[key].attrs['min'] = float(arr_min)
    root_trn[key].attrs['range'] = float(arr_range)
    del root[key]

print(root.tree())