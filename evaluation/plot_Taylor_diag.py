# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure we can import the sibling file 'taylorDiagram.py'
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)

from taylorDiagram import TaylorDiagram

# ----- ----- ----- ----- ----- -----
# Command line arguments
# ----- ----- ----- ----- ----- -----
parser  = argparse.ArgumentParser()
parser.add_argument('--wds', default='anytown', type=str)
parser.add_argument('--file', default='Taylor_metrics_processed.csv', type=str)
parser.add_argument('--extend', default=None, type=float)
parser.add_argument('--smin', default=0, type=float)
parser.add_argument('--smax', default=1.2, type=float)
parser.add_argument('--legend', action='store_true')
parser.add_argument('--fill', action='store_true')
parser.add_argument('--individual', action='store_true', help="Plot individual runs")
parser.add_argument('--nocenter', action='store_true', help="Do NOT plot the average (center) of the runs")
parser.add_argument('--savepdf', action='store_true')
parser.add_argument('--tag', default='def', type=str)
parser.add_argument('--models', nargs='+', type=str, help="List of models to plot (e.g., --models gat cheb1 naive). Defaults to all.")

args = parser.parse_args()

# ----- ----- ----- ----- ----- -----
# DB loading
# ----- ----- ----- ----- ----- -----
csv_path = os.path.join(current_dir, '..', 'experiments', args.file)

if not os.path.exists(csv_path):
    print(f" Error: File not found at {csv_path}")
    print("    Make sure you ran your evaluation and processing scripts first!")
    exit(1)

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip().str.replace(';', '', regex=False)

required_cols = ['wds', 'model', 'sigma_pred', 'obs_rat', 'corr_coeff']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas en el CSV: {missing}. Columnas detectadas: {list(df.columns)}")


wds = args.wds
# Filter for Reference (orig) to get the "1.0" mark
ref_data = df.loc[(df['wds'] == wds) & (df['model'] == 'orig')]

if ref_data.empty:
    print(f" Error: No reference data found for WDS '{wds}'.")
    exit(1)

std_ref = ref_data['sigma_pred'].values[0]

# ----- ----- ----- ----- ----- -----
# Plot assembly
# ----- ----- ----- ----- ----- -----
fig = plt.figure(figsize=(12, 9)) # Slightly wider to fit the big legend
dia = TaylorDiagram(1.0, fig=fig, label='Reference', extend=args.extend, srange=(args.smin, args.smax))
dia.samplePoints[0].set_color('r')
dia.samplePoints[0].set_marker('P')
dia.samplePoints[0].set_markersize(12)
cmap = plt.get_cmap('tab10')

available_ratios = sorted(df['obs_rat'].unique())

# Define ALL possible models
all_models = {
    'cheb1':  {'marker': 'D', 'label': 'Cheb1'},   
    'cheb2':  {'marker': '^', 'label': 'Cheb2'},   
    'cheb3':  {'marker': 'o', 'label': 'Cheb3'}, 
    'gat':    {'marker': 'P', 'label': 'GAT'},
    'gat_hyp':  {'marker': 'v', 'label': 'GAT_hyp'}, 
    'gat2': {'marker': '*', 'label': 'GATv2'},
    'gat2_log': {'marker': 's', 'label': 'GATv2 log'},
    'gat2_weig': {'marker': 'X', 'label': 'GATv2 weig'}    
}

# Filter based on user input
if args.models:
    # Only keep the models that the user requested
    models_to_plot = {k: v for k, v in all_models.items() if k in args.models}
    
    # Safety check if you typed a name wrong
    if not models_to_plot:
        print(f" Warning: None of the requested models {args.models} were recognized. Plotting all.")
        models_to_plot = all_models
else:
    # If no argument is passed, plot everything
    models_to_plot = all_models

# ==========================================
#  Plot Individual Points (Semi-transparent)
# ==========================================
if args.individual:
    for i, obs_rat in enumerate(available_ratios):
        for model_name, props in models_to_plot.items():
            model_df = df[(df['wds'] == wds) & (df['obs_rat'] == obs_rat) & (df['model'] == model_name)]
            
            if not model_df.empty:
                dia.add_sample(
                    model_df['sigma_pred'].values / std_ref, 
                    model_df['corr_coeff'].values,
                    marker=props['marker'], ms=5, ls='', mfc=cmap(i), mec='none', alpha=0.3
                )

# ==========================================
# Plot Averages / Centers (Large, Hollow)
# ==========================================
if not args.nocenter:
    for i, obs_rat in enumerate(available_ratios):
        for model_name, props in models_to_plot.items():
            model_df = df[(df['wds'] == wds) & (df['obs_rat'] == obs_rat) & (df['model'] == model_name)]
            
            if not model_df.empty:
                dia.add_sample(
                    (model_df['sigma_pred'].values / std_ref).mean(), 
                    model_df['corr_coeff'].values.mean(),
                    marker=props['marker'], ms=10, ls='', mfc='none', mec=cmap(i), mew=2.5,
                    label=f"{props['label']} (OR={obs_rat})"
                )

# Add Contours (RMSE)
contours = dia.add_contours(levels=5, colors='0.5', linestyles='dashed', alpha=0.5)
plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

dia.add_grid()
dia._ax.axis[:].major_ticks.set_tick_out(True)

if args.legend:
    # Moved the legend slightly outward so it doesn't overlap the diagram
    plt.legend(loc='upper left', bbox_to_anchor=(0.95, 1.0), fontsize=10)

plt.title(f"Taylor Diagram - {wds.capitalize()}", y=1.05, fontsize=14)
plt.tight_layout()

if args.savepdf:
    plt.savefig(f'taylor-{args.wds}.pdf', format='pdf', bbox_inches='tight')
else:
    plt.show()