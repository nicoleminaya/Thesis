# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse
import os

# ==========================================
# 0. Command-line arguments
# ==========================================
parser = argparse.ArgumentParser(description="Generate GCN summary plots.")
parser.add_argument(
    '--models',
    nargs='+',
    type=str,
    help='List of GCN models to plot (e.g., --models cheb1 gat gat2). If omitted, plots all.'
)
args = parser.parse_args()

# ==========================================
# 1. Load Test and Training Files
# ==========================================
test_files = glob.glob("*_Test_Summary.csv")
train_files = glob.glob("*_Training_Summary.csv")

if not test_files and not train_files:
    print(" Error: No summary CSV files found in the current directory.")
    exit()

print(f"Files found: {len(test_files)} Testing and {len(train_files)} Training.")

# Helper function to load and combine CSVs
def load_and_combine(files_list):
    if not files_list:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files_list]
    df = pd.concat(dfs, ignore_index=True)
    df['GNN_Model'] = df['GNN_Model'].astype(str).str.strip().str.lower()
    return df

df_test = load_and_combine(test_files)
df_train = load_and_combine(train_files)

# Filter by models if requested by the user
if args.models:
    requested_models = [m.lower() for m in args.models]
    if not df_test.empty:
        df_test = df_test[df_test['GNN_Model'].isin(requested_models)]
    if not df_train.empty:
        df_train = df_train[df_train['GNN_Model'].isin(requested_models)]
    print(f" Filtering to show only models: {requested_models}")

# Helper function to center the third subplot, fix missing labels and clean titles
def center_last_subplot(g, xlabel):
    """Centers the 3rd subplot when using col_wrap=2, forces x-labels, and cleans titles."""
    axes = g.axes.flatten()
    
    # Force the x-label, tick labels, and clean titles to be visible on ALL subplots
    for ax in axes:
        # Clean up the title (from "WDS = anytown" to "Anytown" or "BWSN")
        current_title = ax.get_title()
        if "=" in current_title:
            network_name = current_title.split("=")[-1].strip()
        else:
            network_name = current_title
            
        # Format the WDS name correctly
        if network_name.lower() == 'bwsn':
            display_name = "BWSN"
        else:
            display_name = network_name.capitalize()
            
        ax.set_title(display_name, fontsize=13, fontweight='bold')
        
        # Force X-axis labels to appear on all subplots
        ax.set_xlabel(xlabel)
        ax.tick_params(labelbottom=True)
        
    if len(axes) == 3:
        # Get positional data for all 3 subplots
        pos1 = axes[0].get_position()
        pos2 = axes[1].get_position()
        pos3 = axes[2].get_position()
        
        # Calculate the exact center x-coordinate between the first two plots
        new_x0 = (pos1.x0 + pos2.x0) / 2.0
        
        # Apply the new position to the third plot
        axes[2].set_position([new_x0, pos3.y0, pos3.width, pos3.height])

# ==========================================
# 2. Configure Style and Colors
# ==========================================
sns.set_theme(style="whitegrid")

# Your custom color palette to maintain consistency
custom_colors = {
    "cheb1": "#1f77b4",   # Azul
    "cheb2": "#8c564b",   # Brown
    "cheb3": "#ff7f0e" ,  # Orange
    "gat": "#2ca02c"  ,   # Green
    "gat2": "#9467bd",    # Purple
    "gat_hyp": "#d62728"  # Red
}

# Define the exact order for the WDS subplots
wds_order = ['anytown', 'hanoi', 'bwsn']

# ==========================================
# 3. PLOT 1: Performance vs Ratio (Using TEST)
# ==========================================
if not df_test.empty and 'tst_rel_err' in df_test.columns:
    g1 = sns.relplot(
        data=df_test,
        x="Ratio", y="tst_rel_err", hue="GNN_Model", col="WDS",
        col_wrap=2, col_order=wds_order, 
        palette=custom_colors,
        kind="line", marker="o", markersize=8, errorbar="sd",
        linewidth=2, height=5, aspect=1.2, facet_kws={'sharey': False}
    )
    g1.set_axis_labels("Observation Ratio", "Test Rel. Error ")
    g1.fig.suptitle("Model performance vs. Observation ratio - GA sensor placement", y=1.05, fontsize=16, fontweight='bold')
    
    # Apply centering to the bottom plot
    center_last_subplot(g1, "Observation Ratio")

    plot1_name = "Plot_1_Test_Performance.png"
    plt.savefig(plot1_name, bbox_inches='tight', dpi=300)
    print(f" Saved: {plot1_name}")
    plt.close()
else:
    print(" Skipping Plot 1: Missing Test data or 'tst_rel_err' column.")

# ==========================================
# 4. PLOT 2: Stability and Variance (Using TEST)
# ==========================================
if not df_test.empty and 'tst_rel_err' in df_test.columns:
    g2 = sns.catplot(
        data=df_test,
        x="Ratio", y="tst_rel_err", hue="GNN_Model", col="WDS",
        col_wrap=2, col_order=wds_order, 
        palette=custom_colors,
        kind="box", height=5, aspect=1.2, sharey=False,
        boxprops={'alpha': 0.8}
    )
    g2.set_axis_labels("Observation Ratio", "Test Rel. Error ")
    g2.fig.suptitle("Model Stability acros runs - GA sensor placement", y=1.05, fontsize=16, fontweight='bold')
    
    # Apply centering to the bottom plot
    center_last_subplot(g2, "Observation Ratio")

    plot2_name = "Plot_2_Test_Stability.png"
    plt.savefig(plot2_name, bbox_inches='tight', dpi=300)
    print(f" Saved: {plot2_name}")
    plt.close()
else:
    print(" Skipping Plot 2: Missing Test data or 'tst_rel_err' column.")

# ==========================================
# 5. PLOT 3: Convergence Speed (Using TRAINING)
# ==========================================
if not df_train.empty and 'Stopped_At_Epoch' in df_train.columns:
    g3 = sns.catplot(
        data=df_train,
        x="GNN_Model", y="Stopped_At_Epoch", hue="GNN_Model", col="WDS",
        col_wrap=2, col_order=wds_order,  
        palette=custom_colors,
        kind="bar", errorbar="sd", capsize=.1, height=5, aspect=1.2, 
        sharey=False, legend=False
    )
    g3.set_axis_labels("GCN Model", "Epochs to Convergence (Early Stop)")
    g3.fig.suptitle("Convergence Speed - GA sensor placement", y=1.05, fontsize=16, fontweight='bold')
    
    # Apply centering to the bottom plot
    center_last_subplot(g3, "GCN model")

    plot3_name = "Plot_3_Train_Convergence.png"
    plt.savefig(plot3_name, bbox_inches='tight', dpi=300)
    print(f" Saved: {plot3_name}")
    plt.close()
else:
    print(" Skipping Plot 3: Missing Training data or 'Stopped_At_Epoch' column.")

print("\nAll requested plots have been generated!")