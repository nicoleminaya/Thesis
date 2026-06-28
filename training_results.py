# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse

# ==========================================
# 0. Command-line arguments
# ==========================================
parser = argparse.ArgumentParser(description="Generate GNN summary plots.")
parser.add_argument(
    '--models',
    nargs='+',
    type=str,
    help='List of GNN models to plot (e.g. --models cheb1 gat gat2). If omitted, all models are plotted.'
)
args = parser.parse_args()

# ==========================================
# 1. Automatically find all summary files
# ==========================================
summary_files = glob.glob("*_Training_Summary.csv")

if not summary_files:
    print("No summary CSV files found in the current directory.")
    exit()

print(f"Found {len(summary_files)} files. Generating plots...")

# Load and combine all the CSV files into one Master DataFrame
dfs = [pd.read_csv(f) for f in summary_files]
master_df = pd.concat(dfs, ignore_index=True)

# Limpieza básica
master_df['GNN_Model'] = master_df['GNN_Model'].astype(str).str.strip().str.lower()

# ==========================================
# 2. Setup the visual style
# ==========================================
sns.set_theme(style="whitegrid")

custom_colors = {
    'cheb1': '#1f77b4',     # Blue
    'cheb2': '#ff7f0e',     # Orange
    'cheb3': '#2ca02c',     # Green
    'gat': '#d62728',       # Red
    'gat2': '#9467bd',      # Purple
    'gat_hyp': '#8c564b'    # Brown
}

# ==========================================
# 3. Filter by selected models (optional)
# ==========================================
if args.models:
    requested_models = [m.strip().lower() for m in args.models]
    available_models = sorted(master_df['GNN_Model'].dropna().unique())

    master_df = master_df[master_df['GNN_Model'].isin(requested_models)]

    if master_df.empty:
        print(f"Warning: None of the requested models {requested_models} were found.")
        print(f"Available models in the data: {available_models}")
        exit()

    # Ajustar la paleta solo a los modelos seleccionados
    plot_palette = {m: custom_colors[m] for m in requested_models if m in custom_colors}
else:
    plot_palette = custom_colors

# ==========================================
# 4. Plot 1: Performance vs Observation Ratio
# ==========================================
g1 = sns.relplot(
    data=master_df,
    x="Ratio", y="vld_rel_err", hue="GNN_Model", col="WDN",
    palette=plot_palette,
    kind="line", marker="o", markersize=8, errorbar="sd",
    linewidth=2, height=5, aspect=1.2, facet_kws={'sharey': False}
)
g1.set_axis_labels("Observation Ratio", "Node Rel. Error")
g1.fig.suptitle("Model Performance vs Observation Ratio (Lower is Better)", y=1.05, fontsize=16, fontweight='bold')

plot1_name = "Plot_1_Performance_vs_Ratio"
if args.models:
    plot1_name += "_" + "_".join(requested_models)
plot1_name += ".png"

plt.savefig(plot1_name, bbox_inches='tight', dpi=300)
print(f"Saved: {plot1_name}")
plt.close()

# ==========================================
# 5. Plot 2: Model Stability and Variance (Box Plot)
# ==========================================
g2 = sns.catplot(
    data=master_df,
    x="Ratio", y="vld_rel_err", hue="GNN_Model", col="WDN",
    palette=plot_palette,
    kind="box", height=5, aspect=1.2, sharey=False,
    boxprops={'alpha': 0.8}
)
g2.set_axis_labels("Observation Ratio", "Node Rel. Error")
g2.fig.suptitle("Model Stability Across Runs (Spread = Variance)", y=1.05, fontsize=16, fontweight='bold')

plot2_name = "Plot_2_Stability_Boxplots"
if args.models:
    plot2_name += "_" + "_".join(requested_models)
plot2_name += ".png"

plt.savefig(plot2_name, bbox_inches='tight', dpi=300)
print(f"Saved: {plot2_name}")
plt.close()

# ==========================================
# 6. Plot 3: Convergence Speed (Bar Plot)
# ==========================================
g3 = sns.catplot(
    data=master_df,
    x="GNN_Model", y="Stopped_At_Epoch", hue="GNN_Model", col="WDN",
    palette=plot_palette,
    kind="bar", errorbar="sd", capsize=.1, height=5, aspect=1.2,
    sharey=False, legend=False
)
g3.set_axis_labels("GNN Model", "Epochs to Converge (Early Stop)")
g3.fig.suptitle("Convergence Speed: Average Epochs to Stop (Lower = Faster)", y=1.05, fontsize=16, fontweight='bold')

plot3_name = "Plot_3_Convergence_Speed"
if args.models:
    plot3_name += "_" + "_".join(requested_models)
plot3_name += ".png"

plt.savefig(plot3_name, bbox_inches='tight', dpi=300)
print(f"Saved: {plot3_name}")
plt.close()

print("\nAll plots generated successfully!")