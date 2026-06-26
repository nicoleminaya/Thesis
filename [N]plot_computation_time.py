# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# ==========================================
# 1. Command-Line Argument Setup
# ==========================================
parser = argparse.ArgumentParser(description='Plot GNN training computation times.')
parser.add_argument(
    '--wds',
    type=str,
    required=True,
    help='The WDS type (e.g., BWSN, Net3, Richmond, Anytown).'
)

parser.add_argument(
    '--models',
    nargs='+',
    type=str,
    help='List of GNN models to plot (e.g., --models cheb1 gat gat2). If omitted, all models are plotted.'
)

parser.add_argument(
    '--file',
    type=str,
    help='Name of the file.'
)

args = parser.parse_args()
wds_type = args.wds.upper()

# ==========================================
# 2. Configuration & Data Loading
# ==========================================
model_colors = {
    "cheb1": "#1f77b4",   # Azul
    "cheb2": "#8c564b",   # Brown
    "cheb3": "#ff7f0e" ,  # Orange
    "gat": "#2ca02c"  ,   # Green
    "gat2": "#9467bd",    # Purple
    "gat_hyp": "#d62728"  # Red
}

#csv_filename = "training_computation_times.csv"
# Hanoi csv_filename = "[hanoi_noseed_cheb2_v1]training_computation_times_eliminados.csv"
# bwsn csv_filename = "[BWSN_20_RUNS]training_computation_times_eliminados.csv"
# anytown no seed csv_filename = "[ANYTOWN_v2][no_seed]training_computation_times_eliminados.csv"
# anytown seed csv_filename = "[Anytown_SEED]training_computation_times_eliminados.csv"
#csv_filename = "[Anytown_SEED]training_computation_times_eliminados.csv"
#csv_filename = "GENA_training_computation_times_eliminados.csv"
#csv_filename = "RANDOM_training_computation_times.csv"
csv_filename = args.file

try:
    column_names = ['Experiment_Run', 'Network', 'GNN', 'Ratio', 'tag', 'Time_Seconds']
    df = pd.read_csv(csv_filename, names=column_names, skiprows=1)
except FileNotFoundError:
    print(f"Error: '{csv_filename}' not found. Please ensure the file exists.")
    exit()

# Limpieza básica por si hay espacios raros
df['Network'] = df['Network'].astype(str).str.strip()
df['GNN'] = df['GNN'].astype(str).str.strip().str.lower()

# ==========================================
# 3. Filter Data for the Selected WDS ONLY
# ==========================================
df_filtered = df[df['Network'].str.lower() == wds_type.lower()]

if df_filtered.empty:
    print(f"Error: No data found for network '{wds_type}'. Exiting.")
    exit()

# ==========================================
# 4. Filter by selected models (optional)
# ==========================================
if args.models:
    requested_models = [m.strip().lower() for m in args.models]

    # modelos válidos encontrados en el CSV para ese WDS
    available_models = sorted(df_filtered['GNN'].dropna().unique())

    # nos quedamos solo con los modelos pedidos
    df_filtered = df_filtered[df_filtered['GNN'].isin(requested_models)]

    if df_filtered.empty:
        print(f"Warning: None of the requested models {requested_models} were found for WDS '{wds_type}'.")
        print(f"Available models in the data: {available_models}")
        exit()

# ==========================================
# 5. Process Data & Plot
# ==========================================
avg_times = df_filtered.groupby(['Ratio', 'GNN'])['Time_Seconds'].mean().unstack()

plt.figure(figsize=(8, 5))

print(f"--- Plotting Data for {wds_type} ---")
for gnn_model in avg_times.columns:
    clean_name = str(gnn_model).strip().lower()
    line_color = model_colors.get(clean_name, 'black')

    plt.plot(
        avg_times.index,
        avg_times[gnn_model],
        marker='o',
        linewidth=2,
        label=clean_name.upper(),
        color=line_color
    )

# ==========================================
# 6. Format and Save Chart
# ==========================================
plt.title(f'Average Computation Time vs. Observation Ratio ({wds_type})', fontsize=14)
plt.xlabel('Observation Ratio', fontsize=12)
plt.ylabel('Time (Seconds)', fontsize=12)

# ordenar ratios si son numéricos
try:
    sorted_ratios = sorted(df_filtered['Ratio'].dropna().unique())
    plt.xticks(sorted_ratios)
except:
    plt.xticks(df_filtered['Ratio'].unique())

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='GNN Model')
plt.tight_layout()

# nombre del archivo de salida
if args.models:
    model_suffix = "_".join([m.lower() for m in args.models])
else:
    model_suffix = "all_models"

output_image = f'{wds_type}_computation_time_chart_{model_suffix}.png'
plt.savefig(output_image, dpi=300)
print(f"Success! Chart saved as {output_image}")

plt.show()