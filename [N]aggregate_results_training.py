import os
import glob
import pandas as pd

log_dir = os.path.join('experiments', 'logs')

# 1. Find all base CSV files, but EXCLUDE the testing and meta files
all_csvs = glob.glob(os.path.join(log_dir, '*.csv'))
train_files = [f for f in all_csvs if not f.endswith('_tst.csv') 
               and not f.endswith('_meta.csv') 
               and not f.endswith('_sensor_nodes.csv')]

all_results = []

print(f"Found {len(train_files)} training log files. Compiling...")

for file in train_files:
    try:
        # --- A. Read the Epoch Data ---
        df = pd.read_csv(file)
        
        # --- B. Find the Best Epoch ---
        best_epoch_idx = df['vld_loss'].idxmin()
        best_metrics = df.iloc[best_epoch_idx].to_dict()
        best_metrics['Stopped_At_Epoch'] = int(df['epoch'].iloc[best_epoch_idx]) if 'epoch' in df.columns else best_epoch_idx
        
        # --- C. Extract Metadata from Filename ---
        filename = os.path.basename(file)
        name_parts = filename.replace('.csv', '').split('-')
        
        if len(name_parts) >= 7:
            metadata = {
                'WDS': name_parts[0],
                'Deployment': name_parts[1],
                'Ratio': float(name_parts[2]),
                'Adjacency': name_parts[3],
                'GNN_Model': name_parts[4],
                'Tag': name_parts[5],
                'Run_ID': int(name_parts[6])
            }
        else:
            # Fallback for unexpected filenames
            metadata = {'Filename': filename, 'WDS': 'unknown_wds'}
            
        # --- D. Combine and Store ---
        combined_data = {**metadata, **best_metrics}
        all_results.append(combined_data)
        
    except Exception as e:
        print(f" Error processing {file}: {e}")

# 2. Create, Split, and Save DataFrames
if all_results:
    master_df = pd.DataFrame(all_results)
    
    # Sort the data neatly
    if 'Ratio' in master_df.columns and 'GNN_Model' in master_df.columns:
        master_df = master_df.sort_values(by=['Ratio', 'GNN_Model', 'Run_ID'])
        
    # --- NEW LOGIC: Split by WDS ---
    unique_wds = master_df['WDS'].unique()
    print("\n--- Saving Individual WDS Files ---")
    
    for wds_name in unique_wds:
        # Filter the master dataframe for just this one city
        wds_df = master_df[master_df['WDS'] == wds_name]
        
        # Create a specific filename (e.g., "anytown_Training_Summary.csv")
        output_file = f"{wds_name}_Training_Summary.csv"
        
        # Save it!
        wds_df.to_csv(output_file, index=False)
        print(f" Saved {len(wds_df)} runs to -> {output_file}")
        
else:
    print("\n No training files found in the experiments/logs/ directory.")