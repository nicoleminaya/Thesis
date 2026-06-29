import os
import numpy as np
import pandas as pd
import sys

# Get the directory where this script is located (e.g., .../evaluation/)
current_dir = os.path.dirname(os.path.realpath(__file__))

# Define Paths
# Access the 'experiments' folder relative to this script
input_path = os.path.join(current_dir, '..', 'experiments', 'Taylor_metrics.csv')
output_path = os.path.join(current_dir, '..', 'experiments', 'Taylor_metrics_processed.csv')

def process_metrics():
    print(f" Reading raw metrics from: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: File not found. Please run your evaluation script first.")
        return

    # 1. Read the CSV (No headers initially)
    df = pd.read_csv(input_path, header=None)
    
    # 2. Assign standard column names
    df.columns = ['run_id', 'MSEc', 'sigma_pred']
    
    # ADD THESE TWO LINES: Clean up any accidental semicolons from Excel and force them to floats
    df['MSEc'] = df['MSEc'].astype(str).str.replace(';', '').astype(float)
    df['sigma_pred'] = df['sigma_pred'].astype(str).str.replace(';', '').astype(float)
    
    # 3. Initialize calculated columns
    df['sigma_true'] = 0.0

    
    # 3. Initialize calculated columns
    df['sigma_true'] = 0.0
    df['seed'] = 0
    
    print("Parsing run IDs...")
    # 4. Parse the "run_id" string (e.g., anytown-random-2-binary-rep_paper-1-gcn)
    # The author's notebook uses strict index splitting:
    try:
        df['wds'] = [elem.split('-')[0] for elem in df['run_id']]
        df['placement'] = [elem.split('-')[1] for elem in df['run_id']]
        df['obs_rat'] = [elem.split('-')[2] for elem in df['run_id']]
        df['adjacency'] = [elem.split('-')[3] for elem in df['run_id']]
        # Position 4 is now the Training Model (e.g., 'gat' or 'cheb1')
        df['tag'] = [elem.split('-')[-3] for elem in df['run_id']]    
        df['num'] = [elem.split('-')[-2] for elem in df['run_id']]     
        df['model'] = [elem.split('-')[-1] for elem in df['run_id']] 
    except IndexError as e:
        print("\n Error parsing run_id strings!")
        print(f"   Your filenames might not match the format.")
        print(f"   Example of a failing ID: {df['run_id'].iloc[0]}")
        sys.exit(1)

    # 5. Find and broadcast the 'True' Sigma (from the 'orig' model)
    print(" Locating Ground Truth (Reference) values...")
    unique_wds = df['wds'].unique()
    
    for wds in unique_wds:
        # Find the 'orig' row for this city
        ref_rows = df[(df['wds'] == wds) & (df['model'] == 'orig')]
        
        if ref_rows.empty:
            print(f"     Warning: No 'orig' (reference) data found for {wds}!")
            continue
            
        # Get the sigma value (Standard Deviation of the real pressure)
        sigma_true_val = ref_rows['sigma_pred'].iloc[0]
        
        # Apply this value to ALL rows for this city (so we can compare them)
        df['sigma_true'] = df['sigma_true'].astype(float)
        df.loc[df['wds'] == wds, 'sigma_true'] = sigma_true_val
        print(f"    {wds}: Reference Sigma = {sigma_true_val:.4f}")

    # 6. Calculate Correlation Coefficient
    # Formula: Correlation = Covariance / (Sigma_True * Sigma_Pred)
    df['corr_coeff'] = df['MSEc'] / (df['sigma_true'] * df['sigma_pred'])

    # 7. (Optional) Replicate Author's Seed Logic
    # The authors cycle through a fixed list of seeds for visualization purposes
    seeds = np.array([1, 8, 5266, 739, 88867])
    
    # Assign seeds for 'xrandom'
    mask_x = df['placement'] == 'xrandom'
    if mask_x.any():
        df.loc[mask_x, 'seed'] = seeds[df.loc[mask_x, 'num'].values.astype(int) % len(seeds)]
    
    # Assign seeds for 'random'
    mask_r = df['placement'] == 'random'
    if mask_r.any():
        df.loc[mask_r, 'seed'] = seeds[df.loc[mask_r, 'num'].values.astype(int) % len(seeds)]

    # 8. Save
    print(f" Saving processed data to: {output_path}")
    df.to_csv(output_path, index=False) # Author saves with index, but False is cleaner for plotting
    print(" Done!")

if __name__ == "__main__":
    process_metrics()