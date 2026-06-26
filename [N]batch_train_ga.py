import subprocess
import argparse
import time
import csv
import os  # Added to check if the file exists

parser = argparse.ArgumentParser()
parser.add_argument('--wds', default='anytown', type=str)
parser.add_argument('--tag', default='def', type=str)
parser.add_argument('--deploy', default='xrandom', type=str)
parser.add_argument('--batch', default=64, type=int)
parser.add_argument('--adj', default='binary', choices=['binary', 'weighted', 'logarithmic', 'pruned'], type=str, help="Type of adjacency matrix.")
parser.add_argument('--epoch', default=500, type=int)
parser.add_argument('--deterministic', action="store_true", help="Setting random seed for sensor placement.")
parser.add_argument('--gnn', default='cheb1', choices=['cheb1', 'cheb2', 'cheb3','gat', 'gat2','gat_hyp'], type=str)
parser.add_argument('--runs', default=4, type=int)
parser.add_argument('--lr', default=0.00067, type=float)
parser.add_argument('--decay', default=0.000006, type=float)
args = parser.parse_args()

# Configuration
runs = args.runs
ratios = [0.05, 0.1, 0.2, 0.4, 0.8]
csv_filename = "training_computation_times.csv"

print("Start")

# 1. Check if the file exists before we open it
file_exists = os.path.isfile(csv_filename)

# 2. Open the CSV file in append mode ('a' instead of 'w')
with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # 3. Only write the header row if the file is brand new
    if not file_exists:
        writer.writerow(['Experiment_Run', 'GNN', 'Ratio', 'Time_Seconds'])

    for j in range(1, runs + 1):
        for i in ratios:
            print(f"\n------------------------------------------")
            print(f" Starting {args.gnn.upper()} | Ratio #{i} (Run {j}/{runs})")
            print(f"------------------------------------------")
            
            # Build the base command shared by both conditions
            command = [
                "python", "[N]train_ga.py",  
                "--epoch", str(args.epoch),
                "--tag", str(args.tag),
                "--deploy", str(args.deploy),
                "--wds", str(args.wds),
                "--deploy", str(args.deploy),
                "--obsrat", str(i), 
                "--batch", str(args.batch),
                "--gnn", str(args.gnn),
                "--lr", str(args.lr),
                "--decay", str(args.decay)
            ]
            
            # Add the specific arguments based on the deterministic flag
            if args.deterministic: 
                command.extend(["--deterministic", "--adj", str(args.adj)])
            else:
                command.extend(["--adj", "binary"])
                
            try:
                # Start the timer
                start_time = time.perf_counter()
                
                # Execute the command
                subprocess.run(command, check=True)
                
                # Stop the timer
                end_time = time.perf_counter()
                
                # Calculate the elapsed time
                elapsed_time = end_time - start_time
                print(f"\n---> Training for Ratio {i} completed in {elapsed_time:.2f} seconds.")
                
                # Write the row to the CSV
                writer.writerow([j, args.wds ,args.gnn, i, args.tag,round(elapsed_time, 2)])
                
                # Flush the file buffer to ensure it saves to the disk immediately
                file.flush()
                
            except subprocess.CalledProcessError as e:
                print(f" Error occurred in run #{i}. Stopping.")
                break
            except KeyboardInterrupt:
                print("\n Execution stopped by user.")
                break

print("\n Finish")