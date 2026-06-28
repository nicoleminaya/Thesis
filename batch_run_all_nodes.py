import subprocess
import time
import argparse
from epynet import Network

parser = argparse.ArgumentParser()
parser.add_argument('--gnn', default='cheb1', choices=['cheb1','cheb2','gat'], type=str)
parser.add_argument('--wds',
                    default = 'anytown',
                    type    = str,
                    help    = "Water distribution system."
                    )
args = parser.parse_args()

# Load the Anytown network
wds = Network('water_networks/anytown.inp')

# Extract all valid Junction IDs
junction_ids = wds.junctions.index.values

print(f"Found {len(junction_ids)} junctions. Starting experiments...")

for jid in junction_ids:
    print(f"\n=======================================================")
    print(f" Training model with SINGLE sensor at Node: {jid}")
    print(f"=======================================================")
    
    # Define the command as a list of strings
    # Note: subprocess.run expects strings, so I converted integers to str()

    #### for gat I use traiin_gat, for chebnet I use [N]train_obsrat.py
    command = [
        "python", "train_gat.py",
        "--epoch", "500",
        "--adj", "binary",
        "--tag", f"node_{jid}",
        "--idx", str(jid),
        "--batch", "64",
        "--gnn", str(args.gnn)
    ]
    
    try:
        # check=True will raise CalledProcessError if the script fails
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred in run for Node {jid}. Stopping.")
        break
    except KeyboardInterrupt:
        print("\nExecution stopped by user.")
        break

print("\nFinished")