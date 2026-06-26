import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load your newly generated CSV
csv_path = 'experiments/relative_error-anytown-random-0.05.csv'
df = pd.read_csv(csv_path)
df.columns = ['runid'] + list(range(1, 23))

# 2. Set the runid as the index so it doesn't get plotted as a normal column
df.set_index('runid', inplace=True)

# 3. Create the Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df, cmap='Reds', annot=False, cbar_kws={'label': 'Mean Absolute Relative Error'})

plt.title('Anytown: Predictive Error by Single Sensor Placement', fontsize=16)
plt.xlabel('Predicted Junction (Node ID)', fontsize=12)
plt.ylabel('Sensor Location (Run ID)', fontsize=12)

# Save and show
plt.tight_layout()
plt.savefig('experiments/Anytown_Single_Sensor_Heatmap.pdf')
plt.show()