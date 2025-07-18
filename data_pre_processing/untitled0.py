# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 00:39:04 2025

@author: gmfet
"""

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

data_path = r"C:\Users\gmfet\vgd_italy\output\cross_correlations20250708_152435.csv"

chunksize=1000

df = pd.read_csv(data_path)

# for chunk in df:
#     print(chunk)
#     break



cols = [col for col in df.columns if col.startswith('lag_') and 'vgd' in col]

output_df = df[cols]

g = sns.PairGrid(output_df, corner=False, diag_sharey=False)
g.map_lower(sns.scatterplot, s=20, alpha=0.6)
g.map_diag(sns.histplot, kde=True)

plt.tight_layout()
plt.savefig('../output/scatter_matrix_lag.png')



# lag
# lag_columns = [col for col in output_df.columns if col.startswith("lag_")]
overall_lags = output_df.mean().sort_values()
print(overall_lags)


# Create a matrix from lag column names like "lag_temp_to_vgd"
lag_matrix = pd.DataFrame(index=[], columns=[])

for col in cols:
    parts = col.replace("lag_", "").split("_to_")
    src, tgt = parts[0], parts[1]
    lag_value = output_df.mean()
    lag_matrix.loc[src, tgt] = lag_value

lag_matrix = lag_matrix.astype(float)
print(lag_matrix)



plt.figure(figsize=(8, 6))
sns.heatmap(lag_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Average Lag Between Variables")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Lag data
data = {
    'Variable Pair': [
        'temp → vgd', 'drought → vgd', 'vgd → prec', 'vgd → twsan',
        'seismic → vgd', 'vgd → seismic', 'twsan → vgd', 'prec → vgd',
        'vgd → drought', 'vgd → temp'
    ],
    'Lag': [
        -1.390909, -0.753463, -0.261292, -0.044629,
        -0.029980, 0.029975, 0.044629, 0.261292,
        0.753463, 1.390909
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Set up the plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Lag', y='Variable Pair', data=df, palette='coolwarm', orient='h')

# Add a vertical line at 0 lag
plt.axvline(0, color='black', linewidth=1, linestyle='--')
plt.title('Lag Values Between VGD and Climatic/Seismic Variables', fontsize=14)
plt.xlabel('Lag (time units)')
plt.ylabel('Variable Relationship')

# Annotate bars
for index, row in df.iterrows():
    plt.text(row['Lag'] + (0.05 if row['Lag'] >= 0 else -0.35), index, f"{row['Lag']:.2f}", 
             color='black', va='center', fontsize=9)

plt.tight_layout()
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.show()
