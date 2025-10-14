# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 11:40:46 2025

@author: gmfet
"""

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import re

import geopandas as gpd
SPATIAL_ANALYSIS_ENABLED = False

# Load your correlation CSV in chunks
file_path = r"C:\Users\gmfet\vgd_italy\output\cross_correlations20250708_152435.csv"
CHUNK_SIZE = 10000
df_chunks = pd.read_csv(file_path, chunksize=CHUNK_SIZE)

pairs = []
results = {}
total_rows = 0 # To track total number of points

# Read file in chunks and aggregate counts
for i, chunk in enumerate(df_chunks):
    total_rows += len(chunk)
    # Select only Granger p-value columns
    granger_p_cols = [col for col in chunk.columns if col.startswith("granger_") and col.endswith("_p")]

    # Initialize pair info only once (from first chunk)
    if not pairs:
        for col in granger_p_cols:
            match = re.match(r"granger_([^_]+)_to_([^_]+)_p", col)
            if match:
                src, dst = match.groups()
                pairs.append((src, dst, col))
                results[(src, dst)] = []
            else:
                print(f"Skipped malformed column: {col}")

    # Count significant causal relations (p < 0.05)
    for src, dst, col in pairs:
        sig_count = (chunk[col] < 0.05).sum()
        results[(src, dst)].append(sig_count)

print(f"Total points (rows) analyzed: {total_rows}")

# Summarize across all chunks
summary = []
for (src, dst), counts in results.items():
    total_sig = sum(counts)
    percentage = (total_sig / total_rows) * 100 if total_rows > 0 else 0
    summary.append((src, dst, total_sig, percentage))

sig_df = pd.DataFrame(summary, columns=["source", "target", "significant_count", "percentage_of_points"])
sig_df.sort_values("significant_count", ascending=False, inplace=True)

print("\n--- Summary of Granger Causal Relations ---")
print(f"Total points (locations) in dataset: {total_rows}")
print("\nTop 10 Causal Links (by count of significant points):")
print(sig_df.head(10).to_string(index=False))

# -------------------------------
# 1️⃣ Causality matrix heatmap
# -------------------------------
# Check if there's enough data for a pivot
if not sig_df.empty:
    causality_matrix = sig_df.pivot(index="source", columns="target", values="significant_count").fillna(0)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(causality_matrix, cmap="viridis")

    # Add count labels to the heatmap cells
    for (j, i), count in causality_matrix.stack().items():
        if count > 0:
            plt.text(causality_matrix.columns.get_loc(i), causality_matrix.index.get_loc(j),
                     f'{int(count)}', ha='center', va='center', color='white', fontsize=8)

    plt.colorbar(im, label="Count of Significant Points (p < 0.05)")
    plt.xticks(range(len(causality_matrix.columns)), causality_matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(causality_matrix.index)), causality_matrix.index)
    plt.title("Bidirectional Granger Causality Matrix (Point Counts)")
    plt.tight_layout()
    plt.show()

# -------------------------------
# 2️⃣ Directional dominance ratios
# -------------------------------
dominance = []
for src in causality_matrix.index:
    for dst in causality_matrix.columns:
        if src != dst and dst in causality_matrix.index:
            fwd = causality_matrix.loc[src, dst]
            rev = causality_matrix.loc[dst, src]
            if fwd + rev > 0:
                ratio = fwd / (fwd + rev)
                dominance.append({"Pair": f"{src}↔{dst}", "ForwardRatio": ratio})

dominance_df = pd.DataFrame(dominance)
dominance_df.sort_values("ForwardRatio", ascending=False, inplace=True)

# Filter out pairs with zero total counts for clearer plot
dominance_df = dominance_df[dominance_df["ForwardRatio"].isin([0, 1]) | (dominance_df["ForwardRatio"].between(0.01, 0.99))]

plt.figure(figsize=(10, len(dominance_df) * 0.4 + 1))
plt.barh(dominance_df["Pair"], dominance_df["ForwardRatio"], color="teal", alpha=0.7)
plt.xlabel("Forward Direction Ratio (0 = reverse dominates, 1 = forward dominates)")
plt.title("Directional Dominance Between Variable Pairs")
plt.tight_layout()
plt.show()

# -------------------------------
# 3️⃣ Granger causality network
# -------------------------------
G = nx.DiGraph()

for _, row in sig_df.iterrows():
    G.add_edge(row["source"], row["target"], weight=row["significant_count"], count=row["significant_count"])

# Use the count to scale the edge thickness
edge_weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
max_weight = max(edge_weights) if edge_weights else 1
edge_widths = [w / max_weight * 5 for w in edge_weights] # Scale to a max width of 5

plt.figure(figsize=(7, 7))
pos = nx.circular_layout(G) # Using a circular layout for clean display
edges = nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20,
                               edge_color='gray', width=edge_widths, alpha=0.7)
nodes = nx.draw_networkx_nodes(G, pos, node_color='teal', node_size=2000, alpha=0.9)
labels = nx.draw_networkx_labels(G, pos, font_size=10, font_color='white')

# Add edge labels (count) for clarity
edge_labels = nx.get_edge_attributes(G, 'count')
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: int(v) for k, v in edge_labels.items()}, font_color='red')

plt.title("Granger Causality Network (Edge Thickness = # of Significant Points)")
plt.axis("off")
plt.show()

# -------------------------------
# 4️⃣ Spatial Analysis (Requires GeoPandas)
# -------------------------------
if SPATIAL_ANALYSIS_ENABLED:
    print("\n--- Starting Spatial and Lag Analysis ---")
    
    # Reload the full file for spatial analysis (necessary as chunks don't keep easting/northing consistently)
    # WARNING: This step can consume a lot of memory if the file is very large.
    try:
        df_full = pd.read_csv(file_path)
    except Exception as e:
        print(f"ERROR loading full DataFrame for spatial analysis: {e}")
        SPATIAL_ANALYSIS_ENABLED = False # Disable if full load fails

    if SPATIAL_ANALYSIS_ENABLED:
        # Example: Plot significance of Temperature → VGD across Italy
        key_link_p = "granger_temp_to_vgd_p"
        key_link_lag = "granger_temp_to_vgd_lag"

        # Filter valid points where causality is significant
        mask = df_full[key_link_p] < 0.05
        spatial_df = df_full.loc[mask, ["easting", "northing", key_link_p, key_link_lag]]
        
        if not spatial_df.empty:
            # Convert to GeoDataFrame
            # NOTE: Assuming the coordinates are in a projected CRS appropriate for Italy, 
            # such as EPSG:3035 (ETRS89-LAEA) or a UTM zone. Please CONFIRM YOUR CRS.
            try:
                gdf = gpd.GeoDataFrame(
                    spatial_df,
                    geometry=gpd.points_from_xy(spatial_df["easting"], spatial_df["northing"]),
                    crs="EPSG:3035" # Placeholder CRS - Replace with your actual CRS!
                )
            except Exception as e:
                print(f"Error creating GeoDataFrame. Check 'easting'/'northing' columns and CRS: {e}")
                gdf = None

            if gdf is not None:
                # Plot significance (P-value)
                fig, ax = plt.subplots(figsize=(8, 10))
                gdf.plot(column=key_link_p, cmap="Reds_r", markersize=1, legend=True, ax=ax, 
                         legend_kwds={'label': f"P-value of {key_link_p} (Significant Points)"})
                ax.set_title(f"Significant Granger {key_link_p.split('_')[1].upper()} → {key_link_p.split('_')[3].upper()} Relationships")
                plt.show()

                # Plot lag values
                fig, ax = plt.subplots(figsize=(8, 10))
                gdf.plot(column=key_link_lag, cmap="viridis", markersize=1, legend=True, ax=ax,
                         legend_kwds={'label': f"Lag (Time Delay) of {key_link_lag}"})
                ax.set_title(f"Lag (Time Delay) of {key_link_lag.split('_')[1].upper()} → {key_link_lag.split('_')[3].upper()} Causality")
                plt.show()

                # -------------------------------
                # 5️⃣ Interpret the Lag
                # -------------------------------
                # Summarize lag statistics for significant points
                lag_stats = spatial_df[key_link_lag].describe()
                print(f"\nLag statistics for {key_link_p} (significant points, N={len(spatial_df)}):")
                print(lag_stats)

                # Example interpretation:
                mean_lag = lag_stats["mean"]
                print(f"\nInterpretation:")
                print(f"On average, changes in Temperature are followed by VGD changes after approximately {mean_lag:.1f} time units.")
                print(f"The Max/Min lag values show the full range of response times across the country.")
        else:
            print(f"No significant points found for the example link: {key_link_p}")
else:
    print("\nSpatial analysis skipped. Install GeoPandas and uncomment the related code to enable.")