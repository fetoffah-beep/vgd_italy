# -*- coding: utf-8 -*-
"""
Created on Fri May 23 17:45:45 2025

@author: gmfet
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from tqdm import tqdm

# --- CONFIG ---
metadata_path = r"C:\Users\gmfet\Desktop\emilia\data\metadata.csv"
npy_folder = r"C:\Users\gmfet\Desktop\emilia\data\vgd_targets"
max_lag = 24
sample_size = 20 # Note: The plot title says "2000 MPs", but sample_size is 20. Adjust as needed.

# --- Load metadata ---
try:
    metadata = pd.read_csv(metadata_path)
except FileNotFoundError:
    print(f"Error: metadata.csv not found at {metadata_path}. Please check the path.")
    exit() # Exit if metadata is crucial for sampling

# --- Sample MP IDs ---
if len(metadata) < sample_size:
    print(f"Warning: metadata has only {len(metadata)} entries, sampling {len(metadata)} instead of {sample_size}.")
    sample_ids = metadata['mp_id'].tolist()
else:
    sample_ids = metadata.sample(n=sample_size, random_state=42)['mp_id'].tolist()

acf_matrix = []
successful_samples = 0

# --- Loop through sampled IDs ---
for mp_id in tqdm(sample_ids, desc="Computing ACFs"):
    npy_path = os.path.join(npy_folder, f"mp_{mp_id}.npy")
    
    # if not os.path.exists(npy_path):
    #     # print(f"Missing file: {npy_path}") # Suppress for cleaner tqdm output if many missing
    #     acf_matrix.append(np.full(max_lag + 1, np.nan))
    #     continue

    ts = np.load(npy_path)
    
    if len(ts) <= max_lag:
        # print(f"Time series for {mp_id} is too short ({len(ts)} <= {max_lag}). Skipping ACF computation.")
        acf_matrix.append(np.full(max_lag + 1, np.nan))
    else:
        try:
            acf_vals = acf(ts, nlags=max_lag, fft=True)
            acf_matrix.append(acf_vals)
            successful_samples += 1
        except Exception as e:
            print(f"Error computing ACF for {mp_id}: {e}. Appending NaNs.")
            acf_matrix.append(np.full(max_lag + 1, np.nan))


acf_matrix = np.array(acf_matrix)  # shape: (n_samples, max_lag + 1)

if successful_samples == 0:
    print("\nNo valid time series found to compute ACFs. Cannot generate plots.")
else:
    # --- Plot: Mean ACF with ±1 Std Dev ---
    lags = np.arange(max_lag + 1)
    mean_acf = np.nanmean(acf_matrix, axis=0)
    std_acf = np.nanstd(acf_matrix, axis=0)

    # Check if mean_acf is all NaNs before plotting
    if np.all(np.isnan(mean_acf)):
        print("\nMean ACF is all NaNs. Cannot plot Mean ACF with Std Dev.")
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(lags, mean_acf, label='Mean ACF', marker='o')
        # Only plot std dev if it's not all NaNs
        if not np.all(np.isnan(std_acf)):
            plt.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf, color='gray', alpha=0.3, label='±1 Std Dev')
        else:
            print("Standard deviation of ACF is all NaNs. Skipping ±1 Std Dev plot.")

        plt.title(f"Mean Autocorrelation Across {successful_samples} Valid MPs (out of {sample_size} sampled)")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- Histograms of ACF at selected lags ---
    for lag in [1, 12]:
        # Filter out NaNs for histogram plotting
        data_for_hist = acf_matrix[:, lag][~np.isnan(acf_matrix[:, lag])]
        
        if len(data_for_hist) == 0:
            print(f"\nNo valid ACF values for histogram at Lag {lag}. Skipping plot.")
            continue

        plt.figure(figsize=(6, 4))
        plt.hist(data_for_hist, bins=40, color='skyblue', edgecolor='k')
        plt.title(f"Histogram of ACF at Lag {lag}")
        plt.xlabel("ACF Value")
        plt.ylabel("Number of MPs")
        plt.grid(True)
        plt.tight_layout()
        plt.show()