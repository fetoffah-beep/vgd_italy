# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 01:36:48 2025

@author: gmfet
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the two CSV files into DataFrames
try:
    df_2015_2023 = pd.read_csv(r"C:\Users\gmfet\Downloads\Ghana_Daily_Precipitation_2015_2023.csv")
    df_2024_2025 = pd.read_csv(r"C:\Users\gmfet\Downloads\Ghana_Daily_Precipitation_2024_2025.csv")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure both files are uploaded and named correctly.")
else:
    # Concatenate the two DataFrames
    combined_df = pd.concat([df_2015_2023, df_2024_2025], ignore_index=True)
    
    # Check the data types and info
    print("Combined DataFrame Info:")
    combined_df.info()
    print("\nCombined DataFrame Head:")
    print(combined_df.head())
    
    # Convert the 'date' column to datetime objects
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    
    # Sort the DataFrame by date
    combined_df.sort_values(by='date', inplace=True)
    
    # Plot the data
    plt.figure(figsize=(30, 12))
    plt.plot(combined_df['date'], combined_df['precipitation'])
    
    # Add titles and labels with increased font size
    plt.title('Daily Total Precipitation over Ghana (2015-2025)', fontsize=28)
    plt.xlabel('Date', fontsize=22)
    plt.ylabel('Precipitation (mm/day)', fontsize=22)
    
    # Improve x-axis labels for readability
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=15))
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    
    # Add a grid for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('daily_precipitation_merged.png')
    
    # Print success message
    print("Plot 'daily_precipitation_merged.png' has been saved successfully.")
