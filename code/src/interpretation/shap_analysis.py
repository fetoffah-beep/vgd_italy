# -*- coding: utf-8 -*-
"""
Purpose: SHAP-based interpretation of model predictions.

Content:
- Functions to calculate and plot SHAP values.
"""
import torch
import shap
import numpy as np

import matplotlib.pyplot as plt


def compute_shap(model, data_loader, device, pred_vars, static_vars, dataset_name, explainer_type="deep"):
    """
    Compute and visualize SHAP values using different explainers.

    Parameters:
    - model: Trained PyTorch model.
    - data_loader: DataLoader (train, validation, or test).
    - device: 'cuda' or 'cpu'.
    - dataset_name: Name of the dataset.
    - explainer_type: Type of SHAP explainer ('gradient', 'kernel', 'deep', 'tree', or 'auto').
    """
    model.to(device)
    model.eval()  
    
    # Select a batch of data for SHAP analysis (using the first batch for model input)

    for dyn_inputs, static_input, targets in data_loader:
        dyn_inputs, static_input, targets = dyn_inputs.to(device), static_input.to(device), targets.to(device)       
        break


    # Use DeepExplainer for the deep learning model
    explainer = shap.DeepExplainer(model, [dyn_inputs, static_input])
    # Initialize list to store all SHAP values
    all_shap_values = []
    
    for dyn_inputs, static_input, targets in data_loader:
        dyn_inputs, static_input, targets = dyn_inputs.to(device), static_input.to(device), targets.to(device)
         
        shap_values = explainer.shap_values([dyn_inputs, static_input])
        
        dyn_shap_values_all = [np.array(shap_values[i][0]) for i in range(len(shap_values))]  # Dynamic inputs for all outputs
        stat_shap_values_all = [np.array(shap_values[i][1]) for i in range(len(shap_values))]  # Static inputs for all outputs
        
        
        dyn_shap_values_mean = np.mean(dyn_shap_values_all, axis=( 1, 2, 4, 5))  # Mean over height & width
        stat_shap_values_mean = np.mean(stat_shap_values_all, axis=(1, 3, 4))  # Mean for static inputs


        features_shap=np.concatenate([dyn_shap_values_mean,stat_shap_values_mean], axis=1)
        
        all_shap_values.append(features_shap)


        
        
        # len(shap_values)               → 10  (Number of output features)
        # len(shap_values[0])            → 2   (Dynamic and static inputs)
        # len(shap_values[0][0])         → 32  (Batch size)
        # len(shap_values[0][0][0])      → 10  (Time steps)
        # len(shap_values[0][0][0][0])   → 3   (Number of features for input)
        # len(shap_values[0][0][0][0][0]) → 5  (Height)
        # len(shap_values[0][0][0][0][0][0]) → 5  (Width)
        



    # Convert list to numpy array
    all_shap = np.concatenate(all_shap_values, axis=1)  # Transpose for SHAP format
    
    # Create feature names
    feature_names = pred_vars + static_vars
    # Plot summary
    shap.summary_plot(all_shap, feature_names=feature_names, title=f"Global Feature Importance")
    
    
    




# Visualize them to see which features are the most influential.
# Analyze the magnitude and direction of each SHAP value to understand whether a feature is pushing the prediction higher or lower.
# Summarize the global importance of each feature across all predictions.




# That’s a great idea! To analyze **feature importance for each EGMS measurement point** and **visualize it spatially**, here’s the approach:

# ---

# ### **🔹 Workflow for Spatial Feature Importance Mapping**
# 1. **Compute SHAP values** for each point in your dataset. 
# 2. **Extract feature importance** per point. 
# 3. **Aggregate feature importance spatially** (e.g., per region or grid cell). 
# 4. **Visualize feature importance on a map**.

# ---

# ### **🔹 Step-by-Step Implementation**
# #### **1️⃣ Compute SHAP Values Per Measurement Point**
# Since each point has multiple predictors (features), calculate SHAP values for **each feature at each EGMS point**:

# ```python
# import shap
# import numpy as np

# # Assume X_test contains the test data (each row is a point, columns are features)
# explainer = shap.Explainer(model, X_train) # Use the model trained on EGMS data
# shap_values = explainer(X_test) # Compute SHAP values for test points
# ```

# ---

# #### **2️⃣ Extract Feature Importance per Point**
# We want to get the absolute mean SHAP value per feature at each location:

# ```python
# # Compute mean absolute SHAP value for each feature per point
# feature_importance = np.abs(shap_values.values).mean(axis=0)
# ```

# Each entry in `feature_importance` now represents the importance of a feature **averaged across all test samples**.

# ---

# #### **3️⃣ Attach Feature Importance to Spatial Coordinates**
# Your **EGMS dataset should have latitude and longitude for each measurement point**. We need to match feature importance with spatial locations:

# ```python
# import pandas as pd

# # Create a DataFrame linking SHAP values to coordinates
# df_shap = pd.DataFrame({
# 'longitude': lon_test, # Longitudes of test points
# 'latitude': lat_test, # Latitudes of test points
# 'importance': feature_importance # Feature importance per location
# })
# ```

# Now, `df_shap` contains **longitude, latitude, and feature importance**, which can be plotted.

# ---

# #### **4️⃣ Plot Feature Importance on a Map**
# Use **Folium (for interactive maps) or Matplotlib (for static heatmaps)**.

# 🔹 **Interactive Map (Folium)** 
# ```python
# import folium
# from folium.plugins import HeatMap

# # Initialize a map centered at the dataset's mean location
# m = folium.Map(location=[df_shap.latitude.mean(), df_shap.longitude.mean()], zoom_start=6)

# # Add heatmap of feature importance
# heat_data = list(zip(df_shap.latitude, df_shap.longitude, df_shap.importance))
# HeatMap(heat_data).add_to(m)

# # Show the map
# m
# ```

# 🔹 **Static Heatmap (Matplotlib & Basemap)** 
# ```python
# import matplotlib.pyplot as plt
# import geopandas as gpd

# # Convert to a GeoDataFrame
# gdf = gpd.GeoDataFrame(df_shap, geometry=gpd.points_from_xy(df_shap.longitude, df_shap.latitude))

# # Plot using Geopandas
# fig, ax = plt.subplots(figsize=(8, 6))
# gdf.plot(column='importance', cmap='coolwarm', legend=True, ax=ax, markersize=5)
# plt.title("Feature Importance Across EGMS Measurement Points")
# plt.show()
# ```

# ---

# ### **🔹 Bonus: Feature-Specific Influence per Region**
# If you want to **see which feature dominates in different regions**, you can:
# 1. **Group by spatial regions (e.g., administrative boundaries or grid cells).** 
# 2. **Find the most important feature in each region.** 
# 3. **Plot a categorical map showing the dominant feature per region.**

# Example using `geopandas`:

# ```python
# # Group by a spatial region (e.g., provinces, using a shapefile)
# regions = gpd.read_file("regions_shapefile.shp") # Load spatial boundaries
# gdf_joined = gpd.sjoin(gdf, regions, how="left", predicate="intersects")

# # Find the most important feature per region
# most_important_feature_per_region = gdf_joined.groupby("region")["importance"].idxmax()

# # Plot the dominant feature map
# fig, ax = plt.subplots(figsize=(10, 8))
# gdf_joined.loc[most_important_feature_per_region].plot(column="feature_name", cmap="tab10", legend=True, ax=ax)
# plt.title("Dominant Feature Influence in Each Region")
# plt.show()
# ```

