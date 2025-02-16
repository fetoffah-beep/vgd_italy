# -*- coding: utf-8 -*-
"""
Purpose: SHAP-based interpretation of model predictions.

Content:
- Functions to calculate and plot SHAP values.
"""
import torch
import shap
import numpy as np


def compute_shap(model, data_loader, device, dataset_name, explainer_type="auto"):
    """
    Compute and visualize SHAP values using different explainers.

    Parameters:
    - model: Trained PyTorch model.
    - data_loader: DataLoader (train, validation, or test).
    - device: 'cuda' or 'cpu'.
    - dataset_name: Name of the dataset.
    - explainer_type: Type of SHAP explainer ('gradient', 'kernel', 'deep', 'tree', or 'auto').
    """
    model.eval()  # Ensure model is in evaluation mode

    # Dynamically choose SHAP explainer
    explainer = None

    # Select a batch of data for SHAP analysis (using the first batch for model input)
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        break  # Take only the first batch for input reference

    if explainer_type == "auto":
        explainer = shap.GradientExplainer(model, inputs)
    elif explainer_type == "gradient":
        explainer = shap.GradientExplainer(model, inputs)
    elif explainer_type == "kernel":
        explainer = shap.KernelExplainer(
            lambda x: model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().detach().numpy(),
            inputs.cpu().numpy()
        )
    elif explainer_type == "deep":
        explainer = shap.DeepExplainer(model, inputs)
    elif explainer_type == "tree":
        explainer = shap.TreeExplainer(model)
    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")

    print(f"Using {explainer.__class__.__name__} for {dataset_name}")

    # Initialize list to store all SHAP values
    all_shap_values = []

    # Iterate over all data in the data_loader
    for inputs, targets in data_loader:
        inputs = inputs.to(device)

        shap_values = explainer.shap_values(inputs)

        shap_values = np.array(shap_values)
        if shap_values.ndim == 1:  
            shap_values = shap_values.reshape(-1, 1)

        all_shap_values.append(shap_values)

    all_shap_values = np.concatenate(all_shap_values, axis=0)

    inputs_np = inputs.cpu().numpy()
    if inputs_np.ndim == 1:
        inputs_np = inputs_np.reshape(-1, 1)

    shap.summary_plot(all_shap_values, inputs_np, title=f"SHAP Summary for {dataset_name} with {explainer_type}")

    np.save(f"shap_values_{dataset_name}.npy", all_shap_values)

    print(f"SHAP analysis completed for {dataset_name} using {explainer.__class__.__name__}")


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
# explainer = shap.Explainer(model, X_train)  # Use the model trained on EGMS data
# shap_values = explainer(X_test)  # Compute SHAP values for test points
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
#     'longitude': lon_test,  # Longitudes of test points
#     'latitude': lat_test,    # Latitudes of test points
#     'importance': feature_importance  # Feature importance per location
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
# regions = gpd.read_file("regions_shapefile.shp")  # Load spatial boundaries
# gdf_joined = gpd.sjoin(gdf, regions, how="left", predicate="intersects")

# # Find the most important feature per region
# most_important_feature_per_region = gdf_joined.groupby("region")["importance"].idxmax()

# # Plot the dominant feature map
# fig, ax = plt.subplots(figsize=(10, 8))
# gdf_joined.loc[most_important_feature_per_region].plot(column="feature_name", cmap="tab10", legend=True, ax=ax)
# plt.title("Dominant Feature Influence in Each Region")
# plt.show()
# ```

# ---

# ### **🚀 Summary**
# ✔ Compute **SHAP values per EGMS measurement point**.  
# ✔ Extract **feature importance per point**.  
# ✔ Link to **spatial coordinates**.  
# ✔ **Visualize importance** on an interactive **heatmap** or static **map**.  
# ✔ Optionally, **analyze dominant features by region**.

# ---

# ### **💡 Next Steps**
# - Do you want to **analyze different SHAP explainers** for comparison?  
# - Should we **aggregate feature importance** over time as well? 🚀