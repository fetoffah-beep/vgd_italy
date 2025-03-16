# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 13:12:45 2025

@author: 39351
"""
import numpy as np
import shap
import matplotlib.pyplot as plt

def shap_plot(ds):
    
    feature_names = list(ds.data_vars.keys())
    shap_values = np.stack([ds[feature].values.flatten() for feature in feature_names], axis=1)

     ########### Summary plot ###########
    shap.summary_plot(shap_values, feature_names=feature_names, title=f"Global Feature Importance")
    shap.summary_plot(shap_values, feature_names=feature_names, title=f"Global Feature Importance", plot_type="bar")
    
    
    
    ########### Force plot ###########
    shap_values = np.array([ds[feature].values.flatten()[0] for feature in feature_names])
    base_value = np.mean(shap_values)
    feature_values = np.zeros_like(shap_values)
    shap.force_plot(base_value, shap_values, feature_values, feature_names=feature_names, matplotlib=True)
    
    ########### Spatial plot ###########
    for feature_name in ds.data_vars:
        # Extract SHAP values for the current feature
        feature_data = ds[feature_name]

        # Create spatial plot
        plt.figure()
        plt.scatter(ds['easting'], ds['northing'], c=feature_data.values.flatten(), cmap='viridis')
        plt.colorbar(label='SHAP Value')
        plt.xlabel('Easting')
        plt.ylabel('Northing')
        plt.title(f'SHAP Values for Feature: {feature_name}')
        plt.show()
        
        
    ########### Feature wih highest SHAP value Spatial plot  ###########
    shap_data = ds.to_dataframe() #convert to dataframe for ease of use.

    # Find the feature with the highest SHAP value for each point
    highest_shap_feature = []
    for index, row in shap_data.iterrows():
        feature_values = row[feature_names].values
        highest_feature_index = np.argmax(np.abs(feature_values))
        highest_shap_feature.append(feature_names[highest_feature_index])

    shap_data['highest_feature'] = highest_shap_feature

    plt.figure()
    unique_features = shap_data['highest_feature'].unique()
    colors = plt.cm.get_cmap('viridis', len(unique_features)) #create a colormap.

    for i, feature in enumerate(unique_features):
        subset = shap_data[shap_data['highest_feature'] == feature]
        plt.scatter(subset.index.get_level_values('easting'), subset.index.get_level_values('northing'), color=colors(i), label=feature, s=10)

    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.title('Spatial Distribution of Highest SHAP Value Features')
    plt.legend()
    plt.show()
    