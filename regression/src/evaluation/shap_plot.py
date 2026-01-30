# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 13:12:45 2025

@author: 39351
"""
import numpy as np
import shap
import matplotlib.pyplot as plt
import datetime
from line_profiler import profile
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

@profile
def shap_plot(ds):
    
    feature_names = list(ds.data_vars.keys())
    shap_values = np.stack([ds[feature].values.flatten() for feature in feature_names], axis=1)

     ########### Summary plot ###########

    #      We can get a summary (or
    # global explanation) of the feature attribution values for the first 10 examples from
    # our test set with the following
    plt.close()
    shap.summary_plot(shap_values, feature_names=feature_names, title=f"Global Feature Importance")
    plt.savefig(f'output/global_feature_importance_1_{timestamp}.png')
    shap.summary_plot(shap_values, feature_names=feature_names, title=f"Global Feature Importance", plot_type="bar")
    
    plt.savefig(f'output/global_feature_importance_2_{timestamp}.png')
    
    
    
    ########### Force plot ###########
    # The feature attribution values for one example from our fuel efficiency pre‐
    # diction model.
    shap_values = np.array([ds[feature].values.flatten()[0] for feature in feature_names])
    base_value = np.mean(shap_values)
    feature_values = np.zeros_like(shap_values)
    shap.force_plot(base_value, shap_values, feature_values, feature_names=feature_names, matplotlib=True)
    ax = plt.gca()
    ax.tick_params(axis='x', labelrotation=45)
    plt.savefig(f'output/force_plot_{timestamp}.png')
    
    # ########### Spatial plot ###########
    # for feature_name in ds.data_vars:
    #     # Extract SHAP values for the current feature
    #     feature_data = ds[feature_name]

    #     # Create spatial plot
    #     plt.figure()
    #     plt.scatter(ds['easting'], ds['northing'], c=feature_data.values[:len(ds['easting'])], cmap='viridis')
    #     plt.colorbar(label='SHAP Value')
    #     plt.xlabel('Easting')
    #     plt.ylabel('Northing')
    #     plt.title(f'SHAP Values for Feature: {feature_name}')
    #     plt.savefig(f'output/{feature_name}_importance_{timestamp}.png')
        
    # for feature_name in ds.data_vars:
    #     feature_data = ds[feature_name]
    
    #     easting, northing = np.meshgrid(ds['easting'], ds['northing'])
    #     plt.figure()
    #     plt.scatter(
    #         easting.flatten(), 
    #         northing.flatten(), 
    #         c=feature_data.values.flatten(), 
    #         cmap='viridis', s=15
    #     )
    #     plt.colorbar(label='SHAP Value')
    #     plt.xlabel('Easting')
    #     plt.ylabel('Northing')
    #     plt.title(f'SHAP Values for Feature: {feature_name}')
    #     plt.savefig(f'output/{feature_name}_importance_{timestamp}.png')
    


        
        
        
    ########### Feature wih highest SHAP value Spatial plot  ###########
    shap_data = ds.to_dataframe() #convert to dataframe for ease of use.

    # Find the feature with the highest SHAP value for each point
    highest_shap_feature = []
    for index, row in shap_data.iterrows():
        feature_values = row[feature_names].values
        highest_feature_index = np.argmax(np.abs(feature_values))
        highest_shap_feature.append(feature_names[highest_feature_index])

    shap_data['highest_feature'] = highest_shap_feature

    plt.figure(figsize=(20, 8))
    unique_features = shap_data['highest_feature'].unique()
    colors = plt.cm.get_cmap('viridis', len(unique_features)) #create a colormap.

    for i, feature in enumerate(unique_features):
        subset = shap_data[shap_data['highest_feature'] == feature]
        plt.scatter(subset.index.get_level_values('easting'), subset.index.get_level_values('northing'), color=colors(i), label=feature, s=10)

    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.title('Spatial Distribution of Highest SHAP Value Features')
    plt.legend(
        title="Feature",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        frameon=False
    )
    plt.savefig(f'output/feature_importance_distribution{timestamp}.png')
    
    