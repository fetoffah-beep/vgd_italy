# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:25:04 2024

@author: 39351
"""

import time

start_time = time.time()
import os
import yaml
import argparse
import torch
import wandb
import geopandas as gpd
from src.models.hybrid_model import VGDModel
from src.data.dataloader import VGDDataLoader
from src.data.hybrid_dataset import VGDDataset
from src.models.train_model import train_model
from src.interpretation.shap_analysis import compute_shap
from src.interpretation.lime_analysis import compute_lime
from src.models.checkpoint import load_checkpoint
from src.evaluation.evaluation import evaluate_model
from src.evaluation.shap_plot import shap_plot
from src.evaluation.summary_stats import get_summary_stats, display_summary_stats
from src.evaluation.visualization import plot_results
from torchvision.transforms import Compose
from src.data.transforms.transforms import NormalizeTransform
import cProfile
from pstats import Stats


def main(args):

    with open("../data/dynamic_feature.txt", "r") as f:
        dynamic_feature_names = [line.strip() for line in f if line.strip()]
    
    with open("../data/static_feature.txt", "r") as f:
        static_feature_names = [sline.strip() for sline in f if sline.strip()]
    
    

    
    
    with open("config.yaml") as config_file:
        config = yaml.safe_load(config_file)
        # config["data"] = transform_stats
        #
    # # Save updated config
    # with open("config.yaml", "w") as f:
    #     yaml.dump(config, f, sort_keys=False)
    
    dyn_mean = config["data"]['stats']["mean"]["dynamic"]
    dyn_std = config["data"]['stats']["std"]["dynamic"]
    static_mean = config["data"]['stats']["mean"]["static"]
    static_std = config["data"]['stats']["std"]["static"]
    target_mean = config["data"]['stats']["mean"]["target"]
    target_std = config["data"]['stats']["std"]["target"]
    
   
    
    # Configuration
    checkpoint_path = config["checkpoint"]["save_path"]
    hidden_size = config["model"]["hidden_layers"]
    num_epochs = config["training"]["epochs"]
    learning_rate = config["optimizer"]["init_args"]["lr"]
    model_optimizer = config["optimizer"]["class_path"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    device = config["model"]["device"]
    seq_len = config["training"]["seq_len"]
    output_size = config["model"]["output_size"]
    

    target_var = "displacement"
    
    
    pred_vars = [dynamic_feature_names, static_feature_names, target_var]
    

    # continue_from_checkpoint = False


    # Initialize the dataset for train, val, and test splits
    train_dataset = VGDDataset('training',      "../emilia_aoi/train_metadata.csv", 'config.yaml', "../data", seq_len, time_split=False)
    val_dataset   = VGDDataset('validation',    "../emilia_aoi/val_metadata.csv",   'config.yaml', "../data", seq_len, time_split=False)
    test_dataset  = VGDDataset('test',          "../emilia_aoi/test_metadata.csv",  'config.yaml', "../data", seq_len, time_split=False)
    
    # # Example: print the first sample
    # sample = test_dataset[0]

    # print("Static shape:", sample["static"].shape)
    # print("Dynamic shape:", sample["dynamic"].shape)
    # print("Target shape:", sample["target"].shape)
    # print("Coords:", sample["coords"])


    

    # Create DataLoaders for batching and shuffling
    train_loader = VGDDataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader   = VGDDataLoader(val_dataset,   batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader  = VGDDataLoader(test_dataset,  batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Get the data loaders for train, validation, and test sets
    train_loader, val_loader, test_loader = (dl.data_loader for dl in (train_loader, val_loader, test_loader))


    # # # Compute correlations
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)} \n")

    # Initialize the model
    model = VGDModel(test_dataset[0]['dynamic'].shape[1], test_dataset[0]['static'].shape[0], hidden_size, output_size)

    # Load checkpoint or initialize training
    # if continue_from_checkpoint:
    model, optimizer, start_epoch = load_checkpoint(
        checkpoint_path, model, learning_rate, device, optimizer=model_optimizer
    )
    # else:
    #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.02)
    #     start_epoch = 0
        
        
        
    # # print(model)
    # # print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    # # print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # # print('Optimizer: ', optimizer)
    # # print('Start epoch: ', start_epoch)



    # Train the model
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        learning_rate,
        'config.yaml',
        start_epoch=start_epoch,
        num_epochs=num_epochs,
        checkpoint_path=checkpoint_path,

    )
    

    # Test/Evaluate the model
    print("Training complete. Evaluating the model on the test set.")
    model.eval()

    results = evaluate_model(model, test_loader, device=device)
    predictions, ground_truth, residuals = (
        results["predictions"],
        results["ground_truth"],
        results["residuals"],
    )

    # Compute and display statistics
    pred_stats = get_summary_stats(predictions)
    gt_stats = get_summary_stats(ground_truth)
    res_stats = get_summary_stats(residuals)

    print("\n=== Model Evaluation Summary ===")
    display_summary_stats(pred_stats, label="Predictions")
    display_summary_stats(gt_stats, label="Ground Truth")
    display_summary_stats(res_stats, label="Residuals")

    # Visualize results
    plot_results(ground_truth, predictions, residuals)

    # # # Perform SHAP analysis for train, validation, and test sets
    # # train_shap = compute_shap(model, train_loader, device, pred_vars[0], pred_vars[1], "Train")
    # # shap_plot(train_shap)

    # # # val_shap = compute_shap(model, val_loader, device, pred_vars[0], pred_vars[1], "Validation")
    # # # shap_plot(val_shap)

    # # # test_shap = compute_shap(model, test_loader, device, pred_vars[0], pred_vars[1], "Test")
    # # # shap_plot(test_shap)

    # # # Perform LIME analysis for train, validation, and test sets
    # # compute_lime(model, train_loader, device, pred_vars[0], pred_vars[1], "Train")
    # # # compute_lime(model, val_loader, device, pred_vars[0], pred_vars[1], "Validation")
    # # # compute_lime(model, test_loader, device, pred_vars[0], pred_vars[1], "Test")

    print("Time taken:", time.time() - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    main(args)

    
    # do_profiling = True
    # profile_file = "profile_data.prof"
    
    # if do_profiling:
    #     with cProfile.Profile() as pr:
    #         main(args)

    #     with open(profile_file, "w") as stream:
    #         stats = Stats(pr, stream=stream)
    #         stats.strip_dirs()
    #         stats.sort_stats("time")
    #         stats.dump_stats(profile_file)
    #         stats.print_stats(5)

        # 2. Then, from the command line, run:
        # python -m  snakeviz profile_results.prof
        # This will open an interactive visualization in your browser.

    
    


# # Batch size-leraning rate
# # https://arxiv.org/pdf/1612.05086

# # learning curve
# # https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/

# # https://www.thenewatlantis.com/publications/correlation-causation-and-confusion?utm_source=chatgpt.com



