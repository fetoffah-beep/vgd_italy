# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:25:04 2024

@author: 39351
"""

import yaml
import torch
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
import time
from line_profiler import profile
import line_profiler 



start_time = time.time()


print('libraries import done')
profile = line_profiler.LineProfiler()

@profile
def main():    
    
    with open("config.yaml") as config_file:
        config = yaml.safe_load(config_file)
        
   
    
    # Configuration
    checkpoint_path     = config["checkpoint"]["save_path"]
    hidden_size         = config["model"]["hidden_layers"]
    num_epochs          = config["training"]["epochs"]
    learning_rate       = config["optimizer"]["init_args"]["lr"]
    model_optimizer     = None #config["optimizer"]["class_path"]
    batch_size          = config["training"]["batch_size"]
    num_workers         = config["training"]["num_workers"]
    device              = config["model"]["device"]
    seq_len             = config["training"]["seq_len"]
    output_size         = config["model"]["output_size"]

    num_dynamic_features = len(config["data"]['stats']['mean']['dynamic'].keys())
    num_static_features  = len(config["data"]['stats']['mean']['static'].keys())

    # continue_from_checkpoint = False

    print('starting dataset initialisation')

    # preload data
    train_dataset = VGDDataset('training',      "../emilia_aoi/train_metadata.csv", 'config.yaml', "../data", seq_len, time_split=False)
    train_dataset.pre_load_data()

    # Initialize the dataset for train, val, and test splits
    # train_dataset = VGDDataset('training',      "../emilia_aoi/train_metadata.csv", 'config.yaml', "../data", seq_len, time_split=False)
    val_dataset   = VGDDataset('validation',    "../emilia_aoi/val_metadata.csv",   'config.yaml', "../data", seq_len, time_split=False)
    test_dataset  = VGDDataset('test',          "../emilia_aoi/test_metadata.csv",  'config.yaml', "../data", seq_len, time_split=False)


    # base_dataset = VGDDataset(config_path="config.yaml", data_dir="data").pre_load_data()

    # train_dataset = VGDDataset(split="train", ...)
    # train_dataset.dynamic_data = train_dataset.dynamic_data
    # train_dataset.seismic_tree = train_dataset.seismic_tree
    # train_dataset.static_data = train_dataset.static_data

    # val_dataset = VGDDataset(split="val", ...)
    val_dataset.dynamic_data = train_dataset.dynamic_data
    val_dataset.seismic_tree = train_dataset.seismic_tree
    val_dataset.static_data = train_dataset.static_data

    # test_dataset = VGDDataset(split="test", ...)
    test_dataset.dynamic_data = train_dataset.dynamic_data
    test_dataset.seismic_tree = train_dataset.seismic_tree
    test_dataset.static_data = train_dataset.static_data




    
    # # # Example: print the first sample
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
    


    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)} \n")

    # Initialize the model
    model = VGDModel(num_dynamic_features, num_static_features, hidden_size, output_size)

    model_optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=learning_rate,
                        weight_decay=0.02
                    )
    
    # Load checkpoint or initialize training
    model, optimizer, start_epoch = load_checkpoint(
        checkpoint_path, model, learning_rate, device, optimizer=model_optimizer
    )
     
        
    print(model)
    # print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    # print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # print('Optimizer: ', optimizer)
    # print('Start epoch: ', start_epoch)
    
    
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
    
    # results = evaluate_model(model, test_loader, 'config.yaml', device=device)
    # predictions, ground_truth, residuals = (
    #     results["predictions"],
    #     results["ground_truth"],
    #     results["residuals"],
    # )

    # # Compute and display statistics
    # pred_stats = get_summary_stats(predictions)
    # gt_stats = get_summary_stats(ground_truth)
    # res_stats = get_summary_stats(residuals)

    # print("\n=== Model Evaluation Summary ===")
    # display_summary_stats(pred_stats, label="Predictions")
    # display_summary_stats(gt_stats, label="Ground Truth")
    # display_summary_stats(res_stats, label="Residuals")

    # # Visualize results
    # plot_results(ground_truth, predictions, residuals)

    # # # Perform SHAP analysis for train, validation, and test sets
    # dyn_features = ['precipitation', 'drought_code', 'temperature']
    # static_features = ["bulk_density", "clay_content", "dem", "land_cover", "population_density_2020_1km", "sand", "silt", "slope", "soil_organic_carbon", "topo_wetness_index", "vol water content at -10 kPa", "vol water content at -1500 kPa", "vol water content at -33 kPa"]

    # train_shap = compute_shap(model, train_loader, device, dyn_features, static_features, "Train")
    # shap_plot(train_shap)

    # # # # val_shap = compute_shap(model, val_loader, device, dyn_features, static_features, "Validation")
    # # # # shap_plot(val_shap)

    # test_shap = compute_shap(model, test_loader, device, dyn_features, static_features, "Test")
    # shap_plot(test_shap)

    # Perform LIME analysis for train, validation, and test sets
    # compute_lime(model, train_loader, device, dyn_features, static_features, "Train")
    # compute_lime(model, val_loader, device, dyn_features, static_features, "Validation")
    # compute_lime(model, test_loader, device, dyn_features, static_features, "Test")

    print("Time taken:", time.time() - start_time)


if __name__ == "__main__":
    # try:
    #     mp.set_start_method('spawn')
    # except RuntimeError:
    #     pass
    main()
    client.close()
    


# kernprof -l -v main.py    

    
    # pr.disable()

    # # Save the stats to a file
    # pr.dump_stats('profile_stats.prof')
    
    # # Read and print the top 10 bottlenecks
    # stats = pstats.Stats('profile_stats.prof').sort_stats('tottime')
    # stats.print_stats(10)
    


    
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
