# https://docs.xarray.dev/en/stable/user-guide/dask.html
import os
import yaml
import torch
import dask
import distributed
from dask.distributed import Client
from distributed import LocalCluster
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
import line_profiler 
from line_profiler import profile

# from codecarbon import EmissionsTracker


@profile
def main(config, split_pattern, model_type):  

    # Configuration
    checkpoint_path     = config["checkpoint"]["save_path"]
    hidden_size         = config["model"]["hidden_layers"]
    num_epochs          = config["training"]["epochs"]
    learning_rate       = config["optimizer"]["init_args"]["lr"]
    # model_optimizer     = None #config["optimizer"]["class_path"]
    batch_size          = config["training"]["batch_size"]
    num_workers         = config["training"]["num_workers"]
    device              = config["model"]["device"]
    output_size         = config["model"]["output_size"]

    # predictors = config["data"]['stats']

    print('Starting dataset initialisation')

    # preload data
    train_dataset = VGDDataset('training', model_type, config, "./original_data", split_pattern='spatial')
    val_dataset = VGDDataset('validation', model_type, config, "./original_data", split_pattern='spatial')
    test_dataset = VGDDataset('test', model_type, config, "./original_data", split_pattern='spatial')
    
    num_dynamic_features = train_dataset.num_dynamic_features

    num_static_features  = train_dataset.num_static_features

    print(f"Number of static features: {num_static_features} \nNumber of dynamic features: {num_dynamic_features}")
    
    # Create DataLoaders for batching and shuffling
    train_loader = VGDDataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader   = VGDDataLoader(val_dataset,   batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader  = VGDDataLoader(test_dataset,  batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Get the data loaders for train, validation, and test sets
    train_loader, val_loader, test_loader = (dl.data_loader for dl in (train_loader, val_loader, test_loader))
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)} \n")


    # Initialize the model
    model = VGDModel(num_dynamic_features, num_static_features, train_dataset.var_categories, hidden_size, output_size)

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
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print('Optimizer: ', optimizer)
    print('Start epoch: ', start_epoch)
    
    
    # Train the model
    # train_model(
    #     model,
    #     train_loader,
    #     val_loader,
    #     optimizer,
    #     learning_rate,
    #     config,
    #     start_epoch=start_epoch,
    #     num_epochs=num_epochs,
    #     checkpoint_path=checkpoint_path,
    # )
    
    
    # # Test/Evaluate the model
    # print("Training complete. Evaluating the model on the test set.")
    
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

    # # Perform SHAP analysis for train, validation, and test sets
    # dyn_features = train_dataset.dynamic_data.keys()
    # static_features = train_dataset.static_data.keys()
    # # train_shap = compute_shap(model, train_loader, device, dyn_features, static_features, "Train")
    # # shap_plot(train_shap)

    # # val_shap = compute_shap(model, val_loader, device, dyn_features, static_features, "Validation")
    # # shap_plot(val_shap)

    # test_shap = compute_shap(model, test_loader, device, dyn_features, static_features, "Test")
    # shap_plot(test_shap)

    # # Perform LIME analysis for train, validation, and test sets
    # # compute_lime(model, train_loader, device, dyn_features, static_features, "Train")
    # # compute_lime(model, val_loader, device, dyn_features, static_features, "Validation")
    # # compute_lime(model, test_loader, device, dyn_features, static_features, "Test")

    


if __name__ == "__main__":
    with open("config.yaml") as config_file:
        config = yaml.safe_load(config_file)


    split_patterns = ['spatial','temporal', 'spatio_temporal', 'spatial_train_val']
    model_types = [ 'Time_series', 'Explanatory', 'Mixed']
        

    # Disable Dask's GPU monitoring to prevent the NVMLError crash
    # dask.config.set({"distributed.diagnostics.nvml": False})

    profile = line_profiler.LineProfiler()
    # tracker = EmissionsTracker(project_name="vgd_training")
    # tracker.start()
    
    scratch_dir = "./dask_scratch" 
    
    # close any existing clients
    # try:
    #     client = Client.current()
    #     client.close()
    # except ValueError:
    #     pass
    
    # cluster = LocalCluster(n_workers=8, threads_per_worker=1, memory_limit='8GB', local_directory=scratch_dir, scheduler_port=8786, dashboard_address=':8787', silence_logs=50)

    # client = Client(cluster)
    # print(f"Dask Dashboard available at: {client.dashboard_link} \n")
    
    for model_type in model_types:
        for split_pattern in split_patterns:
            start_time = time.time()
            print(f"{model_type} Model experiment training start time: {time.ctime(start_time)} \n")
    
            main(config, split_pattern, model_type)
            break
        # break
    
    
            # print(f"{model_type} Model run end time: {time.ctime()} \n")
            # print(f"Time taken to run {model_type} model for {split_pattern}: ", time.time() - start_time)
    # training code
    
    # emissions = tracker.stop()
    # print(emissions)

    # cluster.close()
    
    # client.close()

    
    


# kernprof -l -v main.py    

    

        # 2. Then, from the command line, run:
        # python -m  snakeviz profile_results.prof
        # This will open an interactive visualization in your browser.

    
    


# # Batch size-leraning rate
# # https://arxiv.org/pdf/1612.05086

# # learning curve
# # https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/

# # https://www.thenewatlantis.com/publications/correlation-causation-and-confusion?utm_source=chatgpt.com
