import unittest
import torch
import numpy as np
from src.data.dataloader.dataloader import VGDDataLoader
from src.data.dataset.dataset import VGDDataset
from src.transforms import NormalizeTransform, ReshapeTransform
from torch.utils.data import DataLoader

class TestDataScripts(unittest.TestCase):

    def test_data_loading(self):
        """
        Test that data is being loaded correctly from a dataset.
        """
        data = np.random.rand(100, 10)  
        targets = np.random.rand(100) 
        dataset = VGDDataset(data, targets) 

        # Test dataset length
        self.assertEqual(len(dataset), 100, "Dataset length is incorrect.")

        # Test if the data and targets are correctly paired
        sample_data, sample_target = dataset[0]
        self.assertEqual(sample_data.shape, (10,), "Data shape is incorrect.")
        self.assertTrue(isinstance(sample_target, float), "Target is not of expected type.")

    def test_transforms(self):
        """
        Test that the transformation functions are applied correctly.
        """
        # Create a sample data point
        sample = np.random.rand(10)  
        
        # Test normalization
        normalize_transform = NormalizeTransform()
        normalized_sample = normalize_transform(sample)
        self.assertEqual(normalized_sample.shape, sample.shape, "Normalized sample shape is incorrect.")
        self.assertAlmostEqual(np.mean(normalized_sample), 0, delta=1e-1, msg="Normalized sample mean is not zero.")
        self.assertAlmostEqual(np.std(normalized_sample), 1, delta=1e-1, msg="Normalized sample standard deviation is not one.")

        # Test reshaping
        reshape_transform = ReshapeTransform(new_shape=(2, 5))
        reshaped_sample = reshape_transform(sample)
        self.assertEqual(reshaped_sample.shape, (2, 5), "Reshaped sample shape is incorrect.")

    def test_data_loader(self):
        """
        Test the data loading mechanism and splitting into batches.
        """
        # Generate synthetic data
        data = np.random.rand(100, 10)  
        targets = np.random.rand(100) 
        dataset = VGDDataset(data, targets)  

        # Initialize the VGDDataLoader with 20% validation and 20% test split
        data_loader = VGDDataLoader(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

        # Get the data loaders for train, validation, and test sets
        train_loader, val_loader, test_loader = data_loader.get_data_loaders()

        # Test that the number of batches in the train loader is correct
        self.assertGreater(len(train_loader), 0, "Training data loader is empty.")
        self.assertGreater(len(val_loader), 0, "Validation data loader is empty.")
        self.assertGreater(len(test_loader), 0, "Test data loader is empty.")

        # Test if batch sizes are correct
        batch_size = 16
        for batch in train_loader:
            inputs, targets = batch
            self.assertEqual(inputs.shape[0], batch_size, "Batch size is incorrect.")
            self.assertEqual(targets.shape[0], batch_size, "Batch size for targets is incorrect.")

    def test_merge_datasets(self):
        """
        Test the merging of multiple datasets into a single DataLoader.
        """
        # Create synthetic datasets
        data1 = np.random.rand(50, 10)
        targets1 = np.random.rand(50)
        dataset1 = VGDDataset(data1, targets1)
        
        data2 = np.random.rand(50, 10)
        targets2 = np.random.rand(50)
        dataset2 = VGDDataset(data2, targets2)
        
        # Merge the datasets
        merged_loader = merge_datasets([dataset1, dataset2], batch_size=16, shuffle=True)
        
        # Test that the merged loader has the correct number of batches
        self.assertGreater(len(merged_loader), 0, "Merged data loader is empty.")
        
        # Test the shape of the data from the merged loader
        for batch in merged_loader:
            inputs, targets = batch
            self.assertEqual(inputs.shape[1], 10, "Merged dataset features are incorrect.")
            self.assertEqual(targets.shape[0], 16, "Batch size for merged dataset is incorrect.")
        
    def test_invalid_data(self):
        """
        Test that invalid data handling works as expected (e.g., empty or corrupt data).
        """
        with self.assertRaises(ValueError, msg="Data cannot be empty or malformed."):
            VGDDataset([], [])

        with self.assertRaises(ValueError, msg="Data dimensions do not match targets."):
            VGDDataset(np.random.rand(100, 10), np.random.rand(99))

if __name__ == '__main__':
    unittest.main()




