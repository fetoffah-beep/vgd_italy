import unittest
import torch
from torch.utils.data import DataLoader
from src.models.train_model import train_model
from src.models import VGDModel
from src.data.dataloader.dataloader import VGDDataLoader
from src.data.dataset.dataset import VGDDataset
from torch.optim import Adam
import numpy as np

class TestModelScripts(unittest.TestCase):

    def test_train_model(self):
        """
        Test that the model training process runs without errors.
        """
        # Generate synthetic dataset
        data = np.random.rand(100, 10)  
        targets = np.random.rand(100)  
        dataset = VGDDataset(data, targets)

        # Create a DataLoader from the dataset
        data_loader = VGDDataLoader(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        train_loader, val_loader, _ = data_loader.get_data_loaders()

        # Initialize the model
        input_size = 10 
        hidden_size = 64 
        output_size = 1 
        model = VGDModel(input_size, hidden_size, output_size)

        # Initialize optimizer
        optimizer = Adam(model.parameters(), lr=0.001)

        # Test training function
        try:
            train_model(model, train_loader, val_loader, num_epochs=2, learning_rate=0.001, checkpoint_path=None)
        except Exception as e:
            self.fail(f"train_model raised an exception: {e}")
    
    def test_model_evaluation(self):
        """
        Test that the model evaluation function produces meaningful results.
        """
        # Generate synthetic dataset
        data = np.random.rand(100, 10)  
        targets = np.random.rand(100)  
        dataset = VGDDataset(data, targets)

        # Create a DataLoader from the dataset
        data_loader = VGDDataLoader(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        _, _, test_loader = data_loader.get_data_loaders()

        # Initialize the model
        input_size = 10  
        hidden_size = 64 
        output_size = 1 
        model = VGDModel(input_size, hidden_size, output_size)

        # Initialize optimizer
        optimizer = Adam(model.parameters(), lr=0.001)

        # Test the model evaluation
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            criterion = torch.nn.MSELoss()
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        
        # Assert that the loss is a non-negative number
        self.assertGreaterEqual(total_loss, 0, "Loss should be non-negative.")

    def test_checkpoint_loading(self):
        """
        Test that the model can load a checkpoint correctly.
        """
        # Generate synthetic dataset
        data = np.random.rand(100, 10)  
        targets = np.random.rand(100)  
        dataset = VGDDataset(data, targets)

        # Create a DataLoader from the dataset
        data_loader = VGDDataLoader(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        train_loader, val_loader, _ = data_loader.get_data_loaders()

        # Initialize the model
        input_size = 10 
        hidden_size = 64  
        output_size = 1 
        model = VGDModel(input_size, hidden_size, output_size)

        # Initialize optimizer
        optimizer = Adam(model.parameters(), lr=0.001)

        # Test loading checkpoint
        checkpoint_path = "test_checkpoint.pth"
        try:
            # Save a model checkpoint before loading it
            torch.save({
                'epoch': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': 0.5
            }, checkpoint_path)

            # Load the checkpoint
            model, optimizer, epoch = load_checkpoint(checkpoint_path, model)

            # Check that the epoch has been properly loaded
            self.assertEqual(epoch, 1, "Epoch from checkpoint is incorrect.")
        except Exception as e:
            self.fail(f"load_checkpoint raised an exception: {e}")

    def test_invalid_checkpoint_loading(self):
        """
        Test that loading an invalid checkpoint raises an appropriate error.
        """
        model = VGDModel(input_size=10, hidden_size=64, output_size=1)
        try:
            model, optimizer, epoch = load_checkpoint("invalid_checkpoint.pth", model)
            self.fail("load_checkpoint should have raised an error for invalid checkpoint.")
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    unittest.main()
