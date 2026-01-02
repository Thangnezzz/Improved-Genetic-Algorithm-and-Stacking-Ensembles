"""
Model architectures for ECG Heartbeat Classification.
Contains LSTM, CNN and CatBoost model definitions.
"""

import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from typing import Optional


class LSTMModel(nn.Module):
    """
    LSTM-based model for ECG classification.
    
    Architecture:
        - Multiple stacked LSTM layers
        - Final linear layer for classification
    """
    
    def __init__(
        self,
        input_size: int = 187,
        hidden_layers: int = 2,
        neurons: int = 64,
        learning_rate: float = 0.001,
        optimizer_name: str = "Adam",
        output_size: int = 5,
        seed: int = 42
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Size of input features (ECG signal length)
            hidden_layers: Number of stacked LSTM layers
            neurons: Number of neurons in each LSTM layer
            learning_rate: Learning rate for optimizer
            optimizer_name: Name of optimizer to use
            output_size: Number of output classes
            seed: Random seed for reproducibility
        """
        super(LSTMModel, self).__init__()
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.neurons = neurons
        
        # Build LSTM layers
        self.lstm_layers = nn.ModuleList()
        current_input_size = input_size
        
        for _ in range(hidden_layers):
            self.lstm_layers.append(
                nn.LSTM(current_input_size, neurons, batch_first=True)
            )
            current_input_size = neurons
        
        self.output_layer = nn.Linear(neurons, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, channels, sequence_length)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Reshape if needed: (batch, 1, 187) -> (batch, 187, 1) for LSTM
        if x.size(1) == 1:
            x = x.transpose(1, 2)
            
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            
        # Take the output from the last timestep
        x = self.output_layer(x[:, -1, :])
        return x


class CNNModel(nn.Module):
    """
    1D CNN-based model for ECG classification.
    
    Architecture:
        - Multiple Conv1D layers with ReLU and MaxPool
        - Final linear layer for classification
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        input_length: int = 187,
        conv_layers: int = 2,
        neurons: int = 64,
        learning_rate: float = 0.001,
        optimizer_name: str = "Adam",
        output_size: int = 5,
        kernel_size: int = 3,
        seed: int = 42
    ):
        """
        Initialize CNN model.
        
        Args:
            input_channels: Number of input channels
            input_length: Length of input sequence
            conv_layers: Number of convolutional layers
            neurons: Number of filters in each conv layer
            learning_rate: Learning rate for optimizer
            optimizer_name: Name of optimizer to use
            output_size: Number of output classes
            kernel_size: Size of convolutional kernel
            seed: Random seed for reproducibility
        """
        super(CNNModel, self).__init__()
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.conv_layers_count = conv_layers
        self.neurons = neurons
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for _ in range(conv_layers):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, neurons, kernel_size=kernel_size, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ))
            in_channels = neurons
        
        # Calculate the size after all pooling operations
        reduced_length = input_length // (2 ** conv_layers)
        self.fc_input_size = neurons * reduced_length
        
        self.output_layer = nn.Linear(self.fc_input_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, channels, sequence_length)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        for conv in self.conv_layers:
            x = conv(x)
        
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)
        return x


def create_catboost_model(
    learning_rate: float = 0.1,
    depth: int = 6,
    l2_leaf_reg: int = 3,
    border_count: int = 128,
    iterations: int = 5000,
    early_stopping_rounds: int = 30,
    random_state: int = 42,
    task_type: str = "GPU",
    devices: str = "0",
    verbose: int = 1000,
    auto_class_weights: str = "Balanced",
    eval_metric: str = "TotalF1"
) -> CatBoostClassifier:
    """
    Create a CatBoost classifier with specified parameters.
    
    Args:
        learning_rate: Learning rate for gradient boosting
        depth: Depth of trees
        l2_leaf_reg: L2 regularization coefficient
        border_count: Number of splits for numerical features
        iterations: Maximum number of trees
        early_stopping_rounds: Early stopping patience
        random_state: Random seed
        task_type: Device type ('GPU' or 'CPU')
        devices: GPU device ID
        verbose: Verbosity level
        auto_class_weights: Class weighting strategy
        eval_metric: Evaluation metric
        
    Returns:
        Configured CatBoostClassifier instance
    """
    return CatBoostClassifier(
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        border_count=border_count,
        iterations=iterations,
        verbose=verbose,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        task_type=task_type,
        devices=devices,
        auto_class_weights=auto_class_weights,
        eval_metric=eval_metric
    )


def get_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float):
    """
    Get PyTorch optimizer by name.
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer
        learning_rate: Learning rate
        
    Returns:
        Configured optimizer instance
    """
    import torch.optim as optim
    
    optimizers = {
        "Adam": lambda: optim.Adam(model.parameters(), lr=learning_rate),
        "SGD": lambda: optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9),
        "AdamW": lambda: optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01),
        "RMSprop": lambda: optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99),
        "Adadelta": lambda: optim.Adadelta(model.parameters(), rho=learning_rate),
        "Adamax": lambda: optim.Adamax(model.parameters(), lr=learning_rate),
        "Nadam": lambda: optim.NAdam(model.parameters(), lr=learning_rate),
    }
    
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {list(optimizers.keys())}")
    
    return optimizers[optimizer_name]()


def save_model(model: nn.Module, filepath: str, optimizer: Optional[torch.optim.Optimizer] = None, 
               epoch: int = 0, val_loss: float = 0.0, val_accuracy: float = 0.0):
    """
    Save PyTorch model checkpoint.
    
    Args:
        model: Model to save
        filepath: Path to save checkpoint
        optimizer: Optional optimizer to save
        epoch: Current epoch number
        val_loss: Validation loss
        val_accuracy: Validation accuracy
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(model: nn.Module, filepath: str, optimizer: Optional[torch.optim.Optimizer] = None):
    """
    Load PyTorch model checkpoint.
    
    Args:
        model: Model to load weights into
        filepath: Path to checkpoint file
        optimizer: Optional optimizer to load state into
        
    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    print(f"Model loaded from {filepath}, epoch {checkpoint.get('epoch', 'N/A')}")
    return checkpoint
