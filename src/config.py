"""
Configuration module for ECG Heartbeat Classification project.
Contains all configurable parameters using dataclasses.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import os


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_path: str = "./data/mitbih_train.csv"
    test_size: float = 0.3
    val_ratio: float = 2/3  # Ratio of validation from test split
    random_state: int = 42
    input_length: int = 187
    num_classes: int = 5
    

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    # LSTM parameters
    lstm_hidden_layers: int = 2
    lstm_neurons: int = 64
    lstm_input_size: int = 187
    
    # CNN parameters
    cnn_conv_layers: int = 2
    cnn_neurons: int = 64
    cnn_input_channels: int = 1
    cnn_input_length: int = 187
    
    # CatBoost parameters
    cat_learning_rate: float = 0.1
    cat_depth: int = 6
    cat_l2_leaf_reg: int = 3
    cat_border_count: int = 128
    cat_iterations: int = 5000
    cat_early_stopping_rounds: int = 30
    cat_task_type: str = "GPU"
    cat_devices: str = "0"
    
    # Output
    output_size: int = 5


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 0.001
    optimizer: Literal["Adam", "SGD", "AdamW", "RMSprop", "Adadelta", "Adamax", "Nadam"] = "Adam"
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 5
    
    # Focal Loss parameters
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    
    # Checkpointing
    save_checkpoint: bool = True
    save_best_only: bool = True
    checkpoint_dir: str = "./checkpoints"
    
    # Device
    device: str = "cuda"


@dataclass
class GAConfig:
    """Configuration for Genetic Algorithm optimization."""
    population_size: int = 10
    num_generations: int = 50
    mutation_rate_base: float = 0.1
    crossover_rate: float = 0.8
    early_stop_count: int = 15
    
    # LSTM search space
    lstm_hidden_layers_range: tuple = (1, 5)
    lstm_neurons_range: tuple = (32, 512)
    lstm_lr_range: tuple = (1e-3, 1e-1)
    
    # CNN search space
    cnn_conv_layers_range: tuple = (1, 7)
    cnn_neurons_range: tuple = (10, 512)
    cnn_lr_range: tuple = (1e-4, 1e-1)
    
    # CatBoost search space
    cat_lr_range: tuple = (1e-3, 0.5)
    cat_depth_range: tuple = (4, 10)
    cat_l2_range: tuple = (3, 15)
    cat_border_range: tuple = (30, 600)
    
    # Optimizer options
    optimizer_options: List[str] = field(default_factory=lambda: [
        "Adam", "SGD", "AdamW", "RMSprop", "Adadelta", "Adamax", "Nadam"
    ])


@dataclass 
class EnsembleConfig:
    """Configuration for stacking ensemble."""
    n_splits: int = 5
    meta_model_type: str = "logistic_regression"
    save_meta_features: bool = True
    meta_features_dir: str = "./meta_features"


@dataclass
class Config:
    """Main configuration class that combines all sub-configurations."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    
    # Logging
    log_dir: str = "./logs"
    experiment_name: str = "ecg_classification"
    verbose: int = 1
    seed: int = 42
    
    def __post_init__(self):
        """Create necessary directories after initialization."""
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.ensemble.meta_features_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    @classmethod
    def from_args(cls, args) -> "Config":
        """Create config from argparse arguments."""
        config = cls()
        
        # Update data config
        if hasattr(args, 'data_path') and args.data_path:
            config.data.data_path = args.data_path
        if hasattr(args, 'test_size') and args.test_size:
            config.data.test_size = args.test_size
            
        # Update training config
        if hasattr(args, 'batch_size') and args.batch_size:
            config.training.batch_size = args.batch_size
        if hasattr(args, 'epochs') and args.epochs:
            config.training.num_epochs = args.epochs
        if hasattr(args, 'learning_rate') and args.learning_rate:
            config.training.learning_rate = args.learning_rate
        if hasattr(args, 'optimizer') and args.optimizer:
            config.training.optimizer = args.optimizer
        if hasattr(args, 'device') and args.device:
            config.training.device = args.device
            
        # Update model config
        if hasattr(args, 'lstm_layers') and args.lstm_layers:
            config.model.lstm_hidden_layers = args.lstm_layers
        if hasattr(args, 'lstm_neurons') and args.lstm_neurons:
            config.model.lstm_neurons = args.lstm_neurons
        if hasattr(args, 'cnn_layers') and args.cnn_layers:
            config.model.cnn_conv_layers = args.cnn_layers
        if hasattr(args, 'cnn_neurons') and args.cnn_neurons:
            config.model.cnn_neurons = args.cnn_neurons
            
        # Update GA config
        if hasattr(args, 'population_size') and args.population_size:
            config.ga.population_size = args.population_size
        if hasattr(args, 'generations') and args.generations:
            config.ga.num_generations = args.generations
            
        # Update general config
        if hasattr(args, 'seed') and args.seed:
            config.seed = args.seed
        if hasattr(args, 'verbose') and args.verbose is not None:
            config.verbose = args.verbose
        if hasattr(args, 'experiment_name') and args.experiment_name:
            config.experiment_name = args.experiment_name
            
        return config
