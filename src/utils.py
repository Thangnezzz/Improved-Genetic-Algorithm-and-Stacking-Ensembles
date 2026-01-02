"""
Utility functions for ECG Heartbeat Classification project.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_device(prefer_gpu: bool = True) -> str:
    """
    Get available device.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if prefer_gpu and torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU")
    return device


def setup_logging(
    log_dir: str = "./logs",
    experiment_name: str = "experiment",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        experiment_name: Name of the experiment
        level: Logging level
        
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(experiment_name)
    logger.info(f"Logging to {log_file}")
    
    return logger


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    axes[0].plot(history.get('train_loss', []), label='Train Loss', color='blue')
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(history['val_loss'], label='Val Loss', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history.get('train_accuracy', []), label='Train Accuracy', color='blue')
    if 'val_accuracy' in history and history['val_accuracy']:
        axes[1].plot(history['val_accuracy'], label='Val Accuracy', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    cmap: str = 'Blues'
):
    """
    Plot confusion matrix heatmap.
    
    Args:
        conf_matrix: Confusion matrix array
        class_names: Names for each class
        save_path: Optional path to save the figure
        show: Whether to display the plot
        cmap: Colormap to use
    """
    if class_names is None:
        class_names = [str(i) for i in range(len(conf_matrix))]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Label',
        xlabel='Predicted Label',
        title='Confusion Matrix'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, format(conf_matrix[i, j], 'd'),
                ha="center", va="center",
                color="white" if conf_matrix[i, j] > thresh else "black"
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_ga_optimization(
    scores: List[float],
    title: str = "GA Optimization Progress",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot genetic algorithm optimization progress.
    
    Args:
        scores: Best scores per generation
        title: Plot title
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(1, len(scores) + 1), scores, marker='o', linewidth=2, markersize=4)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness Score')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Mark best score
    best_gen = np.argmax(scores) + 1
    best_score = max(scores)
    ax.axhline(y=best_score, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_score:.4f}')
    ax.axvline(x=best_gen, color='g', linestyle='--', alpha=0.5, label=f'Gen: {best_gen}')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_results(
    results: Dict[str, Any],
    output_path: str
):
    """
    Save results dictionary to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64)):
            serializable_results[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            serializable_results[key] = int(value)
        else:
            serializable_results[key] = value
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {output_path}")


def load_results(input_path: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Results dictionary
    """
    with open(input_path, 'r') as f:
        results = json.load(f)
    return results


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 1, 187)):
    """
    Print model architecture summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, length)
    """
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(model)
    print("-" * 60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 60 + "\n")


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{int(mins)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{int(hours)}h {int(mins)}m"


class EarlyStopping:
    """
    Early stopping handler to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for accuracy/score
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if score is an improvement."""
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
