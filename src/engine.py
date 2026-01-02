"""
Training and evaluation engine for ECG Heartbeat Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    accuracy_score, classification_report, 
    balanced_accuracy_score, confusion_matrix
)
from typing import Optional, Tuple, List, Dict
import numpy as np
import os


class Trainer:
    """
    Training engine for PyTorch models.
    Handles training loop, validation, early stopping, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        verbose: int = 1
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer for training
            loss_fn: Loss function
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        total_batches = len(train_loader)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if self.verbose >= 2:
                progress = int((batch_idx + 1) / total_batches * 100)
                print(f'\rBatch {batch_idx + 1}/{total_batches} ({progress}%)', end='')
        
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = running_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
        early_stopping: bool = True,
        patience: int = 5,
        save_best: bool = True,
        model_name: str = "model"
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs to train
            val_loader: Optional validation data loader
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            save_best: Whether to save best model
            model_name: Name for saved model file
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)
                
                if self.verbose >= 1:
                    print(f'Epoch {epoch + 1}/{num_epochs} - '
                          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                
                # Save best model
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    self._save_checkpoint(epoch, val_loss, val_acc, model_name)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if early_stopping and patience_counter >= patience:
                    if self.verbose >= 1:
                        print(f'Early stopping at epoch {epoch + 1}')
                    break
            else:
                if self.verbose >= 1:
                    print(f'Epoch {epoch + 1}/{num_epochs} - '
                          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        print('Training finished!')
        return self.history
    
    def _save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, model_name: str):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_acc
        }
        path = os.path.join(self.checkpoint_dir, f"{model_name}_best.pth")
        torch.save(checkpoint, path)
        if self.verbose >= 2:
            print(f'Checkpoint saved to {path}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f'Model loaded from {checkpoint_path}')
        return checkpoint


class Evaluator:
    """
    Evaluation engine for trained models.
    Computes various metrics and generates predictions.
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize evaluator.
        
        Args:
            model: Trained PyTorch model
            device: Device to use for evaluation
        """
        self.model = model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
    def evaluate(
        self,
        test_loader: DataLoader,
        true_labels: np.ndarray = None,
        show_report: bool = True
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            true_labels: Optional ground truth labels for metrics
            show_report: Whether to print classification report
            
        Returns:
            Dictionary containing predictions and metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().tolist())
                all_probabilities.extend(probabilities.cpu().tolist())
        
        accuracy = 100 * correct / total
        
        # Use labels from loader if not provided
        if true_labels is None:
            true_labels = []
            for _, labels in test_loader:
                true_labels.extend(labels.tolist())
            true_labels = np.array(true_labels)
        
        # Compute metrics
        f1 = f1_score(true_labels, all_predictions, average='macro')
        balanced_acc = balanced_accuracy_score(true_labels, all_predictions)
        conf_matrix = confusion_matrix(true_labels, all_predictions)
        
        if show_report:
            print(f'\nAccuracy: {accuracy:.2f}%')
            print(f'Balanced Accuracy: {balanced_acc:.4f}')
            print(f'Macro F1 Score: {f1:.4f}')
            print('\nClassification Report:')
            print(classification_report(true_labels, all_predictions, digits=4))
            print('\nConfusion Matrix:')
            print(conf_matrix)
        
        return {
            "predictions": all_predictions,
            "probabilities": all_probabilities,
            "accuracy": accuracy,
            "f1_score": f1,
            "balanced_accuracy": balanced_acc,
            "confusion_matrix": conf_matrix
        }
    
    def get_probabilities(self, test_loader: DataLoader) -> np.ndarray:
        """
        Get prediction probabilities for ensemble.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Array of prediction probabilities
        """
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().tolist())
        
        return np.array(all_probs)


def train_pytorch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    num_epochs: int = 50,
    early_stopping: bool = True,
    patience: int = 5,
    device: str = "cuda",
    checkpoint_dir: str = "./checkpoints",
    model_name: str = "model",
    verbose: int = 1
) -> Tuple[nn.Module, Dict]:
    """
    Convenience function to train a PyTorch model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        loss_fn: Loss function
        num_epochs: Number of epochs
        early_stopping: Whether to use early stopping
        patience: Early stopping patience
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        model_name: Name for saved model
        verbose: Verbosity level
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=checkpoint_dir,
        verbose=verbose
    )
    
    history = trainer.train(
        train_loader=train_loader,
        num_epochs=num_epochs,
        val_loader=val_loader,
        early_stopping=early_stopping,
        patience=patience,
        save_best=True,
        model_name=model_name
    )
    
    return model, history


def evaluate_pytorch_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
    show_report: bool = True
) -> Dict:
    """
    Convenience function to evaluate a PyTorch model.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use
        show_report: Whether to print report
        
    Returns:
        Evaluation metrics dictionary
    """
    evaluator = Evaluator(model=model, device=device)
    return evaluator.evaluate(test_loader=test_loader, show_report=show_report)
