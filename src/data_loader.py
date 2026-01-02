"""
Data loading and preprocessing module for ECG Heartbeat Classification.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class DataSplit:
    """Container for train/val/test data splits."""
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


class ECGDataLoader:
    """
    Data loader for ECG Heartbeat dataset.
    Handles data loading, preprocessing, and creating PyTorch DataLoaders.
    """
    
    def __init__(
        self,
        data_path: str,
        test_size: float = 0.3,
        val_ratio: float = 2/3,
        random_state: int = 42,
        input_length: int = 187
    ):
        """
        Initialize the ECG data loader.
        
        Args:
            data_path: Path to the CSV data file
            test_size: Proportion of data for test+validation split
            val_ratio: Ratio of validation data from the test split
            random_state: Random seed for reproducibility
            input_length: Length of input ECG signal
        """
        self.data_path = data_path
        self.test_size = test_size
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.input_length = input_length
        
        self.data_split: Optional[DataSplit] = None
        self.df: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file and rename columns.
        
        Returns:
            DataFrame with renamed columns
        """
        self.df = pd.read_csv(self.data_path)
        
        # Rename columns
        new_names = [f"{i}_time" for i in range(1, len(self.df.columns))]
        new_names.append("Targets")
        self.df.columns = new_names[:len(self.df.columns)]
        
        # Rename last column to Targets if needed
        if self.df.columns[-1] != "Targets":
            cols = list(self.df.columns)
            cols[-1] = "Targets"
            self.df.columns = cols
            
        return self.df
    
    def prepare_splits(self) -> DataSplit:
        """
        Split data into train, validation, and test sets.
        
        Returns:
            DataSplit object containing all splits
        """
        if self.df is None:
            self.load_data()
            
        train_features = self.df.drop(["Targets"], axis=1)
        labels = self.df["Targets"]
        
        # First split: train and temp (val + test)
        x_train, x_temp, y_train, y_temp = train_test_split(
            train_features, labels,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Second split: val and test from temp
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp,
            test_size=self.val_ratio,
            random_state=self.random_state
        )
        
        # Convert to float32
        self.data_split = DataSplit(
            x_train=x_train.astype(np.float32).values,
            y_train=y_train.astype(np.float32).values,
            x_val=x_val.astype(np.float32).values,
            y_val=y_val.astype(np.float32).values,
            x_test=x_test.astype(np.float32).values,
            y_test=y_test.astype(np.float32).values
        )
        
        return self.data_split
    
    def get_torch_tensors(self) -> Tuple[torch.Tensor, ...]:
        """
        Convert data splits to PyTorch tensors.
        
        Returns:
            Tuple of (X_train, Y_train, X_val, Y_val, X_test, Y_test) tensors
        """
        if self.data_split is None:
            self.prepare_splits()
        
        # Reshape for 1D Conv/LSTM: (batch, channels, length)
        X_train = torch.from_numpy(self.data_split.x_train).float().reshape(-1, 1, self.input_length)
        Y_train = torch.from_numpy(self.data_split.y_train).long()
        
        X_val = torch.from_numpy(self.data_split.x_val).float().reshape(-1, 1, self.input_length)
        Y_val = torch.from_numpy(self.data_split.y_val).long()
        
        X_test = torch.from_numpy(self.data_split.x_test).float().reshape(-1, 1, self.input_length)
        Y_test = torch.from_numpy(self.data_split.y_test).long()
        
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
    
    def get_data_loaders(
        self,
        batch_size: int = 32,
        shuffle_train: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for train, validation, and test sets.
        
        Args:
            batch_size: Batch size for DataLoaders
            shuffle_train: Whether to shuffle training data
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        X_train, Y_train, X_val, Y_val, X_test, Y_test = self.get_torch_tensors()
        
        train_dataset = TensorDataset(X_train, Y_train)
        val_dataset = TensorDataset(X_val, Y_val)
        test_dataset = TensorDataset(X_test, Y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle_train
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def get_numpy_data(self) -> Tuple[np.ndarray, ...]:
        """
        Get numpy arrays for sklearn-based models (CatBoost).
        
        Returns:
            Tuple of (x_train, y_train, x_val, y_val, x_test, y_test) arrays
        """
        if self.data_split is None:
            self.prepare_splits()
            
        return (
            self.data_split.x_train,
            self.data_split.y_train,
            self.data_split.x_val,
            self.data_split.y_val,
            self.data_split.x_test,
            self.data_split.y_test
        )
    
    def get_stratified_kfold_loaders(
        self,
        n_splits: int = 5,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> List[Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]]:
        """
        Create DataLoaders for stratified K-fold cross-validation.
        
        Args:
            n_splits: Number of folds
            batch_size: Batch size for DataLoaders
            shuffle: Whether to shuffle folds
            
        Returns:
            List of tuples (train_loader, test_loader, train_numpy, test_numpy) for each fold
        """
        X_train, Y_train, _, _, _, _ = self.get_torch_tensors()
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)
        
        fold_loaders = []
        for train_idx, test_idx in skf.split(X_train, Y_train):
            X_train_fold = X_train[train_idx]
            Y_train_fold = Y_train[train_idx]
            X_test_fold = X_train[test_idx]
            Y_test_fold = Y_train[test_idx]
            
            train_dataset = TensorDataset(X_train_fold, Y_train_fold)
            test_dataset = TensorDataset(X_test_fold, Y_test_fold)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Also provide numpy arrays for CatBoost
            x_train_np = self.data_split.x_train[train_idx]
            y_train_np = self.data_split.y_train[train_idx]
            x_test_np = self.data_split.x_train[test_idx]
            y_test_np = self.data_split.y_train[test_idx]
            
            fold_loaders.append((
                train_loader, test_loader,
                (x_train_np, y_train_np),
                (x_test_np, y_test_np)
            ))
            
        return fold_loaders
    
    def get_data_info(self) -> dict:
        """
        Get information about the loaded data.
        
        Returns:
            Dictionary with data statistics
        """
        if self.data_split is None:
            self.prepare_splits()
            
        return {
            "train_size": len(self.data_split.x_train),
            "val_size": len(self.data_split.x_val),
            "test_size": len(self.data_split.x_test),
            "input_length": self.input_length,
            "num_classes": len(np.unique(self.data_split.y_train)),
            "class_distribution_train": dict(zip(
                *np.unique(self.data_split.y_train, return_counts=True)
            ))
        }
