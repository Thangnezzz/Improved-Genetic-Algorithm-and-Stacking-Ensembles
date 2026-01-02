"""
Stacking Ensemble module for ECG Heartbeat Classification.
Combines CNN, LSTM, and CatBoost predictions using a meta-learner.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)
from typing import List, Tuple, Dict, Optional, Any
import joblib
import os

from .models import LSTMModel, CNNModel, create_catboost_model, get_optimizer
from .losses import FocalLoss
from .engine import Trainer, Evaluator


class StackingEnsemble:
    """
    Stacking Ensemble that combines CNN, LSTM, and CatBoost predictions.
    Uses cross-validation to generate meta-features and trains a meta-learner.
    """
    
    def __init__(
        self,
        cnn_params: Dict[str, Any],
        lstm_params: Dict[str, Any],
        catboost_params: Dict[str, Any],
        n_splits: int = 5,
        num_epochs: int = 50,
        device: str = "cuda",
        random_state: int = 42,
        verbose: int = 1
    ):
        """
        Initialize stacking ensemble.
        
        Args:
            cnn_params: CNN hyperparameters (conv_layers, neurons, learning_rate, optimizer)
            lstm_params: LSTM hyperparameters (hidden_layers, neurons, learning_rate, optimizer)
            catboost_params: CatBoost hyperparameters (learning_rate, depth, l2_leaf_reg, border_count)
            n_splits: Number of cross-validation folds
            num_epochs: Training epochs for neural networks
            device: Device for PyTorch models
            random_state: Random seed
            verbose: Verbosity level
        """
        self.cnn_params = cnn_params
        self.lstm_params = lstm_params
        self.catboost_params = catboost_params
        self.n_splits = n_splits
        self.num_epochs = num_epochs
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state
        self.verbose = verbose
        
        self.meta_model = None
        self.trained_models = {
            "cnn": None,
            "lstm": None,
            "catboost": None
        }
        
        # Metrics storage
        self.fold_metrics = {
            "cnn": {"f1": [], "accuracy": []},
            "lstm": {"f1": [], "accuracy": []},
            "catboost": {"f1": [], "accuracy": []}
        }
    
    def _create_cnn(self) -> CNNModel:
        """Create CNN model with stored parameters."""
        return CNNModel(
            conv_layers=self.cnn_params.get("conv_layers", 2),
            neurons=self.cnn_params.get("neurons", 64),
            learning_rate=self.cnn_params.get("learning_rate", 0.001),
            optimizer_name=self.cnn_params.get("optimizer", "Adam")
        )
    
    def _create_lstm(self) -> LSTMModel:
        """Create LSTM model with stored parameters."""
        return LSTMModel(
            hidden_layers=self.lstm_params.get("hidden_layers", 2),
            neurons=self.lstm_params.get("neurons", 64),
            learning_rate=self.lstm_params.get("learning_rate", 0.001),
            optimizer_name=self.lstm_params.get("optimizer", "Adam")
        )
    
    def _create_catboost(self):
        """Create CatBoost model with stored parameters."""
        return create_catboost_model(
            learning_rate=self.catboost_params.get("learning_rate", 0.1),
            depth=self.catboost_params.get("depth", 6),
            l2_leaf_reg=self.catboost_params.get("l2_leaf_reg", 3),
            border_count=self.catboost_params.get("border_count", 128)
        )
    
    def generate_meta_features(
        self,
        X_train_tensor: torch.Tensor,
        Y_train_tensor: torch.Tensor,
        x_train_numpy: np.ndarray,
        y_train_numpy: np.ndarray,
        batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate meta-features using cross-validation.
        
        Args:
            X_train_tensor: Training features as PyTorch tensor
            Y_train_tensor: Training labels as PyTorch tensor
            x_train_numpy: Training features as numpy array
            y_train_numpy: Training labels as numpy array
            batch_size: Batch size for data loaders
            
        Returns:
            Tuple of (meta_features, meta_labels)
        """
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        meta_features_list = []
        meta_labels_list = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_tensor, Y_train_tensor)):
            if self.verbose >= 1:
                print(f"\n=== Fold {fold + 1}/{self.n_splits} ===")
            
            # Split data for this fold
            X_fold_train = X_train_tensor[train_idx]
            Y_fold_train = Y_train_tensor[train_idx]
            X_fold_val = X_train_tensor[val_idx]
            Y_fold_val = Y_train_tensor[val_idx]
            
            x_fold_train = x_train_numpy[train_idx]
            y_fold_train = y_train_numpy[train_idx]
            x_fold_val = x_train_numpy[val_idx]
            y_fold_val = y_train_numpy[val_idx]
            
            # Create data loaders
            train_dataset = TensorDataset(X_fold_train, Y_fold_train)
            val_dataset = TensorDataset(X_fold_val, Y_fold_val)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Train CNN
            if self.verbose >= 1:
                print("Training CNN...")
            cnn_proba = self._train_and_predict_nn(
                self._create_cnn(), train_loader, val_loader, "CNN"
            )
            
            # Train LSTM  
            if self.verbose >= 1:
                print("Training LSTM...")
            lstm_proba = self._train_and_predict_nn(
                self._create_lstm(), train_loader, val_loader, "LSTM"
            )
            
            # Train CatBoost
            if self.verbose >= 1:
                print("Training CatBoost...")
            catboost_proba = self._train_and_predict_catboost(
                x_fold_train, y_fold_train, x_fold_val, y_fold_val
            )
            
            # Combine probabilities as meta-features
            fold_meta_features = np.hstack([cnn_proba, lstm_proba, catboost_proba])
            meta_features_list.append(fold_meta_features)
            meta_labels_list.extend(y_fold_val)
        
        meta_features = np.vstack(meta_features_list)
        meta_labels = np.array(meta_labels_list)
        
        return meta_features, meta_labels
    
    def _train_and_predict_nn(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str
    ) -> np.ndarray:
        """Train neural network and get prediction probabilities."""
        optimizer = get_optimizer(model, model.optimizer_name, model.learning_rate)
        loss_fn = FocalLoss(alpha=1, gamma=2)
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=self.device,
            verbose=0
        )
        
        trainer.train(
            train_loader=train_loader,
            num_epochs=self.num_epochs,
            early_stopping=True,
            patience=5
        )
        
        evaluator = Evaluator(model=model, device=self.device)
        results = evaluator.evaluate(val_loader, show_report=False)
        
        # Store metrics
        key = model_name.lower()
        self.fold_metrics[key]["f1"].append(results["f1_score"])
        self.fold_metrics[key]["accuracy"].append(results["accuracy"])
        
        return np.array(results["probabilities"])
    
    def _train_and_predict_catboost(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray
    ) -> np.ndarray:
        """Train CatBoost and get prediction probabilities."""
        model = self._create_catboost()
        model.fit(x_train, y_train)
        
        predictions = model.predict(x_val)
        probabilities = model.predict_proba(x_val)
        
        # Store metrics
        acc = accuracy_score(y_val, predictions)
        f1 = f1_score(y_val, predictions, average='macro')
        self.fold_metrics["catboost"]["f1"].append(f1)
        self.fold_metrics["catboost"]["accuracy"].append(acc)
        
        return probabilities
    
    def train_meta_model(
        self,
        meta_features: np.ndarray,
        meta_labels: np.ndarray
    ):
        """
        Train the meta-learner on meta-features.
        
        Args:
            meta_features: Meta-features from base models
            meta_labels: Corresponding labels
        """
        if self.verbose >= 1:
            print("\nTraining meta-model (Logistic Regression)...")
        
        self.meta_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        self.meta_model.fit(meta_features, meta_labels)
        
        if self.verbose >= 1:
            print("Meta-model training complete!")
    
    def train_final_models(
        self,
        train_loader: DataLoader,
        x_train: np.ndarray,
        y_train: np.ndarray,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Train final base models on full training data.
        
        Args:
            train_loader: Full training data loader
            x_train: Full training features (numpy)
            y_train: Full training labels (numpy)
            val_loader: Optional validation loader
        """
        if self.verbose >= 1:
            print("\n=== Training Final Models ===")
        
        # Train CNN
        print("Training final CNN model...")
        self.trained_models["cnn"] = self._create_cnn()
        cnn_optimizer = get_optimizer(
            self.trained_models["cnn"],
            self.trained_models["cnn"].optimizer_name,
            self.trained_models["cnn"].learning_rate
        )
        cnn_trainer = Trainer(
            model=self.trained_models["cnn"],
            optimizer=cnn_optimizer,
            loss_fn=FocalLoss(alpha=1, gamma=2),
            device=self.device,
            verbose=self.verbose
        )
        cnn_trainer.train(
            train_loader=train_loader,
            num_epochs=self.num_epochs,
            val_loader=val_loader,
            early_stopping=True,
            patience=5,
            model_name="cnn_final"
        )
        
        # Train LSTM
        print("Training final LSTM model...")
        self.trained_models["lstm"] = self._create_lstm()
        lstm_optimizer = get_optimizer(
            self.trained_models["lstm"],
            self.trained_models["lstm"].optimizer_name,
            self.trained_models["lstm"].learning_rate
        )
        lstm_trainer = Trainer(
            model=self.trained_models["lstm"],
            optimizer=lstm_optimizer,
            loss_fn=FocalLoss(alpha=1, gamma=2),
            device=self.device,
            verbose=self.verbose
        )
        lstm_trainer.train(
            train_loader=train_loader,
            num_epochs=self.num_epochs,
            val_loader=val_loader,
            early_stopping=True,
            patience=5,
            model_name="lstm_final"
        )
        
        # Train CatBoost
        print("Training final CatBoost model...")
        self.trained_models["catboost"] = self._create_catboost()
        self.trained_models["catboost"].fit(x_train, y_train)
    
    def predict(
        self,
        test_loader: DataLoader,
        x_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the stacking ensemble.
        
        Args:
            test_loader: Test data loader
            x_test: Test features (numpy)
            
        Returns:
            Tuple of (predictions, meta_features)
        """
        # Get probabilities from base models
        cnn_evaluator = Evaluator(self.trained_models["cnn"], device=self.device)
        cnn_proba = cnn_evaluator.get_probabilities(test_loader)
        
        lstm_evaluator = Evaluator(self.trained_models["lstm"], device=self.device)
        lstm_proba = lstm_evaluator.get_probabilities(test_loader)
        
        catboost_proba = self.trained_models["catboost"].predict_proba(x_test)
        
        # Combine as meta-features
        meta_features = np.hstack([cnn_proba, lstm_proba, catboost_proba])
        
        # Predict with meta-model
        predictions = self.meta_model.predict(meta_features)
        
        return predictions, meta_features
    
    def evaluate(
        self,
        test_loader: DataLoader,
        x_test: np.ndarray,
        y_test: np.ndarray,
        show_report: bool = True
    ) -> Dict:
        """
        Evaluate the stacking ensemble.
        
        Args:
            test_loader: Test data loader
            x_test: Test features (numpy)
            y_test: Test labels
            show_report: Whether to print classification report
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions, meta_features = self.predict(test_loader, x_test)
        
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='macro')
        balanced_acc = balanced_accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        
        if show_report:
            print("\n=== Stacking Ensemble Results ===")
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print(f"Balanced Accuracy: {balanced_acc:.4f}")
            print(f"Macro F1 Score: {f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, predictions, digits=4))
            print("\nConfusion Matrix:")
            print(conf_matrix)
        
        return {
            "predictions": predictions,
            "meta_features": meta_features,
            "accuracy": accuracy,
            "f1_score": f1,
            "balanced_accuracy": balanced_acc,
            "confusion_matrix": conf_matrix
        }
    
    def save_models(self, output_dir: str = "./models"):
        """
        Save all trained models.
        
        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save PyTorch models
        torch.save(
            self.trained_models["cnn"].state_dict(),
            os.path.join(output_dir, "cnn_model.pth")
        )
        torch.save(
            self.trained_models["lstm"].state_dict(),
            os.path.join(output_dir, "lstm_model.pth")
        )
        
        # Save CatBoost
        self.trained_models["catboost"].save_model(
            os.path.join(output_dir, "catboost_model.cbm")
        )
        
        # Save meta-model
        joblib.dump(self.meta_model, os.path.join(output_dir, "meta_model.joblib"))
        
        # Save parameters
        params = {
            "cnn_params": self.cnn_params,
            "lstm_params": self.lstm_params,
            "catboost_params": self.catboost_params
        }
        joblib.dump(params, os.path.join(output_dir, "model_params.joblib"))
        
        print(f"Models saved to {output_dir}")
    
    def load_models(self, model_dir: str = "./models"):
        """
        Load all trained models.
        
        Args:
            model_dir: Directory containing saved models
        """
        # Load parameters
        params = joblib.load(os.path.join(model_dir, "model_params.joblib"))
        self.cnn_params = params["cnn_params"]
        self.lstm_params = params["lstm_params"]
        self.catboost_params = params["catboost_params"]
        
        # Load PyTorch models
        self.trained_models["cnn"] = self._create_cnn()
        self.trained_models["cnn"].load_state_dict(
            torch.load(os.path.join(model_dir, "cnn_model.pth"), map_location=self.device)
        )
        
        self.trained_models["lstm"] = self._create_lstm()
        self.trained_models["lstm"].load_state_dict(
            torch.load(os.path.join(model_dir, "lstm_model.pth"), map_location=self.device)
        )
        
        # Load CatBoost
        self.trained_models["catboost"] = self._create_catboost()
        self.trained_models["catboost"].load_model(
            os.path.join(model_dir, "catboost_model.cbm")
        )
        
        # Load meta-model
        self.meta_model = joblib.load(os.path.join(model_dir, "meta_model.joblib"))
        
        print(f"Models loaded from {model_dir}")
    
    def save_meta_features(
        self,
        meta_features: np.ndarray,
        meta_labels: np.ndarray,
        output_path: str,
        num_classes: int = 5
    ):
        """
        Save meta-features to CSV file.
        
        Args:
            meta_features: Meta-features array
            meta_labels: Labels array
            output_path: Path to save CSV
            num_classes: Number of classes (for naming columns)
        """
        columns = []
        for model in ["cnn", "lstm", "catboost"]:
            for i in range(num_classes):
                columns.append(f"{model}_proba_{i}")
        
        df = pd.DataFrame(meta_features, columns=columns)
        df["labels"] = meta_labels
        df.to_csv(output_path, index=False)
        
        print(f"Meta-features saved to {output_path}")
    
    def get_fold_metrics_summary(self) -> Dict:
        """Get summary of cross-validation fold metrics."""
        summary = {}
        for model_name, metrics in self.fold_metrics.items():
            summary[model_name] = {
                "mean_f1": np.mean(metrics["f1"]),
                "std_f1": np.std(metrics["f1"]),
                "mean_accuracy": np.mean(metrics["accuracy"]),
                "std_accuracy": np.std(metrics["accuracy"])
            }
        return summary
