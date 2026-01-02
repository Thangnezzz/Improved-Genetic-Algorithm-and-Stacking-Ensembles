#!/usr/bin/env python
"""
Main training script for ECG Heartbeat Classification.
Supports training individual models (CNN, LSTM, CatBoost) and stacking ensemble.
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data_loader import ECGDataLoader
from src.models import LSTMModel, CNNModel, create_catboost_model, get_optimizer
from src.losses import FocalLoss, get_loss_function
from src.engine import Trainer, Evaluator, train_pytorch_model, evaluate_pytorch_model
from src.ensemble import StackingEnsemble
from src.utils import set_seed, get_device, setup_logging, plot_training_history, save_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ECG Heartbeat Classification Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # General arguments
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'ensemble', 'evaluate'],
                        help='Training mode')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'lstm', 'catboost', 'all'],
                        help='Model to train')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='./data/mitbih_train.csv',
                        help='Path to training data CSV')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='Test set ratio')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam', 'SGD', 'AdamW', 'RMSprop', 'Adadelta', 'Adamax', 'Nadam'],
                        help='Optimizer to use')
    parser.add_argument('--early-stopping', action='store_true', default=True,
                        help='Use early stopping')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    
    # Model architecture arguments
    parser.add_argument('--lstm-layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--lstm-neurons', type=int, default=64,
                        help='Number of LSTM neurons')
    parser.add_argument('--cnn-layers', type=int, default=2,
                        help='Number of CNN conv layers')
    parser.add_argument('--cnn-neurons', type=int, default=64,
                        help='Number of CNN filters')
    
    # CatBoost arguments
    parser.add_argument('--cat-depth', type=int, default=6,
                        help='CatBoost tree depth')
    parser.add_argument('--cat-lr', type=float, default=0.1,
                        help='CatBoost learning rate')
    parser.add_argument('--cat-iterations', type=int, default=5000,
                        help='CatBoost iterations')
    
    # Loss function arguments
    parser.add_argument('--loss', type=str, default='focal',
                        choices=['focal', 'cross_entropy', 'label_smoothing'],
                        help='Loss function to use')
    parser.add_argument('--focal-alpha', type=float, default=1.0,
                        help='Focal loss alpha')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma')
    
    # Ensemble arguments
    parser.add_argument('--n-splits', type=int, default=5,
                        help='Number of CV splits for ensemble')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--experiment-name', type=str, default='ecg_experiment',
                        help='Experiment name')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0, 1, or 2)')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save training plots')
    
    return parser.parse_args()


def train_cnn(args, data_loader, config, device):
    """Train CNN model."""
    print("\n" + "="*60)
    print("TRAINING CNN MODEL")
    print("="*60)
    
    train_loader, val_loader, test_loader = data_loader.get_data_loaders(
        batch_size=args.batch_size
    )
    
    model = CNNModel(
        conv_layers=args.cnn_layers,
        neurons=args.cnn_neurons,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer
    )
    
    optimizer = get_optimizer(model, args.optimizer, args.learning_rate)
    loss_fn = get_loss_function(
        args.loss, 
        alpha=args.focal_alpha, 
        gamma=args.focal_gamma
    )
    
    model, history = train_pytorch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=args.epochs,
        early_stopping=args.early_stopping,
        patience=args.patience,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        model_name="cnn",
        verbose=args.verbose
    )
    
    # Evaluate
    print("\n--- CNN Evaluation on Test Set ---")
    results = evaluate_pytorch_model(model, test_loader, device=device)
    
    # Save results
    save_results(
        {"model": "cnn", "accuracy": results["accuracy"], "f1": results["f1_score"],
         "balanced_accuracy": results["balanced_accuracy"]},
        os.path.join(args.output_dir, "cnn_results.json")
    )
    
    if args.save_plots:
        plot_training_history(
            history, 
            save_path=os.path.join(args.output_dir, "cnn_training.png"),
            show=False
        )
    
    return model, results


def train_lstm(args, data_loader, config, device):
    """Train LSTM model."""
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)
    
    train_loader, val_loader, test_loader = data_loader.get_data_loaders(
        batch_size=args.batch_size
    )
    
    model = LSTMModel(
        hidden_layers=args.lstm_layers,
        neurons=args.lstm_neurons,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer
    )
    
    optimizer = get_optimizer(model, args.optimizer, args.learning_rate)
    loss_fn = get_loss_function(
        args.loss,
        alpha=args.focal_alpha,
        gamma=args.focal_gamma
    )
    
    model, history = train_pytorch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=args.epochs,
        early_stopping=args.early_stopping,
        patience=args.patience,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        model_name="lstm",
        verbose=args.verbose
    )
    
    # Evaluate
    print("\n--- LSTM Evaluation on Test Set ---")
    results = evaluate_pytorch_model(model, test_loader, device=device)
    
    # Save results
    save_results(
        {"model": "lstm", "accuracy": results["accuracy"], "f1": results["f1_score"],
         "balanced_accuracy": results["balanced_accuracy"]},
        os.path.join(args.output_dir, "lstm_results.json")
    )
    
    if args.save_plots:
        plot_training_history(
            history,
            save_path=os.path.join(args.output_dir, "lstm_training.png"),
            show=False
        )
    
    return model, results


def train_catboost(args, data_loader, config):
    """Train CatBoost model."""
    print("\n" + "="*60)
    print("TRAINING CATBOOST MODEL")
    print("="*60)
    
    x_train, y_train, x_val, y_val, x_test, y_test = data_loader.get_numpy_data()
    
    model = create_catboost_model(
        learning_rate=args.cat_lr,
        depth=args.cat_depth,
        iterations=args.cat_iterations,
        early_stopping_rounds=args.patience * 5,
        task_type="GPU" if args.device == "cuda" and torch.cuda.is_available() else "CPU"
    )
    
    model.fit(x_train, y_train, eval_set=(x_val, y_val))
    
    # Evaluate
    print("\n--- CatBoost Evaluation on Test Set ---")
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report
    
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    balanced_acc = balanced_accuracy_score(y_test, predictions)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Macro F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, digits=4))
    
    # Save model
    model.save_model(os.path.join(args.checkpoint_dir, "catboost_model.cbm"))
    
    results = {"accuracy": accuracy, "f1_score": f1, "balanced_accuracy": balanced_acc}
    save_results(
        {"model": "catboost", **results},
        os.path.join(args.output_dir, "catboost_results.json")
    )
    
    return model, results


def train_ensemble(args, data_loader, config, device):
    """Train stacking ensemble."""
    print("\n" + "="*60)
    print("TRAINING STACKING ENSEMBLE")
    print("="*60)
    
    # Get data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data_loader.get_torch_tensors()
    x_train, y_train, x_val, y_val, x_test, y_test = data_loader.get_numpy_data()
    
    train_loader, val_loader, test_loader = data_loader.get_data_loaders(
        batch_size=args.batch_size
    )
    
    # Create ensemble
    ensemble = StackingEnsemble(
        cnn_params={
            "conv_layers": args.cnn_layers,
            "neurons": args.cnn_neurons,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer
        },
        lstm_params={
            "hidden_layers": args.lstm_layers,
            "neurons": args.lstm_neurons,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer
        },
        catboost_params={
            "learning_rate": args.cat_lr,
            "depth": args.cat_depth,
            "l2_leaf_reg": 3,
            "border_count": 128
        },
        n_splits=args.n_splits,
        num_epochs=args.epochs,
        device=device,
        random_state=args.seed,
        verbose=args.verbose
    )
    
    # Generate meta-features using cross-validation
    print("\nGenerating meta-features...")
    meta_features, meta_labels = ensemble.generate_meta_features(
        X_train, Y_train, x_train, y_train,
        batch_size=args.batch_size
    )
    
    # Save meta-features
    ensemble.save_meta_features(
        meta_features, meta_labels,
        os.path.join(args.output_dir, "train_meta_features.csv")
    )
    
    # Train meta-model
    ensemble.train_meta_model(meta_features, meta_labels)
    
    # Train final base models
    print("\nTraining final base models...")
    ensemble.train_final_models(train_loader, x_train, y_train, val_loader)
    
    # Evaluate ensemble
    print("\n--- Ensemble Evaluation on Test Set ---")
    results = ensemble.evaluate(test_loader, x_test, y_test)
    
    # Save models
    ensemble.save_models(os.path.join(args.checkpoint_dir, "ensemble"))
    
    # Save results
    save_results(
        {"model": "ensemble", "accuracy": results["accuracy"], 
         "f1": results["f1_score"], "balanced_accuracy": results["balanced_accuracy"]},
        os.path.join(args.output_dir, "ensemble_results.json")
    )
    
    # Print fold metrics summary
    print("\n--- Cross-Validation Fold Metrics ---")
    summary = ensemble.get_fold_metrics_summary()
    for model_name, metrics in summary.items():
        print(f"{model_name.upper()}: F1={metrics['mean_f1']:.4f}±{metrics['std_f1']:.4f}, "
              f"Acc={metrics['mean_accuracy']:.2f}±{metrics['std_accuracy']:.2f}")
    
    return ensemble, results


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device(prefer_gpu=(args.device == 'cuda'))
    
    # Setup logging
    logger = setup_logging(
        log_dir=os.path.join(args.output_dir, "logs"),
        experiment_name=args.experiment_name
    )
    
    # Create config
    config = Config.from_args(args)
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    data_loader = ECGDataLoader(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.seed
    )
    data_loader.load_data()
    data_loader.prepare_splits()
    
    # Print data info
    info = data_loader.get_data_info()
    print(f"Train size: {info['train_size']}")
    print(f"Validation size: {info['val_size']}")
    print(f"Test size: {info['test_size']}")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Class distribution: {info['class_distribution_train']}")
    
    # Training mode
    if args.mode == 'train':
        if args.model == 'cnn':
            train_cnn(args, data_loader, config, device)
        elif args.model == 'lstm':
            train_lstm(args, data_loader, config, device)
        elif args.model == 'catboost':
            train_catboost(args, data_loader, config)
        elif args.model == 'all':
            train_cnn(args, data_loader, config, device)
            train_lstm(args, data_loader, config, device)
            train_catboost(args, data_loader, config)
    
    elif args.mode == 'ensemble':
        train_ensemble(args, data_loader, config, device)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
