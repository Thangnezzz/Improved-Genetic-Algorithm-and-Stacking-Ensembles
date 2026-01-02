#!/usr/bin/env python
"""
Hyperparameter optimization script using Genetic Algorithm.
Optimizes hyperparameters for CNN, LSTM, or CatBoost models.
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
from src.genetic_algorithm import (
    CatBoostGAOptimizer, 
    LSTMGAOptimizer, 
    CNNGAOptimizer,
    save_ga_results
)
from src.utils import set_seed, get_device, setup_logging, plot_ga_optimization


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Genetic Algorithm Hyperparameter Optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model to optimize
    parser.add_argument('--model', type=str, required=True,
                        choices=['cnn', 'lstm', 'catboost'],
                        help='Model to optimize')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='./data/mitbih_train.csv',
                        help='Path to training data CSV')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='Test set ratio')
    
    # GA arguments
    parser.add_argument('--population-size', type=int, default=10,
                        help='Population size')
    parser.add_argument('--generations', type=int, default=50,
                        help='Number of generations')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                        help='Base mutation rate')
    parser.add_argument('--early-stop-count', type=int, default=15,
                        help='Stop if same parents selected this many times')
    
    # Neural network training args
    parser.add_argument('--train-epochs', type=int, default=3,
                        help='Epochs to train each model during evaluation')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    
    # Search space arguments for LSTM
    parser.add_argument('--lstm-layers-min', type=int, default=1,
                        help='Min LSTM layers')
    parser.add_argument('--lstm-layers-max', type=int, default=5,
                        help='Max LSTM layers')
    parser.add_argument('--lstm-neurons-min', type=int, default=32,
                        help='Min LSTM neurons')
    parser.add_argument('--lstm-neurons-max', type=int, default=512,
                        help='Max LSTM neurons')
    
    # Search space arguments for CNN
    parser.add_argument('--cnn-layers-min', type=int, default=1,
                        help='Min CNN layers')
    parser.add_argument('--cnn-layers-max', type=int, default=7,
                        help='Max CNN layers')
    parser.add_argument('--cnn-neurons-min', type=int, default=10,
                        help='Min CNN filters')
    parser.add_argument('--cnn-neurons-max', type=int, default=512,
                        help='Max CNN filters')
    
    # Search space arguments for CatBoost
    parser.add_argument('--cat-lr-min', type=float, default=1e-3,
                        help='Min CatBoost learning rate')
    parser.add_argument('--cat-lr-max', type=float, default=0.5,
                        help='Max CatBoost learning rate')
    parser.add_argument('--cat-depth-min', type=int, default=4,
                        help='Min CatBoost depth')
    parser.add_argument('--cat-depth-max', type=int, default=10,
                        help='Max CatBoost depth')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./ga_results',
                        help='Output directory for GA results')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save optimization plots')
    
    return parser.parse_args()


def optimize_catboost(args, data_loader):
    """Run GA optimization for CatBoost."""
    print("\n" + "="*60)
    print("OPTIMIZING CATBOOST HYPERPARAMETERS")
    print("="*60)
    
    x_train, y_train, x_val, y_val, _, _ = data_loader.get_numpy_data()
    
    optimizer = CatBoostGAOptimizer(
        lr_range=(args.cat_lr_min, args.cat_lr_max),
        depth_range=(args.cat_depth_min, args.cat_depth_max),
        l2_range=(3, 15),
        border_range=(30, 600),
        population_size=args.population_size,
        num_generations=args.generations,
        mutation_rate_base=args.mutation_rate,
        early_stop_count=args.early_stop_count,
        random_state=args.seed,
        verbose=args.verbose
    )
    
    scores, best_params = optimizer.optimize(
        train_data=(x_train, y_train),
        val_data=(x_val, y_val)
    )
    
    return scores, best_params


def optimize_lstm(args, data_loader, device):
    """Run GA optimization for LSTM."""
    print("\n" + "="*60)
    print("OPTIMIZING LSTM HYPERPARAMETERS")
    print("="*60)
    
    train_loader, val_loader, _ = data_loader.get_data_loaders(batch_size=args.batch_size)
    _, _, _, y_val, _, _ = data_loader.get_numpy_data()
    
    optimizer = LSTMGAOptimizer(
        hidden_layers_range=(args.lstm_layers_min, args.lstm_layers_max),
        neurons_range=(args.lstm_neurons_min, args.lstm_neurons_max),
        lr_range=(1e-3, 1e-1),
        train_epochs=args.train_epochs,
        device=device,
        population_size=args.population_size,
        num_generations=args.generations,
        mutation_rate_base=args.mutation_rate,
        early_stop_count=args.early_stop_count,
        random_state=args.seed,
        verbose=args.verbose
    )
    
    scores, best_params = optimizer.optimize(
        train_data=train_loader,
        val_data=(val_loader, y_val)
    )
    
    return scores, best_params


def optimize_cnn(args, data_loader, device):
    """Run GA optimization for CNN."""
    print("\n" + "="*60)
    print("OPTIMIZING CNN HYPERPARAMETERS")
    print("="*60)
    
    train_loader, val_loader, _ = data_loader.get_data_loaders(batch_size=args.batch_size)
    _, _, _, y_val, _, _ = data_loader.get_numpy_data()
    
    optimizer = CNNGAOptimizer(
        conv_layers_range=(args.cnn_layers_min, args.cnn_layers_max),
        neurons_range=(args.cnn_neurons_min, args.cnn_neurons_max),
        lr_range=(1e-4, 1e-1),
        train_epochs=args.train_epochs,
        device=device,
        population_size=args.population_size,
        num_generations=args.generations,
        mutation_rate_base=args.mutation_rate,
        early_stop_count=args.early_stop_count,
        random_state=args.seed,
        verbose=args.verbose
    )
    
    scores, best_params = optimizer.optimize(
        train_data=train_loader,
        val_data=(val_loader, y_val)
    )
    
    return scores, best_params


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(prefer_gpu=(args.device == 'cuda'))
    
    # Setup logging
    logger = setup_logging(
        log_dir=os.path.join(args.output_dir, "logs"),
        experiment_name=f"ga_{args.model}"
    )
    
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
    
    info = data_loader.get_data_info()
    print(f"Train size: {info['train_size']}")
    print(f"Validation size: {info['val_size']}")
    
    # Run optimization
    if args.model == 'catboost':
        scores, best_params = optimize_catboost(args, data_loader)
    elif args.model == 'lstm':
        scores, best_params = optimize_lstm(args, data_loader, device)
    elif args.model == 'cnn':
        scores, best_params = optimize_cnn(args, data_loader, device)
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best score: {max(scores):.4f}")
    print(f"Best parameters: {best_params[0]}")
    print(f"Second best: {best_params[1]}")
    
    # Save results
    save_ga_results(scores, best_params, args.model, args.output_dir)
    
    # Plot optimization progress
    if args.save_plots:
        plot_ga_optimization(
            scores,
            title=f"{args.model.upper()} GA Optimization",
            save_path=os.path.join(args.output_dir, f"{args.model}_optimization.png"),
            show=False
        )
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
