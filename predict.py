#!/usr/bin/env python
"""
Prediction/Inference script for ECG Heartbeat Classification.
Load trained models and make predictions on new data.
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import ECGDataLoader
from src.models import LSTMModel, CNNModel, create_catboost_model, load_model
from src.engine import Evaluator
from src.ensemble import StackingEnsemble
from src.utils import get_device, plot_confusion_matrix


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ECG Heartbeat Classification Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        choices=['cnn', 'lstm', 'catboost', 'ensemble'],
                        help='Model to use for prediction')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to saved model or model directory')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to data CSV for prediction')
    parser.add_argument('--has-labels', action='store_true',
                        help='Whether data has ground truth labels')
    
    # Model architecture (needed for PyTorch models)
    parser.add_argument('--lstm-layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--lstm-neurons', type=int, default=64,
                        help='Number of LSTM neurons')
    parser.add_argument('--cnn-layers', type=int, default=2,
                        help='Number of CNN conv layers')
    parser.add_argument('--cnn-neurons', type=int, default=64,
                        help='Number of CNN filters')
    
    # Output arguments
    parser.add_argument('--output-path', type=str, default='./predictions.csv',
                        help='Path to save predictions')
    parser.add_argument('--save-probabilities', action='store_true',
                        help='Save prediction probabilities')
    
    # Other arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--show-report', action='store_true',
                        help='Show classification report (requires labels)')
    parser.add_argument('--save-confusion-matrix', action='store_true',
                        help='Save confusion matrix plot')
    
    return parser.parse_args()


def load_pytorch_model(model_class, model_path, model_kwargs, device):
    """Load a PyTorch model from checkpoint."""
    model = model_class(**model_kwargs)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def predict_cnn(args, data_loader, device):
    """Make predictions with CNN model."""
    print("\nLoading CNN model...")
    
    model = load_pytorch_model(
        CNNModel,
        args.model_path,
        {
            'conv_layers': args.cnn_layers,
            'neurons': args.cnn_neurons
        },
        device
    )
    
    # Get data
    _, _, test_loader = data_loader.get_data_loaders(batch_size=args.batch_size)
    
    evaluator = Evaluator(model, device=device)
    
    if args.has_labels:
        results = evaluator.evaluate(test_loader, show_report=args.show_report)
        return results['predictions'], results.get('probabilities')
    else:
        probs = evaluator.get_probabilities(test_loader)
        preds = np.argmax(probs, axis=1)
        return preds.tolist(), probs.tolist()


def predict_lstm(args, data_loader, device):
    """Make predictions with LSTM model."""
    print("\nLoading LSTM model...")
    
    model = load_pytorch_model(
        LSTMModel,
        args.model_path,
        {
            'hidden_layers': args.lstm_layers,
            'neurons': args.lstm_neurons
        },
        device
    )
    
    # Get data
    _, _, test_loader = data_loader.get_data_loaders(batch_size=args.batch_size)
    
    evaluator = Evaluator(model, device=device)
    
    if args.has_labels:
        results = evaluator.evaluate(test_loader, show_report=args.show_report)
        return results['predictions'], results.get('probabilities')
    else:
        probs = evaluator.get_probabilities(test_loader)
        preds = np.argmax(probs, axis=1)
        return preds.tolist(), probs.tolist()


def predict_catboost(args, data_loader):
    """Make predictions with CatBoost model."""
    print("\nLoading CatBoost model...")
    
    from catboost import CatBoostClassifier
    
    model = CatBoostClassifier()
    model.load_model(args.model_path)
    
    # Get data
    x_train, y_train, x_val, y_val, x_test, y_test = data_loader.get_numpy_data()
    
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    
    if args.has_labels and args.show_report:
        print("\n" + "="*60)
        print("CATBOOST EVALUATION")
        print("="*60)
        print(classification_report(y_test, predictions, digits=4))
        print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, predictions):.4f}")
    
    return predictions.tolist(), probabilities.tolist()


def predict_ensemble(args, data_loader, device):
    """Make predictions with stacking ensemble."""
    print("\nLoading Stacking Ensemble...")
    
    # Initialize ensemble with dummy params (will be loaded)
    ensemble = StackingEnsemble(
        cnn_params={},
        lstm_params={},
        catboost_params={},
        device=device
    )
    
    # Load trained models
    ensemble.load_models(args.model_path)
    
    # Get data
    _, _, test_loader = data_loader.get_data_loaders(batch_size=args.batch_size)
    _, _, _, _, x_test, y_test = data_loader.get_numpy_data()
    
    if args.has_labels:
        results = ensemble.evaluate(test_loader, x_test, y_test, show_report=args.show_report)
        return results['predictions'].tolist(), results['meta_features'].tolist()
    else:
        predictions, meta_features = ensemble.predict(test_loader, x_test)
        return predictions.tolist(), meta_features.tolist()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get device
    device = get_device(prefer_gpu=(args.device == 'cuda'))
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    data_loader = ECGDataLoader(
        data_path=args.data_path,
        test_size=0.999 if not args.has_labels else 0.3,  # Use all data for prediction if no labels
        random_state=42
    )
    data_loader.load_data()
    data_loader.prepare_splits()
    
    info = data_loader.get_data_info()
    print(f"Data size: {info['test_size']}")
    
    # Make predictions
    print("\n" + "="*60)
    print("MAKING PREDICTIONS")
    print("="*60)
    
    if args.model == 'cnn':
        predictions, probabilities = predict_cnn(args, data_loader, device)
    elif args.model == 'lstm':
        predictions, probabilities = predict_lstm(args, data_loader, device)
    elif args.model == 'catboost':
        predictions, probabilities = predict_catboost(args, data_loader)
    elif args.model == 'ensemble':
        predictions, probabilities = predict_ensemble(args, data_loader, device)
    
    # Save predictions
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create output dataframe
    output_df = pd.DataFrame({'prediction': predictions})
    
    if args.save_probabilities and probabilities is not None:
        probs_array = np.array(probabilities)
        for i in range(probs_array.shape[1]):
            output_df[f'prob_class_{i}'] = probs_array[:, i]
    
    output_df.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")
    
    # Save confusion matrix if labels available
    if args.has_labels and args.save_confusion_matrix:
        _, _, _, _, _, y_test = data_loader.get_numpy_data()
        cm = confusion_matrix(y_test, predictions)
        plot_confusion_matrix(
            cm,
            class_names=['N', 'S', 'V', 'F', 'Q'],
            save_path=args.output_path.replace('.csv', '_confusion_matrix.png'),
            show=False
        )
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE!")
    print("="*60)
    print(f"Total predictions: {len(predictions)}")
    print(f"Class distribution: {dict(zip(*np.unique(predictions, return_counts=True)))}")


if __name__ == '__main__':
    main()
