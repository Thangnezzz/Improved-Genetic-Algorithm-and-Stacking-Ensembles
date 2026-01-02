# ECG Heartbeat Classification with Genetic Algorithm Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive deep learning framework for ECG heartbeat classification using CNN, LSTM, CatBoost, and Stacking Ensemble with Genetic Algorithm hyperparameter optimization.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training Individual Models](#training-individual-models)
  - [Hyperparameter Optimization with GA](#hyperparameter-optimization-with-ga)
  - [Training Stacking Ensemble](#training-stacking-ensemble)
  - [Making Predictions](#making-predictions)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a robust ECG heartbeat classification system that:

1. **Classifies ECG signals** into 5 categories (N, S, V, F, Q)
2. **Uses multiple models**: CNN, LSTM, and CatBoost
3. **Optimizes hyperparameters** using Genetic Algorithm
4. **Combines predictions** using Stacking Ensemble with cross-validation
5. **Handles class imbalance** using Focal Loss and balanced class weights

## ✨ Features

- **Modular Architecture**: Clean, reusable code organized into separate modules
- **Command-Line Interface**: Full control via argparse arguments
- **Genetic Algorithm Optimization**: Automated hyperparameter tuning
- **Focal Loss**: Addresses class imbalance in ECG datasets
- **Stacking Ensemble**: Combines CNN, LSTM, and CatBoost predictions
- **Cross-Validation**: Robust meta-feature generation
- **Checkpointing**: Save and resume training
- **Comprehensive Metrics**: Accuracy, F1-score, Balanced Accuracy, Confusion Matrix

## 📁 Project Structure

```
ecg-heartbeat-classification/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py             # Configuration dataclasses
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── models.py             # CNN, LSTM, CatBoost model definitions
│   ├── losses.py             # Focal Loss and other loss functions
│   ├── engine.py             # Training and evaluation engines
│   ├── genetic_algorithm.py  # GA optimization for hyperparameters
│   ├── ensemble.py           # Stacking ensemble implementation
│   └── utils.py              # Utility functions
├── train.py                  # Main training script
├── optimize.py               # GA hyperparameter optimization script
├── predict.py                # Prediction/inference script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── data/                     # Data directory (create this)
    └── mitbih_train.csv      # MIT-BIH dataset
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ecg-heartbeat-classification.git
cd ecg-heartbeat-classification
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install PyTorch with CUDA** (if using GPU)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Dataset

This project uses the **MIT-BIH Arrhythmia Dataset** from Kaggle:
- [Heartbeat Dataset on Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

### Download and Setup

1. Download `mitbih_train.csv` from Kaggle
2. Create a `data/` directory in the project root
3. Place the CSV file in `data/mitbih_train.csv`

### Class Distribution

| Class | Label | Description |
|-------|-------|-------------|
| 0 | N | Normal beat |
| 1 | S | Supraventricular ectopic beat |
| 2 | V | Ventricular ectopic beat |
| 3 | F | Fusion beat |
| 4 | Q | Unknown beat |

## 💻 Usage

### Training Individual Models

**Train CNN model:**
```bash
python train.py --mode train --model cnn \
    --data-path ./data/mitbih_train.csv \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --cnn-layers 2 \
    --cnn-neurons 64 \
    --device cuda \
    --save-plots
```

**Train LSTM model:**
```bash
python train.py --mode train --model lstm \
    --data-path ./data/mitbih_train.csv \
    --epochs 50 \
    --batch-size 32 \
    --lstm-layers 2 \
    --lstm-neurons 64
```

**Train CatBoost model:**
```bash
python train.py --mode train --model catboost \
    --data-path ./data/mitbih_train.csv \
    --cat-depth 6 \
    --cat-lr 0.1 \
    --cat-iterations 5000
```

**Train all models:**
```bash
python train.py --mode train --model all --data-path ./data/mitbih_train.csv
```

### Hyperparameter Optimization with GA

**Optimize CNN hyperparameters:**
```bash
python optimize.py --model cnn \
    --data-path ./data/mitbih_train.csv \
    --population-size 10 \
    --generations 50 \
    --train-epochs 3 \
    --output-dir ./ga_results \
    --save-plots
```

**Optimize LSTM hyperparameters:**
```bash
python optimize.py --model lstm \
    --data-path ./data/mitbih_train.csv \
    --population-size 10 \
    --generations 50 \
    --lstm-layers-min 1 \
    --lstm-layers-max 5 \
    --lstm-neurons-min 32 \
    --lstm-neurons-max 512
```

**Optimize CatBoost hyperparameters:**
```bash
python optimize.py --model catboost \
    --data-path ./data/mitbih_train.csv \
    --population-size 10 \
    --generations 50 \
    --cat-lr-min 0.001 \
    --cat-lr-max 0.5
```

### Training Stacking Ensemble

```bash
python train.py --mode ensemble \
    --data-path ./data/mitbih_train.csv \
    --n-splits 5 \
    --epochs 50 \
    --batch-size 32 \
    --cnn-layers 6 \
    --cnn-neurons 41 \
    --lstm-layers 3 \
    --lstm-neurons 162 \
    --cat-depth 9 \
    --cat-lr 0.247 \
    --output-dir ./outputs \
    --checkpoint-dir ./checkpoints
```

### Making Predictions

**Predict with CNN:**
```bash
python predict.py --model cnn \
    --model-path ./checkpoints/cnn_best.pth \
    --data-path ./data/test_data.csv \
    --cnn-layers 2 \
    --cnn-neurons 64 \
    --output-path ./predictions.csv \
    --save-probabilities
```

**Predict with Ensemble:**
```bash
python predict.py --model ensemble \
    --model-path ./checkpoints/ensemble \
    --data-path ./data/test_data.csv \
    --has-labels \
    --show-report \
    --save-confusion-matrix
```

## 🏗️ Model Architecture

### CNN Model
- Multiple 1D Convolutional layers
- ReLU activation + MaxPooling
- Fully connected output layer
- Configurable number of layers and filters

### LSTM Model
- Stacked LSTM layers
- Final hidden state for classification
- Configurable layers and hidden units

### CatBoost
- Gradient boosting with GPU support
- Auto class weights for imbalanced data
- TotalF1 evaluation metric

### Stacking Ensemble
1. Generate meta-features using 5-fold CV
2. Train CNN, LSTM, CatBoost on each fold
3. Collect prediction probabilities
4. Train Logistic Regression as meta-learner
5. Final prediction combines all models

## ⚙️ Configuration

All parameters can be configured via command-line arguments. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-path` | `./data/mitbih_train.csv` | Path to dataset |
| `--batch-size` | 32 | Training batch size |
| `--epochs` | 50 | Number of training epochs |
| `--learning-rate` | 0.001 | Learning rate |
| `--optimizer` | Adam | Optimizer (Adam, SGD, AdamW, etc.) |
| `--loss` | focal | Loss function (focal, cross_entropy) |
| `--device` | cuda | Device (cuda or cpu) |
| `--seed` | 42 | Random seed |

See `python train.py --help` for all options.


*Results may vary based on hyperparameters and random seed.*

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516)

---

⭐ Star this repository if you find it helpful!
