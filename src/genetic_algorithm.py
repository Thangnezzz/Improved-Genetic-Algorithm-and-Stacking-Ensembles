"""
Genetic Algorithm module for hyperparameter optimization.
Supports optimization of LSTM, CNN, and CatBoost models.
"""

import random
import numpy as np
from itertools import product
from typing import List, Tuple, Callable, Optional, Dict, Any
from abc import ABC, abstractmethod
import torch
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, f1_score

from .models import LSTMModel, CNNModel, create_catboost_model, get_optimizer
from .losses import FocalLoss
from .engine import Trainer, Evaluator


class GeneticOptimizer(ABC):
    """
    Abstract base class for Genetic Algorithm optimization.
    """
    
    def __init__(
        self,
        population_size: int = 10,
        num_generations: int = 50,
        mutation_rate_base: float = 0.1,
        early_stop_count: int = 15,
        random_state: int = 42,
        verbose: int = 1
    ):
        """
        Initialize genetic optimizer.
        
        Args:
            population_size: Size of population
            num_generations: Maximum number of generations
            mutation_rate_base: Base mutation rate
            early_stop_count: Stop if same parents selected this many times
            random_state: Random seed
            verbose: Verbosity level
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate_base = mutation_rate_base
        self.early_stop_count = early_stop_count
        self.random_state = random_state
        self.verbose = verbose
        
        random.seed(random_state)
        np.random.seed(random_state)
        
    @abstractmethod
    def create_individual(self) -> List:
        """Create a single individual (set of hyperparameters)."""
        pass
    
    @abstractmethod
    def evaluate_fitness(self, individual: List, train_data, val_data) -> float:
        """Evaluate fitness of an individual."""
        pass
    
    @abstractmethod
    def mutate(self, child: List, mutation_strength: int) -> List:
        """Mutate an individual."""
        pass
    
    def create_population(self) -> List[List]:
        """Create initial population."""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def crossover(self, parent1: List, parent2: List) -> List[List]:
        """
        Perform crossover between two parents.
        Generates all possible combinations except exact copies of parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            List of offspring
        """
        combinations = []
        num_genes = len(parent1)
        
        for selection in product([0, 1], repeat=num_genes):
            child = [parent1[i] if selection[i] == 0 else parent2[i] for i in range(num_genes)]
            if child != list(parent1) and child != list(parent2):
                combinations.append(child)
        
        # Remove duplicates
        unique_combinations = [list(item) for item in set(tuple(row) for row in combinations)]
        return unique_combinations
    
    def mutate_population(self, children: List[List], mutation_strength: int) -> List[List]:
        """Mutate all children in population."""
        return [self.mutate(child.copy(), mutation_strength) for child in children]
    
    def optimize(
        self,
        train_data,
        val_data,
        initial_population: Optional[List[List]] = None
    ) -> Tuple[List[float], List]:
        """
        Run genetic algorithm optimization.
        
        Args:
            train_data: Training data
            val_data: Validation data
            initial_population: Optional initial population
            
        Returns:
            Tuple of (score_history, best_parents)
        """
        population = initial_population if initial_population else self.create_population()
        score_history = []
        previous_parents = None
        stagnation_count = 1
        
        for generation in range(self.num_generations):
            if self.verbose >= 1:
                print(f'\n=== Generation {generation + 1}/{self.num_generations} ===')
            
            # Evaluate fitness for all individuals
            fitness_scores = []
            for idx, individual in enumerate(population):
                if self.verbose >= 2:
                    print(f'Evaluating individual {idx + 1}/{len(population)}: {individual}')
                score = self.evaluate_fitness(individual, train_data, val_data)
                fitness_scores.append(score)
            
            if self.verbose >= 1:
                print(f'Fitness scores: {[round(s, 4) for s in fitness_scores]}')
            
            # Record best score
            best_score = max(fitness_scores)
            score_history.append(best_score)
            
            # Select top 2 parents
            sorted_population = [x for _, x in sorted(
                zip(fitness_scores, population), reverse=True
            )]
            parent1, parent2 = sorted_population[0], sorted_population[1]
            
            if self.verbose >= 1:
                print(f'Best score: {best_score:.4f}')
                print(f'Best individual: {parent1}')
            
            # Check for stagnation
            if previous_parents == (tuple(parent1), tuple(parent2)):
                stagnation_count += 1
            else:
                stagnation_count = 1
            previous_parents = (tuple(parent1), tuple(parent2))
            
            if self.verbose >= 2:
                print(f'Stagnation count: {stagnation_count}')
            
            # Early stopping if stuck
            if stagnation_count >= self.early_stop_count:
                if self.verbose >= 1:
                    print(f'\nEarly stopping: Same parents for {self.early_stop_count} generations')
                break
            
            # Create next generation
            children = self.crossover(parent1, parent2)
            
            # Ensure minimum population size
            min_children = self.population_size - 2
            if len(children) < min_children:
                additional = [self.create_individual() for _ in range(min_children - len(children))]
                children.extend(additional)
            
            # Mutate children
            mutated_children = self.mutate_population(children, stagnation_count)
            
            # New population = parents + mutated children
            population = [parent1, parent2] + mutated_children[:self.population_size - 2]
        
        return score_history, [parent1, parent2]


class CatBoostGAOptimizer(GeneticOptimizer):
    """
    Genetic Algorithm optimizer for CatBoost hyperparameters.
    """
    
    def __init__(
        self,
        lr_range: Tuple[float, float] = (1e-3, 0.5),
        depth_range: Tuple[int, int] = (4, 10),
        l2_range: Tuple[int, int] = (3, 15),
        border_range: Tuple[int, int] = (30, 600),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lr_range = lr_range
        self.depth_range = depth_range
        self.l2_range = l2_range
        self.border_range = border_range
        
    def create_individual(self) -> List:
        """Create individual: [learning_rate, depth, l2_leaf_reg, border_count]"""
        return [
            random.uniform(*self.lr_range),
            np.random.randint(*self.depth_range),
            np.random.randint(*self.l2_range),
            np.random.randint(*self.border_range)
        ]
    
    def evaluate_fitness(self, individual: List, train_data, val_data) -> float:
        """
        Evaluate CatBoost model with given hyperparameters.
        
        Args:
            individual: [learning_rate, depth, l2_leaf_reg, border_count]
            train_data: (x_train, y_train)
            val_data: (x_val, y_val)
            
        Returns:
            Balanced accuracy score
        """
        learning_rate, depth, l2_leaf_reg, border_count = individual
        x_train, y_train = train_data
        x_val, y_val = val_data
        
        model = create_catboost_model(
            learning_rate=learning_rate,
            depth=int(depth),
            l2_leaf_reg=int(l2_leaf_reg),
            border_count=int(border_count)
        )
        
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)
        
        return balanced_accuracy_score(y_val, predictions)
    
    def mutate(self, child: List, mutation_strength: int) -> List:
        """Mutate CatBoost hyperparameters."""
        mutation_rate = min(self.mutation_rate_base * mutation_strength, 1.0)
        
        if random.random() < mutation_rate:
            if mutation_strength > 5:
                # Strong mutation: change all parameters
                child = self.create_individual()
            else:
                # Selective mutation
                mut_idx = random.random()
                if mut_idx < 0.2:
                    child[0] = random.uniform(*self.lr_range)
                elif mut_idx < 0.4:
                    child[1] = np.random.randint(*self.depth_range)
                elif mut_idx < 0.6:
                    child[2] = np.random.randint(*self.l2_range)
                elif mut_idx < 0.8:
                    child[3] = np.random.randint(*self.border_range)
                else:
                    child = self.create_individual()
        
        return child


class LSTMGAOptimizer(GeneticOptimizer):
    """
    Genetic Algorithm optimizer for LSTM hyperparameters.
    """
    
    def __init__(
        self,
        hidden_layers_range: Tuple[int, int] = (1, 5),
        neurons_range: Tuple[int, int] = (32, 512),
        lr_range: Tuple[float, float] = (1e-3, 1e-1),
        optimizer_options: List[str] = None,
        train_epochs: int = 3,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_layers_range = hidden_layers_range
        self.neurons_range = neurons_range
        self.lr_range = lr_range
        self.optimizer_options = optimizer_options or [
            "Adam", "SGD", "AdamW", "RMSprop", "Adadelta", "Adamax", "Nadam"
        ]
        self.train_epochs = train_epochs
        self.device = device
        
    def create_individual(self) -> List:
        """Create individual: [hidden_layers, neurons, learning_rate, optimizer]"""
        return [
            np.random.randint(*self.hidden_layers_range),
            np.random.randint(*self.neurons_range),
            random.uniform(*self.lr_range),
            random.choice(self.optimizer_options)
        ]
    
    def evaluate_fitness(self, individual: List, train_data, val_data) -> float:
        """
        Evaluate LSTM model with given hyperparameters.
        
        Args:
            individual: [hidden_layers, neurons, learning_rate, optimizer_name]
            train_data: train_loader
            val_data: (val_loader, y_val)
            
        Returns:
            Balanced accuracy score
        """
        hidden_layers, neurons, learning_rate, optimizer_name = individual
        train_loader = train_data
        val_loader, y_val = val_data
        
        model = LSTMModel(
            hidden_layers=int(hidden_layers),
            neurons=int(neurons),
            learning_rate=learning_rate,
            optimizer_name=optimizer_name
        )
        
        optimizer = get_optimizer(model, optimizer_name, learning_rate)
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
            num_epochs=self.train_epochs,
            early_stopping=False
        )
        
        evaluator = Evaluator(model=model, device=self.device)
        results = evaluator.evaluate(val_loader, show_report=False)
        
        return results["balanced_accuracy"]
    
    def mutate(self, child: List, mutation_strength: int) -> List:
        """Mutate LSTM hyperparameters."""
        mutation_rate = min(self.mutation_rate_base * mutation_strength, 1.0)
        
        if random.random() < mutation_rate:
            if mutation_strength > 5:
                child = self.create_individual()
            else:
                mut_idx = random.random()
                if mut_idx < 0.2:
                    child[0] = np.random.randint(*self.hidden_layers_range)
                elif mut_idx < 0.4:
                    child[1] = np.random.randint(*self.neurons_range)
                elif mut_idx < 0.6:
                    child[2] = random.uniform(*self.lr_range)
                elif mut_idx < 0.8:
                    child[3] = random.choice(self.optimizer_options)
                else:
                    child = self.create_individual()
        
        return child


class CNNGAOptimizer(GeneticOptimizer):
    """
    Genetic Algorithm optimizer for CNN hyperparameters.
    """
    
    def __init__(
        self,
        conv_layers_range: Tuple[int, int] = (1, 7),
        neurons_range: Tuple[int, int] = (10, 512),
        lr_range: Tuple[float, float] = (1e-4, 1e-1),
        optimizer_options: List[str] = None,
        train_epochs: int = 2,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.conv_layers_range = conv_layers_range
        self.neurons_range = neurons_range
        self.lr_range = lr_range
        self.optimizer_options = optimizer_options or [
            "Adam", "SGD", "AdamW", "RMSprop", "Adadelta", "Adamax", "Nadam"
        ]
        self.train_epochs = train_epochs
        self.device = device
        
    def create_individual(self) -> List:
        """Create individual: [conv_layers, neurons, learning_rate, optimizer]"""
        return [
            np.random.randint(*self.conv_layers_range),
            np.random.randint(*self.neurons_range),
            random.uniform(*self.lr_range),
            random.choice(self.optimizer_options)
        ]
    
    def evaluate_fitness(self, individual: List, train_data, val_data) -> float:
        """
        Evaluate CNN model with given hyperparameters.
        """
        conv_layers, neurons, learning_rate, optimizer_name = individual
        train_loader = train_data
        val_loader, y_val = val_data
        
        model = CNNModel(
            conv_layers=int(conv_layers),
            neurons=int(neurons),
            learning_rate=learning_rate,
            optimizer_name=optimizer_name
        )
        
        optimizer = get_optimizer(model, optimizer_name, learning_rate)
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
            num_epochs=self.train_epochs,
            early_stopping=False
        )
        
        evaluator = Evaluator(model=model, device=self.device)
        results = evaluator.evaluate(val_loader, show_report=False)
        
        return results["balanced_accuracy"]
    
    def mutate(self, child: List, mutation_strength: int) -> List:
        """Mutate CNN hyperparameters."""
        mutation_rate = min(self.mutation_rate_base * mutation_strength, 1.0)
        
        if random.random() < mutation_rate:
            if mutation_strength > 5:
                child = self.create_individual()
            else:
                mut_idx = random.random()
                if mut_idx < 0.2:
                    child[0] = np.random.randint(*self.conv_layers_range)
                elif mut_idx < 0.4:
                    child[1] = np.random.randint(*self.neurons_range)
                elif mut_idx < 0.6:
                    child[2] = random.uniform(*self.lr_range)
                elif mut_idx < 0.8:
                    child[3] = random.choice(self.optimizer_options)
                else:
                    child = self.create_individual()
        
        return child


def save_ga_results(
    scores: List[float],
    best_params: List,
    model_name: str,
    output_dir: str = "./"
):
    """
    Save GA optimization results to files.
    
    Args:
        scores: List of best scores per generation
        best_params: Best hyperparameters found
        model_name: Name of the model (e.g., 'catboost', 'lstm', 'cnn')
        output_dir: Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, f"score_{model_name}.txt"), "w") as f:
        for score in scores:
            f.write(f"{score}\n")
    
    with open(os.path.join(output_dir, f"best_params_{model_name}.txt"), "w") as f:
        for params in best_params:
            f.write(f"{params}\n")
    
    print(f"Results saved to {output_dir}")
