"""
Custom loss functions for ECG Heartbeat Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in classification.
    
    Focal Loss reduces the contribution of easy examples and focuses
    on hard examples during training. This is particularly useful for
    imbalanced datasets like ECG heartbeat classification.
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
        weight: torch.Tensor = None
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for the rare class (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
                   Higher gamma = more focus on hard examples
            reduction: Specifies the reduction to apply to output:
                       'none' | 'mean' | 'sum'
            weight: Optional class weights tensor
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            inputs: Predicted logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Focal loss value
        """
        # Standard cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        # Get prediction probability
        pt = torch.exp(-ce_loss)
        
        # Focal loss formula: -alpha * (1 - pt)^gamma * log(pt)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    
    Label smoothing is a regularization technique that softens the 
    target labels, reducing overconfidence in predictions.
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        """
        Initialize Label Smoothing Cross Entropy.
        
        Args:
            smoothing: Smoothing factor (0 = no smoothing)
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Label Smoothing Cross Entropy.
        
        Args:
            inputs: Predicted logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Loss value
        """
        n_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed target distribution
        targets_one_hot = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1
        )
        targets_smoothed = (1 - self.smoothing) * targets_one_hot + \
                          self.smoothing / n_classes
        
        # Compute loss
        loss = -(targets_smoothed * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with per-class weights for extreme class imbalance.
    """
    
    def __init__(
        self,
        class_weights: torch.Tensor,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        """
        Initialize Weighted Focal Loss.
        
        Args:
            class_weights: Tensor of weights for each class
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super(WeightedFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Weighted Focal Loss.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth labels
            
        Returns:
            Weighted focal loss value
        """
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.class_weights.to(inputs.device), 
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(
    loss_type: str = "focal",
    alpha: float = 1.0,
    gamma: float = 2.0,
    smoothing: float = 0.1,
    class_weights: torch.Tensor = None
) -> nn.Module:
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type: Type of loss ('focal', 'cross_entropy', 'label_smoothing', 'weighted_focal')
        alpha: Alpha parameter for focal loss
        gamma: Gamma parameter for focal loss
        smoothing: Smoothing factor for label smoothing
        class_weights: Optional class weights
        
    Returns:
        Loss function module
    """
    if loss_type == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    elif loss_type == "weighted_focal":
        if class_weights is None:
            raise ValueError("class_weights required for weighted_focal loss")
        return WeightedFocalLoss(class_weights, alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
