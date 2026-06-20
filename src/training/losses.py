"""
Custom loss functions for action recognition training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import warnings


def _apply_reduction(loss: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """Apply reduction operation to loss tensor."""
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'")


def _validate_input_tensor(tensor: torch.Tensor, name: str, expected_dim: Optional[int] = None) -> None:
    """Validate input tensor for NaN/Inf and shape."""
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf values")
    if expected_dim is not None and len(tensor.shape) != expected_dim:
        raise ValueError(f"{name} expected {expected_dim}D, got {len(tensor.shape)}D")


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing."""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        if not (0 <= smoothing < 1):
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (N, C)
            target: Ground truth labels (N,)
        
        Returns:
            Loss value
        """
        _validate_input_tensor(pred, "pred", expected_dim=2)
        _validate_input_tensor(target, "target", expected_dim=1)
        
        if target.max() >= self.num_classes or target.min() < 0:
            raise ValueError(f"target values out of range [0, {self.num_classes})")
        
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)
        smooth_dist = torch.full((pred.size(0), self.num_classes), smooth_value, 
                                 device=pred.device, dtype=pred.dtype)
        smooth_dist.scatter_(1, target.unsqueeze(1), confidence)
        
        log_probs = F.log_softmax(pred, dim=1)
        loss = -(smooth_dist * log_probs).sum(dim=1)
        
        return _apply_reduction(loss, self.reduction)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits (N, C)
            targets: Ground truth labels (N,)
        
        Returns:
            Focal loss value
        """
        _validate_input_tensor(inputs, "inputs", expected_dim=2)
        _validate_input_tensor(targets, "targets", expected_dim=1)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return _apply_reduction(focal_loss, self.reduction)


class TripletLoss(nn.Module):
    """Triplet loss for metric learning."""
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embeddings (N, D)
            positive: Positive embeddings (N, D)
            negative: Negative embeddings (N, D)
        
        Returns:
            Triplet loss value
        """
        _validate_input_tensor(anchor, "anchor", expected_dim=2)
        _validate_input_tensor(positive, "positive", expected_dim=2)
        _validate_input_tensor(negative, "negative", expected_dim=2)
        
        if anchor.shape != positive.shape or anchor.shape != negative.shape:
            raise ValueError(f"Shape mismatch: anchor {anchor.shape}, positive {positive.shape}")
        
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        
        return _apply_reduction(losses, self.reduction)


class MultiTaskLoss(nn.Module):
    """Multi-task loss with learnable uncertainty weights."""
    
    def __init__(self, tasks: List[str], loss_weights: Optional[Dict[str, float]] = None,
                 learnable: bool = True):
        """Initialize multi-task loss.
        
        Args:
            tasks: List of task names
            loss_weights: Fixed weights per task (if learnable=False)
            learnable: Whether to use learnable uncertainty weights
        """
        super().__init__()
        if not tasks or len(tasks) != len(set(tasks)):
            raise ValueError("tasks must be non-empty list with unique names")
        
        self.tasks = tasks
        self.learnable = learnable
        
        if learnable:
            # Learnable uncertainty weights (log variance)
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.tensor(0.0)) for task in tasks
            })
        else:
            # Fixed weights
            if loss_weights is None:
                loss_weights = {task: 1.0 for task in tasks}
            
            if set(loss_weights.keys()) != set(tasks):
                raise ValueError(f"loss_weights keys don't match tasks")
            
            for w in loss_weights.values():
                if w <= 0:
                    raise ValueError(f"All weights must be positive")
            
            self.register_buffer('weights', torch.tensor(
                [loss_weights[task] for task in tasks], dtype=torch.float32
            ))
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine task losses with optional learnable weights.
        
        Args:
            losses: Dictionary mapping task names to loss tensors
        
        Returns:
            Combined loss
        """
        total_loss = 0.0
        
        for i, task in enumerate(self.tasks):
            if task not in losses:
                warnings.warn(f"Task '{task}' not in losses")
                continue
            
            task_loss = losses[task]
            _validate_input_tensor(task_loss, f"losses['{task}']", expected_dim=0)
            
            if self.learnable:
                log_var = self.log_vars[task]
                weight = torch.exp(-log_var)
                total_loss += weight * task_loss + 0.1 * log_var  # Small regularization
            else:
                weight = self.weights[i]
                total_loss += weight * task_loss
        
        return total_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for representation learning."""
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Normalized feature embeddings (N, D)
            labels: Ground truth labels (N,)
        
        Returns:
            Contrastive loss value (InfoNCE)
        """
        _validate_input_tensor(features, "features", expected_dim=2)
        _validate_input_tensor(labels, "labels", expected_dim=1)
        
        batch_size = features.size(0)
        
        # Compute similarity matrix (N, N)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same class)
        labels_expanded = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels_expanded, labels_expanded.T).float()
        
        # Remove self-similarity (diagonal)
        mask_no_self = mask - torch.eye(batch_size, device=features.device, dtype=mask.dtype)
        
        # InfoNCE loss with numerical stability
        sim_max = similarity_matrix.max(dim=1, keepdim=True)[0]
        exp_sim = torch.exp(similarity_matrix - sim_max)
        exp_sim_sum = exp_sim.sum(dim=1, keepdim=True)
        
        # Sum of positive pair exponentials (excluding self)
        pos_exp_sum = (exp_sim * mask_no_self).sum(dim=1, keepdim=True)
        pos_exp_sum = torch.clamp(pos_exp_sum, min=1e-8)
        
        # InfoNCE loss
        loss = -torch.log(pos_exp_sum / exp_sim_sum).squeeze(1)
        
        return _apply_reduction(loss, self.reduction)


class PoseEstimationLoss(nn.Module):
    """Combined loss for pose estimation with multi-scale supervision."""
    
    def __init__(self, heatmap_weight: float = 1.0, offset_weight: float = 0.5,
                 intermediate_weight: float = 0.5):
        """Initialize pose estimation loss.
        
        Args:
            heatmap_weight: Weight for main heatmap loss
            offset_weight: Weight for offset loss
            intermediate_weight: Weight for intermediate supervision heatmaps
        """
        super().__init__()
        if heatmap_weight <= 0 or offset_weight < 0 or intermediate_weight < 0:
            raise ValueError("All weights must be non-negative")
        
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.intermediate_weight = intermediate_weight
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute pose estimation loss.
        
        Args:
            predictions: Model predictions containing heatmaps and offsets
            targets: Ground truth heatmaps and offsets
        
        Returns:
            Dictionary with per-task and total losses
        """
        losses = {}
        total_loss = 0.0
        
        # Main heatmap loss
        if 'heatmap' in predictions and 'heatmap' in targets:
            hm_loss = self.mse_loss(predictions['heatmap'], targets['heatmap'])
            losses['heatmap'] = hm_loss
            total_loss += self.heatmap_weight * hm_loss
        
        # Intermediate supervision heatmaps (heatmap1, heatmap2, ...)
        intermediate_idx = 1
        while f'heatmap{intermediate_idx}' in predictions:
            if 'heatmap' in targets:
                int_loss = self.mse_loss(predictions[f'heatmap{intermediate_idx}'], targets['heatmap'])
                losses[f'heatmap{intermediate_idx}'] = int_loss
                total_loss += self.intermediate_weight * int_loss
            intermediate_idx += 1
        
        # Offset loss
        if 'offsets' in predictions and 'offsets' in targets:
            offset_loss = self.l1_loss(predictions['offsets'], targets['offsets'])
            losses['offsets'] = offset_loss
            total_loss += self.offset_weight * offset_loss
        
        losses['total'] = total_loss
        
        return losses


class ActionRecognitionLoss(nn.Module):
    """Main loss function for action recognition with optional auxiliary tasks."""
    
    def __init__(self,
                 num_classes: int,
                 loss_type: str = 'cross_entropy',
                 label_smoothing: float = 0.1,
                 class_weights: Optional[torch.Tensor] = None,
                 use_focal_loss: bool = False,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0,
                 use_pose_loss: bool = False,
                 use_contrastive_loss: bool = False,
                 pose_weight: float = 0.1,
                 contrastive_weight: float = 0.1):
        """Initialize action recognition loss.
        
        Args:
            num_classes: Number of action classes
            label_smoothing: Label smoothing coefficient
            use_focal_loss: Whether to use focal loss
            use_pose_loss: Include pose estimation auxiliary loss
            use_contrastive_loss: Include contrastive auxiliary loss
            pose_weight: Weight for pose loss contribution
            contrastive_weight: Weight for contrastive loss contribution
        """
        super().__init__()
        
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.pose_weight = pose_weight if use_pose_loss else 0.0
        self.contrastive_weight = contrastive_weight if use_contrastive_loss else 0.0
        
        # Classification loss
        if use_focal_loss:
            self.classification_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif label_smoothing > 0:
            self.classification_loss = LabelSmoothingCrossEntropy(
                num_classes, smoothing=label_smoothing
            )
        else:
            self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        # Auxiliary losses (initialized once, not recreated in forward)
        self.pose_loss_fn = PoseEstimationLoss() if use_pose_loss else None
        self.contrastive_loss_fn = ContrastiveLoss() if use_contrastive_loss else None
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute action recognition loss.
        
        Args:
            predictions: Dict with 'logits', 'pose_output' (optional), 'features' (optional)
            targets: Dict with 'label', 'pose_target' (optional)
        
        Returns:
            Dictionary with per-task and total losses
        """
        if 'logits' not in predictions or 'label' not in targets:
            raise ValueError("predictions['logits'] and targets['label'] required")
        
        losses = {}
        
        # Classification loss (always required)
        classification_loss = self.classification_loss(
            predictions['logits'], targets['label']
        )
        losses['classification'] = classification_loss
        
        # Pose estimation losses (optional)
        if self.pose_loss_fn is not None and 'pose_output' in predictions and 'pose_target' in targets:
            pose_losses = self.pose_loss_fn(predictions['pose_output'], targets['pose_target'])
            losses.update({f'pose_{k}': v for k, v in pose_losses.items()})
        
        # Contrastive loss (optional)
        if self.contrastive_loss_fn is not None and 'features' in predictions:
            contrastive_loss = self.contrastive_loss_fn(
                predictions['features'], targets['label']
            )
            losses['contrastive'] = contrastive_loss
        
        # Compute total loss with weights
        total_loss = classification_loss
        
        if 'pose_total' in losses:
            total_loss = total_loss + self.pose_weight * losses['pose_total']
        
        if 'contrastive' in losses:
            total_loss = total_loss + self.contrastive_weight * losses['contrastive']
        
        losses['total'] = total_loss
        
        return losses


def get_loss_function(config: Dict[str, Any]) -> nn.Module:
    """Factory function to get loss functions."""
    loss_type = config.get('loss_type', 'cross_entropy')
    
    if loss_type == 'cross_entropy':
        return ActionRecognitionLoss(**config)
    elif loss_type == 'multi_task':
        return MultiTaskLoss(config['tasks'], config.get('loss_weights'))
    elif loss_type == 'triplet':
        return TripletLoss(margin=config.get('margin', 1.0))
    elif loss_type == 'contrastive':
        return ContrastiveLoss(temperature=config.get('temperature', 0.07))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
