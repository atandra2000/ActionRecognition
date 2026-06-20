"""
Evaluation metrics for action recognition
"""

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def _validate_tensor_input(tensor: torch.Tensor, name: str, expected_dim: Optional[int] = None) -> np.ndarray:
    """Validate and convert tensor to numpy array."""
    if not isinstance(tensor, (torch.Tensor, np.ndarray)):
        raise TypeError(f"{name} must be torch.Tensor or np.ndarray, got {type(tensor)}")
    
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    if expected_dim is not None and len(tensor.shape) != expected_dim:
        raise ValueError(f"{name} expected {expected_dim}D, got {len(tensor.shape)}D with shape {tensor.shape}")
    
    return tensor


def _validate_shape_match(*arrays: np.ndarray, dim: int = 0) -> None:
    """Validate that arrays match on specified dimension."""
    if not arrays:
        return
    
    first_size = arrays[0].shape[dim]
    for i, arr in enumerate(arrays[1:], 1):
        if arr.shape[dim] != first_size:
            raise ValueError(f"Shape mismatch on dimension {dim}: array 0 has {first_size}, array {i} has {arr.shape[dim]}")


class Metrics:
    """Compute and store various evaluation metrics with validation and caching."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        
        if len(self.class_names) != num_classes:
            raise ValueError(f"class_names length {len(self.class_names)} doesn't match num_classes {num_classes}")
        
        # Storage for predictions and targets
        self.predictions = []
        self.targets = []
        self.confidences = []  # Full softmax probabilities (N, C), not just class indices
        
        # Current metrics and cache state
        self.current_metrics = {}
        self._metrics_dirty = True  # Track if metrics need recomputation
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               confidences: Optional[torch.Tensor] = None) -> None:
        """Update metrics with new batch of predictions.
        
        Args:
            predictions: Predicted class indices (N,)
            targets: Ground truth class indices (N,)
            confidences: Softmax probabilities (N, C) for top-k accuracy computation
        """
        # Validate and convert
        predictions = _validate_tensor_input(predictions, "predictions", expected_dim=1)
        targets = _validate_tensor_input(targets, "targets", expected_dim=1)
        
        # Validate shapes match
        _validate_shape_match(predictions, targets, dim=0)
        
        # Validate prediction values
        if predictions.min() < 0 or predictions.max() >= self.num_classes:
            raise ValueError(f"predictions out of range [0, {self.num_classes}): got [{predictions.min()}, {predictions.max()}]")
        if targets.min() < 0 or targets.max() >= self.num_classes:
            raise ValueError(f"targets out of range [0, {self.num_classes}): got [{targets.min()}, {targets.max()}]")
        
        # Convert to int type
        predictions = predictions.astype(np.int64)
        targets = targets.astype(np.int64)
        
        # Validate and store confidences (if provided)
        if confidences is not None:
            confidences = _validate_tensor_input(confidences, "confidences", expected_dim=2)
            if confidences.shape[0] != len(predictions):
                raise ValueError(f"confidences batch size {confidences.shape[0]} doesn't match predictions {len(predictions)}")
            if confidences.shape[1] != self.num_classes:
                raise ValueError(f"confidences classes {confidences.shape[1]} doesn't match num_classes {self.num_classes}")
            self.confidences.extend(confidences)
        
        # Store
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        self._metrics_dirty = True  # Mark metrics as needing recomputation
    
    def compute(self) -> Dict[str, Union[float, np.ndarray]]:
        """Compute all metrics.
        
        Returns:
            Dictionary of computed metrics (cached until update() is called)
        """
        # Return cached metrics if not dirty
        if not self._metrics_dirty and self.current_metrics:
            return self.current_metrics
        
        if len(self.predictions) == 0:
            warnings.warn("No predictions to compute metrics from")
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Compute all P/R/F1 metrics in single call for efficiency
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0, labels=list(range(self.num_classes))
        )
        
        # Compute aggregated metrics from per-class results
        accuracy = accuracy_score(targets, predictions)
        precision_macro = precision_per_class.mean()
        recall_macro = recall_per_class.mean()
        f1_macro = f1_per_class.mean()
        
        # Weighted averages
        weights = support / support.sum()
        precision_weighted = (precision_per_class * weights).sum()
        recall_weighted = (recall_per_class * weights).sum()
        f1_weighted = (f1_per_class * weights).sum()
        
        # Top-k accuracy (only if confidences provided)
        top_k_metrics = {}
        if self.confidences:
            confidences = np.array(self.confidences)
            for k in [3, 5]:
                if k <= self.num_classes:
                    top_k_metrics[f'top{k}_accuracy'] = self._top_k_accuracy(targets, confidences, k=k)
        
        # Confusion matrix
        try:
            cm = confusion_matrix(targets, predictions, labels=list(range(self.num_classes)))
        except Exception as e:
            warnings.warn(f"Failed to compute confusion matrix: {e}")
            cm = np.zeros((self.num_classes, self.num_classes))
        
        # Store metrics
        self.current_metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support': support
        }
        self.current_metrics.update(top_k_metrics)
        self._metrics_dirty = False
        
        return self.current_metrics
    
    def _top_k_accuracy(self, targets: np.ndarray, confidences: np.ndarray, k: int = 5) -> float:
        """Compute top-k accuracy.
        
        Args:
            targets: Ground truth class indices (N,)
            confidences: Softmax probabilities (N, C) - NOT class indices
            k: Top-k value
        
        Returns:
            Top-k accuracy (0-1)
        """
        if confidences.shape[1] != self.num_classes:
            raise ValueError(f"confidences shape {confidences.shape} doesn't match num_classes {self.num_classes}")
        
        if k > self.num_classes:
            warnings.warn(f"k={k} > num_classes={self.num_classes}, using num_classes")
            k = self.num_classes
        
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        
        # Get top-k class indices for each sample
        top_k_indices = np.argsort(confidences, axis=1)[:, -k:]
        
        # Check if true label is in top-k
        correct = np.any(top_k_indices == targets[:, np.newaxis], axis=1)
        
        return float(correct.mean())
    
    def reset(self) -> None:
        """Reset all stored metrics."""
        self.predictions.clear()
        self.targets.clear()
        self.confidences.clear()
        self.current_metrics.clear()
        self._metrics_dirty = True
    
    def get_summary(self) -> str:
        """Get a summary of current metrics."""
        if not self.current_metrics:
            self.compute()
        
        metrics = self.current_metrics
        summary = f"""
        Accuracy: {metrics['accuracy']:.4f}
        Precision (Macro): {metrics['precision_macro']:.4f}
        Recall (Macro): {metrics['recall_macro']:.4f}
        F1-Score (Macro): {metrics['f1_macro']:.4f}
        Precision (Weighted): {metrics['precision_weighted']:.4f}
        Recall (Weighted): {metrics['recall_weighted']:.4f}
        F1-Score (Weighted): {metrics['f1_weighted']:.4f}
        Top-3 Accuracy: {metrics['top3_accuracy']:.4f}
        Top-5 Accuracy: {metrics['top5_accuracy']:.4f}
        """
        return summary
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None, 
                            normalize: bool = True, show: bool = True) -> None:
        """Plot confusion matrix.
        
        Args:
            save_path: Optional path to save figure
            normalize: Normalize by row (true label)
            show: Whether to display plot
        """
        if 'confusion_matrix' not in self.current_metrics:
            self.compute()
        
        cm = self.current_metrics['confusion_matrix'].copy()
        
        if normalize:
            # Avoid division by zero
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Prevent division by zero
            cm = cm.astype('float') / row_sums
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', xticklabels=self.class_names,
                   yticklabels=self.class_names, cbar_kws={'label': 'Count' if not normalize else 'Normalized Count'})
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_per_class_metrics(self, save_path: Optional[str] = None, show: bool = True,
                              top_n: Optional[int] = None) -> None:
        """Plot per-class precision, recall, and F1-score.
        
        Args:
            save_path: Optional path to save figure
            show: Whether to display plot
            top_n: Show only top-n classes by F1 score (useful for large num_classes)
        """
        if not self.current_metrics:
            self.compute()
        
        precision = self.current_metrics['precision_per_class']
        recall = self.current_metrics['recall_per_class']
        f1 = self.current_metrics['f1_per_class']
        
        # Optionally select top-n classes
        if top_n is not None and top_n < len(self.class_names):
            top_indices = np.argsort(f1)[-top_n:]
            precision = precision[top_indices]
            recall = recall[top_indices]
            f1 = f1[top_indices]
            class_names = [self.class_names[i] for i in top_indices]
        else:
            class_names = self.class_names
        
        x = np.arange(len(class_names))
        width = 0.25
        
        plt.figure(figsize=(max(15, len(class_names) * 0.5), 8))
        plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
        plt.bar(x, recall, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        title = 'Per-Class Metrics'
        if top_n is not None:
            title += f' (Top-{top_n})'
        plt.title(title)
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Display training progress."""
    
    def __init__(self, num_batches: int, meters: List[AverageMeter], 
                 prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class PerformanceTracker:
    """Track model performance during training with early stopping."""
    
    def __init__(self, num_classes: int, patience: int = 10, 
                 early_stopping_metric: str = 'val_loss'):
        """Initialize performance tracker.
        
        Args:
            num_classes: Number of classes
            patience: Number of epochs with no improvement to trigger early stopping
            early_stopping_metric: Metric to monitor ('val_loss' or 'val_accuracy')
        """
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if early_stopping_metric not in ['val_loss', 'val_accuracy']:
            raise ValueError(f"early_stopping_metric must be 'val_loss' or 'val_accuracy', got {early_stopping_metric}")
        
        self.num_classes = num_classes
        self.patience = patience
        self.early_stopping_metric = early_stopping_metric
        
        # Best metrics
        self.best_accuracy = 0.0
        self.best_f1 = 0.0
        self.best_loss = float('inf')
        
        # History
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        
        # Early stopping
        self.patience_counter = 0
        self.best_epoch = -1
    
    def update(self, train_loss: float, val_loss: float, 
               train_acc: float, val_acc: float) -> bool:
        """Update performance metrics.
        
        Args:
            train_loss: Training loss
            val_loss: Validation loss
            train_acc: Training accuracy
            val_acc: Validation accuracy
        
        Returns:
            True if improvement detected on early stopping metric, False otherwise
        """
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)
        
        # Update best metrics (independent of early stopping)
        if val_loss < self.best_loss:
            self.best_loss = val_loss
        
        if val_acc > self.best_accuracy:
            self.best_accuracy = val_acc
        
        # Early stopping check on single metric
        improved = False
        if self.early_stopping_metric == 'val_loss':
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                improved = True
                self.best_epoch = len(self.val_loss_history) - 1
            else:
                self.patience_counter += 1
        else:  # val_accuracy
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.patience_counter = 0
                improved = True
                self.best_epoch = len(self.val_acc_history) - 1
            else:
                self.patience_counter += 1
        
        return improved
    
    def should_stop(self) -> bool:
        """Check if training should stop (early stopping)."""
        return self.patience_counter >= self.patience
    
    def plot_training_curves(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """Plot training and validation curves.
        
        Args:
            save_path: Optional path to save figure
            show: Whether to display plot
        """
        if not self.train_loss_history:
            warnings.warn("No training history to plot")
            return
        
        epochs = range(1, len(self.train_loss_history) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(epochs, self.train_loss_history, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.val_loss_history, 'r-', label='Validation Loss', linewidth=2)
        if self.best_epoch >= 0:
            ax1.axvline(x=self.best_epoch + 1, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({self.best_epoch + 1})')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, self.train_acc_history, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.val_acc_history, 'r-', label='Validation Accuracy', linewidth=2)
        if self.best_epoch >= 0:
            ax2.axvline(x=self.best_epoch + 1, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({self.best_epoch + 1})')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


def accuracy(logits_or_probs: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[float]:
    """Compute top-k accuracy for the specified values of k.
    
    Args:
        logits_or_probs: Model logits or probabilities (N, C)
        target: Ground truth class indices (N,)
        topk: Tuple of k values to compute accuracy for
    
    Returns:
        List of top-k accuracies as percentages
    
    Raises:
        ValueError: If inputs have invalid shape or k is invalid
    """
    if len(logits_or_probs.shape) != 2:
        raise ValueError(f"logits_or_probs expected 2D, got shape {logits_or_probs.shape}")
    if len(target.shape) != 1:
        raise ValueError(f"target expected 1D, got shape {target.shape}")
    if logits_or_probs.shape[0] != target.shape[0]:
        raise ValueError(f"logits_or_probs and target batch sizes don't match: {logits_or_probs.shape[0]} vs {target.shape[0]}")
    
    num_classes = logits_or_probs.shape[1]
    maxk = max(topk)
    if maxk > num_classes:
        raise ValueError(f"k={maxk} > num_classes={num_classes}")
    if any(k < 1 for k in topk):
        raise ValueError(f"all k values must be >= 1, got {topk}")
    
    with torch.no_grad():
        batch_size = target.size(0)
        
        _, pred = logits_or_probs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def compute_video_level_accuracy(predictions: List[Union[np.ndarray, int]], 
                                targets: List[int], 
                                video_ids: List[str],
                                aggregation: str = 'mean') -> float:
    """Compute video-level accuracy (for multiple clips per video).
    
    Args:
        predictions: List of predictions (class indices or probability arrays)
        targets: List of ground truth class indices
        video_ids: List of video identifiers
        aggregation: Aggregation method ('mean', 'max', or 'majority')
    
    Returns:
        Video-level accuracy (0-1)
    
    Raises:
        ValueError: If inputs have mismatched lengths or invalid aggregation method
    """
    # Validate inputs
    if not (len(predictions) == len(targets) == len(video_ids)):
        raise ValueError(f"Length mismatch: predictions={len(predictions)}, "
                        f"targets={len(targets)}, video_ids={len(video_ids)}")
    
    if len(predictions) == 0:
        warnings.warn("Empty predictions list")
        return 0.0
    
    if aggregation not in ['mean', 'max', 'majority']:
        raise ValueError(f"aggregation must be 'mean', 'max', or 'majority', got {aggregation}")
    
    # Group predictions by video
    video_predictions = {}
    video_targets = {}
    
    for pred, target, vid_id in zip(predictions, targets, video_ids):
        if vid_id not in video_predictions:
            video_predictions[vid_id] = []
            video_targets[vid_id] = target
        
        # Validate target consistency
        if video_targets[vid_id] != target:
            raise ValueError(f"Conflicting targets for video {vid_id}: {video_targets[vid_id]} vs {target}")
        
        video_predictions[vid_id].append(pred)
    
    # Aggregate predictions per video
    video_level_predictions = []
    video_level_targets = []
    
    for vid_id in video_predictions:
        clip_preds = video_predictions[vid_id]
        
        # Convert to consistent format
        if isinstance(clip_preds[0], np.ndarray):
            # Probability arrays
            stacked = np.array(clip_preds)
            if aggregation == 'mean':
                avg_pred = stacked.mean(axis=0)
            elif aggregation == 'max':
                avg_pred = stacked.max(axis=0)
            final_pred = np.argmax(avg_pred)
        else:
            # Class indices
            if aggregation == 'majority':
                final_pred = int(np.bincount(clip_preds).argmax())
            else:
                raise ValueError(
                    f"aggregation='{aggregation}' is invalid for class-index predictions. "
                    f"Use 'majority' instead."
                )
        
        video_level_predictions.append(final_pred)
        video_level_targets.append(video_targets[vid_id])
    
    return float(accuracy_score(video_level_targets, video_level_predictions))
