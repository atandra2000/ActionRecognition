"""
Main training script for action recognition
"""

import os
import time
import argparse
from typing import Dict, Optional, Tuple, Any
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb

from src.models.action_recognition import create_action_model
from src.data.datasets import create_dataset, create_dataloader
from src.training.losses import get_loss_function, ActionRecognitionLoss
from src.training.metrics import Metrics, PerformanceTracker, AverageMeter, accuracy, ProgressMeter
from src.utils.config import load_config, Config
from src.utils.logger import setup_logger


# Default scheduler parameters
DEFAULT_SCHEDULER_PARAMS = {
    'step': {'step_size': 30, 'gamma': 0.1},
    'plateau': {'factor': 0.5, 'patience': 5}
}


def _get_scheduler_param(config, scheduler_type: str, param_name: str, default_value: Any):
    """Get scheduler parameter from config with fallback to defaults.
    
    Args:
        config: Training config
        scheduler_type: Type of scheduler (step, plateau, cosine)
        param_name: Parameter name (step_size, gamma, factor, patience)
        default_value: Default value if not found
        
    Returns:
        Parameter value from config or default
    """
    # Try to get from config nested scheduler settings
    if hasattr(config.training, f'scheduler_{param_name}'):
        return getattr(config.training, f'scheduler_{param_name}')
    
    # Try to get from default params
    if scheduler_type in DEFAULT_SCHEDULER_PARAMS:
        if param_name in DEFAULT_SCHEDULER_PARAMS[scheduler_type]:
            return DEFAULT_SCHEDULER_PARAMS[scheduler_type][param_name]
    
    return default_value


class Trainer:
    """Main trainer class."""
    
    # Helper static methods to reduce redundancy
    @staticmethod
    def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Tuple[Any, torch.Tensor]:
        """Move batch data and targets to device.
        
        Args:
            batch: Dictionary with 'data' and 'label' keys
            device: Target device
            
        Returns:
            Tuple of (data, targets)
        """
        if isinstance(batch['data'], dict):
            data = {k: v.to(device) for k, v in batch['data'].items()}
        else:
            data = batch['data'].to(device)
        
        targets = batch['label'].to(device)
        return data, targets
    
    @staticmethod
    def _extract_predictions(outputs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract predictions and confidences from model outputs.
        
        Args:
            outputs: Model output (tensor or dict)
            
        Returns:
            Tuple of (predictions, confidences)
        """
        if isinstance(outputs, dict):
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=1)
            confidences = outputs.get('probabilities', F.softmax(logits, dim=1))
        else:
            predictions = torch.argmax(outputs, dim=1)
            confidences = F.softmax(outputs, dim=1)
        
        return predictions, confidences
    
    @staticmethod
    def _compute_loss(loss_fn: nn.Module, outputs: Any, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss from outputs and targets.
        
        Args:
            loss_fn: Loss function
            outputs: Model output (tensor or dict)
            targets: Target labels
            
        Returns:
            Loss tensor
        """
        if isinstance(outputs, dict):
            logits = outputs.get('logits')
            if logits is None:
                raise ValueError("Model output dict missing 'logits' key")
            if isinstance(loss_fn, ActionRecognitionLoss):
                loss_inputs = {'logits': logits}
                loss_targets = {'label': targets}
                losses_dict = loss_fn(loss_inputs, loss_targets)
                return losses_dict['total']
            else:
                return loss_fn(logits, targets)
        else:
            return loss_fn(outputs, targets)
    
    @staticmethod
    def _is_main_process() -> bool:
        """Check if current process is main process (rank 0).
        
        Returns:
            True if main process, False otherwise
        """
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # ── A100 / Ampere GPU optimizations ──────────────────────────────────
        if torch.cuda.is_available():
            # TF32: 8x matmul throughput on Ampere tensor cores (no accuracy loss)
            if getattr(config.training, 'use_tf32', False):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # cuDNN auto-tuner: finds fastest conv algorithm for current shapes
            if getattr(config.training, 'cudnn_benchmark', False):
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            
            # Flash Attention via SDPA backend (2-4x faster MHA on A100)
            if getattr(config.training, 'use_flash_attention', False):
                if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                    torch.backends.cuda.enable_flash_sdp(True)
                if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # Setup logging
        self.logger = setup_logger(
            name=config.project_name,
            log_file=f"{config.output_dir}/logs/training.log",
            level=config.log_level
        )
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(f"{config.output_dir}/logs", exist_ok=True)
        os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{config.output_dir}/visualizations", exist_ok=True)
        
        # Initialize distributed training if needed
        self.distributed = config.distributed and torch.cuda.is_available()
        if self.distributed:
            self._setup_distributed()
        
        # Create model
        self.model = self._create_model()
        
        # Create data loaders
        self.train_loader = self._create_dataloader('train')
        self.val_loader = self._create_dataloader('val')
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Create loss function
        self.loss_fn = self._create_loss_function()
        
        # Metrics and tracking
        self.metrics = Metrics(config.model.num_classes)
        self.tracker = PerformanceTracker(
            config.model.num_classes,
            patience=config.training.early_stopping_patience if hasattr(config.training, 'early_stopping_patience') else 10
        )
        
        # Logging
        if config.tensorboard_enabled:
            self.writer = SummaryWriter(f"{config.output_dir}/tensorboard")
        
        if config.wandb_enabled:
            wandb.init(
                project=config.project_name,
                name=config.experiment_name,
                config=config.__dict__
            )
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_acc = 0.0
        self.best_loss = float('inf')
        
        # Mixed precision training
        self.use_amp = config.training.use_amp if hasattr(config.training, 'use_amp') else False
        self.amp_dtype = getattr(config.training, 'amp_dtype', 'float16')
        if self.use_amp:
            if self.amp_dtype == 'bfloat16':
                self.amp_dtype_obj = torch.bfloat16
            else:
                self.amp_dtype_obj = torch.float16
            self.scaler = GradScaler(enabled=(self.amp_dtype == 'float16'))
        else:
            self.amp_dtype_obj = torch.float32
            self.scaler = None
        
        # Gradient accumulation
        self.accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
        
        # EMA (Exponential Moving Average)
        self.use_ema = getattr(config.training, 'use_ema', True)
        self.ema = None
        if self.use_ema:
            self.ema_decay = getattr(config.training, 'ema_decay', 0.999)
            self.ema = ModelEMA(self.model, decay=self.ema_decay)
            if self._is_main_process():
                self.logger.info(f"EMA enabled with decay={self.ema_decay}")
        
        # Checkpoint management
        self.keep_best_k = getattr(config.training, 'keep_best_k_checkpoints', 5)
        self.checkpoint_dir = f"{config.output_dir}/checkpoints"
    
    def _setup_distributed(self):
        """Setup distributed training."""
        dist.init_process_group(backend='nccl')
        self.local_rank = dist.get_rank()
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
    
    def _create_model(self) -> nn.Module:
        """Create and initialize model."""
        model_config = {
            'num_classes': self.config.model.num_classes,
            'num_keypoints': self.config.model.num_keypoints,
            'pose_dim': self.config.model.pose_dim,
            'use_attention': getattr(self.config.model, 'use_attention', True),
            'multimodal': getattr(self.config.model, 'multimodal', True),
            'streams': getattr(self.config.model, 'streams', ['joint']),
            'model_type': getattr(self.config.model, 'model_type', 'stgcn'),
        }
        
        model = create_action_model(model_config)
        
        # torch.compile: fuse kernels for 30-50% faster forward/backward (PyTorch 2.0+)
        use_compile = getattr(self.config.training, 'use_torch_compile', False)
        if use_compile and hasattr(torch, 'compile'):
            compile_mode = getattr(self.config.training, 'compile_mode', 'default')
            if self._is_main_process():
                self.logger.info(f"Applying torch.compile with mode={compile_mode}")
            model = torch.compile(model, mode=compile_mode)
        
        # Move to device
        model = model.to(self.device)
        
        # Wrap in DDP if distributed
        if self.distributed:
            model = DDP(model, device_ids=[self.local_rank])
        
        return model
    
    def _create_dataloader(self, split: str) -> DataLoader:
        """Create data loader for train/val split."""
        dataset = create_dataset(self.config.data, split)
        return create_dataloader(dataset, self.config.data, split)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with optional fused kernel support (A100)."""
        optimizer_type = self.config.training.optimizer.lower()
        use_fused = getattr(self.config.training, 'use_fused_optimizer', False)
        
        if optimizer_type == 'adam':
            fused = use_fused and torch.cuda.is_available()
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                fused=fused
            )
        elif optimizer_type == 'adamw':
            fused = use_fused and torch.cuda.is_available()
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(self.config.training.beta1, self.config.training.beta2),
                fused=fused
            )
        elif optimizer_type == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler with configurable parameters."""
        scheduler_type = self.config.training.scheduler.lower()
        
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.min_lr
            )
        elif scheduler_type == 'step':
            step_size = _get_scheduler_param(self.config, 'step', 'step_size', 30)
            gamma = _get_scheduler_param(self.config, 'step', 'gamma', 0.1)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type == 'plateau':
            factor = _get_scheduler_param(self.config, 'plateau', 'factor', 0.5)
            patience = _get_scheduler_param(self.config, 'plateau', 'patience', 5)
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=factor,
                patience=patience,
                min_lr=self.config.training.min_lr
            )
        else:
            return None
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function with all config parameters."""
        loss_config = {
            'num_classes': self.config.model.num_classes,
            'loss_type': self.config.training.loss_type,
            'label_smoothing': self.config.training.label_smoothing,
            'use_focal_loss': getattr(self.config.training, 'use_focal_loss', False),
            'focal_alpha': getattr(self.config.training, 'focal_alpha', 1.0),
            'focal_gamma': getattr(self.config.training, 'focal_gamma', 2.0),
            'use_pose_loss': getattr(self.config.training, 'use_pose_loss', False),
            'use_contrastive_loss': getattr(self.config.training, 'use_contrastive_loss', False),
            'pose_weight': getattr(self.config.training, 'pose_weight', 0.1),
            'contrastive_weight': getattr(self.config.training, 'contrastive_weight', 0.1),
        }
        return get_loss_function(loss_config)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with mixed precision and gradient accumulation support."""
        self.model.train()
        
        # Meters
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix=f"Epoch: [{self.epoch}]"
        )
        
        end = time.time()
        accumulation_counter = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            
            # Move data to device
            data, targets = self._move_batch_to_device(batch, self.device)
            
            # Forward pass with optional mixed precision
            with autocast(device_type='cuda', dtype=self.amp_dtype_obj, enabled=self.use_amp):
                outputs = self.model(data)
                loss = self._compute_loss(self.loss_fn, outputs, targets)
                
                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulation_counter += 1
            
            # Optimizer step after accumulation
            if accumulation_counter % self.accumulation_steps == 0:
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Update EMA after each optimizer step
                if self.ema is not None:
                    self.ema.update(self.model.module if self.distributed else self.model)
                
                self.optimizer.zero_grad()
                accumulation_counter = 0
            
            # Extract predictions for metrics
            with torch.no_grad():
                predictions, confidences = self._extract_predictions(outputs)
            
            # Update metrics
            losses.update(loss.item() * self.accumulation_steps, targets.size(0))
            acc1, acc5 = accuracy(confidences, targets, topk=(1, 5))
            top1.update(acc1[0], targets.size(0))
            top5.update(acc5[0], targets.size(0))
            
            # Update batch time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Log progress (only on main process)
            if batch_idx % 100 == 0 and self._is_main_process():
                progress.display(batch_idx)
                
                # Log to tensorboard
                if self.config.tensorboard_enabled:
                    self.writer.add_scalar('train/loss', loss.item() * self.accumulation_steps, self.step)
                    self.writer.add_scalar('train/accuracy_top1', acc1[0], self.step)
                    self.writer.add_scalar('train/accuracy_top5', acc5[0], self.step)
                    self.writer.add_scalar('train/learning_rate', 
                                         self.optimizer.param_groups[0]['lr'], self.step)
            
            self.step += 1
        
        if accumulation_counter > 0:
            if self.config.training.gradient_clip > 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.model.module if self.distributed else self.model)
            self.optimizer.zero_grad()
        
        return {
            'loss': losses.avg,
            'accuracy_top1': top1.avg,
            'accuracy_top5': top5.avg
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model (uses EMA shadow weights if enabled)."""
        # Apply EMA shadow weights before validation
        ema_was_applied = False
        if self.ema is not None:
            self.ema.apply_shadow(self.model.module if self.distributed else self.model)
            ema_was_applied = True
        
        self.model.eval()
        
        # Meters
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        
        # Reset metrics
        self.metrics.reset()
        
        with torch.no_grad():
            end = time.time()
            
            for batch in self.val_loader:
                # Move data to device
                data, targets = self._move_batch_to_device(batch, self.device)
                
                # Forward pass
                with autocast(device_type='cuda', dtype=self.amp_dtype_obj, enabled=self.use_amp):
                    outputs = self.model(data)
                    loss = self._compute_loss(self.loss_fn, outputs, targets)
                
                # Extract predictions
                predictions, confidences = self._extract_predictions(outputs)
                
                # Update metrics
                losses.update(loss.item(), targets.size(0))
                acc1, acc5 = accuracy(confidences, targets, topk=(1, 5))
                top1.update(acc1[0], targets.size(0))
                top5.update(acc5[0], targets.size(0))
                
                # Update batch time
                batch_time.update(time.time() - end)
                end = time.time()
                
                # Update metrics tracker
                self.metrics.update(predictions.cpu(), targets.cpu(), confidences.cpu())
        
        # Restore model weights if EMA was applied
        if ema_was_applied:
            self.ema.restore(self.model.module if self.distributed else self.model)
        
        # Compute final metrics
        final_metrics = self.metrics.compute()
        
        return {
            'loss': losses.avg,
            'accuracy_top1': top1.avg,
            'accuracy_top5': top5.avg,
            'accuracy': final_metrics.get('accuracy', 0.0),
            'f1_macro': final_metrics.get('f1_macro', 0.0)
        }
    
    def _cleanup_old_checkpoints(self) -> None:
        """Keep only the best K checkpoints to save disk space."""
        if not self._is_main_process():
            return
        
        # Find all epoch checkpoints (not _best or final)
        checkpoint_files = glob.glob(f"{self.checkpoint_dir}/epoch_*.pth")
        checkpoint_files = [f for f in checkpoint_files if '_best' not in f and 'final' not in f]
        
        if len(checkpoint_files) > self.keep_best_k:
            # Sort by modification time and keep only the most recent K
            checkpoint_files.sort(key=os.path.getmtime)
            for old_checkpoint in checkpoint_files[:-self.keep_best_k]:
                try:
                    os.remove(old_checkpoint)
                    self.logger.debug(f"Removed old checkpoint: {old_checkpoint}")
                except OSError as e:
                    self.logger.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")
    
    def save_checkpoint(self, filepath: str, is_best: bool = False) -> None:
        """Save model checkpoint (only on main process)."""
        if not self._is_main_process():
            return
        
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'ema_state_dict': self.ema.state_dict() if self.ema else None,
            'best_acc': self.best_acc,
            'best_loss': self.best_loss,
            'config': self.config.__dict__
        }
        
        try:
            torch.save(checkpoint, filepath)
            if is_best:
                best_path = filepath.replace('.pth', '_best.pth')
                torch.save(checkpoint, best_path)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {filepath}: {e}")
            raise
    
    def load_checkpoint(self, filepath: str, strict: bool = True) -> None:
        """Load model checkpoint with error handling and partial loading support.
        
        Args:
            filepath: Path to checkpoint file
            strict: Whether to strictly match checkpoint keys
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails critically
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint file {filepath}: {e}")
        
        try:
            # Load model with optional strict mode
            incompatible_keys = self.model.load_state_dict(
                checkpoint['model_state_dict'], 
                strict=strict
            )
            if incompatible_keys.missing_keys:
                self.logger.warning(f"Missing keys in model: {incompatible_keys.missing_keys[:5]}...")
            if incompatible_keys.unexpected_keys:
                self.logger.warning(f"Unexpected keys in model: {incompatible_keys.unexpected_keys[:5]}...")
        except RuntimeError as e:
            if strict:
                raise RuntimeError(f"Failed to load model state dict (strict={strict}): {e}")
            else:
                self.logger.warning(f"Loaded model with incompatible keys: {e}")
        
        try:
            # Load optimizer and scheduler (optional)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except Exception as e:
            self.logger.warning(f"Failed to load optimizer/scheduler state: {e}")
        
        if self.ema and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        # Load training state
        self.epoch = checkpoint.get('epoch', 0)
        self.step = checkpoint.get('step', 0)
        self.best_acc = checkpoint.get('best_acc', 0.0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        # Validate checkpoint compatibility
        if 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            checkpoint_num_classes = (
                checkpoint_config.get('model', {}).get('num_classes')
                if isinstance(checkpoint_config, dict) else None
            )
            if checkpoint_num_classes is not None and checkpoint_num_classes != self.config.model.num_classes:
                raise ValueError(
                    f"Checkpoint num_classes ({checkpoint_num_classes}) "
                    f"!= config num_classes ({self.config.model.num_classes})"
                )
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict) -> None:
        """Centralized metric logging.
        
        Args:
            epoch: Current epoch number
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
        """
        # Log to file (only main process)
        if self._is_main_process():
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy_top1']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy_top1']:.2f}%"
            )
        
        # Log to tensorboard
        if self.config.tensorboard_enabled and self._is_main_process():
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/train_accuracy', train_metrics['accuracy_top1'], epoch)
            self.writer.add_scalar('epoch/val_accuracy', val_metrics['accuracy_top1'], epoch)
        
        # Log to wandb
        if self.config.wandb_enabled and self._is_main_process():
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_accuracy': train_metrics['accuracy_top1'],
                'val_accuracy': val_metrics['accuracy_top1']
            })
    
    def train(self) -> None:
        """Main training loop with distributed training support."""
        if self._is_main_process():
            self.logger.info("Starting training...")
            self.logger.info(f"Model: {self.model.__class__.__name__}")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Total epochs: {self.config.training.num_epochs}")
            self.logger.info(f"Batch size: {self.config.data.batch_size}")
            self.logger.info(f"Learning rate: {self.config.training.learning_rate}")
            if self.use_amp:
                self.logger.info(f"Mixed precision training enabled (dtype={self.amp_dtype})")
            if self.accumulation_steps > 1:
                self.logger.info(f"Gradient accumulation: {self.accumulation_steps} steps")
            if getattr(self.config.training, 'use_torch_compile', False):
                self.logger.info(f"torch.compile enabled (mode={getattr(self.config.training, 'compile_mode', 'default')})")
            if getattr(self.config.training, 'use_tf32', False):
                self.logger.info("TF32 matmul acceleration enabled")
            if getattr(self.config.training, 'cudnn_benchmark', False):
                self.logger.info("cuDNN benchmark auto-tuner enabled")
            if getattr(self.config.training, 'use_flash_attention', False):
                self.logger.info("Flash Attention (SDPA) enabled")
            if getattr(self.config.training, 'use_fused_optimizer', False):
                self.logger.info("Fused optimizer kernels enabled")
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, self.config.training.num_epochs):
            self.epoch = epoch
            
            warmup_epochs = getattr(self.config.training, 'warmup_epochs', 0)
            if epoch < warmup_epochs:
                lr_scale = (epoch + 1) / warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.config.training.learning_rate * lr_scale
            
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Update tracker
            self.tracker.update(
                train_metrics['loss'],
                val_metrics['loss'],
                train_metrics['accuracy_top1'],
                val_metrics['accuracy_top1']
            )
            
            # Log metrics (centralized with rank check)
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint (only on main process)
            if self._is_main_process():
                save_freq = getattr(self.config, 'save_frequency', 1)
                if epoch % save_freq == 0 or epoch == self.config.training.num_epochs - 1:
                    checkpoint_path = f"{self.checkpoint_dir}/epoch_{epoch}.pth"
                    is_best = val_metrics['accuracy_top1'] > self.best_acc
                    if is_best:
                        self.best_acc = val_metrics['accuracy_top1']
                    self.save_checkpoint(checkpoint_path, is_best=is_best)
                    self._cleanup_old_checkpoints()
            
            # Early stopping
            if self.tracker.should_stop():
                if self._is_main_process():
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Final logging
        if self._is_main_process():
            self.logger.info(f"Training completed!")
            self.logger.info(f"Best validation accuracy: {self.best_acc:.2f}%")
            
            # Save final model
            final_path = f"{self.checkpoint_dir}/final_model.pth"
            self.save_checkpoint(final_path)
            
            # Close logging
            if self.config.tensorboard_enabled:
                self.writer.close()
            
            if self.config.wandb_enabled:
                wandb.finish()


class ModelEMA:
    """Exponential Moving Average of model parameters.

    Keeps a shadow copy of the model that averages parameters over time,
    producing a smoother, more accurate model at validation time.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module):
        """Update shadow parameters with exponential decay."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone().detach()

    def apply_shadow(self, model: nn.Module):
        """Copy shadow parameters to model (for validation)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone().detach()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original parameters after validation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.shadow = state_dict


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Action Recognition Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        trainer.logger.info(f"Resumed from checkpoint: {args.resume}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()