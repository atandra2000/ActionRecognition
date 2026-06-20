"""
Configuration management system for the action recognition project.
"""

import os
import yaml
from typing import Dict, Any, Optional, Tuple, List
import dataclasses
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Pose Estimator
    pose_estimator: str = "hrnet"
    pose_estimator_pretrained: bool = False
    pose_estimator_weights: Optional[str] = None
    num_keypoints: int = 25
    pose_dim: int = 3  # 2D or 3D
    
    # Feature Extractor
    feature_extractor: str = "stgcn"
    hidden_dim: int = 256
    num_layers: int = 10
    dropout: float = 0.1
    
    # Action Recognition
    num_classes: int = 120  # NTU RGB+D 120
    temporal_window: int = 64  # frames (matches training data max_frame_num)
    graph_layout: str = "ntu-rgb+d"
    use_attention: bool = True
    multimodal: bool = False
    streams: List[str] = field(default_factory=lambda: ["joint", "bone"])  # two-stream by default


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 120
    warmup_epochs: int = 10
    
    # Optimizer
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Scheduler
    scheduler: str = "cosine"
    min_lr: float = 1e-6
    
    # Loss
    loss_type: str = "cross_entropy"
    label_smoothing: float = 0.1
    use_focal_loss: bool = False
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16" (A100 prefers bfloat16)
    
    # torch.compile (PyTorch 2.0+ kernel fusion)
    use_torch_compile: bool = False
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    
    # TF32 (Ampere tensor core acceleration for matmul)
    use_tf32: bool = False
    
    # cuDNN auto-tuner
    cudnn_benchmark: bool = False
    
    # Flash Attention via SDPA backend
    use_flash_attention: bool = False
    
    # Fused optimizer (reduces kernel launches)
    use_fused_optimizer: bool = False
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # Regularization
    gradient_clip: float = 5.0
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 20
    keep_best_k_checkpoints: int = 3


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    dataset: str = "ntu_rgbd120"
    data_root: str = "/data/ntu_rgbd120"
    protocol: str = "xsub"
    num_workers: int = 8
    pin_memory: bool = True
    
    # Preprocessing
    image_size: Tuple[int, int] = (256, 256)
    skeleton_normalization: bool = True
    temporal_sample_rate: int = 1
    max_frame_num: int = 64
    sampling_strategy: str = "uniform"
    
    # Dataloader
    batch_size: int = 32
    prefetch_factor: int = 2  # Prefetch batches per worker (higher = less GPU idle time)
    
    # Augmentation
    random_scale: bool = True
    random_rotation: bool = True
    random_translation: bool = True
    temporal_jitter: bool = True


@dataclass
class InferenceConfig:
    """Inference configuration."""
    device: str = "cuda"
    batch_size: int = 1
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    
    # Real-time settings
    camera_id: int = 0
    display_fps: bool = True
    save_output: bool = False
    output_path: str = "./outputs"
    
    # Model settings
    model_weights: str = "./checkpoints/best_model.pth"
    temporal_window: int = 64
    stride: int = 16


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Global settings
    project_name: str = "action_recognition"
    experiment_name: str = "default"
    output_dir: str = "./outputs"
    seed: int = 42
    debug: bool = False
    
    # Hardware
    device: str = "cuda"
    num_gpus: int = 1
    distributed: bool = False
    
    # Logging
    log_level: str = "INFO"
    wandb_enabled: bool = False
    tensorboard_enabled: bool = True
    save_frequency: int = 10
    
    # Version tracking
    config_version: str = "1.0"


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file with validation."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}. Make sure to provide a valid YAML config file.")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML config file {config_path}: {e}")
    
    if not isinstance(config_dict, dict):
        raise ValueError(f"Config file must contain a dictionary, got {type(config_dict).__name__}")
    
    # Create config with defaults
    config = Config()
    
    # Load nested configs with validation
    try:
        if 'model' in config_dict:
            model_fields = {f.name for f in dataclasses.fields(ModelConfig)}
            model_kwargs = {k: v for k, v in config_dict.get('model', {}).items() if k in model_fields}
            config.model = ModelConfig(**model_kwargs)
        if 'training' in config_dict:
            training_fields = {f.name for f in dataclasses.fields(TrainingConfig)}
            training_kwargs = {k: v for k, v in config_dict.get('training', {}).items() if k in training_fields}
            config.training = TrainingConfig(**training_kwargs)
        if 'data' in config_dict:
            data_fields = {f.name for f in dataclasses.fields(DataConfig)}
            data_kwargs = {k: v for k, v in config_dict.get('data', {}).items() if k in data_fields}
            config.data = DataConfig(**data_kwargs)
        if 'inference' in config_dict:
            inference_fields = {f.name for f in dataclasses.fields(InferenceConfig)}
            inference_kwargs = {k: v for k, v in config_dict.get('inference', {}).items() if k in inference_fields}
            config.inference = InferenceConfig(**inference_kwargs)
    except TypeError as e:
        raise ValueError(f"Invalid configuration field: {e}")
    
    # Update global settings
    global_keys = {'project_name', 'experiment_name', 'output_dir', 'seed', 'debug', 'device', 
                   'num_gpus', 'distributed', 'log_level', 'wandb_enabled', 'tensorboard_enabled', 
                   'save_frequency', 'config_version', 'use_torch_compile', 'compile_mode',
                   'use_tf32', 'cudnn_benchmark', 'use_flash_attention', 'use_fused_optimizer',
                   'amp_dtype', 'prefetch_factor'}
    for key, value in config_dict.items():
        if key in global_keys and hasattr(config, key):
            try:
                setattr(config, key, value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid value for {key}: {e}")
    
    return config


def save_config(config: Config, save_path: str) -> None:
    """Save configuration to YAML file using dataclass introspection."""
    config_dict = asdict(config)
    
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    
    try:
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)
    except IOError as e:
        raise IOError(f"Failed to save config to {save_path}: {e}")


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
    """Merge override config into base config, properly handling nested dataclasses."""
    base_dict = asdict(base_config)
    
    # Recursively merge nested dictionaries
    def deep_merge(base: Dict, override: Dict) -> Dict:
        merged = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    merged_dict = deep_merge(base_dict, override_config)
    
    # Reconstruct Config with nested dataclasses
    try:
        model_fields = {f.name for f in dataclasses.fields(ModelConfig)}
        training_fields = {f.name for f in dataclasses.fields(TrainingConfig)}
        data_fields = {f.name for f in dataclasses.fields(DataConfig)}
        inference_fields = {f.name for f in dataclasses.fields(InferenceConfig)}
        
        config = Config(
            model=ModelConfig(**{k: v for k, v in merged_dict.get('model', {}).items() if k in model_fields}),
            training=TrainingConfig(**{k: v for k, v in merged_dict.get('training', {}).items() if k in training_fields}),
            data=DataConfig(**{k: v for k, v in merged_dict.get('data', {}).items() if k in data_fields}),
            inference=InferenceConfig(**{k: v for k, v in merged_dict.get('inference', {}).items() if k in inference_fields}),
        )
        # Update global settings
        global_keys = {'project_name', 'experiment_name', 'output_dir', 'seed', 'debug', 'device', 
                       'num_gpus', 'distributed', 'log_level', 'wandb_enabled', 'tensorboard_enabled', 
                       'save_frequency', 'config_version', 'use_torch_compile', 'compile_mode',
                       'use_tf32', 'cudnn_benchmark', 'use_flash_attention', 'use_fused_optimizer',
                       'amp_dtype', 'prefetch_factor'}
        for key in global_keys:
            if key in merged_dict:
                setattr(config, key, merged_dict[key])
        return config
    except TypeError as e:
        raise ValueError(f"Failed to merge configs: {e}")


# Example configuration files
def create_example_configs() -> Dict[str, Config]:
    """Create example configuration objects."""
    
    # NTU RGB+D 120 ST-GCN configuration
    ntu120_config = Config()
    ntu120_config.model.num_classes = 120
    ntu120_config.model.feature_extractor = "stgcn"
    ntu120_config.model.graph_layout = "ntu-rgb+d"
    ntu120_config.data.dataset = "ntu_rgbd120"
    ntu120_config.data.data_root = "/data/ntu_rgbd120"
    ntu120_config.training.num_epochs = 120
    ntu120_config.experiment_name = "ntu120_stgcn"
    
    # Real-time inference configuration
    inference_config = Config()
    inference_config.inference.batch_size = 1
    inference_config.inference.temporal_window = 64
    inference_config.inference.stride = 16
    inference_config.inference.confidence_threshold = 0.7
    inference_config.experiment_name = "real_time_inference"
    
    return {
        'ntu120_stgcn.yaml': ntu120_config,
        'inference.yaml': inference_config
    }
