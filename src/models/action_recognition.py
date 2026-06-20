"""
Action Recognition Model - Main model with 2-stream (joint + bone) support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from .pose_estimator import PoseEstimator2D, PoseEstimator3D, heatmaps_to_keypoints
from .pose_extractor import EnhancedPoseFeatureExtractor, normalize_skeleton
from .layers import compute_bone_features


def _build_mlp_classifier(input_dim: int, output_dim: int, 
                          hidden_dims: List[int] = None, dropout: float = 0.5) -> nn.Module:
    """Build a flexible MLP classifier.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension (number of classes)
        hidden_dims: List of hidden layer dimensions. Defaults to [512, 256]
        dropout: Dropout probability
    
    Returns:
        Sequential module with MLP layers
    """
    if hidden_dims is None:
        hidden_dims = [512, 256]
    
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        ])
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class ActionRecognitionModel(nn.Module):
    """Action recognition model with optional 2-stream (joint + bone) support."""
    
    def __init__(self, 
                 num_classes: int = 120,
                 num_keypoints: int = 25,
                 pose_dim: int = 3,
                 use_attention: bool = True,
                 multimodal: bool = True,
                 streams: List[str] = None,
                 attention_heads: int = 8,
                 fusion_method: str = 'attention',
                 classifier_hidden_dims: List[int] = None,
                 dropout: float = 0.5,
                 **kwargs):
        super().__init__()
        
        if not (1 <= num_classes <= 10000):
            raise ValueError(f"num_classes must be 1-10000, got {num_classes}")
        if not (1 <= num_keypoints <= 100):
            raise ValueError(f"num_keypoints must be 1-100, got {num_keypoints}")
        if pose_dim not in [2, 3]:
            raise ValueError(f"pose_dim must be 2 or 3, got {pose_dim}")
        if fusion_method not in ['attention', 'concat', 'add']:
            raise ValueError(f"fusion_method must be 'attention', 'concat', or 'add', got {fusion_method}")
        
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.pose_dim = pose_dim
        self.multimodal = multimodal
        self.fusion_method = fusion_method
        self.streams = streams or ['joint']
        
        for s in self.streams:
            if s not in ('joint', 'bone'):
                raise ValueError(f"Unknown stream '{s}'. Choose from 'joint', 'bone'.")
        
        self.pose_feature_dim = 256
        self.rgb_feature_dim = 256
        
        # Pose estimation component
        if pose_dim >= 2:
            self.pose_estimator_2d = PoseEstimator2D(num_keypoints=num_keypoints)
        
        if pose_dim == 3:
            self.pose_estimator_3d = PoseEstimator3D(num_keypoints=num_keypoints, use_temporal_refinement=False)
        
        # Feature extraction — one backbone per stream
        self.backbones = nn.ModuleDict()
        backbone_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ('dropout', 'use_multiscale_temporal')
        }
        for s in self.streams:
            self.backbones[s] = EnhancedPoseFeatureExtractor(
                in_channels=pose_dim,
                num_keypoints=num_keypoints,
                use_attention=use_attention,
                **backbone_kwargs
            )
        
        # RGB feature extraction (if multimodal)
        rgb_feature_dim = 0
        if multimodal:
            self.rgb_feature_extractor = self._build_rgb_extractor()
            rgb_feature_dim = self.rgb_feature_dim
        
        # Fusion layer
        stream_dim = self.pose_feature_dim * len(self.streams)
        if multimodal and fusion_method == 'attention':
            total_feature_dim = stream_dim + rgb_feature_dim
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=total_feature_dim,
                num_heads=min(attention_heads, total_feature_dim),
                batch_first=True,
                dropout=dropout
            )
            fusion_output_dim = total_feature_dim
        elif multimodal and fusion_method == 'concat':
            fusion_output_dim = stream_dim + rgb_feature_dim
        elif multimodal and fusion_method == 'add':
            if stream_dim != rgb_feature_dim:
                self.fusion_projection = nn.Linear(rgb_feature_dim, stream_dim)
            fusion_output_dim = stream_dim
        else:
            fusion_output_dim = stream_dim
        
        # Classification head — dimension must match actual features at forward time
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [512, 256]
        
        # When multimodal=True, fusion_output_dim includes rgb_feature_dim.
        # At inference time RGB may be absent, so we project pose features up.
        self._pose_only_classifier = multimodal  # needs projection when RGB absent
        if multimodal:
            self.pose_to_fusion = nn.Linear(stream_dim, fusion_output_dim)
        self.classifier = _build_mlp_classifier(
            fusion_output_dim, num_classes, 
            hidden_dims=classifier_hidden_dims, 
            dropout=dropout
        )
        
        self._initialize_weights()
    
    def _build_rgb_extractor(self) -> nn.Module:
        """Build RGB feature extractor (3D CNN with configurable architecture)."""
        return nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            
            nn.Conv3d(128, self.rgb_feature_dim, kernel_size=1),
            nn.BatchNorm3d(self.rgb_feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with optional 2-stream (joint + bone) and multi-modal fusion.

        Args:
            x: Dictionary containing:
                - 'skeleton': joint data (N, C, T, V)
                - 'bone': bone features (N, C, T, V) — optional, computed if missing
                - 'video': RGB video (N, C, T, H, W) — optional

        Returns:
            Dictionary with logits and probabilities
        """
        if not isinstance(x, dict):
            raise ValueError(f"Expected dict input, got {type(x)}")

        outputs = {}

        rgb_video = x.get('video', None)
        skeleton_data = x.get('skeleton', None)
        bone_data = x.get('bone', None)

        # Pose estimation from RGB if skeleton not provided
        if skeleton_data is None and rgb_video is not None:
            skeleton_data = self._extract_pose_from_video(rgb_video)

        stream_features = []
        if skeleton_data is not None:
            if skeleton_data.dim() != 4:
                raise ValueError(f"Expected skeleton shape (N, C, T, V), got {skeleton_data.shape}")
            skeleton_data = normalize_skeleton(skeleton_data)

            if 'joint' in self.streams:
                j_feat = self.backbones['joint'](skeleton_data)
                stream_features.append(j_feat)

            if 'bone' in self.streams:
                if bone_data is None:
                    bone_data = compute_bone_features(skeleton_data)
                b_feat = self.backbones['bone'](bone_data)
                stream_features.append(b_feat)

        if stream_features:
            pose_features = torch.cat(stream_features, dim=1)
            outputs['pose_features'] = pose_features
        else:
            pose_features = None

        rgb_features = None
        if self.multimodal and rgb_video is not None:
            rgb_features = self._extract_rgb_features(rgb_video)
            outputs['rgb_features'] = rgb_features

        if pose_features is not None and rgb_features is not None:
            combined_features = self._fuse_features(pose_features, rgb_features)
        elif pose_features is not None:
            if self._pose_only_classifier:
                combined_features = self.pose_to_fusion(pose_features)
            else:
                combined_features = pose_features
        elif rgb_features is not None:
            combined_features = rgb_features
        else:
            raise ValueError("No valid input features found (need video or skeleton)")

        logits = self.classifier(combined_features)
        outputs['logits'] = logits
        outputs['probabilities'] = F.softmax(logits, dim=1)

        return outputs
    
    def _fuse_features(self, pose_features: torch.Tensor, 
                       rgb_features: torch.Tensor) -> torch.Tensor:
        """Fuse pose and RGB features based on configured fusion method.
        
        Args:
            pose_features: (N, D_pose)
            rgb_features: (N, D_rgb)
        
        Returns:
            Fused features (N, D_out)
        """
        if self.fusion_method == 'concat':
            return torch.cat([pose_features, rgb_features], dim=1)
        
        elif self.fusion_method == 'add':
            if hasattr(self, 'fusion_projection'):
                rgb_features = self.fusion_projection(rgb_features)
            return pose_features + rgb_features
        
        elif self.fusion_method == 'attention':
            # Stack features for attention
            stacked = torch.stack([pose_features, rgb_features], dim=0)  # (2, N, D)
            stacked = stacked.transpose(0, 1)  # (N, 2, D)
            
            # Apply attention
            attended, _ = self.fusion_attention(stacked, stacked, stacked)
            
            # Global average pooling
            return attended.mean(dim=1)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _extract_pose_from_video(self, video: torch.Tensor) -> torch.Tensor:
        """Extract pose from video frames using batch processing.
        
        Args:
            video: (N, C, T, H, W) - batch of video sequences
        
        Returns:
            Skeleton data (N, C, T, V)
        
        Raises:
            ValueError: If video shape is invalid
        """
        if video.dim() != 5:
            raise ValueError(f"Expected 5D video tensor (N, C, T, H, W), got shape {video.shape}")
        
        batch_size, channels, temporal, height, width = video.shape
        
        # Reshape to (N*T, C, H, W) for batch 2D pose estimation
        frames = video.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
        frames = frames.view(batch_size * temporal, channels, height, width)
        
        # 2D pose estimation (batched)
        pose_2d_output = self.pose_estimator_2d(frames)
        pose_2d = heatmaps_to_keypoints(pose_2d_output['heatmap'], stride=4)  # (N*T, K, 2)
        
        # Extract confidence scores
        heatmap_flat = pose_2d_output['heatmap'].view(batch_size * temporal, self.num_keypoints, -1)
        confidence = torch.max(heatmap_flat, dim=2)[0]  # (N*T, K)
        
        # Concatenate with confidence
        pose_2d_conf = torch.cat([pose_2d, confidence.unsqueeze(-1)], dim=-1)  # (N*T, K, 3)
        
        # Reshape back to (N, T, K, C)
        pose_2d_conf = pose_2d_conf.view(batch_size, temporal, self.num_keypoints, 3)
        
        skeleton_data = pose_2d_conf.permute(0, 3, 1, 2).contiguous()
        
        if hasattr(self, 'pose_estimator_3d'):
            poses_2d_seq = skeleton_data[:, :2].permute(0, 2, 3, 1).contiguous()
            N, T, K, _ = poses_2d_seq.shape
            poses_2d_flat = poses_2d_seq.view(N * T, K * 2)
            confidence_flat = pose_2d_conf[:, 2].view(N * T, K)
            
            poses_2d_with_conf = torch.cat([
                poses_2d_flat.view(N * T, K, 2),
                confidence_flat.unsqueeze(-1)
            ], dim=-1)
            
            poses_2d_input = poses_2d_with_conf.view(N * T, -1)
            with torch.no_grad():
                poses_3d_flat = self.pose_estimator_3d.lifting_network(poses_2d_input)
            poses_3d = poses_3d_flat.view(N, T, K, 3)
            
            if hasattr(self.pose_estimator_3d, 'temporal_refiner'):
                poses_3d = self.pose_estimator_3d.temporal_refiner(poses_3d)
            
            skeleton_data = poses_3d.permute(0, 3, 1, 2).contiguous()
        
        return skeleton_data
    
    def _extract_rgb_features(self, video: torch.Tensor) -> torch.Tensor:
        """Extract RGB features using 3D CNN."""
        rgb_features = self.rgb_feature_extractor(video)
        
        # Global average pooling
        rgb_features = F.adaptive_avg_pool3d(rgb_features, 1)
        rgb_features = rgb_features.squeeze(-1).squeeze(-1).squeeze(-1)
        
        return rgb_features


class TemporalActionRecognitionModel(nn.Module):
    """Temporal action recognition model with LSTM/Transformer/GRU."""
    
    def __init__(self, 
                 num_classes: int = 120,
                 num_keypoints: int = 25,
                 pose_dim: int = 3,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 model_type: str = 'lstm',
                 classifier_hidden_dims: List[int] = None,
                 num_heads: int = 8,
                 dropout: float = 0.3,
                 **kwargs):
        super().__init__()
        
        # Validate inputs
        if not (1 <= num_classes <= 10000):
            raise ValueError(f"num_classes must be 1-10000, got {num_classes}")
        if not (1 <= num_keypoints <= 100):
            raise ValueError(f"num_keypoints must be 1-100, got {num_keypoints}")
        if pose_dim not in [2, 3]:
            raise ValueError(f"pose_dim must be 2 or 3, got {pose_dim}")
        if model_type not in ['lstm', 'transformer', 'gru']:
            raise ValueError(f"model_type must be 'lstm', 'transformer', or 'gru', got {model_type}")
        
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.pose_dim = pose_dim
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        
        # Input projection
        input_feature_dim = num_keypoints * pose_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        if model_type == 'lstm':
            self.temporal_model = nn.LSTM(
                hidden_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
            )
            temporal_output_dim = hidden_dim * 2  # bidirectional
            
        elif model_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=min(num_heads, hidden_dim),
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='relu'
            )
            self.temporal_model = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
            temporal_output_dim = hidden_dim
            
        elif model_type == 'gru':
            self.temporal_model = nn.GRU(
                hidden_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
            )
            temporal_output_dim = hidden_dim * 2  # bidirectional
        
        # Classification head
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [512, 256]
        
        self.classifier = _build_mlp_classifier(
            temporal_output_dim, num_classes,
            hidden_dims=classifier_hidden_dims,
            dropout=dropout
        )
    
    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        """Forward pass for temporal action recognition.
        
        Args:
            skeleton: Skeleton data with shape:
                - (N, C, T, V) for shape (batch, channels, time, keypoints) or
                - (N, T, V, C) for shape (batch, time, keypoints, channels)
        
        Returns:
            Action class logits (N, num_classes)
        
        Raises:
            ValueError: If input shape is invalid
        """
        if skeleton.dim() != 4:
            raise ValueError(f"Expected 4D skeleton tensor, got {skeleton.dim()}D with shape {skeleton.shape}")
        
        # Handle different input formats
        if skeleton.size(1) in [2, 3] and skeleton.size(-1) != 2 and skeleton.size(-1) != 3:
            # Input is (N, C, T, V)
            N, C, T, V = skeleton.shape
            skeleton = skeleton.permute(0, 2, 3, 1).contiguous()  # (N, T, V, C)
        else:
            # Input is (N, T, V, C)
            N, T, V, C = skeleton.shape
        
        # Validate dimensions
        if V != self.num_keypoints:
            raise ValueError(f"Expected {self.num_keypoints} keypoints, got {V}")
        
        # Flatten spatial dimensions for temporal model
        skeleton_flat = skeleton.view(N, T, -1)  # (N, T, V*C)
        
        # Project to hidden dimension
        skeleton_proj = self.input_projection(skeleton_flat)  # (N, T, H)
        
        # Temporal modeling
        if self.model_type in ['lstm', 'gru']:
            temporal_out, _ = self.temporal_model(skeleton_proj)
            # Use last hidden state for classification
            final_features = temporal_out[:, -1, :]  # (N, H*2)
        
        elif self.model_type == 'transformer':
            temporal_out = self.temporal_model(skeleton_proj)
            # Global average pooling over time dimension
            final_features = temporal_out.mean(dim=1)  # (N, H)
        
        # Classification
        logits = self.classifier(final_features)
        
        return logits


class EnsembleActionModel(nn.Module):
    """Ensemble of multiple action recognition models with weighted averaging."""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        
        if not models:
            raise ValueError("At least one model is required for ensemble")
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(models)})")
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
        
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through ensemble.
        
        Args:
            x: Input dictionary or tensor depending on model types
        
        Returns:
            Ensemble logits (N, num_classes)
        """
        logits_list = []
        
        for model in self.models:
            try:
                outputs = model(x)
                
                # Extract logits based on output format
                if isinstance(outputs, dict):
                    if 'logits' not in outputs:
                        raise ValueError(
                            f"Model output dict missing 'logits' key. "
                            f"Available keys: {list(outputs.keys())}"
                        )
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                logits_list.append(logits)
            except Exception as e:
                raise RuntimeError(f"Error in ensemble model forward pass: {e}")
        
        if not logits_list:
            raise RuntimeError("No valid outputs from ensemble models")
        
        # Stack logits and compute weighted average
        stacked_logits = torch.stack(logits_list, dim=0)  # (num_models, N, num_classes)
        
        # Apply weights: (num_models,) x (num_models, N, num_classes) -> (N, num_classes)
        weighted_logits = (self.weights.view(-1, 1, 1) * stacked_logits).sum(dim=0)
        
        return weighted_logits


def create_action_model(config: Dict[str, Any]) -> nn.Module:
    """Factory function to create action recognition models.
    
    Args:
        config: Configuration dictionary with:
            - 'model_type': Type of model ('stgcn', 'temporal', or 'ensemble')
            - Other model-specific parameters
    
    Returns:
        Initialized model
    
    Raises:
        ValueError: If model_type is unknown or config is invalid
    """
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config)}")
    
    model_type = config.get('model_type', 'stgcn').lower()
    
    try:
        if model_type == 'stgcn':
            # Default to single-stream if not specified
            if 'streams' not in config:
                config['streams'] = ['joint']
            return ActionRecognitionModel(**config)
        
        elif model_type == 'temporal':
            return TemporalActionRecognitionModel(**config)
        
        elif model_type == 'ensemble':
            if 'sub_models' not in config:
                raise ValueError("Ensemble requires 'sub_models' in config")
            
            sub_configs = config.get('sub_models', [])
            models = [create_action_model(cfg) for cfg in sub_configs]
            weights = config.get('weights', None)
            
            return EnsembleActionModel(models, weights)
        
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Choose from: 'stgcn', 'temporal', 'ensemble'"
            )
    except Exception as e:
        raise RuntimeError(f"Failed to create model: {e}")
