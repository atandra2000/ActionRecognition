"""
Human Pose Estimator from Scratch.
This module implements a custom pose estimator. It includes:
- HRNet backbone with multi-scale feature fusion
- 2D pose estimation head with heatmaps, confidence, and offsets
- 3D pose lifting network with confidence weighting
- Multi-person pose estimation with detection and cropping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, activation: str = "relu"):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, channels: int):
        super().__init__()

        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out += residual
        return self.activation(out)


class SqueezeExcitationBlock(nn.Module):
    """Channel Attention (Squeeze-and-Excitation) block.
    
    Recalibrates channel-wise feature responses by modeling inter-dependencies
    between channels. Lightweight and effective for multi-scale feature fusion.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Squeeze: Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: Channel-wise gating with FC layers
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Attention-weighted output (B, C, H, W)
        """
        # Squeeze: Global average pooling
        b, c, h, w = x.shape
        squeeze = self.avg_pool(x).view(b, c)  # (B, C)
        
        # Excitation: Learn channel weights
        excitation = self.fc(squeeze).view(b, c, 1, 1)  # (B, C, 1, 1)
        
        # Scale input by channel attention
        return x * excitation


class HRNetBlock(nn.Module):
    """High-Resolution Net block with parallel multi-scale branches."""
    
    def __init__(self, channels_list: List[int], num_branches: int, use_attention: bool = True):
        super().__init__()
        self.num_branches = num_branches
        self.channels_list = channels_list
        self.use_attention = use_attention
        
        # Branches at different resolutions
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            self.branches.append(nn.Sequential(
                ResidualBlock(channels_list[i]),
                ResidualBlock(channels_list[i])
            ))
        
        # Fusing connections (downsampling and upsampling)
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(nn.Identity())
                elif i < j:  # Downsample to coarser resolution
                    downsample = nn.Sequential(
                        nn.Conv2d(channels_list[j], channels_list[i], 3, 2**( j - i), 1),
                        nn.BatchNorm2d(channels_list[i])
                    )
                    fuse_layer.append(downsample)
                else:  # Upsample to finer resolution
                    upsample = nn.Sequential(
                        nn.Conv2d(channels_list[j], channels_list[i], 1),
                        nn.Upsample(scale_factor=2**(i - j), mode='nearest')
                    )
                    fuse_layer.append(upsample)
            self.fuse_layers.append(fuse_layer)
        
        # Channel attention for each branch (lightweight SE blocks)
        if use_attention:
            self.se_blocks = nn.ModuleList([
                SqueezeExcitationBlock(channels, reduction=16)
                for channels in channels_list
            ])
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass with channel attention on fused features.
        
        Args:
            x_list: List of features at different scales
        
        Returns:
            Updated feature list with attention-weighted fusion
        """
        # Process each branch
        out_list = []
        for i, branch in enumerate(self.branches):
            out_list.append(branch(x_list[i]))
        
        # Fuse features from different scales with channel attention
        fused_list = []
        for i in range(self.num_branches):
            fused = torch.zeros_like(out_list[i])
            for j in range(self.num_branches):
                fused = fused + self.fuse_layers[i][j](out_list[j])
            fused = self.relu(fused)
            
            # Apply channel attention to fused features
            if self.use_attention:
                fused = self.se_blocks[i](fused)
            
            fused_list.append(fused)
        
        return fused_list


class HRNet(nn.Module):
    """High-Resolution Network for pose estimation.
    
    HRNet maintains high-resolution representations throughout the network
    using parallel multi-scale branches with continuous information exchange.
    """
    
    def __init__(self, num_branches: int = 4, num_blocks: int = 4,
                 base_channels: int = 64, use_attention: bool = True):
        super().__init__()
        self.num_branches = num_branches
        self.num_blocks = num_blocks
        self.base_channels = base_channels
        self.use_attention = use_attention
        
        # Stem: Initial convolution to reduce spatial dimensions and increase channels
        self.stem = nn.Sequential(
            ConvBlock(3, base_channels // 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(base_channels // 2, base_channels, kernel_size=3, stride=2, padding=1)
        )
        
        # Initial layer to create multi-scale branches
        self.layer1 = nn.Sequential(
            ResidualBlock(base_channels),
            ResidualBlock(base_channels)
        )
        
        # Create channel list for branches [64, 128, 256, 512]
        self.channels_list = [base_channels * (2 ** i) for i in range(num_branches)]
        
        # Layer to expand from single scale to multiple scales
        self.transition1 = nn.ModuleList()
        for i in range(num_branches):
            if i == 0:
                self.transition1.append(nn.Identity())
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.channels_list[i-1], self.channels_list[i], 3, 2, 1),
                    nn.BatchNorm2d(self.channels_list[i]),
                    nn.ReLU(inplace=True)
                )
                self.transition1.append(downsample)
        
        # HRNet blocks for continuous multi-scale processing
        self.hr_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.hr_blocks.append(
                HRNetBlock(self.channels_list, num_branches, use_attention=use_attention)
            )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through HRNet.
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            List of multi-scale feature maps maintaining high resolution
        """
        # Stem layers
        x = self.stem(x)  # (B, 64, H/4, W/4)
        x = self.layer1(x)
        
        # Create multi-scale branches
        x_list = []
        for i, transition in enumerate(self.transition1):
            if i == 0:
                x_list.append(transition(x))
            else:
                x_list.append(transition(x_list[i-1] if i > 0 else x))
        
        # Process through HRNet blocks
        for hr_block in self.hr_blocks:
            x_list = hr_block(x_list)
        
        return x_list


class PoseEstimator2D(nn.Module):
    """2D Human Pose Estimator using HRNet architecture.
    
    HRNet maintains high-resolution representations throughout the network,
    enabling accurate pose estimation with better keypoint localization.
    """
    
    def __init__(self, num_keypoints: int = 25, input_channels: int = 3,
                 num_branches: int = 4, num_blocks: int = 4, base_channels: int = 64,
                 use_attention: bool = True):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_branches = num_branches
        self.base_channels = base_channels
        self.use_attention = use_attention
        
        # HRNet backbone with optional channel attention
        self.hrnet = HRNet(num_branches=num_branches, num_blocks=num_blocks,
                          base_channels=base_channels, use_attention=use_attention)
        
        # Calculate output channels from HRNet (list of channels at different scales)
        self.hrnet_channels = [base_channels * (2 ** i) for i in range(num_branches)]
        
        # Prediction heads for each resolution scale
        self.heatmap_heads = nn.ModuleList()
        self.confidence_heads = nn.ModuleList()
        self.offset_heads = nn.ModuleList()
        
        for i, channels in enumerate(self.hrnet_channels):
            # Heatmap prediction
            self.heatmap_heads.append(nn.Sequential(
                ConvBlock(channels, channels),
                nn.Conv2d(channels, num_keypoints, 1)
            ))
            
            # Confidence prediction
            self.confidence_heads.append(nn.Sequential(
                ConvBlock(channels, channels),
                nn.Conv2d(channels, num_keypoints, 1)
            ))
            
            # Offset prediction for refinement
            self.offset_heads.append(nn.Sequential(
                ConvBlock(channels, channels),
                nn.Conv2d(channels, num_keypoints * 2, 1)
            ))
        
        # Final fusion layer to combine multi-scale predictions
        self.fusion = nn.Conv2d(
            num_keypoints * num_branches,
            num_keypoints,
            kernel_size=1
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with multi-scale prediction and fusion.
        
        Args:
            x: Input image tensor (B, C, H, W)
        
        Returns:
            Dictionary with fused heatmaps, confidence scores, and offsets
        """
        # HRNet backbone produces multi-scale features
        features_list = self.hrnet(x)
        
        # Generate predictions at each scale
        heatmap_preds = []
        confidence_preds = []
        offset_preds = []
        intermediate_heatmaps = []
        
        # Base resolution from first scale
        base_shape = features_list[0].shape[2:]
        
        for i, (features, hm_head, conf_head, offset_head) in enumerate(
            zip(features_list, self.heatmap_heads, self.confidence_heads, self.offset_heads)
        ):
            # Generate predictions
            heatmap = hm_head(features)
            confidence = conf_head(features)
            offsets = offset_head(features)
            
            # Upsample to base resolution for fusion
            if i > 0:
                scale_factor = 2 ** i
                heatmap = F.interpolate(
                    heatmap, size=base_shape,
                    mode='bilinear', align_corners=False
                )
                confidence = F.interpolate(
                    confidence, size=base_shape,
                    mode='bilinear', align_corners=False
                )
                offsets = F.interpolate(
                    offsets, size=base_shape,
                    mode='bilinear', align_corners=False
                )
            
            heatmap_preds.append(heatmap)
            confidence_preds.append(confidence)
            offset_preds.append(offsets)
            intermediate_heatmaps.append(heatmap)
        
        # Fuse multi-scale heatmap predictions
        heatmap_fused = torch.cat(heatmap_preds, dim=1)  # (B, K*num_branches, H, W)
        final_heatmap = self.fusion(heatmap_fused)
        
        # Average confidence across scales
        confidence_fused = torch.stack(confidence_preds, dim=0).mean(dim=0)
        
        # Use offset from highest resolution (first scale)
        final_offsets = offset_preds[0]
        
        return {
            'intermediate_heatmaps': intermediate_heatmaps,
            'heatmap': final_heatmap,
            'confidence': torch.sigmoid(confidence_fused),
            'offsets': final_offsets
        }
    

class TemporalPoseRefiner(nn.Module):
    """Temporal refinement for 3D poses with variable sequence length."""
    
    def __init__(self, num_joints: int = 25, hidden_dim: int = 256,
                 refinement_scale: float = 0.1):
        super().__init__()
        self.num_joints = num_joints
        self.refinement_scale = refinement_scale
        
        # Temporal encoder
        self.temporal_encoder = nn.LSTM(
            num_joints * 3, hidden_dim, 
            num_layers=2, bidirectional=True, batch_first=True
        )
        
        # Refinement head
        self.refiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_joints * 3)
        )
    
    def forward(self, poses_3d: torch.Tensor) -> torch.Tensor:
        """Refine 3D poses using temporal information.
        
        Args:
            poses_3d: (B, T, J, 3) where T >= 1
        
        Returns:
            Refined poses with same shape as input
        """
        
        B, T, J, _ = poses_3d.shape
        
        # Flatten spatial dimensions
        x = poses_3d.view(B, T, -1)
        
        # Temporal encoding
        encoded, _ = self.temporal_encoder(x)
        
        # Residual refinement
        refinement = self.refiner(encoded).view(B, T, J, 3)
        
        return poses_3d + self.refinement_scale * refinement


class PoseEstimator3D(nn.Module):
    """3D Human Pose Estimator that lifts 2D poses to 3D using confidence weighting."""
    
    def __init__(self, num_keypoints: int = 25, use_temporal_refinement: bool = True):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.use_temporal_refinement = use_temporal_refinement
        
        # 2D Pose Estimator
        self.pose_2d_estimator = PoseEstimator2D(num_keypoints)
        
        # 3D Lifting Network with confidence-weighted input
        self.lifting_network = nn.Sequential(
            nn.Linear(num_keypoints * 3, 1024),  # 2D coords + confidence
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_keypoints * 3)
        )
        
        # Temporal Refiner (optional)
        if use_temporal_refinement:
            self.temporal_refiner = TemporalPoseRefiner(num_keypoints)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate 3D poses with confidence weighting.
        
        Args:
            x: Input image tensor (B, C, H, W)
        
        Returns:
            Dictionary with 2D keypoints, 3D keypoints, and confidence scores
        """
        # 2D Pose Estimation
        pose_2d_outputs = self.pose_2d_estimator(x)
        heatmaps = pose_2d_outputs['heatmap']
        confidence = pose_2d_outputs['confidence']
        
        # Convert heatmaps to 2D keypoints (HRNet stride = 4)
        keypoints_2d = heatmaps_to_keypoints(heatmaps, stride=4)  # (B, J, 2)
        confidence_scores = heatmaps_to_keypoints(confidence, stride=4)[:, :, 0]  # (B, J)
        
        B, J, _ = keypoints_2d.shape
        
        # Concatenate 2D keypoints with confidence scores
        keypoints_2d_with_conf = torch.cat([
            keypoints_2d, 
            confidence_scores.unsqueeze(-1)
        ], dim=-1)  # (B, J, 3)
        
        # Lift to 3D with confidence weighting
        keypoints_2d_flat = keypoints_2d_with_conf.view(B, -1)
        keypoints_3d_flat = self.lifting_network(keypoints_2d_flat)
        keypoints_3d = keypoints_3d_flat.view(B, J, 3)
        
        # Optional temporal refinement
        if self.use_temporal_refinement:
            keypoints_3d_seq = keypoints_3d.unsqueeze(1)  # (B, T=1, J, 3)
            keypoints_3d = self.temporal_refiner(keypoints_3d_seq).squeeze(1)
        
        return {
            'heatmap': heatmaps,
            'keypoints_2d': keypoints_2d,
            'confidence_2d': confidence_scores,
            'keypoints_3d': keypoints_3d
        }

    def forward_temporal(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate 3D poses from video sequence.
        
        Args:
            x: (B, T, C, H, W) video frames
        
        Returns:
            Dict with keypoints_3d of shape (B, T, J, 3)
        """
        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)
        pose_2d_outputs = self.pose_2d_estimator(frames)
        keypoints_2d = heatmaps_to_keypoints(pose_2d_outputs['heatmap'], stride=4)
        confidence = heatmaps_to_keypoints(pose_2d_outputs['confidence'], stride=4)[:, :, 0]
        keypoints_2d_with_conf = torch.cat([keypoints_2d, confidence.unsqueeze(-1)], dim=-1)
        keypoints_2d_flat = keypoints_2d_with_conf.view(B * T, -1)
        keypoints_3d_flat = self.lifting_network(keypoints_2d_flat)
        keypoints_3d = keypoints_3d_flat.view(B, T, self.num_keypoints, 3)
        if self.use_temporal_refinement:
            keypoints_3d = self.temporal_refiner(keypoints_3d)
        return {'keypoints_3d': keypoints_3d}


class MultiPersonPoseEstimator(nn.Module):
    """Multi-person pose estimator with detection and pose estimation."""
    
    def __init__(self, num_keypoints: int = 25, max_people: int = 10,
                 conf_threshold: float = 0.5, crop_size: int = 256):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.max_people = max_people
        self.conf_threshold = conf_threshold
        self.crop_size = crop_size
        
        # Person detection network (simplified YOLO-like)
        self.detector = nn.Sequential(
            ConvBlock(3, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, max_people * 5),
            nn.Sigmoid()
        )
        
        # Single person pose estimator
        self.pose_estimator = PoseEstimator2D(num_keypoints)

    def _extract_crop(self, image: torch.Tensor, det: torch.Tensor) -> torch.Tensor:
        """Extract image crop based on detection with proper tensor handling.
        
        Args:
            image: (1, C, H, W) - single image from batch
            det: (5,) - [x, y, w, h, conf] as tensor
        
        Returns:
            Cropped and resized image (1, C, crop_size, crop_size)
        """
        # Ensure det is on same device
        det = det.to(image.device)
        
        # Extract coordinates as floats
        x, y, w, h = det[:4].float()
        _, _, H, W = image.shape
        H, W = float(H), float(W)
        
        # Convert normalized coordinates to pixel coordinates with safe casting
        x1 = torch.clamp(torch.round((x - w / 2) * W).long(), min=0, max=int(W)-1)
        y1 = torch.clamp(torch.round((y - h / 2) * H).long(), min=0, max=int(H)-1)
        x2 = torch.clamp(torch.round((x + w / 2) * W).long(), min=x1+1, max=int(W))
        y2 = torch.clamp(torch.round((y + h / 2) * H).long(), min=y1+1, max=int(H))
        
        # Extract crop
        crop = image[:, :, y1:y2, x1:x2]
        
        # Resize to standard size
        crop = F.interpolate(
            crop, size=(self.crop_size, self.crop_size), 
            mode='bilinear', align_corners=False
        )
        
        return crop
    
    def _reorganize_poses(self, poses: Dict[str, torch.Tensor], crop_indices: List[int], 
                          batch_size: int) -> Dict[str, list]:
        """Reorganize poses back to batch structure efficiently.
        
        Args:
            poses: Dictionary with 'heatmap', 'confidence', 'offsets' tensors
            crop_indices: List of batch indices for each crop
            batch_size: Size of original batch
        
        Returns:
            Dictionary with lists of tensors per batch item
        """
        output = {
            'heatmaps': [[] for _ in range(batch_size)],
            'confidence': [[] for _ in range(batch_size)],
            'offsets': [[] for _ in range(batch_size)]
        }
        
        # Group poses by batch index
        for i, batch_idx in enumerate(crop_indices):
            output['heatmaps'][batch_idx].append(poses['heatmap'][i])
            output['confidence'][batch_idx].append(poses['confidence'][i])
            output['offsets'][batch_idx].append(poses['offsets'][i])
        
        # Stack lists into tensors where available
        for b in range(batch_size):
            output['heatmaps'][b] = (
                torch.stack(output['heatmaps'][b]) 
                if output['heatmaps'][b] else None
            )
            output['confidence'][b] = (
                torch.stack(output['confidence'][b]) 
                if output['confidence'][b] else None
            )
            output['offsets'][b] = (
                torch.stack(output['offsets'][b]) 
                if output['offsets'][b] else None
            )
        
        return output
    
    def _empty_output(self, batch_size: int) -> Dict[str, list]:
        """Return empty output structure with None values."""
        return {
            'heatmaps': [None] * batch_size,
            'confidence': [None] * batch_size,
            'offsets': [None] * batch_size
        }
    
    def forward(self, x: torch.Tensor) -> Dict[str, list]:
        """Batch-parallel multi-person pose estimation.
        
        Args:
            x: (B, C, H, W) - batch of images
        
        Returns:
            Dictionary with lists of detection outputs per batch item
        """
        
        batch_size = x.size(0)
    
        # Detect all people at once
        detections = self.detector(x)  # (B, max_people * 5)
        detections = detections.view(batch_size, self.max_people, 5)
    
        # Filter valid detections
        valid_mask = detections[..., 4] > self.conf_threshold
    
        # Batch process all valid crops
        all_crops = []
        crop_indices = []
    
        for b in range(batch_size):
            valid_dets = detections[b][valid_mask[b]]
            for det in valid_dets:
                # Extract crop with proper device handling
                crop = self._extract_crop(x[b:b+1], det)
                all_crops.append(crop)
                crop_indices.append(b)
    
        if all_crops:
            # Batch inference on all crops
            crops_tensor = torch.cat(all_crops, dim=0)
            poses = self.pose_estimator(crops_tensor)
        
            # Reorganize by batch
            return self._reorganize_poses(poses, crop_indices, batch_size)
    
        return self._empty_output(batch_size)


def heatmaps_to_keypoints(heatmaps: torch.Tensor, stride: int = 4) -> torch.Tensor:
    """Convert heatmaps to keypoint coordinates using soft-argmax.
    
    Args:
        heatmaps: Heatmap tensor (B, K, H, W)
        stride: Downsampling stride of the backbone (HRNet default: 4)
        
    Returns:
        Keypoints in original image coordinates (B, K, 2)
    """
    batch_size, num_keypoints, height, width = heatmaps.shape
    
    # Create coordinate grids in heatmap space
    y_coords = torch.arange(height, dtype=torch.float32, device=heatmaps.device)
    x_coords = torch.arange(width, dtype=torch.float32, device=heatmaps.device)
    
    # Normalize heatmaps to probabilities
    probs = F.softmax(heatmaps.view(batch_size, num_keypoints, -1), dim=-1)
    probs = probs.view(batch_size, num_keypoints, height, width)
    
    # Calculate expected coordinates in heatmap space
    y_coords = y_coords.view(1, 1, height, 1)
    x_coords = x_coords.view(1, 1, 1, width)
    
    y_keypoints = (probs * y_coords).sum(dim=[2, 3])
    x_keypoints = (probs * x_coords).sum(dim=[2, 3])
    
    # Stack coordinates (in heatmap space)
    keypoints = torch.stack([x_keypoints, y_keypoints], dim=-1)
    
    # Scale to original image space by stride
    keypoints = keypoints * stride
    
    return keypoints


def create_pose_model(model_type: str = "2d", num_keypoints: int = 25, **kwargs) -> nn.Module:
    """Factory function to create pose estimation models using HRNet backbone.
    
    Args:
        model_type: '2d', '3d', or 'multi_person'
        num_keypoints: Number of keypoints (typically 17 or 25)
        **kwargs: Additional arguments passed to model constructor:
                 - num_branches: Number of HRNet branches (default: 4)
                 - num_blocks: Number of HRNet blocks (default: 4)
                 - base_channels: Base channel count (default: 64)
    
    Returns:
        Initialized pose estimation model with HRNet backbone
    
    Raises:
        ValueError: If model_type is unknown
    """
    model_type = model_type.lower()
    
    if model_type == "2d":
        return PoseEstimator2D(num_keypoints, **kwargs)
    elif model_type == "3d":
        return PoseEstimator3D(num_keypoints, **kwargs)
    elif model_type == "multi_person":
        return MultiPersonPoseEstimator(num_keypoints, **kwargs)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from: '2d', '3d', 'multi_person'"
        )
