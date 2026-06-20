"""
Pose Feature Extractor - Converts skeleton data to meaningful features for action recognition.
This module implements a spatio-temporal graph convolutional network (ST-GCN) to extract features from skeleton sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .skeleton import SKELETON_CONNECTIONS


class AdaptiveGraphConvolution(nn.Module):
    """Graph convolution with learnable adjacency."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 num_nodes: int, num_subsets: int = 3):
        super().__init__()
        
        # Base adjacency (structural prior)
        self.register_buffer('A_base', self._get_base_adjacency(num_nodes))
        
        # Learnable adjacency offset
        self.A_offset = nn.Parameter(torch.zeros(num_subsets, num_nodes, num_nodes))
        
        # Attention for dynamic edges (heads must divide embed_dim)
        attn_heads = max(1, min(4, in_channels))
        while in_channels % attn_heads != 0:
            attn_heads -= 1
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels, 
            num_heads=attn_heads, 
            batch_first=True
        )
        
        self.conv = nn.Conv2d(
            in_channels * num_subsets, 
            out_channels, 
            kernel_size=1
        )
    
    @staticmethod
    def _get_base_adjacency(num_nodes: int) -> torch.Tensor:
        """Build 3-subset spatial graph (same as ST-GCN paper)."""
        from .skeleton import SKELETON_CONNECTIONS

        center = 0
        edges = [(s - 1, t - 1) for (s, t) in SKELETON_CONNECTIONS]

        hop_dis = torch.full((num_nodes, num_nodes), float('inf'))
        for i, j in edges:
            hop_dis[i, j] = 1
            hop_dis[j, i] = 1
        for k in range(num_nodes):
            d = hop_dis[:, k:k+1] + hop_dis[k:k+1, :]
            hop_dis = torch.minimum(hop_dis, d)
        hop_dis[hop_dis == float('inf')] = 999

        A = torch.zeros(3, num_nodes, num_nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                h = int(hop_dis[j, i].item())
                if h == 0:
                    A[0, j, i] = 1.0
                elif h == 1 and hop_dis[j, center].item() < hop_dis[i, center].item():
                    A[1, j, i] = 1.0
                elif h == 1 and hop_dis[j, center].item() > hop_dis[i, center].item():
                    A[2, j, i] = 1.0
        return A
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, V)
        """
        B, C, T, V = x.shape
        
        # Compute dynamic adjacency via attention
        x_flat = x.permute(0, 2, 3, 1).reshape(B * T, V, C)
        attn_out, attn_weights = self.attention(x_flat, x_flat, x_flat)
        A_dynamic = attn_weights.mean(0)  # (V, V)
        
        # Combine static and dynamic adjacency: (S, V, V)
        A = self.A_base + self.A_offset + 0.1 * A_dynamic
        A = F.softmax(A, dim=-1)
        
        # Graph convolution
        x_out = []
        for subset in range(A.size(0)):
            # (B, C, T, V) @ (V, V) -> (B, C, T, V)
            x_subset = torch.einsum('bctv,vw->bctw', x, A[subset])
            x_out.append(x_subset)
        
        x_out = torch.cat(x_out, dim=1)
        return self.conv(x_out)


class STConvBlock(nn.Module):
    """Spatio-Temporal Convolutional Block with multi-scale temporal feature extraction."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: tuple = (9, 3), stride: int = 1, 
                 residual: bool = True, dropout: float = 0.5,
                 use_multiscale_temporal: bool = True):
        super().__init__()
        
        self.gcn = AdaptiveGraphConvolution(in_channels, out_channels, num_nodes=25)
        self.use_multiscale = use_multiscale_temporal
        
        if use_multiscale_temporal:
            # Multi-scale temporal branches
            self.temporal_branches = nn.ModuleList([
                nn.Conv2d(out_channels, out_channels // 4, (3, 1), padding=(1, 0)),
                nn.Conv2d(out_channels, out_channels // 4, (5, 1), padding=(2, 0)),
                nn.Conv2d(out_channels, out_channels // 4, (7, 1), padding=(3, 0)),
                nn.Conv2d(out_channels, out_channels // 4, (9, 1), padding=(4, 0))
            ])
            
            self.tcn = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            )
            
            # Fusion conv matches stride so residual add works
            self.temporal_fusion = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            )
        else:
            # Standard single-scale temporal convolution
            self.tcn = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 
                    kernel_size=kernel_size, 
                    stride=(stride, 1), 
                    padding=((kernel_size[0] - 1) // 2, 0)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            )
        
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 
                    kernel_size=1, 
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x)
        
        if self.use_multiscale:
            # Pre-processing
            x = self.tcn(x)
            
            # Multi-scale temporal branches
            multi_scale_features = [branch(x) for branch in self.temporal_branches]
            
            # Concatenate and fuse
            x = torch.cat(multi_scale_features, dim=1)
            x = self.temporal_fusion(x)
        else:
            # Standard temporal convolution
            x = self.tcn(x)
        
        x += res
        return F.relu(x, inplace=True)


class PoseFeatureExtractor(nn.Module):
    """Spatio-Temporal Graph Convolutional Network for Pose Feature Extraction."""
    
    def __init__(self, in_channels: int = 3, num_keypoints: int = 25,
                 dropout: float = 0.5, use_multiscale_temporal: bool = True):
        super().__init__()
        
        if not (1 <= num_keypoints <= 100):
            raise ValueError(f"num_keypoints must be between 1 and 100, got {num_keypoints}")
        if in_channels < 1:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        
        self.num_keypoints = num_keypoints
        
        # ST-GCN layers
        self.layers = nn.ModuleList([
            STConvBlock(in_channels, 64, residual=False, dropout=dropout, use_multiscale_temporal=use_multiscale_temporal),
            STConvBlock(64, 128, stride=2, dropout=dropout, use_multiscale_temporal=use_multiscale_temporal),
            STConvBlock(128, 256, stride=2, dropout=dropout, use_multiscale_temporal=use_multiscale_temporal),
            STConvBlock(256, 256, dropout=dropout, use_multiscale_temporal=use_multiscale_temporal),
            STConvBlock(256, 256, dropout=dropout, use_multiscale_temporal=use_multiscale_temporal)
        ])
        
        # Final pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor shape."""
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (N, C, T, V), got {x.dim()}D tensor")
        if x.shape[1] < 1:
            raise ValueError(f"Expected at least 1 channel, got {x.shape[1]}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, V) - batch, channels, time, vertices
        """
        self._validate_input(x)
        
        # Forward through ST-GCN layers
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling
        x = self.pool(x)
        x = x.flatten(1)  # Flatten all dimensions except batch
        
        return x


class AttentionModule(nn.Module):
    """Spatial and temporal attention modules."""
    
    def __init__(self, channels: int, num_nodes: int):
        super().__init__()
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(channels, 1, 1)
        
        # Temporal attention
        self.temporal_conv = nn.Conv2d(channels, 1, (9, 1), padding=(4, 0))
        
        # Channel attention
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, V)
        """
        # Spatial attention
        spatial_att = self.spatial_conv(x).squeeze(1)  # (N, T, V)
        spatial_att = F.softmax(spatial_att, dim=-1).unsqueeze(1)  # (N, 1, T, V)
        
        # Temporal attention
        temporal_att = self.temporal_conv(x).squeeze(1)  # (N, T, V)
        temporal_att = F.softmax(temporal_att, dim=1).unsqueeze(1)  # (N, 1, T, V)
        
        # Channel attention
        channel_att = self.channel_conv(self.channel_avg_pool(x))  # (N, C, 1, 1)
        
        # Apply attention
        x = x * spatial_att * temporal_att * channel_att
        
        return x


class EnhancedPoseFeatureExtractor(PoseFeatureExtractor):
    """Enhanced feature extractor with attention mechanisms."""
    
    def __init__(self, in_channels: int = 3, num_keypoints: int = 25,
                 use_attention: bool = True, dropout: float = 0.5,
                 use_multiscale_temporal: bool = True):
        super().__init__(in_channels, num_keypoints, dropout=dropout, use_multiscale_temporal=use_multiscale_temporal)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionModule(256, num_keypoints)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        self._validate_input(x)
        
        # Forward through ST-GCN layers (all but last two)
        for layer in self.layers[:-2]:
            x = layer(x)
        
        # Apply attention before final layers
        if self.use_attention:
            x = self.attention(x)
        
        # Final layers
        for layer in self.layers[-2:]:
            x = layer(x)
        
        # Global average pooling
        x = self.pool(x)
        x = x.flatten(1)  # Flatten all dimensions except batch
        
        return x


def normalize_skeleton(skeleton: torch.Tensor, center_joint: int = 20) -> torch.Tensor:
    """Normalize skeleton coordinates.

    Args:
        skeleton: (N, C, T, V) skeleton tensor
        center_joint: 0-indexed center joint (NTU spine base = joint 21 -> index 20)

    Returns:
        Normalized skeleton tensor
    """    # skeleton: (N, C, T, V)
    
    # Center around the specified joint
    center = skeleton[:, :, :, center_joint:center_joint+1].clone()
    skeleton = skeleton - center
    
    # Scale to unit length
    scales = torch.norm(skeleton, dim=1, keepdim=True)
    scales = scales.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    skeleton = skeleton / (scales + 1e-8)
    
    return skeleton
