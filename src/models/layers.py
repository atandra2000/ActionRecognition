"""
Graph convolutional layers for skeleton-based action recognition.

Implements:
- AdaptiveGraphConv (AGCN-style): learned adjacency with structural prior
- CTRGCBlock (CTR-GCN): channel-wise topology refinement
- MultiScaleTemporalModule (MPM): multi-dilation temporal conv
- EfficientGraphBlock: combined spatio-temporal building block
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .skeleton import SKELETON_CONNECTIONS


# ── Adjacency helpers ──────────────────────────────────────────────────────────

def _edges_to_adjacency(num_nodes: int, edges: List[Tuple[int, int]]) -> torch.Tensor:
    """Build binary adjacency matrix from edge list (0-indexed edges)."""
    A = torch.zeros(num_nodes, num_nodes)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def _build_normalized_adjacency(num_nodes: int) -> torch.Tensor:
    """Build the full NTU adjacency matrix with self-loops, then normalize."""
    # Edges from SKELETON_CONNECTIONS are 1-indexed
    edges = [(s - 1, t - 1) for (s, t) in SKELETON_CONNECTIONS]
    A = _edges_to_adjacency(num_nodes, edges)
    A = A + torch.eye(num_nodes)  # self-loops
    # Symmetric normalize:  D^{-1/2} A D^{-1/2}
    d = A.sum(dim=1).clamp(min=1e-8)
    d_inv_sqrt = d.pow(-0.5)
    return d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]


# ── Graph Convolution Layers ──────────────────────────────────────────────────

class AdaptiveGraphConv(nn.Module):
    """Graph convolution with learnable adjacency (AGCN style).

    A = normalize(A_fixed + A_learned)
    Both A_fixed and A_learned contribute. A_learned is regularised via softmax.
    """

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes

        # Fixed structural prior  (3 subsets: root, centripetal, centrifugal)
        self.register_buffer('A_fixed', self._build_fixed_adjacency(num_nodes))

        # Learnable residual adjacency per subset
        self.A_learned = nn.Parameter(torch.zeros(3, num_nodes, num_nodes))

        # Convolution: aggregate subsets
        self.conv = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1)

        self.reset_parameters()

    @staticmethod
    def _build_fixed_adjacency(num_nodes: int) -> torch.Tensor:
        """Build 3-subset spatial graph (same as ST-GCN paper)."""
        center = 0  # spine base (1-indexed → 0)
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
                # h >= 2 is ignored (only immediate neighbors are connected)
        # Symmetric normalize
        for s in range(3):
            d = A[s].sum(dim=1).clamp(min=1e-8)
            d_inv_sqrt = d.pow(-0.5)
            A[s] = d_inv_sqrt[:, None] * A[s] * d_inv_sqrt[None, :]
        return A

    def reset_parameters(self):
        nn.init.zeros_(self.A_learned)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, T, V) input skeleton features

        Returns:
            (B, C_out, T, V) graph-convolved features
        """
        B, C, T, V = x.shape

        # Combined adjacency: fixed + learned, then softmax over columns
        A = self.A_fixed + self.A_learned          # (3, V, V)
        A = F.softmax(A, dim=-1)                    # normalize per column

        # Vectorized graph conv: (B, C, T, V) @ (3, V, V) -> (B, 3, C, T, V)
        # Reshape x to (B, 1, C, T, V) and use batched matmul
        x_expanded = x.unsqueeze(1)                 # (B, 1, C, T, V)
        # A: (3, V, V) -> (3, 1, V, V) for broadcasting
        A_expanded = A.unsqueeze(1)                 # (3, 1, V, V)
        # matmul: (B, 1, C, T, V) @ (3, 1, V, V) -> (B, 3, C, T, V)
        out = torch.matmul(x_expanded, A_expanded.transpose(-2, -1))
        out = out.reshape(B, 3 * C, T, V)           # (B, 3*C, T, V)

        return self.conv(out)                        # (B, C_out, T, V)


class CTRGCBlock(nn.Module):
    """Channel-wise Topology Refinement Graph Convolution (CTR-GCN).

    Different channels learn different spatial topologies independently.
    Features are grouped → each group gets its own learned adjacency.
    """

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int,
                 num_groups: int = 8, num_subsets: int = 3):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_groups = num_groups
        self.num_subsets = num_subsets
        self.group_channels = in_channels // num_groups

        # Fixed structural prior (shared across groups)
        self.register_buffer('A_fixed', self._build_fixed_adjacency(num_nodes, num_subsets))

        # Learned refinement per group per subset
        self.A_refine = nn.Parameter(
            torch.zeros(num_groups, num_subsets, num_nodes, num_nodes)
        )

        # Norm + activation after graph conv
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Channel fusion conv
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Learnable scaling per channel group
        self.gamma = nn.Parameter(torch.zeros(num_groups, num_subsets))

        self.reset_parameters()

    @staticmethod
    def _build_fixed_adjacency(num_nodes: int, num_subsets: int) -> torch.Tensor:
        """Build base adjacency (used as mask/prior for all groups)."""
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

        A = torch.zeros(num_subsets, num_nodes, num_nodes)
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

    def reset_parameters(self):
        nn.init.zeros_(self.A_refine)
        nn.init.zeros_(self.gamma)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        self.bn.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, T, V) input

        Returns:
            (B, C_out, T, V) graph-convolved features
        """
        B, C, T, V = x.shape
        g = self.num_groups

        # Split channels into groups
        x_grouped = x.view(B, g, self.group_channels, T, V)  # (B, G, C_g, T, V)

        # Vectorized per-group graph convolution
        # A_fixed: (S, V, V), A_refine: (G, S, V, V), gamma: (G, S)
        # Compute adjacency for all groups at once
        A_g = self.A_fixed.unsqueeze(0) + self.gamma.view(g, self.num_subsets, 1, 1) * self.A_refine  # (G, S, V, V)
        A_g = F.softmax(A_g, dim=-1)  # (G, S, V, V)

        # x_grouped: (B, G, C_g, T, V)
        # Reshape for batched einsum: x (B, G, C_g, T, V), A (G, S, V, V)
        # Want output: (B, G, S, C_g, T, V) then sum over S -> (B, G, C_g, T, V)
        # Use einsum: 'bgctv,gsvw->bgsc tw'
        x_flat = x_grouped  # (B, G, C_g, T, V)
        out = torch.einsum('bgctv,gsvw->bgstcw', x_flat, A_g)  # (B, G, S, T, C_g, V)
        out = out.permute(0, 1, 2, 4, 3, 5)  # (B, G, S, C_g, T, V)
        out = out.sum(dim=2)  # Sum over subsets: (B, G, C_g, T, V)
        out = out.reshape(B, C, T, V)

        # Channel projection + norm + activation
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class MultiScaleTemporalModule(nn.Module):
    """Multi-Progress Module (MPM) with multiple temporal kernel scales.

    Uses parallel 1D temporal convs with increasing dilation rates
    to capture motions at different speeds.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_sizes: Tuple[int, ...] = (3, 5, 7, 9),
                 dilations: Tuple[int, ...] = (1, 2, 3, 4),
                 dropout: float = 0.0):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), "kernel_sizes and dilations must match"

        num_branches = len(kernel_sizes)
        branch_out = out_channels // num_branches

        self.branches = nn.ModuleList()
        for ks, d in zip(kernel_sizes, dilations):
            pad = (ks - 1) * d // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_out, kernel_size=(ks, 1),
                              padding=(pad, 0), dilation=(d, 1)),
                    nn.BatchNorm2d(branch_out),
                    nn.ReLU(inplace=True),
                )
            )

        # Residual connection if shapes match
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.res_bn = nn.BatchNorm2d(out_channels) if in_channels != out_channels else nn.Identity()

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, T, V) input

        Returns:
            (B, C_out, T, V) multi-scale temporal features
        """
        branch_outs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outs, dim=1)

        # Residual
        res = self.residual(x)
        if not isinstance(self.res_bn, nn.Identity):
            res = self.res_bn(res)

        out = out + res
        out = self.dropout(out)
        return F.relu(out, inplace=True)


# ── Combined Blocks ───────────────────────────────────────────────────────────

class EfficientGraphBlock(nn.Module):
    """Efficient spatio-temporal building block.

    Architecture: CTRGC (spatial) → MPM (temporal)
    """

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int,
                 stride: int = 1, dropout: float = 0.0,
                 num_groups: int = 8, temporal_kernel_sizes: Tuple[int, ...] = (3, 5, 7, 9),
                 temporal_dilations: Tuple[int, ...] = (1, 2, 3, 4)):
        super().__init__()

        # Spatial graph convolution (CTR-GCN style)
        self.spatial = CTRGCBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            num_nodes=num_nodes,
            num_groups=num_groups,
        )

        # Temporal convolution (multi-scale)
        self.temporal = MultiScaleTemporalModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_sizes=temporal_kernel_sizes,
            dilations=temporal_dilations,
            dropout=dropout,
        )

        # Temporal downsampling if stride > 1
        self.temporal_down = (
            nn.Conv2d(out_channels, out_channels, kernel_size=(stride, 1),
                      stride=(stride, 1))
            if stride > 1
            else nn.Identity()
        )

        # Residual connection
        self.residual = nn.Identity() if (in_channels == out_channels and stride == 1) else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, T, V) input

        Returns:
            (B, C_out, T_out, V) features
        """
        res = self.residual(x)

        x = self.spatial(x)
        x = self.temporal(x)
        x = self.temporal_down(x)

        # Align residual temporal dimension if needed
        if res.shape[-2] != x.shape[-2]:
            res = F.adaptive_avg_pool2d(res, (x.shape[-2], x.shape[-1]))

        x = x + res
        return F.relu(x, inplace=True)


class STGCNBackbone(nn.Module):
    """Full ST-GCN backbone with modern improvements.

    Stack of EfficientGraphBlocks with increasing channel dimensions.
    """

    def __init__(self, in_channels: int, num_nodes: int, num_classes: int,
                 channels: Tuple[int, ...] = (64, 128, 256, 256),
                 strides: Tuple[int, ...] = (1, 2, 2, 1),
                 dropout: float = 0.5):
        super().__init__()

        assert len(channels) == len(strides), "channels and strides must match"

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # ST-GCN blocks
        blocks = []
        prev_c = channels[0]
        for i, (c, s) in enumerate(zip(channels, strides)):
            blocks.append(EfficientGraphBlock(
                in_channels=prev_c,
                out_channels=c,
                num_nodes=num_nodes,
                stride=s,
                dropout=dropout,
                num_groups=min(8, c // 4),
            ))
            prev_c = c
        self.blocks = nn.Sequential(*blocks)

        # Global pooling + classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(prev_c, num_classes),
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, T, V) skeleton input

        Returns:
            (B, num_classes) action logits
        """
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def compute_bone_features(skeleton: torch.Tensor) -> torch.Tensor:
    """Compute bone vectors from skeleton joints.

    Each bone = child_joint - parent_joint. Uses NTU skeleton topology.
    Output has the same (B, C, T, V) shape with bone vectors at joint positions
    (bone vector stored at the child joint location).

    Args:
        skeleton: (B, C, T, V) joint coordinates

    Returns:
        (B, C, T, V) bone features
    """
    B, C, T, V = skeleton.shape
    # NTU skeleton parent mapping (1-indexed → 0-indexed)
    parent_map = torch.full((V,), -1, dtype=torch.long, device=skeleton.device)
    for s, t in SKELETON_CONNECTIONS:
        parent_map[t - 1] = s - 1

    bones = torch.zeros_like(skeleton)
    for j in range(V):
        p = parent_map[j]
        if p >= 0:
            bones[:, :, :, j] = skeleton[:, :, :, j] - skeleton[:, :, :, p]
        else:
            bones[:, :, :, j] = skeleton[:, :, :, j]  # root joint

    return bones
