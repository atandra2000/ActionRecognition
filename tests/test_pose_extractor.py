import torch
import numpy as np
import pytest
from src.models.pose_extractor import normalize_skeleton


class TestNormalizeSkeleton:
    def test_center_joint_default(self):
        skeleton = torch.randn(2, 3, 64, 25)
        normalized = normalize_skeleton(skeleton)
        center = normalized[:, :, :, 20:21]
        assert torch.allclose(center, torch.zeros_like(center), atol=1e-6)

    def test_output_shape_preserved(self):
        skeleton = torch.randn(4, 3, 32, 25)
        normalized = normalize_skeleton(skeleton)
        assert normalized.shape == skeleton.shape

    def test_spatial_normalization(self):
        skeleton = torch.zeros(1, 3, 1, 3)
        skeleton[0, :, 0, 0] = torch.tensor([0.0, 0.0, 0.0])
        skeleton[0, :, 0, 1] = torch.tensor([1.0, 0.0, 0.0])
        skeleton[0, :, 0, 2] = torch.tensor([0.0, 1.0, 0.0])
        normalized = normalize_skeleton(skeleton, center_joint=0)
        norm1 = torch.norm(normalized[0, :, 0, 1])
        norm2 = torch.norm(normalized[0, :, 0, 2])
        assert torch.allclose(norm1, norm2, atol=1e-4)

    def test_no_nan_output(self):
        skeleton = torch.randn(2, 3, 64, 25)
        normalized = normalize_skeleton(skeleton)
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()
