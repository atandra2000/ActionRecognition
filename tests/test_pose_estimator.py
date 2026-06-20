import torch
import numpy as np
import pytest
from src.models.pose_estimator import heatmaps_to_keypoints


class TestHeatmapsToKeypoints:
    def test_output_shape(self):
        heatmaps = torch.randn(2, 25, 64, 64)
        keypoints = heatmaps_to_keypoints(heatmaps, stride=4)
        assert keypoints.shape == (2, 25, 2)

    def test_stride_scaling(self):
        heatmaps = torch.randn(1, 1, 16, 16)
        heatmaps[:, :, 8, 8] = 100.0
        keypoints = heatmaps_to_keypoints(heatmaps, stride=4)
        assert torch.allclose(keypoints[0, 0], torch.tensor([32.0, 32.0]), atol=2.0)

    def test_batch_independence(self):
        heatmaps = torch.randn(4, 10, 32, 32)
        keypoints = heatmaps_to_keypoints(heatmaps, stride=4)
        assert keypoints.shape == (4, 10, 2)
        assert not torch.allclose(keypoints[0], keypoints[1])
