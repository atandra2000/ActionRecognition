import torch
import numpy as np
import pytest
from src.models.layers import compute_bone_features


class TestComputeBoneFeatures:
    def test_output_shape(self):
        skeleton = torch.randn(2, 3, 64, 25)
        bones = compute_bone_features(skeleton)
        assert bones.shape == skeleton.shape

    def test_bone_vector_is_difference(self):
        skeleton = torch.zeros(1, 3, 1, 25)
        skeleton[0, :, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
        skeleton[0, :, 0, 1] = torch.tensor([3.0, 0.0, 0.0])
        bones = compute_bone_features(skeleton)
        assert torch.allclose(bones[0, :, 0, 1], torch.tensor([2.0, 0.0, 0.0]))

    def test_root_joint_is_self(self):
        skeleton = torch.randn(2, 3, 64, 25)
        bones = compute_bone_features(skeleton)
        root_joints = [3, 15, 19, 21, 23]
        for rj in root_joints:
            assert torch.allclose(bones[:, :, :, rj], skeleton[:, :, :, rj])
