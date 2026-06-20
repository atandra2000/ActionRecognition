import torch
import numpy as np
import pytest
import tempfile
import pickle
import os
from src.data.datasets import SkeletonDataset


class TestSkeletonDataset:
    def test_getitem_output_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'skeleton_xsub_train.pkl')
            samples = [np.random.randn(3, 100, 25).astype(np.float32) for _ in range(10)]
            labels = list(range(10))
            with open(data_file, 'wb') as f:
                pickle.dump({'samples': samples, 'labels': labels}, f)

            dataset = SkeletonDataset(
                data_root=tmpdir,
                split='train',
                protocol='xsub',
                max_frames=64,
                normalize=False
            )
            item = dataset[0]
            assert 'data' in item
            assert 'label' in item
            assert 'skeleton' in item['data']

    def test_getitem_skeleton_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, 'skeleton_xsub_train.pkl')
            samples = [np.random.randn(3, 100, 25).astype(np.float32) for _ in range(5)]
            labels = [0, 1, 2, 3, 4]
            with open(data_file, 'wb') as f:
                pickle.dump({'samples': samples, 'labels': labels}, f)

            dataset = SkeletonDataset(
                data_root=tmpdir,
                split='train',
                protocol='xsub',
                max_frames=64,
                normalize=False
            )
            item = dataset[0]
            skeleton = item['data']['skeleton']
            assert skeleton.shape == (3, 64, 25)
