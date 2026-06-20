import torch
import numpy as np
import pytest
from src.training.metrics import accuracy, Metrics


class TestAccuracy:
    def test_top1_perfect(self):
        output = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        target = torch.tensor([0, 1])
        acc = accuracy(output, target, topk=(1,))
        assert acc[0] == 100.0

    def test_top5_covers_true(self):
        output = torch.randn(4, 10)
        target = torch.tensor([0, 1, 2, 3])
        output[:, target] = 100.0
        acc = accuracy(output, target, topk=(5,))
        assert acc[0] == 100.0

    def test_top1_partial(self):
        output = torch.tensor([[10.0, 0.0], [0.0, 10.0], [0.0, 10.0]])
        target = torch.tensor([0, 1, 0])
        acc = accuracy(output, target, topk=(1,))
        assert 0 < acc[0] < 100.0


class TestMetrics:
    def test_update_compute_reset_cycle(self):
        metrics = Metrics(num_classes=5)
        preds = torch.tensor([0, 1, 2])
        targets = torch.tensor([0, 1, 2])
        confs = torch.softmax(torch.randn(3, 5), dim=1)
        metrics.update(preds, targets, confs)
        result = metrics.compute()
        assert 'accuracy' in result
        assert result['accuracy'] == 1.0
        metrics.reset()
        result2 = metrics.compute()
        assert result2 == {}

    def test_empty_metrics_returns_empty(self):
        metrics = Metrics(num_classes=5)
        result = metrics.compute()
        assert result == {}
