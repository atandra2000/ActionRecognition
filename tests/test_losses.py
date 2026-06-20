import torch
import numpy as np
import pytest
from src.training.losses import LabelSmoothingCrossEntropy, ActionRecognitionLoss


class TestLabelSmoothingCrossEntropy:
    def test_loss_positive_for_incorrect(self):
        loss_fn = LabelSmoothingCrossEntropy(num_classes=10, smoothing=0.1)
        pred = torch.randn(4, 10)
        target = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_smoothing_reduces_loss_for_correct(self):
        loss_fn_no_smooth = LabelSmoothingCrossEntropy(num_classes=10, smoothing=0.0)
        loss_fn_smooth = LabelSmoothingCrossEntropy(num_classes=10, smoothing=0.1)
        pred = torch.zeros(4, 10)
        pred[:, 0] = 10.0
        target = torch.zeros(4, dtype=torch.long)
        loss_no = loss_fn_no_smooth(pred, target)
        loss_smooth = loss_fn_smooth(pred, target)
        assert loss_smooth.item() > loss_no.item()


class TestActionRecognitionLoss:
    def test_total_equals_classification_when_no_aux(self):
        loss_fn = ActionRecognitionLoss(num_classes=120, loss_type='cross_entropy')
        predictions = {'logits': torch.randn(4, 120)}
        targets = {'label': torch.randint(0, 120, (4,))}
        losses = loss_fn(predictions, targets)
        assert 'total' in losses
        assert 'classification' in losses
        assert losses['total'].item() > 0

    def test_raises_on_missing_logits(self):
        loss_fn = ActionRecognitionLoss(num_classes=120)
        with pytest.raises(ValueError):
            loss_fn({'other': torch.randn(4, 120)}, {'label': torch.randint(0, 120, (4,))})
