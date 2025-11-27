"""
Loss functions for Stage 2 Prompt Learning
Following SpikeCLIP paper Equation 7-8
https://arxiv.org/abs/2501.04477
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptLoss(nn.Module):
    """
    Binary Cross-Entropy loss for HQ/LQ classification

    From paper Equation 7:
    L_initial = CrossEntropy(y, y_hat)

    Where:
    - y = 0 for LQ, 1 for HQ
    - y_hat = softmax similarity between image and prompts
    """

    def __init__(self, label_smoothing=0.0):
        """
        :param label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, labels):
        """
        Compute cross-entropy loss.

        :param logits: Model output (B, 2) - [LQ_score, HQ_score]
        :param labels: Ground truth (B,) - 0 for LQ, 1 for HQ
        :return: loss value
        """
        return self.criterion(logits, labels)


def compute_accuracy(logits, labels):
    """
    Compute classification accuracy.

    :param logits: Model output (B, 2)
    :param labels: Ground truth (B,)
    :return: accuracy (float)
    """
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total