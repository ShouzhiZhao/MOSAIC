import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss
    """

    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, pos_scores, neg_scores):
        """
        pos_scores: (batch_size,)
        neg_scores: (batch_size,)
        """
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + self.gamma).mean()
        return loss
