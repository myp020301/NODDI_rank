import torch
import torch.nn as nn

class RelativeDifferenceLoss(nn.Module):
    def __init__(self):
        super(RelativeDifferenceLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Calculates the loss based on the relative differences between predictions and targets for all voxel pairs.
        """

        n = predictions.size(0)
        indices = torch.combinations(torch.arange(n))

        pred_a, pred_b = predictions[indices[:, 0]], predictions[indices[:, 1]]
        target_a, target_b = targets[indices[:, 0]], targets[indices[:, 1]]

        diff_pred = pred_a - pred_b
        diff_gt = target_a - target_b

        loss = torch.mean((diff_pred - diff_gt) ** 2)
        return loss
