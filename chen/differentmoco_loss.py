import torch
import torch.nn as nn

class different_moco_loss(nn.Module):
    def __init__(self):
        super(different_moco_loss, self).__init__()

    def forward(self, predictions, targets):
        device = predictions.device
        original_n = predictions.size(0)
        
        if predictions.dim() == 3 and predictions.size(1) == 3:  # [batchsize, 3, 1]
            uniform_labels = torch.linspace(0, 1, steps=original_n).unsqueeze(-1).unsqueeze(-1).to(device)
            uniform_labels = uniform_labels.expand(-1, 3, -1) 
            predictions_enhanced = torch.cat([predictions, uniform_labels], dim=0)
            targets_enhanced = torch.cat([targets, uniform_labels], dim=0)
        elif predictions.dim() == 2 and predictions.size(1) == 1:  # [batchsize, 1]
            uniform_labels = torch.linspace(0, 1, steps=original_n).unsqueeze(-1).to(device)
            predictions_enhanced = torch.cat([predictions, uniform_labels], dim=0)
            targets_enhanced = torch.cat([targets, uniform_labels], dim=0)
        else:
            raise ValueError("Unsupported input dimensions for predictions and targets")

        original_indices = torch.combinations(torch.arange(original_n, dtype=torch.long))
        additional_indices = torch.cartesian_prod(torch.arange(original_n, dtype=torch.long), torch.arange(original_n, original_n * 2 , dtype=torch.long))
        all_indices = torch.cat([original_indices, additional_indices], dim=0)
        
        pred_a, pred_b = predictions_enhanced[all_indices[:, 0]], predictions_enhanced[all_indices[:, 1]]
        target_a, target_b = targets_enhanced[all_indices[:, 0]], targets_enhanced[all_indices[:, 1]]

        diff_pred = pred_a - pred_b
        diff_gt = target_a - target_b

        loss = torch.mean((diff_pred - diff_gt) ** 2)
        return loss
