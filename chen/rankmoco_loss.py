import torch
import numpy as np
from sodeep_ import SpearmanLoss, load_sorter

"""
Ranking loss function for brain age estimation
"""


# ===== loss function of combine rankg loss, age difference loss adn MSE ========= #
class rank_moco_loss(torch.nn.Module):
    def __init__(self, sorter_checkpoint_path=None, beta=1):
        """
        ['Ranking loss', which including Sprear man's ranking loss and age difference loss]

        Args:
            bate (float, optional):
            [used as a weighte between ranking loss and age difference loss.
            Since ranking loss is in (0,1),but age difference is relative large.
            In order to banlance these two loss functions, beta is set in (0,1)].
            Defaults to 1.
        """
        super(rank_moco_loss, self).__init__()
        if sorter_checkpoint_path is not None:
            self.spearman_loss = SpearmanLoss(*load_sorter(sorter_checkpoint_path))
        else:
            self.spearman_loss = SpearmanLoss('exa', 256, None)  

        self.beta = beta

    def forward(self, mem_pred, mem_gt):
        #device = mem_pred.device  
        #uniform_labels = torch.linspace(0, 1, steps=mem_pred.size(0)).unsqueeze(-1).to(device)  # (64,1)
        
        #mem_pred_enhanced = torch.cat([mem_pred, uniform_labels], dim=0)  # (128, 1)
        #mem_gt_enhanced = torch.cat([mem_gt, uniform_labels], dim=0)  # (128, 1)

        
        rankmoco_loss = self.spearman_loss(mem_pred, mem_gt)
        
        return rankmoco_loss
