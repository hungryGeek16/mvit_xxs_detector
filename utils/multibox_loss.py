import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch import distributed as dist
from typing import Union, Optional, Tuple

from .ssd_utils import *

def reduce_tensor(inp_tensor):
    size = float(dist.get_world_size())
    inp_tensor_clone = inp_tensor.clone()
    dist.barrier()
    dist.all_reduce(inp_tensor_clone, op=dist.ReduceOp.SUM)
    inp_tensor_clone /= size
    return inp_tensor_clone


def tensor_to_python_float(inp_tensor, is_distributed):
    if is_distributed and isinstance(inp_tensor, torch.Tensor):
        inp_tensor = reduce_tensor(inp_tensor=inp_tensor)

    if isinstance(inp_tensor, torch.Tensor) and inp_tensor.numel() > 1:
        # For IOU, we get a C-dimensional tensor (C - number of classes)
        # so, we convert here to a numpy array
        return inp_tensor.cpu().numpy()
    elif hasattr(inp_tensor, 'item'):
        return inp_tensor.item()
    elif isinstance(inp_tensor, (int, float)):
        return inp_tensor * 1.0
    else:
        raise NotImplementedError("The data type is not supported yet in tensor_to_python_float function")

class multibox_loss(nn.Module):
  
  def __init__(self):
    super(multibox_loss, self).__init__()
    self.unscaled_reg_loss = 1e-7
    self.unscaled_conf_loss = 1e-7
    self.neg_pos_ratio = 3
    self.wt_loc = 1.0
    self.curr_iter = 0
    self.max_iter = -1
    self.update_inter = 200
    self.is_distributed = False

    self.reset_unscaled_loss_values()

  # confidence: (batch_size, num_priors, num_classes)
  # predicted_locations :(batch_size, num_priors, 4)
  def reset_unscaled_loss_values(self):
    # initialize with very small float values
    self.unscaled_conf_loss = 1e-7
    self.unscaled_reg_loss = 1e-7

  def forward(self, x_hat, y_hat, x, y):

    confidence, predicted_locations = y_hat, x_hat

    gt_labels = y
    gt_locations = x

    num_classes = confidence.shape[-1]
    num_coordinates = predicted_locations.shape[-1]

    with torch.no_grad():
        loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
        mask = hard_negative_mining(loss, gt_labels, self.neg_pos_ratio)


    confidence = confidence[mask, :]
    classification_loss = F.cross_entropy(
        input=confidence.reshape(-1, num_classes),
        target=gt_labels[mask].long(),
        reduction="sum"
    )

    pos_mask = gt_labels > 0
    
    predicted_locations = predicted_locations[pos_mask, :].view(-1, num_coordinates)
    gt_locations = gt_locations[pos_mask, :].view(-1, num_coordinates)
    num_pos = gt_locations.shape[0]

    total_loss = classification_loss / num_pos
    return total_loss
