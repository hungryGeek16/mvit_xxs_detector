
import torch
import numpy as np
from .ssd_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def convert_boxes_to_locations(gt_boxes, prior_boxes, center_variance,size_variance):
    if prior_boxes.dim() + 1 == gt_boxes.dim():  ## Problem is somewhere over here 
        prior_boxes = prior_boxes.unsqueeze(0)

    target_centers = ((gt_boxes[..., :2] - prior_boxes[..., :2]) / prior_boxes[..., 2:]) /center_variance
    target_size = torch.log(gt_boxes[..., 2:] / prior_boxes[..., 2:]) / size_variance
    return torch.cat((target_centers, target_size), dim=-1)

def corner_form_to_center_form(boxes):
    return torch.cat(
        (
            (boxes[..., :2] + boxes[..., 2:]) * 0.5,
            boxes[..., 2:] - boxes[..., :2]
        ),
        dim=-1
    )

def convert_locations_to_boxes(pred_locations, anchor_boxes, center_variance=0.1, size_variance=0.2):

    # priors can have one dimension less.
    if anchor_boxes.dim() + 1 == pred_locations.dim():
        anchor_boxes = anchor_boxes.unsqueeze(0)

    # T_w = log(g_w/d_w) / size_variance ==> g_w = exp(T_w * size_variance) * d_w
    # T_h = log(g_h/d_h) / size_variance ==> g_h = exp(T_h * size_variance) * d_h
    pred_size = (
        torch.exp(pred_locations[..., 2:] * size_variance) * anchor_boxes[..., 2:]
    )
    # T_cx = ((g_cx - d_cx) / d_w) / center_variance ==> g_cx = ((T_cx * center_variance) * d_w) + d_cx
    # T_cy = ((g_cy - d_cy) / d_w) / center_variance ==> g_cy = ((T_cy * center_variance) * d_h) + d_cy
    pred_center = (
        pred_locations[..., :2] * center_variance * anchor_boxes[..., 2:]
    ) + anchor_boxes[..., :2]

    return torch.cat((pred_center, pred_size), dim=-1)



def center_form_to_corner_form(boxes):
    return torch.cat(
        (
            boxes[..., :2] - (boxes[..., 2:] * 0.5),
            boxes[..., :2] + (boxes[..., 2:] * 0.5),
        ),
        dim=-1,
    )

def convert_to_boxes(pred_locations, anchors):
  # decode boxes in center form
  boxes = convert_locations_to_boxes(pred_locations,anchors)
  # convert boxes from center form [c_x, c_y] to corner form [x, y]
  boxes = center_form_to_corner_form(boxes)
  return boxes


def match(reference_boxes_ctr, annotations, center_variance = 0.1, size_variance = 0.2, iou_threshold = 0.5):
  
  boxes = []
  labels = []

  for element in annotations:

    gt_boxes_cor = element['boxes'].to(device)
    gt_labels = element['labels'].to(device)


    if isinstance(gt_boxes_cor, np.ndarray):
        gt_boxes_cor = torch.from_numpy(gt_boxes_cor)
    if isinstance(gt_labels, np.ndarray):
        gt_labels = torch.from_numpy(gt_labels)

    # convert box priors from center [c_x, c_y] to corner_form [x, y]
    reference_boxes_cor = center_form_to_corner_form(boxes=reference_boxes_ctr)

    matched_boxes_cor, matched_labels = assign_priors(
        gt_boxes_cor, # gt_boxes are in corner form [x, y, w, h]
        gt_labels,
        reference_boxes_cor, # priors are in corner form [x, y, w, h]
        iou_threshold
    )
      
    # convert the matched boxes to center form [c_x, c_y]
    matched_boxes_ctr = corner_form_to_center_form(matched_boxes_cor)

    # Eq.(2) in paper https://arxiv.org/pdf/1512.02325.pdf

    boxes_for_regression = convert_boxes_to_locations(
        gt_boxes=matched_boxes_ctr, # center form
        prior_boxes=reference_boxes_ctr, # center form
        center_variance=center_variance,
        size_variance=size_variance
    )


    boxes.append(boxes_for_regression)
    labels.append(matched_labels)

  return torch.stack(boxes).unsqueeze(0), torch.stack(labels).unsqueeze(0)

