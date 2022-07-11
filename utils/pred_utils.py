import torch
from torch import nn
from torchvision.ops import nms as torch_nms
from torch import Tensor
from typing import NamedTuple


def predict(confidences, locations, anchors):
    scores = nn.Softmax(dim=0)(confidences)
    boxes = convert_to_boxes(pred_locations=locations, anchors=anchors)
    return scores, boxes

def nms(boxes, scores, nms_threshold, top_k=200):
    """
    Args:
        boxes (N, 4): boxes in corner-form.
        scores (N): probabilities
        nms_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked: Boxes and scores
    """
    keep = torch_nms(boxes, scores, nms_threshold)
    if top_k > 0:
        keep = keep[:top_k]
    return boxes[keep], scores[keep]

def predict(boxes, scores):
  DetectionPredTuple = NamedTuple(
    'DetectionPredTuple', [
        ('labels', Tensor),
        ('scores', Tensor),
        ('boxes', Tensor)
    ])
  boxes = boxes[0] # remove the batch dimension
  scores = scores[0]

  boxes = boxes.squeeze(0)
  scores = scores.squeeze(0)
  
  num_classes = scores.shape[1]

  object_labels = []
  object_boxes = []
  object_scores = []
  for class_index in range(1, num_classes):
      probs = scores[:, class_index]
      mask = probs > 0.3
      probs = probs[mask]
      if probs.size(0) == 0:
          continue
      masked_boxes = boxes[mask, :]

      filtered_boxes, filtered_scores = nms(
          scores=probs.reshape(-1),
          boxes=masked_boxes,
          nms_threshold=0.3,
          top_k= 300
      )
      object_boxes.append(filtered_boxes)
      object_scores.append(filtered_scores)
      object_labels.extend([class_index] * filtered_boxes.size(0))

  # no object detected
  if not object_scores:
      return DetectionPredTuple(
          labels=torch.empty(0),
          scores=torch.empty(0),
          boxes=torch.empty(0, 4)
      )

  # concatenate all results
  object_scores = torch.cat(object_scores)
  object_boxes = torch.cat(object_boxes)
  object_labels = torch.tensor(object_labels)

  return DetectionPredTuple(
      labels=object_labels,
      scores=object_scores,
      boxes=object_boxes
  )

