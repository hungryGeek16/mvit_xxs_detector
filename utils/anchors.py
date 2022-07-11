import torch
from itertools import product
from typing import List
import numpy as np

def generate_anchors():
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  output_strides = [16, 32, 64, 128, 256, -1]
  dims = [20, 10, 5, 3, 2, 1]
  
  aspect_ratios = [[2, 3]] * (len(output_strides)-1)
  aspect_ratios.append([2])

  min_ratio = 0.1
  max_ratio = 1.05
  no_clipping = False

  step = [1]
  if isinstance(step, int):
      step = [step] * len(output_strides)
  if isinstance(step, List) and len(step) <= len(output_strides):
      step = step + [1] * (len(output_strides) - len(step))

  aspect_ratios = [list(set(ar)) for ar in aspect_ratios]
  output_strides_aspect_ratio = dict()
  for k, v in zip(output_strides, aspect_ratios):
      output_strides_aspect_ratio[k] = v
  output_strides_aspect_ratio = output_strides_aspect_ratio
  #print(output_strides_aspect_ratio)
  output_strides = output_strides
  anchors_dict = dict()

  num_output_strides = len(output_strides)
  num_aspect_ratios = len(aspect_ratios)

  scales = np.linspace(min_ratio, max_ratio, len(output_strides) + 1)
  sizes = dict()
  for i, s in enumerate(output_strides):
      sizes[s] = {
          "min": scales[i],
          "max": (scales[i] * scales[i + 1]) ** 0.5,
          "step": step[i],
      }
  clip = not no_clipping
  min_scale_ratio = min_ratio
  max_scale_ratio = max_ratio

  anchors = []

  for output_stride, dim in zip(output_strides, dims):
    height = width = dim
    min_size_h = sizes[output_stride]["min"]
    min_size_w = sizes[output_stride]["min"]

    max_size_h = sizes[output_stride]["max"]
    max_size_w = sizes[output_stride]["max"]
    aspect_ratio = output_strides_aspect_ratio[output_stride]

    step = max(1, sizes[output_stride]["step"])

    default_anchors_ctr = []

    start_step = max(0, step // 2)

    # Note that feature maps are in NCHW format
    for y, x in product(
        range(start_step, height, step), range(start_step, width, step)
    ):

        # [x, y, w, h] format
        cx = (x + 0.5) / width
        cy = (y + 0.5) / height

        # small box size
        default_anchors_ctr.append([cx, cy, min_size_w, min_size_h])

        # big box size
        default_anchors_ctr.append([cx, cy, max_size_w, max_size_h])

        # change h/w ratio of the small sized box based on aspect ratios
        for ratio in aspect_ratio:
            ratio = ratio ** 0.5
            default_anchors_ctr.extend(
                [
                    [cx, cy, min_size_w * ratio, min_size_h / ratio],
                    [cx, cy, min_size_w / ratio, min_size_h * ratio],
                ]
            )

    default_anchors_ctr = torch.tensor(
        default_anchors_ctr, dtype=torch.float, device=device
    )
    if clip:
        default_anchors_ctr = torch.clamp(default_anchors_ctr, min=0.0, max=1.0)
    anchors.append(default_anchors_ctr)

  anchors = torch.cat([anchors[0], anchors[1] ,anchors[2] ,anchors[3] ,anchors[4] ,anchors[5]], dim=0)
  return anchors