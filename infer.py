# Required imports
import torch
from torch import nn
import pandas as pd
import numpy as np
import argparse
import cv2
import copy
import os
import sys

from model.mobilevit_xxs import MobileDetector
from utils.box_utils import convert_to_boxes
from utils.anchors import generate_anchors
from utils.pred_utils import predict

# Number of classes being detected
object_names =    ['background',
                   'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                   'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush'
                   ]

# initializing varibales for further use
LABEL_COLOR = [255, 255, 255]
TEXT_THICKNESS = 1
RECT_BORDER_THICKNESS = 2
FONT_SIZE = cv2.FONT_HERSHEY_PLAIN

def preprocess_image(image_path, device):
  print("Processing image:",image_path)
  input_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
  if input_img is None:
    sys.exit("Invalid Image found. Exiting")
    
  im_copy = copy.deepcopy(input_img)
  orig_h, orig_w = im_copy.shape[:2]
  # Resize the image to the resolution that detector supports
  res_h, res_w = (320, 320)
  input_img = cv2.resize(input_img, (res_h, res_w), interpolation=cv2.INTER_LINEAR)

  # HWC --> CHW --> normalize
  input_img = np.transpose(input_img, (2, 0, 1))
  input_img = (
      torch.div(
          torch.from_numpy(input_img).float(), # convert to float tensor
          255.0 # convert from [0, 255] to [0, 1]
      ).unsqueeze(dim=0) # add a dummy batch dimension
  )
  input_img = input_img.to(device)
  return input_img, im_copy, orig_h, orig_w,


def infer(input_img, im_copy, model, anchors, orig_h, orig_w):
  # Inference image
  with torch.no_grad():
    x = model(input_img)

  # Collect box-offsets and scores per box 
  boxes, scores = x[0], x[1]

  # Apply offsets to anchor boxes
  co_ord = convert_to_boxes(boxes, anchors)
  scores = nn.Softmax(dim=-1)(scores)

  # Post-process box co-ordinates
  tupes = predict(co_ord,scores)
  boxes = tupes.boxes.cpu().numpy()
  labels = tupes.labels.cpu().numpy()
  scores = tupes.scores.cpu().numpy()

  # Resize co-ordinates to original size
  boxes[..., 0::2] = boxes[..., 0::2] * orig_w
  boxes[..., 1::2] = boxes[..., 1::2] * orig_h
  boxes[..., 0::2] = np.clip(a_min=0, a_max=orig_w, a=boxes[..., 0::2])
  boxes[..., 1::2] = np.clip(a_min=0, a_max=orig_h, a=boxes[..., 1::2])

  # If scores found, plot boxes in the image
  if len(scores) != 0:
    for label, score, coords in zip(labels, scores, boxes):
        c1 = (int(coords[0]), int(coords[1]))
        c2 = (int(coords[2]), int(coords[3]))

        cv2.rectangle(im_copy, c1, c2, (255, 0, 0), thickness=1)
        if object_names is not None:
          label_text = "{label}: {score:.2f}".format(label=object_names[label], score=score)

          t_size = cv2.getTextSize(label_text, FONT_SIZE, 1, TEXT_THICKNESS)[0]
          new_c2 = np.uint(c1[0] + t_size[0] + 3), np.uint(c1[1] + t_size[1] + 4)
          cv2.rectangle(im_copy, c1, new_c2, (0, 0, 255), -1)

          cv2.putText(im_copy, label_text, (np.uint(c1[0]), np.uint(c1[1] + t_size[1] + 4)), FONT_SIZE, 1, LABEL_COLOR, TEXT_THICKNESS)
    print("Detected some objects")
    return im_copy
  else:
    print("No detections found in this image")
    return np.array([])


# Parsing command line arguements
parser = argparse.ArgumentParser()

parser.add_argument("--img_path", type=str, default="", help="path to input image")
parser.add_argument("--batch", type=str, default="", help="path to batch of images")
parser.add_argument("--model_path", type=str, default="mvit_det_81.pt", help="path to input image")
parser.add_argument("--classes", type=int, default=80, help="Total classes that are required")

args = parser.parse_args()

if args.img_path == "" and args.batch == "":
  sys.exit("No path to image/batch of images is given. Exiting...")

if args.img_path != "" and args.batch != "":
  sys.exit("Path to image and batch, both are given, ambiguous inputs . Exiting...")


# Load weights in the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = args.classes + 1 # Adding 1 for the backgorunf class
model = MobileDetector(classes) # Intialize mobile_vit_xxs with 81 classes
t = torch.load(args.model_path, map_location=device) 
model.load_state_dict(t)
model.eval()
model = model.to(device)
print("Weights Initialized!!!")
print('<<<<<<< Model loaded >>>>>>>')
print(model)
print('<<<<<<<>>>>>>>')

#Generate anchors
anchors = generate_anchors()

if args.batch != "":
  if os.path.exists("dets_full"):
    print("Image saving folder exists")
  else:
    os.mkdir("dets_full")

  for i in os.listdir(args.batch):
    if i.count(".jpg") == 0:
      continue

    path = os.path.join(args.batch, i)
    image, im_copy, orig_h, orig_w = preprocess_image(path, device)
    final_img = infer(image, im_copy, model, anchors, orig_h, orig_w)
    if final_img.size != 0:
      cv2.imwrite("dets_full/detect_"+i, final_img)

else:
  image, im_copy, orig_h, orig_w = preprocess_image(args.img_path, device)
  final_img = infer(image, im_copy, model, anchors, orig_h, orig_w)
  if final_img.size != 0:
    cv2.imwrite("detected.jpg", final_img)
