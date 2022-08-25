import torch, torchvision
import torch.nn as nn
import cv2
import torch.optim as optim
import torchvision.models as models
import time, os, copy
import numpy as np
import torch.backends.cudnn as cudnn
import argparse
import math
import matplotlib.pyplot as plt
from collections import OrderedDict

from model.mobilevit_xxs import *
from utils.box_utils import *
from utils.anchors import generate_anchors
from utils.pred_utils import *
from utils.ssd_utils import *
from utils.multibox_loss import *
from utils.misc import *
from data.read_data import CustomCOCODataset

  

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, default="mvit_og.pt", help="path to input image")
parser.add_argument("--classes", type=int, default=80, help="Total classes that are required")
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
parser.add_argument("--epochs", type=int, default=30, help="Total number of epochs")
parser.add_argument("--lr", type=float, default = 1e-3, help ="learning rate")
parser.add_argument("--grad_clip", type=bool, default= False, help= "clip gradients")
parser.add_argument("--cudnn_benchmark", type=bool, default= False, help= "CUDA SUPPORT")
parser.add_argument("--print_freq", type=int, default = 1, help ="Print loss value frequency")
parser.add_argument("--path_train_annotations", type=str, default = "annotations_train.json", help ="Path to train coco jsons folder")
parser.add_argument("--path_test_annotations", type=str, default = "annotations_test.json", help ="Path to test coco jsons folder")
parser.add_argument("--path_to_images", type=str, default = "images/", help ="Path to train-test folder directory")
args = parser.parse_args()


# path to your own data and coco file
data_dir = args.path_to_images
train_coco = args.path_train_annotations
test_coco = args.path_test_annotations

train_dir = os.path.join(data_dir,"train")
test_dir = os.path.join(data_dir,"valid")

# create own Dataset
train = CustomCOCODataset(root=train_dir,
                          annotation=train_coco)

test = CustomCOCODataset(root=test_dir,
                          annotation=test_coco)

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

# own DataLoader
train_data_loader = torch.utils.data.DataLoader(train,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=2,
                                          collate_fn=collate_fn)

test_data_loader = torch.utils.data.DataLoader(test,
                                          batch_size=16,
                                          shuffle=True,
                                          num_workers=2,
                                          collate_fn=collate_fn)


# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Model intializing
classes = args.classes + 1
detector = MobileDetector_Tr(classes)
t = torch.load(args.model_path, map_location=device)
layers = list(t.keys())

for layer in layers:
  if layer.count("class") == 1:
    del t[layer]
  else:
    k = t[layer]
    del t[layer]
    t[layer.replace("model.","")] = k

detector.model.load_state_dict(t)
for p in detector.model.parameters():
    p.requires_grad = False  

# Anchors
anchors = generate_anchors()
# Learning parameters 
params = [p for p in detector.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 7, 8, 9, 10], gamma=0.1)

criterion = multibox_loss().to(device)
detector = detector.to(device)

train_size = len(train_data_loader.dataset)
test_size = len(test_data_loader.dataset)

iter = math.ceil((train_size) / 32) 
iter_test = math.ceil((test_size) / 16)

loop = 0
loss_list = []

#Plot graphs of the loss at every iteration
def plot_loss(loss):
  x = [*range(1,len(loss)+1)]
  plt.plot(x, loss)
  plt.xlabel('Iterations')
  plt.ylabel('Loss')
  plt.title('Loss vs Iterations')
  plt.savefig("loss_trend.jpg")

# DataLoader is iterable over Dataset
for epoch in range(1,args.epochs+1):
  detector.train()
  batch_time = AverageMeter()  # forward prop. + back prop. time
  data_time = AverageMeter()  # data loading time
  losses = AverageMeter()  # loss
  start = time.time()
  i = 1
  if epoch % 5 == 0:
    phase = "test"
  else:
    phase = "train"
   
  for imgs, annotations in train_data_loader:
    imgs = torch.stack(imgs, dim=0).to(device)
    z = detector(imgs)
    x_hat, y_hat = z[0].squeeze(0), z[1].squeeze(0)
    x, y = match(anchors, annotations)
    x , y = x.squeeze(0), y.squeeze(0)
    
    loss = criterion(x_hat, y_hat, x, y)  # scalar
    # Backward prop.
    optimizer.zero_grad()
    loss.backward()

    if args.grad_clip is not False:
      clip_gradient(optimizer, grad_clip)

    optimizer.step()

    losses.update(loss.item(), imgs.size(0))
    batch_time.update(time.time() - start)

    start = time.time()
    # Print status
    if i % args.print_freq == 0:
        print('Epoch: [{0}][{1}/{2}]\t'
              'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, iter,
                                                              batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    
    loop+=1
    i+=1
    loss_list.append(loss.item())
  
  if phase == "test":
    print('Validating...........')
    losses = AverageMeter()
    print("Validation after epoch:",epoch)
    for imgs, annotations in test_data_loader:
      imgs = torch.stack(imgs, dim=0).to(device)
      z = detector(imgs)
      x_hat, y_hat = z[0].squeeze(0), z[1].squeeze(0)
      x, y = match(anchors, annotations)
      x , y = x.squeeze(0), y.squeeze(0)
      loss = criterion(x_hat, y_hat, x, y)  # scalar
      losses.update(loss.item(), imgs.size(0))

    print("Validation loss:", losses.avg)
#Plot loss graph
plot_loss(loss_list)

# Save Model    
torch.save(detector.state_dict(),"mvit.pt")
