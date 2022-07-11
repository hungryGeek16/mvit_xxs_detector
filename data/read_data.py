import os
import torch
import torch.utils.data
import torchvision
import cv2
import numpy as np
from pycocotools.coco import COCO

class CustomCOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image

        img = cv2.imread(os.path.join(self.root, path),cv2.IMREAD_COLOR)
        
        height, width, channels = img.shape
        img = cv2.resize(img, dsize=(320, 320))
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img = torch.div(img, 255.0)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        num_objs = len(coco_annotation)
        
        boxes = []
        labels = []
        for i in range(num_objs):
          if not(None in coco_annotation[i]['bbox']):
            xmin = (coco_annotation[i]['bbox'][0]) / width
            ymin = (coco_annotation[i]['bbox'][1]) / height
            xmax = (coco_annotation[i]['bbox'][0] + coco_annotation[i]['bbox'][2]) / width
            ymax = (coco_annotation[i]['bbox'][1] + coco_annotation[i]['bbox'][3]) / height
            category = int(coco_annotation[i]['category_id'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(category)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        #boxes = boxes[keep]
        
        labels = torch.as_tensor(labels, dtype=torch.float32)
        #labels = labels[keep]
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        
        return img, my_annotation

    def __len__(self):
        return len(self.ids)