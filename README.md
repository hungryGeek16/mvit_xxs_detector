# MobileVit XXS Inference instructions:

* Required Python Version: >= 3.7

* Installation of necessary packages can be done by running the command given below:

```bash
python install -r requirements.txt
```

* After completing the installation step, one could inference the model on images by executing the command given below:

```bash
python infer.py --img_path /path/to/your/image --model_path /path/to/the/model --classes number of classes
```

* The original model that was trained on 81 coco classes is present in repository itself by the name ```mvit_og.pt```, to execute this model, we can simply ignore **model_path** and **classes** arguments, since they are passed to the program as default parameters. Hence the final command becomes:

```bash
python infer.py --img_path /path/to/your/image 
```
* Image with the name detected.jpg would be saved in the current folder.
* For inferencing the model on batch of images, one could use the batch argument, example is given below:

```
python infer.py --batch /path/to/your/image_folder
```
* All detected images would be saved in the folder named ```dets_full``` in the same directory.

* ```Some detected results:```
<p align="center">
  <img src="dets.png" width = 1000>
</p>

# MobileVit XXS Training instructions:

* To transfer learn MobileVIT detector on a dataset then it must be in COCO format.
* Dataset strutcture should be in the format as shown below:
```bash
--Dataset
     |---> train
     |---> valid
```
* To train the your dataset, please follow the command given below:
```bash
python train.py --path_to_images path/to/dataset --lr 0.01 --epochs 10 --classes no_of_classes_present --batch_size 32 --path_test_annotations path/to/test/annotations.json --path_train_annotations path/to/test/annotations.json --model_path mvit_og.pt
```

* After training, the file will output two files: a. loss_trends.jpg: training loss graph, b. mvit.pt: learned model file
* The mvit.pt file can be inferred using the infer.py file, you just have to pass model path, images or bath path and number of classes it detects.

```bash
python infer.py --model_path path/to/model --classes no_of_classes --img_path /path/to/image/
```

## Citation

```bibtex
@article{mehta2021mobilevit,
  title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2110.02178},
  year={2021}
}
```

### Code credits: [ml_cvnets](https://github.com/apple/ml-cvnets/tree/cvnets-v0.1), [chinhsuanwu](https://github.com/chinhsuanwu/mobilevit-pytorch)
