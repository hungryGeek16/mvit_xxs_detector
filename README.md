# MobileVit XXS Inference instructions:

* Required Python Version: >= 3.7

* Installation of necessary packages can be done by running the command given below:

```bash
python install -r requirement.txt
```

* After completing the installation step, one could inference the model on images by executing the command given below:

```bash
python infer.py --img_path /path/to/your/image --model_path /path/to/the/model --classes number of classes
```

* The original model that was trained on 81 coco classes is present in repository itself by the name ```mvit_det_81.pt```, to execute this model, we can simply ignore **model_path** and **classes** arguments, since they are passed to the program as default parameters. Hence the final command becomes:

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

## Credits:

### Code credits: [ml_cvnets](https://github.com/apple/ml-cvnets/tree/cvnets-v0.1), [chinhsuanwu](https://github.com/chinhsuanwu/mobilevit-pytorch), [sgrvinod](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)
### Paper: [Mobile_VIT](https://arxiv.org/abs/2110.02178)
