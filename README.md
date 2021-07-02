ResNet-50-SKFAC Example
=========================

## Description

This is a Mindspore implementation of our paper [SKFAC: Training Neural Networks With Faster Kronecker-Factored Approximate Curvature](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_SKFAC_Training_Neural_Networks_With_Faster_Kronecker-Factored_Approximate_Curvature_CVPR_2021_paper.html)
<!-- 
This is an example of training ResNet-50 V1.5 with ImageNet2012 dataset by second-order optimizer [SKFAC](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_SKFAC_Training_Neural_Networks_With_Faster_Kronecker-Factored_Approximate_Curvature_CVPR_2021_paper.html). This example is based on modifications to the [THOR optimizer](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/cv/resnet_thor) on the [MindSpore framework](https://www.mindspore.cn/en). -->


<!-- TOC -->

- [ResNet-50-SKFAC Example](#resnet-50-skfac-example)
 	- [Description](#description)
 	- [Model Architecture](#model-architecture)
 	- [Dataset](#dataset)
 	- [Environment Requirements](#environment-requirements)
 		- Hardware
 		- Framework
 	- [Quick Start](#quick-start)
 	- [Script Parameters](#script-parameters)
	- [Optimize Performance](#optimize-performance)
	- [References](#references)

<!-- TOC -->



## Model Architecture

The overall network architecture of ResNet-50 is show below:[link](https://arxiv.org/pdf/1512.03385.pdf)

## Dataset

Dataset used: ImageNet2012

- Dataset size 224*224 colorful images in 1000 classes
    - Train：1,281,167 images
    - Test： 50,000 images

- Data format：jpeg
    - Note：Data will be processed in dataset.py

- Download the dataset ImageNet2012

> Unzip the ImageNet2012 dataset to any path you want and the folder structure should include train and eval dataset as follows:

```shell
    ├── ilsvrc                  # train dataset
    └── ilsvrc_eval             # infer dataset
```

## Environment Requirements

- Hardware
	- The GPU processor which supports CUDA 10.1.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)
> This example was built and tested only on MindSpore 1.1.1 (GPU) installed in Docker. It may not work on other different environments.

## Quick Start
After installing MindSpore correctly, you can use this example with shell statements below:
```shell
# run training example
python train.py --dataset_path='DATASET_PATH'
# run evaluation example
python eval.py --dataset_path='DATASET_PATH' --checkpoint_path='CHECKPOINT_PATH'

#For example:
python train.py --dataset_path='ImageNet2012/train'
python eval.py --dataset_path='ImageNet2012/val' --checkpoint_path='resnet-2_830.ckpt'
```
>You can also edit default arguments in train.py and eval.py and run them directly.

## Script Parameters

Parameters for both training and inference can be set in config.py.

- Parameters for GPU

```shell
"class_num": 1001,                # dataset class number
"batch_size": 32,                 # batch size of input tensor
"loss_scale": 1,                  # loss scale (not used in SKFAC)
"momentum": 0.9,                  # momentum of SKFAC optimizer
"weight_decay": 5e-4,             # weight decay
"epoch_size": 40,                 # only valid for training, which is always 1 for inference
"save_checkpoint": True,          # whether save checkpoint or not
"save_checkpoint_epochs": 1,      # the epoch interval between two checkpoints. By default, the checkpoint will be saved every epoch
"keep_checkpoint_max": 15,        # only keep the last keep_checkpoint_max checkpoint
"save_checkpoint_path": "./",     # path to save checkpoint relative to the executed path
"use_label_smooth": True,         # label smooth
"label_smooth_factor": 0.1,       # label smooth factor
"lr_init": 0.00625,               # learning rate init value
"lr_decay": 0.87,                 # learning rate decay rate value
"lr_end_epoch": 50,               # learning rate end epoch value
"damping_init": 0.03,             # damping init value for Fisher information matrix
"frequency": 834,                 # the step interval to update second-order information matrix
```
> Edit config_gpu if you want to change settings.

## Optimize Performance

|  Epoch   | Top 1 acc  |
|  ----  | ----  |
| 1  | 0.3707 |
| 2  | 0.4450 |
| 3  | 0.4904 |
| 4  | 0.5191 |
| 5  | 0.5411 |
| 6  | 0.5584 |
| 7  | 0.5697 |
| 8  | 0.5915 |
| 9  | 0.5994 |
| 10 | 0.6129 |
| 11 | 0.6255 |
| 12 | 0.6348 |
| 13 | 0.6466 |
| 14 | 0.6575 |
| 15 | 0.6626 |
| 16 | 0.6692 |
| 17 | 0.6798 |
| 18 | 0.6859 |
| 19 | 0.6916 |
| 20 | 0.6992 |
| 21 | 0.7021 |
| 22 | 0.7095 |
| 23 | 0.7125 |
| 24 | 0.7156 |
| 25 | 0.7209 |
| 26 | 0.7240 |
| 27 | 0.7289 |
| 28 | 0.7304 |
| 29 | 0.7351 |
| 30 | 0.7379 |
| 31 | 0.7399 |
| 32 | 0.7407 |
| 33 | 0.7446 |
| 34 | 0.7469 |
| 35 | 0.7471 |
| 36 | 0.7480 |
| 37 | 0.7500 |
| 38 | 0.7525 |
| 39 | 0.7535 |
| 40 | 0.7537 |
| 41 | 0.7549 |
| 42 | 0.7549 |
| 43 | 0.7549 |
| 44 | 0.7559 |
| 45 | 0.7565 |
| 46 | 0.7582 |
| 47 | 0.7569 |
| 48 | 0.7585 |
| 49 | 0.7595 |


## Cite

Please cite [our paper](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_SKFAC_Training_Neural_Networks_With_Faster_Kronecker-Factored_Approximate_Curvature_CVPR_2021_paper.html) (and the respective papers of the methods used) if you use this code in your own work:

```
@InProceedings{Tang_2021_CVPR,
    author    = {Tang, Zedong and Jiang, Fenlong and Gong, Maoguo and Li, Hao and Wu, Yue and Yu, Fan and Wang, Zidong and Wang, Min},
    title     = {SKFAC: Training Neural Networks With Faster Kronecker-Factored Approximate Curvature},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {13479-13487}
}
```

## References
- This example is based on modifications to [THOR optimizer](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/cv/resnet_thor).
- This example runs on [MindSpore framework](https://www.mindspore.cn/en).
