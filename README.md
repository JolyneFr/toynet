# ToyNet: Lightweight image classification network

> this CNN is optimized specifically for CIFAR10 dataset

## Environment
Make sure all your code environments match the `requirements.txt`.
```
keras==2.7.0
numpy==1.19.5
tensorflow==2.7.0
tensorflow_macos==2.7.0
```
A specific conda-env is recommended.

## Training
To train ToyNet using CIFAR10 dataset, just type
```shell
python train.py <case-name>
```
`case-name` denotes the name of current training case, which should be unique.
> Example:\
> `python train.py 1221_cos_decay`\
> **1221** shows the block_number of each ToyStack\
> **cos_decay** is the mainly changed feature of this case

## Result Visualization
After training was done, run
```shell
tensorboard --logdir training/<case_name>/log
```
Then you can see your actual training result at `http://localhost:6006/` by deafult.

## Model Introduction
ToyNet is inspired by [ResNet](https://arxiv.org/abs/1512.03385), using shortcut to achieve deeper layers.\
...\
TODO: add model image and statistics after completing my final project of elective course.

