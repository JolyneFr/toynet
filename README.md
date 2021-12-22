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
`case-name` denotes the name of current training case, which should be unique.\
It may take 1+ hour to complete training.

## Result Visualization
After training was done, run
```shell
tensorboard --logdir training/toynet20/<case_name>/log
```
Then you can see your actual training result at `http://localhost:6006/` by deafult.
