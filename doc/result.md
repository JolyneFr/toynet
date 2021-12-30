# 实验结果

本部分主要包括预测准确率，训练过程，实验调参过程及结果分析。

## 预测准确率

## 训练过程

## 实验调参过程及结果分析

1_init 参考ResNet20实现的初始模型，阶梯形初始学习率曲线

<div align="center">
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\1_init\learning_rate.png" alt="learning_rate" style="zoom:50%;" />
</div>
<div align="center">
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\1_init\epoch_accuracy.png" alt="epoch_accuracy" style="zoom:50%;" />
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\1_init\epoch_loss.png" alt="epoch_loss" style="zoom:50%;" />
</div>

2_half_lr 发现acc曲线有拐点，尝试减半学习率

<div align="center">
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\2_half_lr\learning_rate.png" alt="learning_rate" style="zoom:50%;" />
</div>
<div align="center">
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\2_half_lr\epoch_accuracy.png" alt="epoch_accuracy" style="zoom:50%;" />
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\2_half_lr\epoch_loss.png" alt="epoch_loss" style="zoom:50%;" />
</div>

3_add_dropout 由于过拟合较为严重，向每一个TinyToyStack最后加入Dropout层，减少过拟合（可以放和上一次尝试的acc和loss的图片对比）

<div align="center">
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\3_add_dropout\learning_rate.png" alt="learning_rate" style="zoom:50%;" />
</div>
<div align="center">
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\3_add_dropout\epoch_accuracy.png" alt="epoch_accuracy" style="zoom:50%;" />
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\3_add_dropout\epoch_loss.png" alt="epoch_loss" style="zoom:50%;" />
</div>

4_preact 发现上一个case减少过拟合之后准确度较低，尝试重构模型。

原模型：(filter, block_num): (16, 3) (32, 3) (64, 3)

新模型：(64, 2) (128, 2) (256, 2) (512, 2)

并且将Conv2D->BN->ReLU结构改为了BN->ReLU->Conv2D（preact形式）

preact的灵感来源参考[preact_paper](https://arxiv.org/abs/1603.05027)

由于是新模型，去掉了dropout

<div align="center">
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\4_preact\learning_rate.png" alt="learning_rate" style="zoom:50%;" />
</div>
<div align="center">
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\4_preact\epoch_accuracy.png" alt="epoch_accuracy" style="zoom:50%;" />
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\4_preact\epoch_loss.png" alt="epoch_loss" style="zoom:50%;" />
</div>

5_e_lr 将阶梯形学习率换为指数下降学习率，使acc和loss曲线更平滑

<div align="center">
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\5_e_lr\learning_rate.png" alt="learning_rate" style="zoom:50%;" />
</div>
<div align="center">
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\5_e_lr\epoch_accuracy.png" alt="epoch_accuracy" style="zoom:50%;" />
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\5_e_lr\epoch_loss.png" alt="epoch_loss" style="zoom:50%;" />
</div>

6_cos_lr 为防止指数下降学习率收敛太快而达到局部最优，换用余弦下降学习率

<div align="center">
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\6_cos_lr\learning_rate.png" alt="learning_rate" style="zoom:50%;" />
</div>
<div align="center">
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\6_cos_lr\epoch_accuracy.png" alt="epoch_accuracy" style="zoom:50%;" />
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\6_cos_lr\epoch_loss.png" alt="epoch_loss" style="zoom:50%;" />
</div>

7_cos_less_layer 降低层数减少过拟合

<div align="center">
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\7_cos_less_layers\learning_rate.png" alt="learning_rate" style="zoom:50%;" />
</div>
<div align="center">
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\7_cos_less_layers\epoch_accuracy.png" alt="epoch_accuracy" style="zoom:50%;" />
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\7_cos_less_layers\epoch_loss.png" alt="epoch_loss" style="zoom:50%;" />
</div>

8_cos_dropout 在降低层数的基础上加入更多dropout（final）

<div align="center">
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\8_cos_dropout\learning_rate.png" alt="learning_rate" style="zoom:50%;" />
</div>
<div align="center">
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\8_cos_dropout\epoch_accuracy.png" alt="epoch_accuracy" style="zoom:50%;" />
    <img src="D:\Schoolwork\2021_autumn\Machine Learning\ToyNet\code\toynet\doc\graph\8_cos_dropout\epoch_loss.png" alt="epoch_loss" style="zoom:50%;" />
</div>

