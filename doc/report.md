# CNN Image Classifier: ToyNet
Team Members: 杨镇宇, 潘浩楠, 李卓龙

本项目主要致力于探索各种对模型的修改和训练方法的调整如何影响CNN。为此，我们构建了一个基于`ResNet`的残差神经网络：`ToyNet`，在之上对*CIFAR10*数据集进行训练和测试。

<img src="./model_images/toynet.png" alt="keras" style="zoom:55%;" />

`ToyNet`这个轻量级的残差神经网络允许我们对其进行模块化的修改和超参数调整，**Keras**提供的简明API也为整个实验过程带来方便，这些会在之后的章节详细介绍。我们先从数据预处理开始。

## 数据处理

这是独立于模型设计与训练之外的步骤，是图像识别的第一步。

### 数据增强（Data Augmentation）

> *CIFAR10*的数据量为$32 * 32 * 3 * 60000$，其中50000张训练集，10000张测试集。
> 较小的尺寸加上较少的图片数，必须进行**数据增强**才能让数据量达到有利于训练的水准。

在使用CNN进行图像识别时，数据增强的有以下作用：

- 增大dataset，提高模型的泛化能力，防止过拟合。
- 增加噪声干扰数据，提升模型的鲁棒性
- 通过增强从原始数据集中提取出更多的信息，使得增强后的数据集代表更为全面的数据集合，进而缩小训练集和验证集之间的差距。

项目中采用了**Keras**提供的`ImageDataGenerator`类，通过这个类可以通过实时数据增强生成张量图像数据批次，并且可以循环迭代。`ImageDateGenerator()`是一个图片生成器，负责生成一个批次一个批次的图片，以生成器的形式给模型训练，同时对每个批次的训练图片，适时地进行数据增强处理。

实际的数据增强代码如下：其中的`datagen`负责增强并标准化**训练集**，`standardizer`则把同样的标准化参数应用于**测试集**，以保证训练和测试集处理过程的一致性。

```python
from keras.preprocessing.image import ImageDataGenerator
# enlarge dataset
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    featurewise_center=True,
    featurewise_std_normalization=True,
    validation_split=0.2
)

# standardize test data
standardizer = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
)
```

参数设计与解释：

+ `rotation_range`：随机旋转的度数范围。
+ `width_shift_range / height_shift_range`：随机水平、垂直平移范围，单位为图像宽度。
+ `horizontal_flip`：布尔值，设为 True 表示随机对图片执行水平翻转操作。
+ `featurewise_center`：布尔值，设为 True 表示使数据集去中心化（使得其均值为0）。
+ `featurewise_std_normalization`：布尔值，设为 True 表示将输入的每个样本除以其自身的标准差。两个 featurewise 函数从数据集整体上对每张图片进行了标准化（*Z-Score*）处理。
+ `validation_split`：将训练集的一部分单独分出作为验证集，之后详细说明

一张图片经过以上的**随机旋转、随机反转、随机平移**可能得到如下结果：

<img src="./data_images/datagen.png" alt="cifar10_datagen" style="zoom:33%;" />

然后可以将两个`ImageDataGenerator`对象作用于`x_train`，以便之后完成标准化：

```python
# enlarged training set, and standardize whole dataset. 
datagen.fit(x_train)
standardizer.fit(x_train)
```

###  标签向量化处理

多类分类问题与二类分类问题类似，需要将类别变量（*categorical function*）的输出标签转化为数值变量。

在多分类问题中我们将转化为虚拟变量（*dummy variable*）：即用`one-hot-encoding`方法将输出标签的向量转化为只在出现对应标签的那一列为1，其余为0的布尔矩阵。

```python
from keras.datasets import cifar10
from keras.utils import np_utils

def load_processed_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # make label categorical
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)
```

对网络的结构的影响：

```python
# 同时在网络中，损失函数对应选择 categorical_crossentropy
model.compile(loss='categorical_crossentropy', ...)
```

## 系统设计

关于**ToyNet**的完整实现，可以到[GitHub仓库](https://github.com/JolyneFr/toynet)查看最新的版本（不一定是acc最高的）。

### 模型设计

> Slides指导我们要站在巨人的肩膀上，所以我们选了个100,000+引用的肩膀。

#### 概述

**ToyNet**的主要灵感来源于**ResNet**([Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385))，其思路是通过加入shortcut 路径，使更深层的网络层能够学习到上层的原始函数。单元block的实现上参考了BasicBlock，没有参考BottleneckBlock 的原因是目标数据集较为简单，没必要使用为更深层神经网络优化的 BottleneckBlock。

由于测试准确率还有上升空间，在阅读了关于**ResNet**的网络层优化的另一篇文章([Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027))后，采用了其中提到的 pre-activation 优化。~~事实上该优化对于层数更深的残差网络（比如ResNet152）作用才比较明显，这里只是尝试一下。~~

为了解决训练过程中Loss曲线显示的过拟合问题，在全连接层上加入了`Dropout`，提高了**ToyNet**在不同数据之间的泛化能力。（过程中也尝试了在卷积层加dropout的无理操作）

针对*CIFAR10*数据集，我们还额外调整了网络层数，有**ToyNet18**和**ToyNet14**两个版本。

#### 具体实现

网络中的最基本模块是**ToyBlock**，根据shortcut是否包含卷积层分为两类。其中shortcut包含卷积层的**ToyBlock**用于调整通道数，以便进行与主路进行相加运算。相比于普通CNN中的 *Conv->BN->ReLU*结构，pre-activation优化后的结构为*BN->ReLU->Conv*。

<img src="./model_images/toyblock.png" alt="toyblock" style="zoom:40%;" />

典型的CNN（比如VGG、ResNet、DenseNet、EfficientNet...）都是让通道数随着网络的加深而逐渐翻倍的，因此**ToyNet**也效仿了这样的做法，将卷积层通道数相同的**ToyBlock**由统一的结构**ToyStack**聚合在一起管理。

在网络最后，相对于传统的`AveragePooling + Flatten`，使用了论文 [Network In Network](https://arxiv.org/abs/1312.4400) 中提出的`GlobalAveragePooling`结构，被认为能够提供更好的过拟合抗性。

下图左侧描述了**ToyStack**的结构，右侧是**ToyStack18**的全局结构。

<img src="./model_images/toystack.png" alt="toystack_net" style="zoom:45%;" />

**ToyStack14**的block_num配置调整为[1, 2, 2, 1]。

### 训练方法

**ToyNet**的训练中有如下要点，训练细节参考源代码中的`train.py`

#### 数据增强、划分与标准化

数据增强在**第一部分：数据处理**中已经提到过了，这里主要介绍数据划分。

原始的*CIFAR10*数据集只分为了`train`和`test`两个集合，但是从语义层面考虑，CNN模型在训练时不应得知任何有关测试集的信息，因此我们单独从测试集中分割出$20\%$作为验证集（*validation*），用以在模型训练过程中验证准确率和Loss大小，从而观察模型的泛化能力。测试集只在模型训练完后的`evaluate`阶段使用。

```python
datagen = ImageDataGenerator(..., validation_split=0.2)
# split to 2 subset: training & validation
traing_data = datagen.flow(x_train, y_train, subset='training')
validation_data = datagen.flow(x_train, y_train, subset='validation')
...
# only use test_data when evaluating model
scores = model.evaluate(standardizer.flow(x_test, y_test), verbose=1)
```

#### 条件性学习率衰减

当训练过程中某一指标（这里使用`val_loss`）一直处于平台期时，以某一倍数降低学习率，从而优化训练过程，据说在实践中很有用。

```python
ReduceLROnPlateau(factor=0.15, monitor='val_loss', patience=5)
```

下面是一段指数型(*ExponentialDecay*)学习率/epoch曲线，中间下降的点发生了条件性学习率衰减。

<img src="./model_images/lr_drop.png" alt="lr_drop" style="zoom:43%;" />

#### 余弦衰减学习率

在正常训练过程中采用了余弦学习率。相比于指数衰减学习率，余弦衰减的速度更慢，因此能在前期更好的寻找全局最优而非直接进入局部最优。

```python
LearningRateScheduler(CosineDecay(initial_learning_rate=1e-3))
```
除此之外，我们还尝试了其他许多种学习率衰减策略（指数衰减、含重启的余弦退火学习率等）。这些会在**调参实验和结果分析**中详述

#### 学习策略

选用了经验上效果最好的`Adam Optimizer` ([Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980))。这是一种可以替代传统随机梯度下降过程的一阶优化算法，它能基于训练数据迭代地更新神经网络权重，通过计算梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应性学习率。

> 总而言之就是效果很好，应该。

#### Loss函数与指标

选用了`categorical_crossentropy`作为Loss函数，`accuracy`作为训练时的指标。这部分的设计比较一般化，项目结束后有时间时可能会尝试`focal_loss`的效果，但是期末考试实在复习不完了。

## 实验结果

事实上，我们花在训练+调参的时间比构建模型多得多。

### 预测准确率

由于**ToyNet**基本上脱胎于残差神经网络，因此将**ResNet**在*CIFAR10*数据集上的结果作为实验的baseline是一件很自然的事情。为此，我们实现了一个符合[原始论文](https://arxiv.org/abs/1512.03385)参数与设计的轻量级残差神经网络：**PureResNet**。

**PureResNet**的实现在`toynet/pure_model`中，有**PureResNet18, 34, ..., 152**多种网络层数的版本。为了适应*CIFAR10*的低数据量，选用18层的网络作为baseline，训练100个epoch后测得：

```shell
Test accuracy: 92.440% Test Loss: 0.438
```

![baseline](graph/0_baseline/output.png)

以此为依据，可以预测**ToyNet**在*CIFAR10*上的表现：

+ 为防止过拟合，采用的**ToyNet**层数小于18层，因此最终的准确率可能不会优于**PureResNet**
+ 前提同上，**ToyNet**由此能拥有更强的泛化能力，我们有理由期待Test Loss值低于baseline

### 训练过程

