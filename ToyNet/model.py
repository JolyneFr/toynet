from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, Dropout,
    Activation, Add, GlobalAveragePooling2D, Dense, Input
)

eps = 1e-3


def ToyBlock(x, filters, stride=1, shortcut_kind='Pool', name=None):
    """
    Base on basic_block of ResNet, 
    which includes 2 x (3 x 3) Conv2D-BN-ReLU units (BN-ReLU-Conv2D after preact).
    Preact optmize: put bn & relu before each conv2, it just works.
    To avoid overfitting, add dropout after Conv2D.
    As Recommended, dropout are added after ReLU layer, rate between 0.1-0.2.
    """

    x = BatchNormalization(axis=-1, epsilon=eps, name=name+'_1_bn')(x)
    x = Activation('relu', name=name+'_1_relu')(x)

    if shortcut_kind == 'Conv':
        shortcut = Conv2D(filters, 1, strides=stride, use_bias=False, padding='same', name=name+'_shortcut_conv')(x)
    elif shortcut_kind == 'Pool':
        shortcut = MaxPooling2D(1, strides=stride, name=name+'_shortcut_pool')(x) if stride > 1 else x

    x = Conv2D(filters, 3, strides=stride, use_bias=False, padding='same', name=name+'_1_conv')(x)
    
    x = BatchNormalization(axis=-1, epsilon=eps, name=name+'_2_bn')(x)
    x = Activation('relu', name=name+'_2_relu')(x)
    x = Conv2D(filters, 3, strides=1, use_bias=False, padding='same', name=name+'_2_conv')(x)

    x = Add(name=name+'_merge')([shortcut, x])
    return x

def ToyStack(x, filters, block_num, first_shortcut=True, name=None):
    """
    First stack of ToyNet not include conv-shortcut.
    If frist block has shortcut, its stride is 2.
    See https://arxiv.org/abs/1512.03385
    """
    if first_shortcut:
        # first block has conv-shortcut
        x = ToyBlock(x, filters, stride=2, shortcut_kind='Conv', name=name+'_block1')
    else:
        x = ToyBlock(x, filters, name=name+'_block1')
    for idx in range(2, block_num + 1):
        x = ToyBlock(x, filters, name=name+'_block'+str(idx))
    return x

def ToyNet(input_shape, block_nums, classes=10, model_name='my_toynet'):
    """
    Base on ResNet, add preact-opt and dropout,\\
    Using CIFAR10 as default dataset, so classes is 10.
    """
    inputs = Input(shape=input_shape)

    x = Conv2D(64, 3, strides=1, padding='same', use_bias=False, name='conv1_conv')(inputs)

    x = ToyStack(x, 64, block_nums[0], first_shortcut=False, name='conv2')
    x = ToyStack(x, 128, block_nums[1], name='conv3')
    x = ToyStack(x, 256, block_nums[2], name='conv4')
    x = ToyStack(x, 512, block_nums[3], name='conv5')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.5, name='pool_dropout')(x)
    outputs = Dense(classes, activation='softmax', kernel_initializer='he_normal', name='predictions')(x)

    return Model(inputs, outputs, name=model_name)


def ToyNet14(input_shape, classes=10):
    return ToyNet(input_shape, [1, 2, 2, 1], classes=classes, model_name='toynet14')


def ToyNet18(input_shape, classes=10):
    return ToyNet(input_shape, [2, 2, 2, 2], classes=classes, model_name='toynet20')


def ToyNet34(input_shape, classes=10):
    return ToyNet(input_shape, [3, 4, 6, 3], classes=classes, model_name='toynet34')






