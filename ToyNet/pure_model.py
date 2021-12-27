from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, Dropout,
    Activation, Add, GlobalAveragePooling2D, Dense, Input
)

def BasicBlock(x, filters, stride=1, conv_sc=False):
    '''
    Basic Block of ResNet
    '''
    if conv_sc:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(x)
    else:
        shortcut = MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # ReLU after this unit
    x = Conv2D(filters, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

def BottleneckBlock(x, filters, stride=1, conv_sc=False):

    return x

def ResStack(block_fn, x, filters, block_num, first_conv):

    if first_conv:
        x = block_fn(x, filters, stride=2, conv_sc=True)
    else:
        x = block_fn(x, filters)
    for _ in range(2, block_num + 1):
        x = block_fn(x, filters)
    return x

def ResNet(block_fn, block_nums, input_shape, classes):

    inputs = Input(shape=input_shape)

    x = Conv2D(64, 3, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ResStack(block_fn, x, 64, block_nums[0], False)
    x = ResStack(block_fn, x, 128, block_nums[1], True)
    x = ResStack(block_fn, x, 256, block_nums[2], True)
    x = ResStack(block_fn, x, 512, block_nums[3], True)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(classes, activation='softmax')(x)

    return Model(inputs, outputs)

def PureResNet18(input_shape, classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], input_shape, classes)