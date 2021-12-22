from tensorflow.keras import Model
from tensorflow.python.keras.layers.core import Activation
from tensorflow.keras.layers import (
    ZeroPadding2D, Conv2D, MaxPooling2D, BatchNormalization, Flatten,
    Activation, Add, GlobalAveragePooling2D, Dense, Input
)

eps = 1e-3

def ToyBlock(x, filters, stride=1, kernal_size=3, shortcut_kind='Pool', name=None):
    # preact
    preact = BatchNormalization(axis=-1, epsilon=eps, name=name+'_preact_bn')(x)
    preact = Activation('relu', name=name+'_preact_relu')(preact)

    # 2 kinds of short-cut
    if shortcut_kind == 'Conv':
        shortcut = Conv2D(4 * filters, 1, strides=stride, name=name+'_shortcut_conv')(preact)
    elif shortcut_kind == 'Pool':
        shortcut = MaxPooling2D(1, strides=stride, name=name+'_shortcut_pool')(x) if stride > 1 else x

    x = Conv2D(filters, 1, strides=1, use_bias=False, name=name+'_1_conv')(preact)
    x = BatchNormalization(axis=-1, epsilon=eps, name=name+'_1_bn')(x)
    x = Activation('relu', name=name+'_1_relu')(x)

    # Add zero-padding to make shape same
    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=name+'_2_pad')(x)
    x = Conv2D(filters, kernal_size, strides=stride, use_bias=False, name=name+'_2_conv')(x)
    x = BatchNormalization(axis=-1, epsilon=eps, name=name+'_2_bn')(x)
    x = Activation('relu', name=name+'_2_relu')(x)

    x = Conv2D(4 * filters, 1, name=name+'_3_conv')(x)
    x = Add(name=name+'_merge')([shortcut, x])
    return x

def ToyStack(x, filters, block_num, final_stride=2, name=None):
    x = ToyBlock(x, filters, shortcut_kind='Conv', name=name+'_block1')
    for idx in range(2, block_num):
        x = ToyBlock(x, filters, name=name+'_block'+str(idx))
    x = ToyBlock(x, filters, stride=final_stride, name=name+'_block'+str(block_num))
    return x
    

def ToyNet50(input_shape, classes=10, model_name='my_toynet'):
    """
    Base on ResNet50_v2 
    """
    inputs = Input(shape=input_shape)

    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(inputs)
    x = Conv2D(64, 7, strides=2, name='conv1_conv')(x)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = ToyStack(x, 64, 3, name='conv2')
    x = ToyStack(x, 128, 4, name='conv3')
    x = ToyStack(x, 256, 6, name='conv4')
    x = ToyStack(x, 512, 3, final_stride=1, name='conv5')

    x = BatchNormalization(-1, epsilon=eps, name='final_bn')(x)
    x = Activation('relu', name='final_relu')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = Dense(classes, activation='softmax', name='predictions')(x)

    return Model(inputs, outputs, name=model_name)




def TinyToyBlock(x, filters, stride=1, kernel_size=3, shortcut_kind='Pool', name=None):
    """
    Base on ResNet20_v1, 
    which includes 2 x (3 x 3) Conv2D-BN-ReLU units.
    Consider matching the output between shortcut & main
    """
    
    if shortcut_kind == 'Conv':
        shortcut = Conv2D(filters, 1, strides=stride, padding='same', name=name+'_shortcut_conv')(x)
    elif shortcut_kind == 'Pool':
        shortcut = MaxPooling2D(1, strides=stride, name=name+'_shortcut_pool')(x) if stride > 1 else x

    x = Conv2D(filters, kernel_size, strides=stride, padding='same', name=name+'_1_conv')(x)
    x = BatchNormalization(axis=-1, epsilon=eps, name=name+'_1_bn')(x)
    x = Activation('relu', name=name+'_1_relu')(x)

    # ReLU after this unit
    x = Conv2D(filters, kernel_size, strides=1, padding='same', name=name+'_2_conv')(x)
    x = BatchNormalization(axis=-1, epsilon=eps, name=name+'_2_bn')(x)

    x = Add(name=name+'_merge')([shortcut, x])
    x = Activation('relu', name=name+'_out_relu')(x)
    return x

def TinyToyStack(x, filters, block_num, first_shortcut=True, name=None):
    """
    First stack of ToyNet20 not include conv-shortcut
    """
    if first_shortcut:
        # first block has conv-shortcut
        x = TinyToyBlock(x, filters, stride=2, shortcut_kind='Conv', name=name+'_block1')
    else:
        x = TinyToyBlock(x, filters, name=name+'_block1')
    for idx in range(2, block_num + 1):
        x = TinyToyBlock(x, filters, name=name+'_block'+str(idx))
    return x

def ToyNet20(input_shape, classes=10, model_name='my_toynet'):
    """
    Base on ResNet20_v1
    """
    inputs = Input(shape=input_shape)

    x = Conv2D(16, 3, strides=1, padding='same', name='conv1_conv')(inputs)
    x = BatchNormalization(axis=-1, epsilon=eps, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)

    x = TinyToyStack(x, 16, 3, first_shortcut=False, name='conv2')
    x = TinyToyStack(x, 32, 3, name='conv3')
    x = TinyToyStack(x, 64, 3, name='conv4')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = Dense(classes, activation='softmax', kernel_initializer='he_normal', name='predictions')(x)

    return Model(inputs, outputs, name=model_name)





