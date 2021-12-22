import numpy as np
import tensorflow as tf
import random, os, sys
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizer_v2.adam import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (
    LearningRateScheduler, ReduceLROnPlateau, 
    ModelCheckpoint, TensorBoard
)
from tensorflow.keras.utils import plot_model

from ToyNet.model import ToyNet20

epochs = 100

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

def lr_schedule(epoch):
    lrate = 5e-4
    if epoch > 90:
        lrate *= 2e-4
    elif epoch > 80:
        lrate *= 5e-4
    elif epoch > 65:
        lrate *= 1e-3
    elif epoch > 50:
        lrate *= 1e-2
    elif epoch > 30:
        lrate *= 1e-1
    return lrate

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_processed_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # categorical
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def training_callbacks(case_name):
    ckpt_path = f'training/toynet20/{case_name}/ckpt'
    log_path = f'training/toynet20/{case_name}/log'

    # checkpoint = ModelCheckpoint(filepath=ckpt_path,
    #                          monitor='accuracy',
    #                          verbose=1,
    #                          save_freq=10,
    #                          save_best_only=True)

    return [
        ReduceLROnPlateau(patience=5, min_lr=0.5e-6), 
        LearningRateScheduler(lr_schedule), 
        TensorBoard(log_dir=log_path, histogram_freq=1),
    ]

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: python train.py <case-name>')
        quit()
    
    case_name = sys.argv[1]

    # set random seed
    set_seed(seed=2021)

    (x_train, y_train), (x_test, y_test) = load_processed_cifar10()

    model = ToyNet20((32, 32, 3), 10)
    model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    model.save_weights(f'training/toynet20/{case_name}/model.h5') 
    plot_model(model)

    callbacks=training_callbacks(case_name)
    datagen.fit(x_train)
    
    model.fit(x=datagen.flow(x_train, y_train, subset='training'), 
        validation_data=datagen.flow(x_train, y_train, subset='validation'), 
        verbose=1, callbacks=callbacks, epochs=epochs)

    scores = model.evaluate(datagen.flow(x_test, y_test), verbose=1)
    print('\nTest result: %.3f loss: %.3f' % (scores[1] * 100,scores[0]))
