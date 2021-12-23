import numpy as np
import tensorflow as tf
import random, os, sys
from keras.datasets import cifar10
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (
    LearningRateScheduler, ReduceLROnPlateau, 
    ModelCheckpoint, TensorBoard
)
from tensorflow.keras.utils import plot_model

from ToyNet.model import ToyNet18

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

# standardize test data
standardizer = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
)

def lr_schedule_fn(epoch):
    lrate = 6e-4
    if epoch > 90:
        lrate *= 2e-4
    elif epoch > 80:
        lrate *= 4e-4
    elif epoch > 70:
        lrate *= 8e-4
    elif epoch > 55:
        lrate *= 2e-3
    elif epoch > 35:
        lrate *= 3e-2
    elif epoch > 15:
        lrate *= 4e-1
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
        ReduceLROnPlateau(factor=0.15, patience=5), 
        LearningRateScheduler(lr_schedule_fn),
        TensorBoard(log_dir=log_path, histogram_freq=1)
    ]

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: python train.py <case-name>')
        quit()
    
    case_name = sys.argv[1]

    # set random seed
    set_seed(seed=2021)

    (x_train, y_train), (x_test, y_test) = load_processed_cifar10()

    model = ToyNet18((32, 32, 3), 10)

    lr_scheduler = ExponentialDecay(initial_learning_rate=1e-3, decay_steps=epochs, decay_rate=0.96)
    model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(learning_rate=6e-4), 
                    metrics=['accuracy'])
    model.summary()

    datagen.fit(x_train)
    standardizer.fit(x_train)
    callbacks=training_callbacks(case_name)
    
    model.fit(x=datagen.flow(x_train, y_train, subset='training'), 
        validation_data=datagen.flow(x_train, y_train, subset='validation'), 
        verbose=1, callbacks=callbacks, epochs=epochs)

    # save training result
    model.save_weights(f'training/toynet20/{case_name}/model.h5') 
    plot_model(model, to_file=f'training/toynet20/{case_name}/model.png', show_shapes=True)

    scores = model.evaluate(standardizer.flow(x_test, y_test, ), verbose=1)
    print('\nTest result: %.3f loss: %.3f' % (scores[1] * 100,scores[0]))
    