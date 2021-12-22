import numpy as np
import tensorflow as tf
import random, os, sys
from tqdm.keras import TqdmCallback
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

batch_size = 32
epochs = 100

# enlarge dataset
datagen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_processed_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # z-score
    mean = np.mean(x_train, axis=0)
    x_train = (x_train - mean)
    x_test = (x_test - mean)

    # categorical
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def training_callbacks(case_name):
    ckpt_path = f'training/toynet20/{case_name}/ckpt'
    log_path = f'training/toynet20/{case_name}/log'

    def lr_schedule(epoch):
        lrate = 1e-3
        if epoch > 90:
            lrate *= 2e-4
        elif epoch > 80:
            lrate *= 5e-4
        elif epoch > 70:
            lrate *= 1e-3
        elif epoch > 55:
            lrate *= 1e-2
        elif epoch > 40:
            lrate *= 1e-1
        return lrate

    # checkpoint = ModelCheckpoint(filepath=ckpt_path,
    #                          monitor='accuracy',
    #                          verbose=1,
    #                          save_freq=10,
    #                          save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

    tensorboard = TensorBoard(log_dir=log_path, histogram_freq=1)

    tqdm_process = TqdmCallback(verbose=2)

    return [lr_reducer, lr_scheduler, tensorboard, tqdm_process]

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
    model.save_weights('model.h5') 
    plot_model(model)

    callbacks=training_callbacks(case_name)
    datagen.fit(x_train)
    
    model.fit(x=datagen.flow(x_train, y_train, batch_size=batch_size), 
        epochs=epochs, validation_data=(x_test, y_test), 
        verbose=0, callbacks=callbacks)

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
