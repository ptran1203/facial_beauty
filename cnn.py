import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json

from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras import backend as K
# from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import ImageDataGenerator

import const
import data_processing as dp
import numpy as np

# Alexnet

class AlexNet:
    def __init__(self):
        model = Sequential()
        model.add(Conv2D(kernel_size=(11, 11),
                        activation='relu',
                        strides=(4, 4),
                        filters=96,
                        padding='valid',
                        input_shape=(227,227,3)))
        model.add(MaxPooling2D(pool_size=(2, 2),
                            strides=(2, 2),
                            padding='valid'))
        model.add(Conv2D(filters=256,
                        kernel_size=(11, 11),
                        strides=(1, 1),
                        padding='valid',
                        activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),
                            strides=(2, 2),
                            padding='valid'))
        model.add(Conv2D(filters=384,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='valid',
                        activation='relu'))
        model.add(Conv2D(filters=384,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='valid',
                        activation='relu'))
        model.add(Conv2D(filters=256,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='valid',
                        activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),
                            strides=(2, 2),
                            padding='valid'))
        model.add(Flatten())
        # fully connected
        model.add(Dense(9216 ,input_shape=(224, 224, 3), activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.4))
        # output
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_absolute_percentage_error',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model

if __name__ == "__main__":
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_df, test_df = dp.split(dp.dataframe(), 0.7)

    train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory='dataset/images',
            x_col="filename",
            y_col="score",
            target_size=(227, 227),
            class_mode='sparse')


    validation_generator = test_datagen.flow_from_dataframe(
            dataframe=test_df,
            directory='dataset/images',
            x_col="filename",
            y_col="score",
            target_size=(227, 227),
            class_mode='sparse')
    alex_net = AlexNet()
    model = alex_net.model
    model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800)
