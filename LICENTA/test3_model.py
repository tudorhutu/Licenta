import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard



# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)

# pickle_in = open("y.pickle","rb")
# y = pickle.load(pickle_in)

X=np.load("features.npy")
y=np.load("labels.npy")

#X = X/255.0

print(X.shape,y.shape)
# X=np.array(X)
# y=np.array(y)
dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(2))  # 2 outputs
            model.add(Activation('softmax'))  # Turns it into a probability distribution

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss="sparse_categorical_crossentropy",, metrics=['accuracy'] optimizer='adam')

            model.fit(X, y,
                      batch_size=64,
                      epochs=12,
                      validation_split=0.3,
                      callbacks=[tensorboard])
model.save('cat_and_dog_model')