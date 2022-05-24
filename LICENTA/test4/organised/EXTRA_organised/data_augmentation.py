import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import preprocessing
import pathlib
img_height = 200
img_width = 200

def augmentation_zoom_and_rotation():
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomFlip("vertical",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )
    return data_augmentation

def augmentation_color():
    data_augmentation = keras.Sequential(
        [
          layers.RandomContrast(0.1)
        ]
    )
    return data_augmentation

def augmentation_translation():
    data_augmentation = keras.Sequential(
        [
        layers.RandomTranslation(0, 0.2),
        layers.RandomTranslation(0.2, 0),
        ]
    )
    return data_augmentation

