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
import pathlib
import shutil

img_height = 224
img_width = 224
batch_size = 32

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(
    'flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)


new_data_dir = 'C:\\users\\Tudor\\Downloads\\monkeys'
os.path.relpath(data_dir)
new_categories=os.listdir(new_data_dir)
for sub_dir in new_categories:
  dir_to_move=os.path.join(new_data_dir,sub_dir)
  shutil.move(dir_to_move, data_dir)

def length_equalizer():
    min_len =1000
    for sub_dir in os.listdir(data_dir):
        if len(os.listdir(os.path.join(data_dir,sub_dir)))<min_len and len(os.listdir(os.path.join(data_dir,sub_dir)))>100:
            min_len = len(os.listdir(os.path.join(data_dir,sub_dir)))

    for sub_dir in os.listdir(data_dir):
        files = os.listdir(os.path.join(data_dir,sub_dir))
        if len(os.listdir(os.path.join(data_dir,sub_dir)))>min_len:
            for file in files[min_len:]:
                if file:   
                    os.remove(os.path.join(os.path.join(data_dir,sub_dir),file))
        

length_equalizer()


train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)

