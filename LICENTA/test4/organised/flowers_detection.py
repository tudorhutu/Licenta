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
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
monkey_data_dir='C:\\users\\Tudor\\Downloads\\monkeys'

batch_size = 32
img_height = 200
img_width = 200

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

monkey_train_ds = tf.keras.utils.image_dataset_from_directory(
  monkey_data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
monkey_val_ds = tf.keras.utils.image_dataset_from_directory(
  monkey_data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

monkey_class_names = monkey_train_ds.class_names

model = tf.keras.models.load_model("./monkeys_model_saved")
sunflower_url = "https://i.guim.co.uk/img/media/3a9f56c9aeed31f4b87aae361bca1f37dda723ff/247_0_4857_2916/master/4857.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=f0386f1da4e055665fec47acc62ad30a"
sunflower_path = tf.keras.utils.get_file('babuin', origin=sunflower_url)

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(monkey_class_names[np.argmax(score)], 100 * np.max(score))
)

sunflower_url = "https://upload.wikimedia.org/wikipedia/commons/2/20/Saimiri_sciureus-1_Luc_Viatour.jpg"
sunflower_path = tf.keras.utils.get_file('como', origin=sunflower_url)

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(monkey_class_names[np.argmax(score)], 100 * np.max(score))
)



#

model = tf.keras.models.load_model("./flowers_model_saved")
# sunflower_url = "https://cache.desktopnexus.com/thumbseg/2352/2352605-bigthumbnail.jpg"
# sunflower_path = tf.keras.utils.get_file('bouquet', origin=sunflower_url)
path_2="C:\\Users\\Tudor\\Desktop\\rosesandtulips.jpg"

img = tf.keras.utils.load_img(
    path_2, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(predictions)
print(class_names)
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)