from operator import imod
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
#monkey_data_dir='C:\\users\\Tudor\\Downloads\\monkeys'

batch_size = 16
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names



# model = tf.keras.models.load_model("./mega_model_saved")
# sunflower_url = "https://www.animenewsnetwork.com/thumbnails/crop1200x630gRB/cms/the-list/101603/monkeygintama.jpg"
# sunflower_path = tf.keras.utils.get_file('gintamag', origin=sunflower_url)

# img = tf.keras.utils.load_img(
#     sunflower_path, target_size=(img_height, img_width)
# )
# img_array1 = tf.keras.utils.img_to_array(img)
# img_array1 = tf.expand_dims(img_array1, 0) # Create a batch


# sunflower_url = "https://upload.wikimedia.org/wikipedia/commons/2/20/Saimiri_sciureus-1_Luc_Viatour.jpg"
# sunflower_path = tf.keras.utils.get_file('coasdasdmo', origin=sunflower_url)

# img = tf.keras.utils.load_img(
#     sunflower_path, target_size=(img_height, img_width)
# )
# img_array2 = tf.keras.utils.img_to_array(img)
# img_array2 = tf.expand_dims(img_array2, 0) # Create a batch





# #

model = tf.keras.models.load_model("./mega_model_saved")
# sunflower_url = "https://cache.desktopnexus.com/thumbseg/2352/2352605-bigthumbnail.jpg"
# sunflower_path = tf.keras.utils.get_file('bouquet', origin=sunflower_url)
path_2="C:\\Users\\Tudor\\Desktop\\rosesandtulips.jpg"

img = tf.keras.utils.load_img(
    path_2, target_size=(img_height, img_width)
)
img_array3 = tf.keras.utils.img_to_array(img)
img_array3 = tf.expand_dims(img_array3, 0) # Create a batch


def pretdict(path_to_predict):
    model = tf.keras.models.load_model("./mega_model_saved_2")
    image = tf.keras.utils.load_img(
    path_to_predict, target_size=(img_height, img_width)
    )
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)
    predictions = model.predict(image_array)
    score = tf.nn.softmax(predictions[0])
    print(class_names)
    print(predictions)
    print(score)
    final_class = class_names[np.argmax(score)]
    final_score = 100 * np.max(score)
    return final_class

# if __name__ == "main":
# predictions = model.predict(img_array1)
# score = tf.nn.softmax(predictions[0])
# final_class = class_names[np.argmax(score)]
# final_score = 100 * np.max(score)
# print(class_names)
# print(predictions)
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )


# predictions = model.predict(img_array2)
# score = tf.nn.softmax(predictions[0])

# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )


# predictions = model.predict(img_array3)
# score = tf.nn.softmax(predictions[0])
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )

