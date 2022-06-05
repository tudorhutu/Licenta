import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
from data_augmentation import *
from dataset_loading import *
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers,optimizers
from tensorflow.keras.models import Sequential
import visualkeras
import pathlib

batch_size = 16

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

input_shape=(200,200,3)
# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
#   for i in range(9):
#     augmented_images = augmentation_zoom_and_rotation()(images)
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(augmented_images[0].numpy().astype("uint8"))
#     plt.axis("off")
# plt.show()

# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
#   for i in range(9):
#     augmented_images = augmentation_color()(images)
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(augmented_images[0].numpy().astype("uint8"))
#     plt.axis("off")
# plt.show()

# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
#   for i in range(9):
#     augmented_images = augmentation_translation()(images)
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(augmented_images[0].numpy().astype("uint8"))
#     plt.axis("off")
# plt.show()

model = Sequential([
  augmentation_zoom_and_rotation(),
  augmentation_zoom_and_rotation(),
  augmentation_color(),
  augmentation_translation(),
  layers.Rescaling(1./255),
  layers.Conv2D(16, 5, padding='same', activation='elu'),
  layers.MaxPooling2D(),
  layers.Conv2D(16, 3, padding='same', activation='elu'),
  layers.Conv2D(16, 3, padding='same', activation='elu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='elu'),
  layers.Conv2D(32, 3, padding='same', activation='elu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='elu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='elu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='elu'),
  layers.Dense(num_classes)
])


def fit_model():
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  model.summary()

  epochs = 80
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    workers=8,
    use_multiprocessing=True
  )

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()

  model.save("./mega_model_saved_2", save_format='h5')
  visualkeras.layered_view(model, to_file='output.png')

if __name__ == "main":
  fit_model()