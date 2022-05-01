import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist =  tf.keras.datasets.mnist

(x_train, y_train),(x_test,y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis = 1)
# x_test = tf.keras.utils.normalize(x_test, axis = 1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5)
# model.save('handwritten_number_model')



# val_loss, val_acc = model.evaluate(x_test,y_test)
# print(val_loss,val_acc)
new_model=tf.keras.models.load_model('handwritten_number_model')
predictions = new_model.predict([x_test])
print(np.argmax(predictions[0]))

