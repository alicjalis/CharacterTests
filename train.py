from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape) #piksele
print(len(train_labels)) #tyle ile obrazow wychodzi
print((train_labels))
# plt.figure()
# plt.imshow(train_images[0])
# plt.gray()
# plt.grid(False)
# plt.show()

train_images = train_images / 255.0 # potem wartosci pikseli sa od 0 do 1 bo tak sie robi zeby nie bylo za duze
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #warstwy sieci neuronowej, jej architektura
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# saving
checkpoint_path = "saved_model/model.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, callbacks=[cp_callback])

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)