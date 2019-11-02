from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from model import model
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from utils import rotate_images

print(tf.__version__)

train_ds = tfds.load("emnist", split=tfds.Split.TRAIN, batch_size=-1)
train_ds = tfds.as_numpy(train_ds)
train_images, train_labels = train_ds["image"], train_ds["label"]
train_images = train_images / 255.0
train_images = rotate_images(train_images)

test_ds = tfds.load("emnist", split=tfds.Split.TEST, batch_size=-1)
test_ds = tfds.as_numpy(test_ds)
test_images, test_labels = test_ds["image"], test_ds["label"]
test_images = test_images / 255.0
test_images = rotate_images(test_images)

emnist_builder = tfds.builder('emnist')
info = emnist_builder.info
num_classes = info.features['label'].num_classes

print(train_images.shape) # liczba obrazkow, piksele wysokosc, piksele szerokosc, liczba kanalow (1 - szary)
print(test_images.shape)

# saving
checkpoint_path = "saved_model/model.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, callbacks=[cp_callback], batch_size=1024)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2, batch_size=1024)

print('\nTest accuracy:', test_acc)