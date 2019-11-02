import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from model import model
from utils import rotate_images
import cv2

# >>>>>>>> all of this can be replaced with tensor of custom data, of shape [batch, 28, 28, 1] (with data in range 0...1)
test_ds = tfds.load("emnist", split=tfds.Split.TEST, batch_size=-1)
test_ds = tfds.as_numpy(test_ds)
test_images, test_labels = test_ds["image"], test_ds["label"]
test_images = test_images / 255.0
test_images = rotate_images(test_images)
emnist_builder = tfds.builder('emnist')
info = emnist_builder.info
num_classes = info.features['label'].num_classes
# >>>>>>>>  ^^ all of this can be replaced with tensor of custom data, of shape [batch, 28, 28, 1] (with data in range 0...1)


checkpoint_dir = 'saved_model'
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

# inference after loading data
images = test_images[:1000]
labels = test_labels[:1000]

i = 458
print(labels[i])
cv2.imshow('', images[i])
cv2.waitKey(-1)

#
# # network assigns 0..1 score for each label
# prediction_scores = model.predict(images)
# pred_labels = np.argmax(prediction_scores, -1)
#
# print('true labels', labels)
# print('pred labels', pred_labels)
