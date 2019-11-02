import numpy as np
import tensorflow as tf
from tensorflow import keras

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

checkpoint_dir = 'saved_model'
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

# inference after loading data
images = test_images[:10]
labels = test_labels[:10]

# network assigns 0..1 score for each label
prediction_scores = model.predict(images)
pred_labels = np.argmax(prediction_scores, -1)

print('true labels', labels)
print('pred labels', pred_labels)