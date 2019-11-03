import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from model import model
from utils import block_thresh

img = cv2.imread('sample2.jpg', cv2.IMREAD_GRAYSCALE)
img_pixels = img.shape[0] * img.shape[1]

img_thresh = block_thresh(img, 10, 15, max_ratio=0.9)

# both emnist and further functions use black backgrounds and white letters
img_thresh = 255 - img_thresh

num_components, components = cv2.connectedComponents(img_thresh)

grid_component = None
filtered_components = []
print('filtering size')
for i in range(num_components):
    # convert component to binary mask with 0's and 255's
    component = (components == i).astype(np.uint8) * 255

    # possibly remove the component if it's too big (in terms of pixel occupation)
    # since it's probably just blob of incorrectly thresholded pixels
    large_blob_threshold = 0.5
    if np.count_nonzero(component) / img_pixels > large_blob_threshold:
        continue

    # also remove too small blobs
    small_blob_threshold = 40  # in pixels
    if np.count_nonzero(component) <= small_blob_threshold:
        continue

    filtered_components.append(component)

# now try to find the grid
grid_id = None
grid = None
max_bbox = None
print('searching for grid')

for i, component in enumerate(filtered_components):
    # let's find the bounding box for each component. We assume that the grid should have the largest bounding box of
    # them all
    bbox = cv2.boundingRect(component)

    if grid is None:
        grid = component
        grid_id = i
        max_bbox = bbox
        continue
    x, y, w, h = bbox
    if w >= max_bbox[2] and h >= max_bbox[3]:
        grid = component
        grid_id = i
        max_bbox = bbox

# assume that all letters should be contained within grid - maybe some offset
# if offset is negative, then all the letters should NOT exceed the grid
offset = 0.95
grid_x, grid_y, grid_w, grid_h = max_bbox
grid_xc = grid_x + grid_w / 2
grid_yc = grid_y + grid_h / 2

# the possible letters cannot exceed this boundaries
min_x = grid_xc - (grid_w / 2) * offset
max_x = grid_xc + (grid_w / 2) * offset
min_y = grid_yc - (grid_h / 2) * offset
max_y = grid_yc + (grid_h / 2) * offset

# remove all components which lie outside boundaries
components_within_grid = []
print('removing outliers')
for i, component in enumerate(filtered_components):
    # grid is also among filtered components
    if i == grid_id:
        continue

    bbox = cv2.boundingRect(component)
    x, y, w, h = bbox

    # remove components outside boundaries
    if x < min_x:
        continue
    if x + w > max_x:
        continue
    if y < min_y:
        continue
    if y + h > max_y:
        continue

    components_within_grid.append(component)

# at this point, some letters are split into separate components. Lets merge them together and apply closing
# since we care only about the positions of the letters, we could distort them heavily and get letters back later
img_letters = np.max(np.array(components_within_grid), 0)
kernel = np.ones((9, 9), np.uint8)

# todo -> don't forget to cut letters out of original thresh image!
# todo -> a little blurring might be useful since emnist is not striclty binary
img_letters_dil = cv2.dilate(img_letters, kernel)  # pogrubia
num_components_let, components_let = cv2.connectedComponents(img_letters_dil)

positions = []
images = []

# iterate from 1, since first component is the whole image
for i in range(1, num_components_let):
    # convert component to binary mask with 0's and 255's
    components_let_ = (components_let == i).astype(np.uint8) * 255
    bbox = cv2.boundingRect(components_let_)
    positions.append(bbox)
    x, y, w, h = bbox

    letter = img_letters[y:y + h, x:x + w]
    letter = letter / 255
    letter = letter.astype(np.float32)

    # pad and resize 
    l_h, l_w = letter.shape
    long_side = max(l_h, l_w)

    # due to integer division (//) there might be 1 pixel difference between true square, but that could be dealt with resize
    # paddings
    # todo: this padd ofset yields best results when inr ange 0.05 to 0.15 depending on image
    pad_offset = 0.15
    h_pad = (long_side - l_h) // 2 + + int(long_side * pad_offset)
    w_pad = (long_side - l_w) // 2 + int(long_side * pad_offset)

    letter = np.pad(letter, [(h_pad, h_pad), (w_pad, w_pad)])
    letter = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_CUBIC)

    images.append(letter)

images = np.array(images)
images = np.expand_dims(images, -1)
checkpoint_dir = 'saved_model'

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

# network assigns 0..1 score for each label
prediction_scores = model.predict(images)
pred_labels = np.argmax(prediction_scores, -1)

vocabulary = list(str(i) for i in range(0, 10)) + [chr(c) for c in range(ord('A'), ord('Z') + 1)] + [chr(c) for c in range(ord('a'), ord('z') + 1)]
pred_letters = [vocabulary[label] for label in pred_labels]

result_img = np.copy(img)

for bbox, letter in zip(positions, pred_letters):
    cv2.putText(result_img, letter, (bbox[0] - 30, bbox[1]), cv2.FONT_HERSHEY_COMPLEX, 3, 0, 2, cv2.LINE_AA)

# for vis purposes
result_img = cv2.resize(result_img, (600, 600))
cv2.imshow('im', result_img)
cv2.waitKey(-1)
