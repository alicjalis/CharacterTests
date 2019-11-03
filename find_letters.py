import cv2
import numpy as np

img = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
img_pixels = img.shape[0] * img.shape[1]

_, img_thresh = cv2.threshold(img, 127, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)  # ensure binary
# gaussian thresh inverts colors, which is actually ok for further computations, since connectedComponents works on non zero pixels

num_components, components = cv2.connectedComponents(img_thresh)

grid_component = None
filtered_components = []
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
#  since we care only about the positions of the letters, we could distort them heavily and get letters back later
img_letters = np.max(np.array(components_within_grid), 0)
kernel = np.ones((9, 9), np.uint8)
img_letters = cv2.dilate(img_letters, kernel)

img_letters = cv2.resize(img_letters, (600, 600))
cv2.imwrite('chars.jpg', img_letters)
cv2.imshow('', img_letters)
cv2.waitKey(-1)