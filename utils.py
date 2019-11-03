import cv2
import numpy as np


def rotate_images(imgs):
    # np.rot90 works on first 2 dimensions, so move temporarily batch to the last one
    imgs = np.transpose(imgs, [1, 2, 3, 0])
    imgs = np.rot90(imgs, -1)
    imgs = np.fliplr(imgs)
    # reverse transposition
    imgs = np.transpose(imgs, [3, 0, 1, 2])
    return imgs


def block_thresh(img, y_blocks, x_blocks, max_ratio):
    """Aplies thresholding in blocks on image"""
    # crop image to match num_blocks x block size
    block_height = img.shape[0] // y_blocks
    block_width = img.shape[1] // x_blocks

    cropped_img = img[:block_height * y_blocks, :block_width * x_blocks]

    # we first splt image into columns then into blocks
    blocks = []
    for y in range(y_blocks):
        for x in range(x_blocks):
            block = cropped_img[y * block_height:(y + 1) * block_height, x * block_width:(x + 1) * block_width]
            blocks.append(block)

    # put blocks together
    thresh_img = np.empty(cropped_img.shape, dtype=cropped_img.dtype)
    i = 0
    for y in range(y_blocks):
        for x in range(x_blocks):
            _, thresh_block = cv2.threshold(blocks[i], 127, 255, cv2.THRESH_OTSU)
            # when dealing with almost white images, otsu still tries to find threshold, but it result in salt and pepper
            # image. So if ratio of black and white pixels is below some threshold, then whiten all block
            if np.count_nonzero(thresh_block) / (block_height * block_width) < max_ratio:
                thresh_block = 255
            thresh_img[y * block_height:(y + 1) * block_height, x * block_width:(x + 1) * block_width] = thresh_block
            i += 1

    return thresh_img