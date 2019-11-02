import numpy as np


def rotate_images(imgs):
    # np.rot90 works on first 2 dimensions, so move temporarily batch to the last one
    imgs = np.transpose(imgs, [1, 2, 3, 0])
    imgs = np.rot90(imgs, -1)
    imgs = np.fliplr(imgs)
    # reverse transposition
    imgs = np.transpose(imgs, [3, 0, 1, 2])
    return imgs