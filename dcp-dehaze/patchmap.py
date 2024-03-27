import numpy as np
import cv2


def get_gradient(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    gradient_x = np.abs(cv2.Sobel(image, -1, 1, 0, ksize=3)).sum(axis=2)
    gradient_y = np.abs(cv2.Sobel(image, -1, 0, 1, ksize=3)).sum(axis=2)
    return gradient_x + gradient_y


def fast_summation(image, layer):
    ROW, COL = image.shape
    sums = np.zeros((ROW, COL, layer), dtype=image.dtype)

    _sum = image + np.vstack((image[[0], :], image[:-1, :]))
    _sum = _sum + np.hstack((_sum[:, [0]], _sum[:, :-1]))
    sums[:, :, 0] = _sum

    offset = 1
    for i in range(layer - 1):
        _sum = np.vstack((_sum[[0] * offset, :], _sum[:-offset, :])) + np.vstack((_sum[offset:, :], _sum[[-1] * offset, :]))
        _sum = np.hstack((_sum[:, [0] * offset], _sum[:, :-offset])) + np.hstack((_sum[:, offset:], _sum[:, [-1] * offset]))
        sums[:, :, i + 1] = _sum
        offset *= 2
    return sums


def get_patchmap(image, layer, threshold=50):
    ROW, COL, _ = image.shape

    gradient = get_gradient(image)
    sums = fast_summation(gradient, layer)

    bigger = sums > threshold

    patchmap = np.full((ROW, COL), layer, dtype=np.uint8)
    unseen = np.full((ROW, COL), True, dtype=bool)
    for i in range(1, layer):
        idxs = bigger[:, :, i] & unseen
        patchmap[idxs] = i
        unseen[idxs] = False

    return patchmap
