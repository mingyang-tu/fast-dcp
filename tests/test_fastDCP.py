import cv2
import numpy as np

from fastDCP import fastDCP


def test_fastDCP():
    image = cv2.imread("images/lena512.bmp")
    layer = 7

    # Proposed method
    dcps = fastDCP(image, layer)

    # OpenCV erode
    rgb_min = np.minimum(image[:, :, 0], np.minimum(image[:, :, 1], image[:, :, 2]))
    for i in range(layer):
        dcp = cv2.erode(rgb_min, np.ones((2 ** (i + 1), 2 ** (i + 1)), np.uint8))
        assert np.all(dcps[:, :, i] == dcp)
