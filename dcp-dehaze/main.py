import numpy as np
import cv2
import time

from patchmap import get_patchmap
from adaptive_dcp import adaptive_dcp
from airlight import get_airlight
from transmission import get_transmission, refinement
from reconstruct import reconstruct


def resize(img, max_size):
    M = max(img.shape)
    ratio = float(max_size) / float(M)
    if M > max_size:
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
    return img


def dehaze(image, layer):
    image = image.astype(np.float32) / 255.0

    patchmap = get_patchmap(image, layer)
    adcp = adaptive_dcp(image, patchmap, layer)
    airlight = get_airlight(adcp, image)
    tmap = get_transmission(image, airlight, patchmap, layer)
    tmap_ref = refinement(image, tmap)
    output = reconstruct(image, airlight, tmap_ref)

    return output


if __name__ == "__main__":
    for i in range(1, 6):
        image = cv2.imread(f"test-images/test{i}.jpeg")
        image = resize(image, 512)
        layer = 7

        start = time.time()
        output = dehaze(image, layer)
        end = time.time()
        print(f"Size: {image.shape}, Elapsed time: {end-start:.4f} s")

        cv2.imshow("original", image)
        cv2.imshow("dehaze", output)
        cv2.waitKey()
