import numpy as np
import cv2
from adaptive_dcp import adaptive_dcp


def refinement(image, tmap, radius=30, epsilon=0.01):
    guided_filter = cv2.ximgproc.createGuidedFilter(image, radius, epsilon)
    return np.clip(guided_filter.filter(tmap), 0, 1)


def get_transmission(image, airlight, patchmap, layer, omega=0.9):
    tmap = 1 - omega * adaptive_dcp(image / airlight, patchmap, layer)
    return tmap
