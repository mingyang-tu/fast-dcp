import numpy as np


def get_airlight(dark_channel, image, top_p=0.1):
    threshold = np.percentile(dark_channel, 100 - top_p)
    candidates = dark_channel >= threshold
    return np.median(image[candidates, :], axis=0)
