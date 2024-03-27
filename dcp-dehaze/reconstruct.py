import numpy as np


def reconstruct(image, airlight, tmap, t0=0.1):
    nonzero_t = np.maximum(tmap, t0)
    return (image - airlight) / nonzero_t[..., np.newaxis] + airlight
