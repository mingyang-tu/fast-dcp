import numpy as np


def adaptive_dcp(image, patchmap, layer):
    ROW, COL, _ = image.shape
    dcps = fastDCP(image, layer)
    adcp = np.zeros((ROW, COL), dtype=image.dtype)
    for i in range(layer):
        idxs = patchmap == (i + 1)
        adcp[idxs] = dcps[idxs, i]
    return adcp


def fastDCP(image, layer):
    """
    Args:
        image (numpy.ndarray): Input color image with shape (M, N, 3).
        layer (int): Number of patch sizes.

    Returns:
        numpy.ndarray: Numpy array with shape (M, N, layer) containing the dark channels
                       using patch sizes: 2^i x 2^i, where i = 1, 2, ..., layer.
    """
    M, N, _ = image.shape

    dcps = np.zeros((M, N, layer), dtype=image.dtype)

    rgb_min = np.minimum(image[:, :, 0], np.minimum(image[:, :, 1], image[:, :, 2]))

    dcp = np.minimum(rgb_min, np.vstack([rgb_min[[0], :], rgb_min[:-1, :]]))
    dcp = np.minimum(dcp, np.hstack([dcp[:, [0]], dcp[:, :-1]]))
    dcps[:, :, 0] = dcp

    offset = 1
    for i in range(layer - 1):
        dcp = np.minimum(
            np.vstack((dcp[[0] * offset, :], dcp[:-offset, :])),
            np.vstack((dcp[offset:, :], dcp[[-1] * offset, :])),
        )
        dcp = np.minimum(
            np.hstack((dcp[:, [0] * offset], dcp[:, :-offset])),
            np.hstack((dcp[:, offset:], dcp[:, [-1] * offset])),
        )
        dcps[:, :, i + 1] = dcp
        offset *= 2
    return dcps
