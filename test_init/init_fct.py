import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import cdist

def harris(img, patch_size, kappa):
    sobel_para = np.array([-1, 0, 1])
    sobel_orth = np.array([1, 2, 1])

    Ix = signal.convolve2d(img, sobel_para[None, :], mode="valid")
    Ix = signal.convolve2d(Ix, sobel_orth[:, None], mode="valid").astype(float)

    Iy = signal.convolve2d(img, sobel_para[:, None], mode="valid")
    Iy = signal.convolve2d(Iy, sobel_orth[None, :], mode="valid").astype(float)

    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix*Iy

    patch = np.ones([patch_size, patch_size])
    pr = patch_size // 2
    sIxx = signal.convolve2d(Ixx, patch, mode="valid")
    sIyy = signal.convolve2d(Iyy, patch, mode="valid")
    sIxy = signal.convolve2d(Ixy, patch, mode="valid")

    scores = (sIxx * sIyy - sIxy ** 2) - kappa * ((sIxx + sIyy) ** 2)

    scores[scores < 0] = 0

    scores = np.pad(scores, [(pr+1, pr+1), (pr+1, pr+1)], mode='constant', constant_values=0)

    return scores


def selectKeypoints(scores, num, r):
    keypoints = np.zeros([2, num])
    temp_scores = np.pad(scores, [(r, r), (r, r)], mode='constant', constant_values=0)

    for i in range(num):
        kp = np.unravel_index(temp_scores.argmax(), temp_scores.shape)
        keypoints[:, i] = np.array(kp) - r
        temp_scores[(kp[0] - r):(kp[0] + r + 1), (kp[1] - r):(kp[1] + r + 1)] = 0

    return keypoints

def describeKeypoints(img, keypoints, r):
    N = keypoints.shape[1]
    desciptors = np.zeros([(2*r+1)**2, N])
    padded = np.pad(img, [(r, r), (r, r)], mode='constant', constant_values=0)

    for i in range(N):
        kp = keypoints[:, i].astype(np.int16) + r
        desciptors[:, i] = padded[(kp[0] - r):(kp[0] + r + 1), (kp[1] - r):(kp[1] + r + 1)].flatten()

    return desciptors

def matchDescriptors(query_descriptors, database_descriptors, match_lambda):
    dists = cdist(query_descriptors.T, database_descriptors.T, 'euclidean')
    matches = np.argmin(dists, axis=1)
    dists = dists[np.arange(matches.shape[0]), matches]
    min_non_zero_dist = dists.min()

    matches[dists >= match_lambda * min_non_zero_dist] = -1

    # remove double matches
    unique_matches = np.ones_like(matches) * -1
    _, unique_match_idxs = np.unique(matches, return_index=True)
    unique_matches[unique_match_idxs] = matches[unique_match_idxs]

    return unique_matches

def plotMatches(matches, query_keypoints, database_keypoints):
    query_indices = np.nonzero(matches >= 0)[0]
    match_indices = matches[query_indices]

    x_from = query_keypoints[0, query_indices]
    x_to = database_keypoints[0, match_indices]
    y_from = query_keypoints[1, query_indices]
    y_to = database_keypoints[1, match_indices]

    for i in range(x_from.shape[0]):
        plt.plot([y_from[i], y_to[i]], [x_from[i], x_to[i]], 'g-', linewidth=3)
