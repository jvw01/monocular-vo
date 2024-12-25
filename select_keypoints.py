import numpy as np


def selectKeypoints(scores, num, r):
    """
    Selects the num best scores as keypoints and performs non-maximum supression of a (2r + 1)*(2r + 1) box around
    the current maximum.
    """
    keypoints = []
    for i in range(num):
        max_idx = np.unravel_index(np.argmax(scores), scores.shape)
        keypoints.append(max_idx)
        
        # Non-maximum suppression
        # Setting pixel values to 0 in a square around the current max index
        # Use bounding to make sure we are not trying to access elements outside of R
        x, y = max_idx
        scores[max(0, x-r):min(scores.shape[0], x+r+1), max(0, y-r):min(scores.shape[1], y+r+1)] = 0 # note: slicing is exclusive of the endpoint
    
    return np.array(keypoints).T
