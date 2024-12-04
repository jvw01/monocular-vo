import cv2
import numpy as np

# TODO:
# - might need to pass lk_params to cv2.calcOpticalFlowPyrLK in order to optimize tracking for our case
def processFrame(img, img_prev, S_prev) -> tuple[dict, np.ndarray]:
    """
    This function implements the continuous visual odometry pipeline in a Markovian way.
    Args:
        img: current frame (query image)
        img_prev: previous frame (database image)
        S_prev: state of previous frame (i.e., the keypoints in the previous frame and the 3D landmarks associated to them)
    Returns:
        S: state of current frame (i.e., the keypoints in the current frame and the 3D landmarks associated to them)
        T_WC: current pose
    """
    # track keypoints from previous frame to current frame with KLT (i.e. pixel coordinates)
    keypoints_prev = S_prev["keypoints"]
    keypoints_prev = keypoints_prev.T.reshape(-1, 1, 2) # calcOpticalFlowPyrLK expects shape (N, 1, 2) where N is the number of keypoints
    keypoints_curr, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=img_prev, nextImg=img, prevPts=keypoints_prev, nextPts=None)
    keypoints = keypoints_curr[status == 1] # filter valid keypoints (note: status is set to 1 if the flow for the corresponding features has been found)
    keypoints = keypoints.reshape(2, -1) # dim: 2xK

    # TODO
    S = None
    T_WC = None

    return S, T_WC