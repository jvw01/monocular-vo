import cv2
import os
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
    keypoints, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=img_prev, nextImg=img, prevPts=keypoints_prev, nextPts=None)

    # filter valid keypoints (note: status is set to 1 if the flow for the corresponding features has been found)
    keypoints_prev = keypoints_prev[status == 1].T # dim: 2xK

    # extract the pose using P3P
    K = np.loadtxt(os.path.join("", "data_VO/K.txt")) # camera matrix
    # cv2.solvePnPRansac
    # _, rotation_vectors, translation_vectors = cv2.solveP3P(landmark_sample, keypoint_sample.T, K, None, flags=cv2.SOLVEPNP_P3P)
    # t_C_W_guess = []
    # R_C_W_guess = []
    # for rotation_vector in rotation_vectors:
    #     rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    #     for translation_vector in translation_vectors:
    #         R_C_W_guess.append(rotation_matrix)
    #         t_C_W_guess.append(translation_vector)

    S = {
            "keypoints": keypoints,
            "landmarks": S_prev["landmarks"]
        }
    
    T_WC = None

    return S, T_WC