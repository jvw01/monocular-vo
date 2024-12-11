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
    object_points = S_prev["landmarks"]
    keypoints_prev = keypoints_prev.T.reshape(-1, 1, 2) # calcOpticalFlowPyrLK expects shape (N, 1, 2) where N is the number of keypoints
    object_points = object_points.T.reshape(-1, 1, 3) # shape (N, 1, 3)
    keypoints, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=img_prev, nextImg=img, prevPts=keypoints_prev, nextPts=None)

    # filter valid keypoints (note: status is set to 1 if the flow for the corresponding features has been found)
    keypoints = keypoints[status == 1] # dim: Kx2
    object_points = object_points[status == 1] # dim: Kx3

    # extract the pose using P3P
    K = np.loadtxt(os.path.join("", "data_VO/K.txt")) # camera matrix
    _, rvec_CW, tvec_CW, inliers = cv2.solvePnPRansac(objectPoints=object_points, imagePoints=keypoints, cameraMatrix=K, distCoeffs=None, flags=cv2.SOLVEPNP_P3P) # rvec, tvec are the rotation and translation vectors from world frame to camera frame
    
    keypoints = keypoints[inliers].squeeze().T # dim: 2xK
    object_points = object_points[inliers].squeeze().T # dim: 3xK

    S = {
            "keypoints": keypoints, # dim: 2xK
            "landmarks": S_prev["landmarks"] # dim: 3xK
        }
    
    rotation_matrix_CW, _ = cv2.Rodrigues(rvec_CW)
    rotation_matrix_WC = rotation_matrix_CW.T
    tvec_WC = -tvec_CW
    T_WC = np.vstack((np.hstack((rotation_matrix_WC, tvec_WC)), np.array([0, 0, 0, 1])))

    return S, T_WC