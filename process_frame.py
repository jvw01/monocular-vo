import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

L_m = 4
verbose = True

# TODO:
# - define input parameters of cv2 functions and not always use the default values
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
    # ------------------------------------------------------ 4.1: Associating keypoints
    # track keypoints from previous frame to current frame with KLT (i.e. pixel coordinates)
    keypoints_prev = S_prev["keypoints"]
    landmarks = S_prev["landmarks"]
    keypoints_prev = keypoints_prev.T # calcOpticalFlowPyrLK expects shape (N, 1, 2) where N is the number of keypoints
    landmarks = landmarks.T # shape (N, 1, 3)
    keypoints, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=img_prev, nextImg=img, prevPts=keypoints_prev, nextPts=None)

    # filter valid keypoints (note: status is set to 1 if the flow for the corresponding features has been found)
    # keypoints = keypoints[status].T # dim: 2xK
    keypoints = np.expand_dims(keypoints, 1)[status == 1].T # dim: 2xK
    landmarks = np.expand_dims(landmarks, 1)[status == 1].T # dim: 3xK

    #################### DEBUG START ####################
    if verbose:
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        keypoints_prev = S_prev["keypoints"]

        # Plot the previous image with previous keypoints
        axs[0].imshow(img_prev, cmap='gray')
        axs[0].scatter(keypoints_prev[0, :], keypoints_prev[1, :], s=5)
        axs[0].set_title('Keypoints in previous image')
        axs[0].axis('equal')

        # Plot the current image with tracked keypoints
        axs[1].imshow(img, cmap='gray')
        axs[1].scatter(keypoints[0, :], keypoints[1, :], s=5)
        axs[1].set_title('Keypoints tracked')
        axs[1].axis('equal')

        plt.tight_layout()
        plt.show()
    #################### DEBUG END ####################

    # ------------------------------------------------------ 4.2: Estimating current pose
    # extract the pose using P3P
    K = np.loadtxt(os.path.join("", "data_VO/K.txt")) # camera matrix
    _, rvec_CW, tvec_CW, inliers = cv2.solvePnPRansac(objectPoints=landmarks.T, imagePoints=keypoints.T, cameraMatrix=K, distCoeffs=None, flags=cv2.SOLVEPNP_P3P) # rvec, tvec are the rotation and translation vectors from world frame to camera frame
    
    keypoints = keypoints[:, inliers].squeeze() # dim: 2xK
    landmarks = landmarks[:, inliers].squeeze() # dim: 3xK

    rotation_matrix_CW, _ = cv2.Rodrigues(rvec_CW)
    rotation_matrix_WC = rotation_matrix_CW.T
    tvec_WC = -tvec_CW
    T_WC = np.hstack((rotation_matrix_WC, tvec_WC))

    # ------------------------------------------------------ 4.3: Triangulating new landmarks
    if S_prev["candidate_keypoints"]:
        # ------------------ Check existing candidate keypoints from previous frame(s) 
        candidate_keypoints_prev = S_prev["candidate_keypoints"]
        candidate_keypoints_prev = candidate_keypoints_prev.T.reshape(-1, 1, 2)
        candidate_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=img_prev, nextImg=img, prevPts=candidate_keypoints_prev, nextPts=None)

        # remove candidate keypoints that were not tracked successfully
        candidate_keypoints = candidate_keypoints[status == 1] # dim: Kx2
        candidate_keypoints_prev = candidate_keypoints_prev[status == 1]
        S_prev["pose_at_first_observation"] = S_prev["pose_at_first_observation"][status == 1]
        S_prev["first_observations"] = S_prev["first_observations"][status == 1]

        # ------------------ Promote keypoints
        S_prev["first_observations"] = S_prev["first_observations"][status == 1] + 1

        promoted_keypoints = candidate_keypoints[S_prev["first_observations"] > L_m]
        n_promoted_keypoints = promoted_keypoint_poses.shape[0] # ????????????????????????
        keypoints = np.vstack((keypoints, candidate_keypoints[S_prev["first_observations"] > L_m]))
        candidate_keypoints = candidate_keypoints[S_prev["first_observations"] <= L_m]

        promoted_keypoint_poses_prev = S_prev["pose_at_first_observation"][S_prev["first_observations"] > L_m]
        promoted_keypoint_poses = np.tile(T_WC,(n_promoted_keypoints,1))
        # TODO: CHECK DIMENSIONS OF INPUT TO FUNCTION
        promoted_candidate_landmarks = cv2.triangulatePoints(promoted_keypoint_poses_prev,promoted_keypoint_poses, candidate_keypoints_prev[S_prev["first_observations"] > L_m], promoted_keypoints)

        landmarks = np.vstack((landmarks, promoted_candidate_landmarks))

        first_observation = S_prev["first_observations"][S_prev["first_observations"] <= L_m]
        pose_at_first_observation = S_prev["pose_at_first_observation"][S_prev["first_observations"] <= L_m]
    
    else:
        candidate_keypoints = np.empty((2, 0))
        first_observation = np.empty((2, 0)) # TODO: why should the counter have dim 2xM???????
        pose_at_first_observation = np.empty((12, 0))

    # ------------------ Extract new keypoints and remove duplicates
    new_keypoints = cv2.goodFeaturesToTrack(img, mask=None, maxCorners=None, qualityLevel=0.3, minDistance=None, useHarrisDetector=True).squeeze().T # dim: 2xK
    distance_threshold = 2.0 # TODO: tune this parameter
    is_duplicate = np.any(np.linalg.norm(new_keypoints[:, :, None] - keypoints[:, None, :], axis=0) < distance_threshold, axis=1) # note: for broadcasting, dimensions have to match or be one
    new_candidate_keypoints = new_keypoints[:, ~is_duplicate]
    candidate_keypoints = np.hstack((candidate_keypoints, new_candidate_keypoints))
    first_observation = np.hstack((first_observation, np.ones(new_candidate_keypoints.shape)))
    pose_at_first_observation = np.hstack((pose_at_first_observation, np.tile(T_WC.flatten(), (new_candidate_keypoints.shape[1],1)).T))

    S = {
            "keypoints": keypoints, # dim: 2xK
            "landmarks": landmarks, # dim: 3xK
            "candidate_keypoints": candidate_keypoints, # dim: 2xM with M = # candidate keypoints
            "first_observations": first_observation, # dim: 2xM with M = # candidate keypoints
            "pose_at_first_observation": pose_at_first_observation # dim: 12xM with M = # candidate keypoints and 12 since the transformation matrix has dim 3x4 (omit last row)
        }

    return S, T_WC
