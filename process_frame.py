import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation

L_m = 4
angle_threshold_for_triangulation = 10 # in degrees
angle_threshold_for_triangulation *= np.pi / 180 # convert to radians
verbose = False

# TODO:
# - define input parameters of cv2 functions and not always use the default values
def processFrame(img: np.ndarray, img_prev: np.ndarray, S_prev: dict, K: np.ndarray) -> tuple[dict, np.ndarray]:
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
    keypoints_prev = S_prev["keypoints"] # dim: Kx2
    landmarks = S_prev["landmarks"] # dim: Kx3
    keypoints, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=img_prev, nextImg=img, prevPts=keypoints_prev, nextPts=None)

    # filter valid keypoints (note: status is set to 1 if the flow for the corresponding features has been found)
    keypoints = np.expand_dims(keypoints, 1)[status == 1] # dim: Kx2
    landmarks = np.expand_dims(landmarks, 1)[status == 1] # dim: Kx3
    n_tracked_keypoints = keypoints.shape[0]

    # #################### DEBUG START ####################
    # if verbose:
    #     fig, axs = plt.subplots(2, 1, figsize=(7, 7))
    #     # Plot the previous image with previous keypoints
    #     axs[0].imshow(img_prev, cmap='gray')
    #     axs[0].scatter(keypoints_prev[:, 0], keypoints_prev[:, 1], s=5)
    #     axs[0].set_title('Keypoints in previous image')
    #     axs[0].axis('equal')

    #     # Plot the current image with tracked keypoints
    #     axs[1].imshow(img, cmap='gray')
    #     axs[1].scatter(keypoints[:, 0], keypoints[:, 1], s=5)
    #     axs[1].set_title('Keypoints tracked')
    #     axs[1].axis('equal')

    #     plt.tight_layout()
    #     plt.show(block=False)
    #     plt.pause(2)
    #     plt.close()
    # #################### DEBUG END ####################

    # ------------------------------------------------------ 4.2: Estimating current pose
    # extract the pose using P3P
    _, rvec_CW, tvec_CW, inliers = cv2.solvePnPRansac(objectPoints=landmarks, imagePoints=keypoints, cameraMatrix=K, distCoeffs=None, flags=cv2.SOLVEPNP_P3P) # rvec, tvec are the rotation and translation vectors from world frame to camera frame
    
    keypoints = keypoints[inliers, :].squeeze() # dim: Kx2
    landmarks = landmarks[inliers, :].squeeze() # dim: Kx3

    rotation_matrix_CW, _ = cv2.Rodrigues(rvec_CW)
    rotation_matrix_WC = rotation_matrix_CW.T.astype(np.float32)
    tvec_WC = -tvec_CW.astype(np.float32)
    T_WC = np.hstack((rotation_matrix_WC, tvec_WC))

    # ------------------------------------------------------ 4.3: Triangulating new landmarks
    if isinstance(S_prev["candidate_keypoints"], np.ndarray):
        # ------------------ Check existing candidate keypoints from previous frame(s) 
        candidate_keypoints_prev = S_prev["candidate_keypoints"]
        candidate_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=img_prev, nextImg=img, prevPts=candidate_keypoints_prev, nextPts=None)

        # remove candidate keypoints that were not tracked successfully
        candidate_keypoints = np.expand_dims(candidate_keypoints, 1)[status == 1] # dim: Kx2
        candidate_keypoints_prev = np.expand_dims(candidate_keypoints_prev, 1)[status == 1] # These are required for triangulation
        S_prev["pose_at_first_observation"] = np.expand_dims(S_prev["pose_at_first_observation"], 1)[status == 1]
        S_prev["first_observations"] = np.expand_dims(S_prev["first_observations"], 1)[status == 1] + 1

        # ------------------ Promote keypoints
        promoted_keypoints = candidate_keypoints[S_prev["first_observations"] > L_m]
        keypoints = np.vstack((keypoints, promoted_keypoints))
        candidate_keypoints = candidate_keypoints[S_prev["first_observations"] <= L_m]
        candidate_keypoints_prev = candidate_keypoints_prev[S_prev["first_observations"] > L_m]
        promoted_keypoint_poses_prev = S_prev["pose_at_first_observation"][S_prev["first_observations"] > L_m]
        n_promoted_keypoints = promoted_keypoints.shape[0]

        if n_promoted_keypoints > 0:
            print("There are promotable keypoints")
            # If any keypoint has been promoted, perform triangulation
            promoted_candidate_landmarks = np.empty((3, n_promoted_keypoints), dtype=np.float32)
            for i in range(n_promoted_keypoints):
                # TODO: ADD MINIMUM ANGLE THAT HAS TO BE SPANNED
                landmark = cv2.triangulatePoints(promoted_keypoint_poses_prev.reshape(-1, 3, 4)[i,:,:], T_WC, candidate_keypoints_prev[i], promoted_keypoints[i])
                promoted_candidate_landmarks[:,i] = np.squeeze(cv2.convertPointsFromHomogeneous(landmark.T)).T # convert landmarks to non-homogeneous coordinates

            landmarks = np.vstack((landmarks, promoted_candidate_landmarks.T))

        first_observations = S_prev["first_observations"][S_prev["first_observations"] <= L_m]
        pose_at_first_observation = S_prev["pose_at_first_observation"][S_prev["first_observations"] <= L_m]
    
    else:
        candidate_keypoints = np.empty((0, 2), dtype=np.float32)
        first_observations = np.empty(0, dtype=np.int64)
        pose_at_first_observation = np.empty((0, 12), dtype=np.float32)
        n_promoted_keypoints = 0

    # ------------------ Extract new keypoints and remove duplicates
    new_keypoints = cv2.goodFeaturesToTrack(img, mask=None, maxCorners=None, qualityLevel=0.05, minDistance=None, useHarrisDetector=True).squeeze() # dim: Kx2

    #################### DEBUG START ####################
    if verbose:
        # Plot the current image with new keypoints
        plt.imshow(img, cmap='gray')
        plt.scatter(new_keypoints.T[0, :], new_keypoints.T[1, :], s=5)
        plt.title('New Keypoints')
        plt.axis('equal')

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    #################### DEBUG END ####################

    distance_threshold = 10.0 # TODO: tune this parameter

    # remove duplicates in keypoints
    is_duplicate_kp = np.any(np.linalg.norm(new_keypoints[:, None, :] - keypoints[None, :, :], axis=2) < distance_threshold, axis=1) # note: for broadcasting, dimensions have to match or be one
    new_keypoints = new_keypoints[~is_duplicate_kp, :]

    # remove duplicates in candidate keypoints
    is_duplicate_ckp = np.any(np.linalg.norm(new_keypoints[:, None, :] - candidate_keypoints[None, :, :], axis=2) < distance_threshold, axis=1)
    new_candidate_keypoints = new_keypoints[~is_duplicate_ckp, :]

    candidate_keypoints = np.vstack((candidate_keypoints, new_candidate_keypoints))
    first_observations = np.hstack((first_observations, np.ones(new_candidate_keypoints.shape[0], dtype=np.int64)))
    pose_at_first_observation = np.vstack((pose_at_first_observation, np.array([T_WC.flatten()]*new_candidate_keypoints.shape[0])))

    S = {
            "keypoints": keypoints, # dim: Kx2
            "landmarks": landmarks, # dim: Kx3
            "candidate_keypoints": candidate_keypoints, # dim: Mx2 with M = # candidate keypoints
            "first_observations": first_observations, # dim: Mx2 with M = # candidate keypoints
            "pose_at_first_observation": pose_at_first_observation # dim: Mx12 with M = # candidate keypoints and 12 since the transformation matrix has dim 3x4 (omit last row)
        }

    return S, T_WC, n_tracked_keypoints, n_promoted_keypoints
