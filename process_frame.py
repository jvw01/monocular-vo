import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
L_m = 4

verbose = True

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
    # ------------------------------------------------------ 4.1: Associating keypoints
    # track keypoints from previous frame to current frame with KLT (i.e. pixel coordinates)
    keypoints_prev = S_prev["keypoints"]
    object_points = S_prev["landmarks"]
    # if verbose:
    #     plt.imshow(img_prev, cmap='gray')
    #     plt.scatter(keypoints_prev[1, :], keypoints_prev[0, :], s=5)
    #     plt.title('Keypoints in previous image')
    #     plt.show()
    keypoints_prev = keypoints_prev.T.reshape(-1, 1, 2) # calcOpticalFlowPyrLK expects shape (N, 1, 2) where N is the number of keypoints
    object_points = object_points.T.reshape(-1, 1, 3) # shape (N, 1, 3)
    keypoints, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=img_prev, nextImg=img, prevPts=keypoints_prev, nextPts=None)

    # filter valid keypoints (note: status is set to 1 if the flow for the corresponding features has been found)
    keypoints = keypoints[status == 1] # dim: Kx2
    if verbose:
        plt.imshow(img, cmap='gray')
        plt.scatter(keypoints[:, 0], keypoints[:, 1], s=5)
        plt.title('Keypoints tracked')
        plt.show()
    object_points = object_points[status == 1] # dim: Kx3

    # ------------------------------------------------------ 4.2: Estimating current pose
    # extract the pose using P3P
    K = np.loadtxt(os.path.join("", "data_VO/K.txt")) # camera matrix
    _, rvec_CW, tvec_CW, inliers = cv2.solvePnPRansac(objectPoints=object_points, imagePoints=keypoints, cameraMatrix=K, distCoeffs=None, flags=cv2.SOLVEPNP_P3P) # rvec, tvec are the rotation and translation vectors from world frame to camera frame
    
    keypoints = keypoints[inliers].squeeze().T # dim: 2xK
    object_points = object_points[inliers].squeeze().T # dim: 3xK

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

        S = {}
        first_observation = S_prev["first_observations"][S_prev["first_observations"] <= L_m]
        pose_at_first_observation = S_prev["pose_at_first_observation"][S_prev["first_observations"] <= L_m]

        # ------------------ Extract new keypoints and remove duplicates
        # TODO: USE SIFT TO EXTRACT NEW CANDIDATE KEYPOINTS OR HARRIS???????????????
        sift = cv2.SIFT_create()
        new_keypoints = sift.detect(img, None)
        new_candidate_keypoints = np.array([kp for kp in new_keypoints if kp.tolist() not in keypoints.tolist()]) # remove duplicates
        new_candidate_keypoints = new_candidate_keypoints.reshape(2, -1) # dim: 2xM
        candidate_keypoints = np.hstack((candidate_keypoints, new_candidate_keypoints))
        first_observation = np.hstack((first_observation, np.ones(new_candidate_keypoints.shape[1])))
        pose_at_first_observation = np.hstack((pose_at_first_observation, np.tile(T_WC,(new_candidate_keypoints.shape[1],1))))

        S = {
                "keypoints": keypoints, # dim: 2xK
                "landmarks": landmarks, # dim: 3xK
                "candidate_keypoints": candidate_keypoints, # dim: 2xM with M = # candidate keypoints
                "first_observations": first_observation, # dim: 2xM with M = # candidate keypoints
                "pose_at_first_observation": pose_at_first_observation # dim: 12xM with M = # candidate keypoints and 12 since the transformation matrix has dim 3x4 (omit last row)
            }
        
    else: # note: this is the case when we are extracting candidate keypoints for the first time
        # ------------------ Extract new keypoints and remove duplicates
        sift = cv2.SIFT_create()
        new_keypoints = sift.detect(img)
        kp_converter = cv2.KeyPoint()
        new_keypoints = kp_converter.convert(new_keypoints).T
        if verbose:
            plt.imshow(img, cmap='gray')
            plt.scatter(keypoints[0, :], keypoints[1, :])
            plt.show()

        # new_candidate_keypoints = np.array([kp for kp in new_keypoints if kp.tolist() not in keypoints.tolist()]) # remove duplicates
        [item for item in new_keypoints.T if np.all(np.linalg.norm(keypoints - item) > 4)]
        new_keypoints = new_keypoints[np.all(np.any((new_keypoints-keypoints[:, None]), axis=1), axis=0)]
        first_observation = np.ones(new_candidate_keypoints.shape[1])
        pose_at_first_observation = np.tile(T_WC,(new_candidate_keypoints.shape[1],1))
        S = {
                "keypoints": keypoints, # dim: 2xK
                "landmarks": object_points, # dim: 3xK
                "candidate_keypoints": new_candidate_keypoints, # dim: 2xM with M = # candidate keypoints 
                "first_observations": first_observation, # dim: 2xM with M = # candidate keypoints
                "pose_at_first_observation": pose_at_first_observation, # dim: 12xM with M = # candidate keypoints and 12 since the transformation matrix has dim 3x4 (omit last row)
            }

    return S, T_WC