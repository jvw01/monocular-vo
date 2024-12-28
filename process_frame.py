import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from harris import harris
from select_keypoints import selectKeypoints
# from scipy.spatial.transform import Rotation

L_m = 2
min_depth = 1 # TODO: tune this parameter
max_depth = 80 # TODO: tune this parameter
angle_threshold_for_triangulation = 5 # in degrees TODO: tune this parameter
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
        K: camera matrix
    Returns:
        S: state of current frame (i.e., the keypoints in the current frame and the 3D landmarks associated to them)
        T_WC: current pose
        n_tracked_keypoints: number of tracked keypoints
        n_promoted_keypoints: number of promoted keypoints
    """

    debug_dict = {}

    # ------------------------------------------------------ 4.1: Associating keypoints
    # track keypoints from previous frame to current frame with KLT (i.e. pixel coordinates)
    keypoints_prev = S_prev["keypoints"] # dim: Kx2
    landmarks = S_prev["landmarks"] # dim: Kx3
    keypoints, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=img_prev, nextImg=img, prevPts=keypoints_prev, nextPts=None)

    # filter valid keypoints (note: status is set to 1 if the flow for the corresponding features has been found)
    debug_dict["untrackable_keypoints"] = np.expand_dims(keypoints, 1)[status == 0]
    keypoints = np.expand_dims(keypoints, 1)[status == 1] # dim: Kx2
    landmarks = np.expand_dims(landmarks, 1)[status == 1] # dim: Kx3
    debug_dict["n_tracked_keypoints"] = keypoints.shape[0]

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
    
    outlier_mask = np.ones(keypoints.shape[0], dtype=bool)
    outlier_mask[inliers] = False
    debug_dict["trackable_outlier_keypoints"] = keypoints[outlier_mask]
    keypoints = keypoints[inliers].squeeze() # dim: Kx2
    debug_dict["trackable_keypoints"] = keypoints
    landmarks = landmarks[inliers].squeeze() # dim: Kx3

    rotation_matrix_CW, _ = cv2.Rodrigues(rvec_CW)
    rotation_matrix_WC = rotation_matrix_CW.T.astype(np.float32)
    tvec_WC = -tvec_CW.astype(np.float32)
    T_WC = np.hstack((rotation_matrix_WC, tvec_WC)) # dim: 3x4

    # ------------------------------------------------------ 4.3: Triangulating new landmarks
    if isinstance(S_prev["candidate_keypoints"], np.ndarray):
        # ------------------ Check existing candidate keypoints from previous frame(s) 
        candidate_keypoints_prev = S_prev["candidate_keypoints"]
        candidate_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=img_prev, nextImg=img, prevPts=candidate_keypoints_prev, nextPts=None)

        # remove candidate keypoints that were not tracked successfully
        debug_dict["untrackable_candidate_keypoints"] = np.expand_dims(candidate_keypoints, 1)[status == 0]
        candidate_keypoints = np.expand_dims(candidate_keypoints, 1)[status == 1] # dim: Kx2
        debug_dict["n_trackable_candidate_keypoints"] = candidate_keypoints.shape[0]
        debug_dict["n_untrackable_candidate_keypoints"] = candidate_keypoints_prev.shape[0] - candidate_keypoints.shape[0]
        S_prev["pose_at_first_observations"] = np.expand_dims(S_prev["pose_at_first_observations"], 1)[status == 1]
        S_prev["first_observations"] = np.expand_dims(S_prev["first_observations"], 1)[status == 1]
        S_prev["keypoint_tracker"] = np.expand_dims(S_prev["keypoint_tracker"], 1)[status == 1] + 1

        # ------------------ Promote keypoints
        promotable_keypoints = candidate_keypoints[S_prev["keypoint_tracker"] > L_m]
        n_promotable_keypoints = promotable_keypoints.shape[0]
        debug_dict["n_promotable_before_triangulation"] = n_promotable_keypoints
        # debug_dict["promotable_keypoints"] = promotable_keypoints

        # remove promotable keypoints from candidate list -> keypoints that can eventually not be promoted will be added again
        candidate_keypoints = candidate_keypoints[S_prev["keypoint_tracker"] <= L_m]
        debug_dict["trackable_unpromotable_candidate_keypoints"] = candidate_keypoints
        first_observations = S_prev["first_observations"][S_prev["keypoint_tracker"] <= L_m]
        pose_at_first_observations = S_prev["pose_at_first_observations"][S_prev["keypoint_tracker"] <= L_m]
        keypoint_tracker = S_prev["keypoint_tracker"][S_prev["keypoint_tracker"] <= L_m]

        # If any keypoint has been promoted, attempt to perform triangulation (requires minimum baseline)
        if n_promotable_keypoints > 0:
            print(f"There are promotable keypoints ({n_promotable_keypoints})")
            promotable_keypoints_first_observations = S_prev["first_observations"][S_prev["keypoint_tracker"] > L_m]
            promotable_keypoints_initial_poses = S_prev["pose_at_first_observations"][S_prev["keypoint_tracker"] > L_m]
            promotable_keypoints_tracker = S_prev["keypoint_tracker"][S_prev["keypoint_tracker"] > L_m]

            # use bearing vectors to calculate the angle between first observation and current observation -> v1 and v2 point to the same landmark
            K_inv = np.linalg.inv(K)
            T_WC_first_observation = promotable_keypoints_initial_poses.reshape(-1, 3, 4)
            R_CC = T_WC[:3,:3].T @ T_WC_first_observation[:,:3,:3] # rotation matrix from first camera frame to current camera frame
            R_CC_transposed = np.transpose(R_CC, axes=(0, 2, 1))
            v1 = (np.matmul(R_CC_transposed, K_inv) @ (np.hstack((promotable_keypoints_first_observations, np.ones((n_promotable_keypoints, 1)))))[:,:,None]).squeeze().T # v1 = R⁽⁻¹⁾ * K⁽⁻¹⁾ * [u1; v1; 1]; dim 3xK
            v2 = K_inv @ np.vstack((promotable_keypoints.T, np.ones((1, n_promotable_keypoints)))) # v2 = K⁽⁻¹⁾ * [u2; v2; 1] (no need for rotation since we are already in correct frame); dim 3xK
            alpha = np.arccos(np.sum(v1 * v2, axis=0) / (np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0))) # angle between bearing vectors in radians
            triangulate = alpha > angle_threshold_for_triangulation # mask that indicates if the angle between the bearing vectors is large enough for triangulation

            # filter keypoints that can be promoted (angle between bearing vectors is large enough)
            debug_dict["untriangulatable_promotable_candidate_keypoints"] = promotable_keypoints[~triangulate]
            promotable_keypoints_after_angle_threshold = promotable_keypoints[triangulate]
            debug_dict["n_lost_candidates_at_angle_filtering"] = promotable_keypoints.shape[0] - promotable_keypoints_after_angle_threshold.shape[0]
            n_promoted_keypoints = promotable_keypoints_after_angle_threshold.shape[0]
            debug_dict["n_promoted_after_angle_filter"] = n_promoted_keypoints


            # triangulate landmarks with least squares approximation
            promoted_keypoints_first_observations = promotable_keypoints_first_observations[triangulate]
            promoted_keypoint_poses_prev = promotable_keypoints_initial_poses[triangulate]
            promoted_keypoint_tracker = promotable_keypoints_tracker[triangulate]
            T_WC_first_observation = promoted_keypoint_poses_prev.reshape(-1, 3, 4)
            
            promoted_landmarks = np.empty((n_promoted_keypoints, 3), dtype=np.float32)
            T_CW = np.linalg.inv(np.vstack((T_WC, np.array([0,0,0,1]))))[:3, :] # transformation matrix from world to camera frame
            M2 = K @ T_CW

            for i in range(n_promoted_keypoints):
                T_CW_first_observation = np.linalg.inv(np.vstack((T_WC_first_observation[i], np.array([0,0,0,1]))))[:3, :] # transformation matrix from world to camera frame
                M1 = K @ T_CW_first_observation
                landmark = cv2.triangulatePoints(projMatr1=M1, projMatr2=M2, projPoints1=promoted_keypoints_first_observations[i], projPoints2=promotable_keypoints_after_angle_threshold[i])
                landmark = (landmark / landmark[3])[:3].squeeze() # normalize homogeneous coordinates
                if (landmark[2]-T_WC[2,3] > min_depth and landmark[2]-T_WC[2,3] < max_depth): # and (landmark[0] > min_width and landmark[0] < max_width)
                    promoted_landmarks[i, :] = landmark # convert landmarks to non-homogeneous coordinates
                else:
                    promoted_landmarks[i, :] = np.nan
        
            mask = ~np.isnan(promoted_landmarks).any(axis=1)
            add_keypoints = promotable_keypoints_after_angle_threshold[mask]
            debug_dict["promotable_candidate_keypoints_outside_thresholds"] = promotable_keypoints_after_angle_threshold[~mask]
            debug_dict["promoted_candidate_keypoints"] = add_keypoints
            debug_dict["n_lost_candidates_at_cartesian_mask"] = promotable_keypoints_after_angle_threshold.shape[0] - add_keypoints.shape[0]
            debug_dict["n_promoted_keypoints"] = add_keypoints.shape[0]
            # Promote the keypoints
            keypoints = np.vstack((keypoints, add_keypoints))
            add_landmarks = promoted_landmarks[mask]
            landmarks = np.vstack((landmarks, add_landmarks))

            readd_keypoints = promotable_keypoints_after_angle_threshold[~mask]
            readd_first_observations = promoted_keypoints_first_observations[~mask]
            readd_pose_at_first_observations = promoted_keypoint_poses_prev[~mask]
            readd_keypoint_tracker = promoted_keypoint_tracker[~mask]

            # add keypoints that cannot be promoted back to candidate list
            candidate_keypoints = np.vstack((candidate_keypoints, readd_keypoints))
            first_observations = np.vstack((first_observations, readd_first_observations))
            pose_at_first_observations = np.vstack((pose_at_first_observations, readd_pose_at_first_observations))
            keypoint_tracker = np.hstack((keypoint_tracker, readd_keypoint_tracker))
    
        else:

            debug_dict["n_promoted_keypoints"] = 0
            debug_dict["n_lost_candidates_at_angle_filtering"] = 0
            debug_dict["n_promoted_after_angle_filter"] = 0
            debug_dict["n_lost_candidates_at_cartesian_mask"] = 0

            debug_dict["untriangulatable_promotable_candidate_keypoints"] = np.empty((0, 2), dtype=np.float32)
            debug_dict["promotable_candidate_keypoints_outside_thresholds"] = np.empty((0, 2), dtype=np.float32)
            debug_dict["promoted_candidate_keypoints"] = np.empty((0, 2), dtype=np.float32)

    else:
        candidate_keypoints = np.empty((0, 2), dtype=np.float32)
        first_observations = np.empty((0, 2), dtype=np.float32)
        pose_at_first_observations = np.empty((0, 12), dtype=np.float32)
        keypoint_tracker = np.empty(0, dtype=np.int64)
        debug_dict["n_promoted_keypoints"] = 0
        debug_dict["n_lost_candidates_at_angle_filtering"] = 0
        debug_dict["n_promoted_after_angle_filter"] = 0
        debug_dict["n_lost_candidates_at_cartesian_mask"] = 0
        debug_dict["trackable_unpromotable_candidate_keypoints"] = np.empty((0, 2), dtype=np.float32)
        debug_dict["untrackable_candidate_keypoints"] = np.empty((0, 2), dtype=np.float32)

        debug_dict["untriangulatable_promotable_candidate_keypoints"] = np.empty((0, 2), dtype=np.float32)
        debug_dict["promotable_candidate_keypoints_outside_thresholds"] = np.empty((0, 2), dtype=np.float32)
        debug_dict["promoted_candidate_keypoints"] = np.empty((0, 2), dtype=np.float32)

    # ------------------ Extract new keypoints and remove duplicates
    # OPTION 1: goodFeaturesToTrack
    new_candidate_keypoints = cv2.goodFeaturesToTrack(img, maxCorners=500, qualityLevel=0.01, minDistance=10, mask=None, blockSize=9, useHarrisDetector=True).squeeze() # dim: Kx2
    # #################### DEBUG START ####################
    # if verbose:
    #     # Plot the current image with new keypoints
    #     plt.imshow(img, cmap='gray')
    #     plt.scatter(new_candidate_keypoints.T[0, :], new_candidate_keypoints.T[1, :], s=5)
    #     plt.title('New Keypoints')
    #     plt.axis('equal')

    #     plt.tight_layout()
    #     plt.show(block=False)
    #     plt.pause(5)
    #     plt.close()
    # #################### DEBUG END ####################

    # OPTION 2: use functions from exercise 03
    # corner_patch_size = 9
    # harris_kappa = 0.08
    # num_keypoints = 200
    # nonmaximum_supression_radius = 8
    # harris_scores = harris(img, corner_patch_size, harris_kappa)
    # new_candidate_keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius).T

    # remove duplicates in keypoints and candidate keypoints
    distance_threshold = 2 # TODO: tune this parameter
    # Filter new candidate keypoints that are duplicates of existing keypoints
    is_duplicate_kp = np.any(np.linalg.norm(new_candidate_keypoints[:, None, :] - keypoints[None, :, :], axis=2) < distance_threshold, axis=1) # note: for broadcasting, dimensions have to match or be one
    debug_dict["candidate_keypoints_duplicate_with_keypoints"] = new_candidate_keypoints[is_duplicate_kp]
    new_candidate_keypoints = new_candidate_keypoints[~is_duplicate_kp]
    # Filter new candidate keypoints that are duplicates of previous candidate keypoints
    is_duplicate_ckp = np.any(np.linalg.norm(new_candidate_keypoints[:, None, :] - candidate_keypoints[None, :, :], axis=2) < distance_threshold, axis=1)
    debug_dict["candidate_keypoints_duplicate_with_prev_candidate_keypoints"] = new_candidate_keypoints[is_duplicate_ckp]
    new_candidate_keypoints = new_candidate_keypoints[~is_duplicate_ckp]
    debug_dict["new_candidate_keypoints"] = new_candidate_keypoints

    # update state
    candidate_keypoints = np.vstack((candidate_keypoints, new_candidate_keypoints))
    keypoint_tracker = np.hstack((keypoint_tracker, np.ones(new_candidate_keypoints.shape[0], dtype=np.int64)))
    first_observations = np.vstack((first_observations, new_candidate_keypoints))
    pose_at_first_observations = np.vstack((pose_at_first_observations, np.array([T_WC.flatten()]*new_candidate_keypoints.shape[0])))

    S = {
            "keypoints": keypoints, # dim: Kx2
            "landmarks": landmarks, # dim: Kx3
            "candidate_keypoints": candidate_keypoints, # dim: Mx2 with M = # candidate keypoints
            "first_observations": first_observations, # dim: Mx2 with M = # candidate keypoints
            "pose_at_first_observations": pose_at_first_observations, # dim: Mx12 with M = # candidate keypoints and 12 since the transformation matrix has dim 3x4 (omit last row)
            "keypoint_tracker": keypoint_tracker # dim: M with M = # candidate keypoints
        }

    return S, T_WC, debug_dict