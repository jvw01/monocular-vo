import cv2
import numpy as np


def processFrame(
    img: np.ndarray, img_prev: np.ndarray, S_prev: dict, params: dict
) -> tuple[dict, np.ndarray]:
    """
    This function implements the continuous visual odometry pipeline in a Markovian way.

    Parameters:
        img: current frame (query image)
        img_prev: previous frame (database image)
        S_prev: state of previous frame (i.e., the keypoints in the previous frame and the 3D landmarks associated to them)
        params: parameters for VO pipeline (e.g., camera matrix)

    Returns:
        S: state of current frame (i.e., the keypoints in the current frame and the 3D landmarks associated to them)
        T_WC: current pose
        debug_dict: dictionary containing debug information
    """

    # tuned parameters
    K = params["K"]
    L_m = params["L_m"]
    min_depth = params["min_depth"]
    max_depth = params["max_depth"]
    angle_threshold_for_triangulation = params["angle_threshold_for_triangulation"]
    distance_threshold = params["distance_threshold"]

    klt_parameters = {
        "winSize": (10, 10),
        "maxLevel": 5,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.05),
    }

    # dict for visualization and debugging
    debug_dict = {}

    # ------------------------------------------------------ 4.1: Associating keypoints
    # track keypoints from previous frame to current frame with KLT (i.e. pixel coordinates)
    keypoints_prev = S_prev["keypoints"]  # dim: Kx2
    landmarks = S_prev["landmarks"]  # dim: Kx3
    keypoints, status, _ = cv2.calcOpticalFlowPyrLK(
        prevImg=img_prev,
        nextImg=img,
        prevPts=keypoints_prev,
        nextPts=None,
        **klt_parameters,
    )

    # filter valid keypoints (note: status is set to 1 if the flow for the corresponding features has been found)
    debug_dict["untrackable_keypoints"] = np.expand_dims(keypoints, 1)[status == 0]
    keypoints = np.expand_dims(keypoints, 1)[status == 1]  # dim: Kx2
    landmarks = np.expand_dims(landmarks, 1)[status == 1]  # dim: Kx3
    debug_dict["n_tracked_keypoints"] = keypoints.shape[0]

    # ------------------------------------------------------ 4.2: Estimating current pose
    # extract the pose using P3P
    _, rvec_CW, tvec_CW, inliers = cv2.solvePnPRansac(
        objectPoints=landmarks,
        imagePoints=keypoints,
        cameraMatrix=K,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_P3P,
    )  # rvec, tvec are the rotation and translation vectors from world frame to camera frame

    outlier_mask = np.ones(keypoints.shape[0], dtype=bool)
    outlier_mask[inliers] = False
    debug_dict["trackable_outlier_keypoints"] = keypoints[outlier_mask]
    keypoints = keypoints[inliers].squeeze()  # dim: Kx2
    debug_dict["trackable_keypoints"] = keypoints
    landmarks = landmarks[inliers].squeeze()  # dim: Kx3

    rotation_matrix_CW, _ = cv2.Rodrigues(rvec_CW)

    rotation_matrix_WC = rotation_matrix_CW.T.astype(np.float32)
    tvec_WC = -rotation_matrix_WC @ tvec_CW.astype(np.float32)
    T_WC = np.hstack((rotation_matrix_WC, tvec_WC))  # dim: 3x4

    # ------------------------------------------------------ 4.3: Triangulating new landmarks
    if isinstance(S_prev["candidate_keypoints"], np.ndarray):
        # ------------------ Check existing candidate keypoints from previous frame(s)
        candidate_keypoints_prev = S_prev["candidate_keypoints"]
        candidate_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(
            prevImg=img_prev,
            nextImg=img,
            prevPts=candidate_keypoints_prev,
            nextPts=None,
            **klt_parameters,
        )

        # remove candidate keypoints that were not tracked successfully
        debug_dict["untrackable_candidate_keypoints"] = np.expand_dims(
            candidate_keypoints, 1
        )[status == 0]
        candidate_keypoints = np.expand_dims(candidate_keypoints, 1)[
            status == 1
        ]  # dim: Kx2
        debug_dict["n_trackable_candidate_keypoints"] = candidate_keypoints.shape[0]
        debug_dict["n_untrackable_candidate_keypoints"] = (
            candidate_keypoints_prev.shape[0] - candidate_keypoints.shape[0]
        )
        S_prev["pose_at_first_observations"] = np.expand_dims(
            S_prev["pose_at_first_observations"], 1
        )[status == 1]
        S_prev["first_observations"] = np.expand_dims(S_prev["first_observations"], 1)[
            status == 1
        ]
        S_prev["keypoint_tracker"] = (
            np.expand_dims(S_prev["keypoint_tracker"], 1)[status == 1] + 1
        )

        # ------------------ Promote keypoints
        promotable_keypoints = candidate_keypoints[S_prev["keypoint_tracker"] > L_m]
        n_promotable_keypoints = promotable_keypoints.shape[0]
        debug_dict[
            "n_promotable_keypoints_before_angle_filtering"
        ] = n_promotable_keypoints

        # remove promotable keypoints from candidate list -> keypoints that can eventually not be promoted will be added again
        candidate_keypoints = candidate_keypoints[S_prev["keypoint_tracker"] <= L_m]
        debug_dict["trackable_unpromotable_candidate_keypoints"] = candidate_keypoints
        first_observations = S_prev["first_observations"][
            S_prev["keypoint_tracker"] <= L_m
        ]
        pose_at_first_observations = S_prev["pose_at_first_observations"][
            S_prev["keypoint_tracker"] <= L_m
        ]
        keypoint_tracker = S_prev["keypoint_tracker"][S_prev["keypoint_tracker"] <= L_m]

        # If any keypoint has been promoted, attempt to perform triangulation (requires minimum baseline)
        if n_promotable_keypoints > 0:
            print(f"There are promotable keypoints ({n_promotable_keypoints})")
            promotable_keypoints_first_observations = S_prev["first_observations"][
                S_prev["keypoint_tracker"] > L_m
            ]
            promotable_keypoints_initial_poses = S_prev["pose_at_first_observations"][
                S_prev["keypoint_tracker"] > L_m
            ]
            promotable_keypoints_tracker = S_prev["keypoint_tracker"][
                S_prev["keypoint_tracker"] > L_m
            ]

            # use bearing vectors to calculate the angle between first observation and current observation -> v1 and v2 point to the same landmark
            K_inv = np.linalg.inv(K)
            T_WC_first_observation = promotable_keypoints_initial_poses.reshape(
                -1, 3, 4
            )
            R_CC = (
                T_WC[:3, :3].T @ T_WC_first_observation[:, :3, :3]
            )  # rotation matrix from first camera frame to current camera frame
            R_CC_transposed = np.transpose(R_CC, axes=(0, 2, 1))
            v1 = (
                (
                    np.matmul(R_CC_transposed, K_inv)
                    @ (
                        np.hstack(
                            (
                                promotable_keypoints_first_observations,
                                np.ones((n_promotable_keypoints, 1)),
                            )
                        )
                    )[:, :, None]
                )
                .squeeze()
                .T
            )  # v1 = R⁽⁻¹⁾ * K⁽⁻¹⁾ * [u1; v1; 1]; dim 3xK
            v2 = (
                K_inv
                @ np.hstack(
                    (promotable_keypoints, np.ones((n_promotable_keypoints, 1)))
                ).T
            )  # v2 = K⁽⁻¹⁾ * [u2; v2; 1] (no need for rotation since we are already in correct frame); dim 3xK
            alpha = np.arccos(
                np.sum(v1 * v2, axis=0)
                / (np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0))
            )  # angle between bearing vectors in radians
            triangulate = (
                alpha > angle_threshold_for_triangulation
            )  # mask that indicates if the angle between the bearing vectors is large enough for triangulation
            # filter keypoints that can be promoted (angle between bearing vectors is large enough)
            debug_dict[
                "untriangulatable_promotable_candidate_keypoints"
            ] = promotable_keypoints[~triangulate]
            promotable_keypoints_after_angle_threshold = promotable_keypoints[
                triangulate
            ]
            debug_dict["n_lost_candidates_at_angle_filtering"] = (
                promotable_keypoints.shape[0]
                - promotable_keypoints_after_angle_threshold.shape[0]
            )
            n_promoted_keypoints = promotable_keypoints_after_angle_threshold.shape[0]
            debug_dict["n_promoted_after_angle_filter"] = n_promoted_keypoints

            # Re-add unpromotable keypoints due to angle threshold (the angle might increase in next frames, and then they can be triangulated)
            readd_keypoints = promotable_keypoints[~triangulate]
            readd_first_observations = promotable_keypoints_first_observations[
                ~triangulate
            ]
            readd_pose_at_first_observations = promotable_keypoints_initial_poses[
                ~triangulate
            ]
            readd_keypoint_tracker = promotable_keypoints_tracker[~triangulate]

            # add keypoints that cannot be promoted back to candidate list
            candidate_keypoints = np.vstack((candidate_keypoints, readd_keypoints))
            first_observations = np.vstack(
                (first_observations, readd_first_observations)
            )
            pose_at_first_observations = np.vstack(
                (pose_at_first_observations, readd_pose_at_first_observations)
            )
            keypoint_tracker = np.hstack((keypoint_tracker, readd_keypoint_tracker))

            if n_promoted_keypoints > 0:
                # triangulate landmarks with least squares approximation
                promoted_keypoints_first_observations = (
                    promotable_keypoints_first_observations[triangulate]
                )
                promoted_keypoint_poses_prev = promotable_keypoints_initial_poses[
                    triangulate
                ]
                promoted_keypoint_tracker = promotable_keypoints_tracker[triangulate]
                T_WC_first_observation = promoted_keypoint_poses_prev.reshape(-1, 3, 4)

                promoted_landmarks = np.empty(
                    (n_promoted_keypoints, 3), dtype=np.float32
                )
                R_CW = T_WC[:3, :3].T
                t_CW = -R_CW @ T_WC[:3, 3]
                T_CW = np.hstack((R_CW, t_CW[:, None]))
                M2 = K @ T_CW

                # Loop through all groups of candidate keypoints (groups were found in the same original image and hence can be triangulated together)
                debug_dict["n_promoted_keypoints"] = 0
                debug_dict[
                    "promotable_candidate_keypoints_outside_thresholds"
                ] = np.empty((0, 2), dtype=np.float32)
                debug_dict["promoted_candidate_keypoints"] = np.empty(
                    (0, 2), dtype=np.float32
                )
                debug_dict["n_lost_candidates_at_cartesian_mask"] = 0
                for i in range(n_promoted_keypoints):

                    R_CW_first_observation = T_WC_first_observation[i, :3, :3].T
                    t_CW_first_observation = (
                        -R_CW_first_observation
                        @ T_WC_first_observation[i, :3, 3][:, None]
                    )
                    T_CW_first_observation = np.hstack(
                        (R_CW_first_observation, t_CW_first_observation)
                    )
                    M1 = K @ T_CW_first_observation
                    promoted_landmark = cv2.triangulatePoints(
                        projMatr1=M1,
                        projMatr2=M2,
                        projPoints1=promoted_keypoints_first_observations[i],
                        projPoints2=promotable_keypoints_after_angle_threshold[i],
                    )
                    promoted_landmark = (promoted_landmark / promoted_landmark[3])[
                        :3
                    ].squeeze()  # normalize homogeneous coordinates

                    promoted_landmark_C2_frame = T_CW @ np.hstack(
                        (promoted_landmark, np.ones(1))
                    )
                    mask = (promoted_landmark_C2_frame[2] > min_depth) & (
                        promoted_landmark_C2_frame[2] < max_depth
                    )

                    if mask == True:
                        add_keypoint = promotable_keypoints_after_angle_threshold[i]

                        debug_dict["promoted_candidate_keypoints"] = np.vstack(
                            (debug_dict["promoted_candidate_keypoints"], add_keypoint)
                        )
                        debug_dict["n_promoted_keypoints"] += 1

                        # Promote the keypoints
                        keypoints = np.vstack((keypoints, add_keypoint))
                        add_landmark = promoted_landmark
                        landmarks = np.vstack((landmarks, add_landmark))

                    else:
                        debug_dict[
                            "promotable_candidate_keypoints_outside_thresholds"
                        ] = np.vstack(
                            (
                                debug_dict[
                                    "promotable_candidate_keypoints_outside_thresholds"
                                ],
                                promotable_keypoints_after_angle_threshold[i],
                            )
                        )
                        debug_dict["n_lost_candidates_at_cartesian_mask"] += 1

                        # add keypoints that cannot be promoted back to candidate list
                        candidate_keypoints = np.vstack(
                            (
                                candidate_keypoints,
                                promotable_keypoints_after_angle_threshold[i],
                            )
                        )
                        first_observations = np.vstack(
                            (
                                first_observations,
                                promoted_keypoints_first_observations[i],
                            )
                        )
                        pose_at_first_observations = np.vstack(
                            (
                                pose_at_first_observations,
                                promoted_keypoint_poses_prev[i],
                            )
                        )
                        keypoint_tracker = np.hstack(
                            (keypoint_tracker, promoted_keypoint_tracker[i])
                        )
            else:
                debug_dict[
                    "promotable_candidate_keypoints_outside_thresholds"
                ] = np.empty((0, 2), dtype=np.float32)
                debug_dict["promoted_candidate_keypoints"] = np.empty(
                    (0, 2), dtype=np.float32
                )
                debug_dict["n_lost_candidates_at_cartesian_mask"] = 0
                debug_dict["n_promoted_keypoints"] = 0
        else:
            debug_dict["n_promoted_keypoints"] = 0
            debug_dict["n_lost_candidates_at_angle_filtering"] = 0
            debug_dict["n_promoted_after_angle_filter"] = 0
            debug_dict["n_lost_candidates_at_cartesian_mask"] = 0

            debug_dict["untriangulatable_promotable_candidate_keypoints"] = np.empty(
                (0, 2), dtype=np.float32
            )
            debug_dict["promotable_candidate_keypoints_outside_thresholds"] = np.empty(
                (0, 2), dtype=np.float32
            )
            debug_dict["promoted_candidate_keypoints"] = np.empty(
                (0, 2), dtype=np.float32
            )

    else:
        candidate_keypoints = np.empty((0, 2), dtype=np.float32)
        first_observations = np.empty((0, 2), dtype=np.float32)
        pose_at_first_observations = np.empty((0, 12), dtype=np.float32)
        keypoint_tracker = np.empty(0, dtype=np.int64)
        debug_dict["n_promotable_keypoints_before_angle_filtering"] = 0
        debug_dict["n_promoted_keypoints"] = 0
        debug_dict["n_lost_candidates_at_angle_filtering"] = 0
        debug_dict["n_promoted_after_angle_filter"] = 0
        debug_dict["n_lost_candidates_at_cartesian_mask"] = 0
        debug_dict["trackable_unpromotable_candidate_keypoints"] = np.empty(
            (0, 2), dtype=np.float32
        )
        debug_dict["untrackable_candidate_keypoints"] = np.empty(
            (0, 2), dtype=np.float32
        )

        debug_dict["untriangulatable_promotable_candidate_keypoints"] = np.empty(
            (0, 2), dtype=np.float32
        )
        debug_dict["promotable_candidate_keypoints_outside_thresholds"] = np.empty(
            (0, 2), dtype=np.float32
        )
        debug_dict["promoted_candidate_keypoints"] = np.empty((0, 2), dtype=np.float32)

    # ------------------ Extract new keypoints and remove duplicates
    new_candidate_keypoints = cv2.goodFeaturesToTrack(
        img, maxCorners=1400, qualityLevel=0.1, minDistance=10
    ).squeeze()  # dim: Kx2

    # remove duplicates in keypoints and candidate keypoints
    is_duplicate_kp = np.any(
        np.linalg.norm(
            new_candidate_keypoints[:, None, :] - keypoints[None, :, :], axis=2
        )
        <= distance_threshold,
        axis=1,
    )  # note: for broadcasting, dimensions have to match or be one
    debug_dict[
        "candidate_keypoints_duplicate_with_keypoints"
    ] = new_candidate_keypoints[is_duplicate_kp]
    new_candidate_keypoints = new_candidate_keypoints[~is_duplicate_kp]
    is_duplicate_ckp = np.any(
        np.linalg.norm(
            new_candidate_keypoints[:, None, :] - candidate_keypoints[None, :, :],
            axis=2,
        )
        <= distance_threshold,
        axis=1,
    )
    debug_dict[
        "candidate_keypoints_duplicate_with_prev_candidate_keypoints"
    ] = new_candidate_keypoints[is_duplicate_ckp]
    new_candidate_keypoints = new_candidate_keypoints[~is_duplicate_ckp]
    debug_dict["new_candidate_keypoints"] = new_candidate_keypoints

    # update state
    if new_candidate_keypoints.shape[0] > 0:
        candidate_keypoints = np.vstack((candidate_keypoints, new_candidate_keypoints))
        keypoint_tracker = np.hstack(
            (
                keypoint_tracker,
                np.ones(new_candidate_keypoints.shape[0], dtype=np.int64),
            )
        )
        first_observations = np.vstack((first_observations, new_candidate_keypoints))
        pose_at_first_observations = np.vstack(
            (
                pose_at_first_observations,
                np.array([T_WC.flatten()] * new_candidate_keypoints.shape[0]),
            )
        )

    S = {
        "keypoints": keypoints,  # dim: Kx2
        "landmarks": landmarks,  # dim: Kx3
        "candidate_keypoints": candidate_keypoints,  # dim: Mx2 with M = # candidate keypoints
        "first_observations": first_observations,  # dim: Mx2 with M = # candidate keypoints
        "pose_at_first_observations": pose_at_first_observations,  # dim: Mx12 with M = # candidate keypoints and 12 since the transformation matrix has dim 3x4 (omit last row)
        "keypoint_tracker": keypoint_tracker,  # dim: M with M = # candidate keypoints
    }

    return S, T_WC, debug_dict
