import numpy as np
import cv2
import matplotlib.pyplot as plt


def bootstrapping(img0, img1, K):
    """
    Perform bootstrapping for visual odometry using two initial images.

    This function initializes the visual odometry pipeline by detecting and matching keypoints
    between two consecutive images, estimating the essential matrix, recovering the pose, and
    triangulating 3D points.

    Parameters:
        img0 (ndarray): The first image (grayscale).
        img1 (ndarray): The second image (grayscale).
        K (ndarray): The camera calibration matrix.

    Returns:
        matched_pts2 (ndarray): Matched keypoints from the second image.
        P (ndarray): Triangulated 3D points.
    """

    # -----------------------------
    # 1) Keypoint Detection
    # -----------------------------
    maxCorners = 1300
    qualityLevel = 0.005
    minDistance = 10
    blockSize = 9

    pts1 = cv2.goodFeaturesToTrack(
        image=img0,
        maxCorners=maxCorners,
        qualityLevel=qualityLevel,
        minDistance=minDistance,
    )
    pts2 = cv2.goodFeaturesToTrack(
        image=img1,
        maxCorners=maxCorners,
        qualityLevel=qualityLevel,
        minDistance=minDistance,
    )

    pts1 = np.squeeze(pts1).T  #  (2, N)
    pts2 = np.squeeze(pts2).T  #  (2, N)

    # -----------------------------
    # 2) Descriptor Extraction
    #    Using SIFT
    # -----------------------------
    sift = cv2.SIFT_create()

    # Convert corners to KeyPoint objects
    kp1 = [cv2.KeyPoint(x=float(x), y=float(y), size=blockSize) for (x, y) in pts1.T]
    kp2 = [cv2.KeyPoint(x=float(x), y=float(y), size=blockSize) for (x, y) in pts2.T]

    # Compute descriptors for these specific keypoints
    kp1, desc1 = sift.compute(img0, kp1)
    kp2, desc2 = sift.compute(img1, kp2)

    # -----------------------------
    # 3) Descriptor Matching
    # -----------------------------
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)  # sort by distance

    # Extract matched keypoints in (2,N) shape
    matched_pts1 = []
    matched_pts2 = []
    for m in matches:
        matched_pts1.append(kp1[m.queryIdx].pt)
        matched_pts2.append(kp2[m.trainIdx].pt)

    matched_pts1 = np.array(matched_pts1).T  # shape (2, numMatches)
    matched_pts2 = np.array(matched_pts2).T  # shape (2, numMatches)

    # -----------------------------
    # 4) Triangulation or 3D Landmarks
    # -----------------------------
    # P is shape (3, N), R is (3,3) rotation, T is (3,1) translation
    P, matched_pts2 = triangulation(matched_pts1, matched_pts2, K)

    # We return the *matched keypoints from image1* (2,N) and the 3D points
    return matched_pts2, P


def triangulation(keypoints, keypoints_2, K):
    E, mask = cv2.findEssentialMat(
        keypoints.T, keypoints_2.T, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    keypoints = np.expand_dims(keypoints.T, 1)[mask == 1].T
    keypoints_2 = np.expand_dims(keypoints_2.T, 1)[mask == 1].T
    _, R, T, _ = cv2.recoverPose(E, keypoints.T, keypoints_2.T, K)

    M1 = np.dot(K, np.eye(3, 4))
    M2 = np.dot(K, np.c_[R, T])
    P = cv2.triangulatePoints(M1, M2, keypoints, keypoints_2)
    P = (P / P[3])[:3].squeeze()

    mask = (P[2] > 0) & (P[2] < 100)
    P = P[:, mask]
    keypoints_2 = keypoints_2[:, mask]

    return P, keypoints_2
