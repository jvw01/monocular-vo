import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import zscore

from test_init.init_fct import harris, selectKeypoints
from test_init.init_fct import describeKeypoints, matchDescriptors, linearTriangulation
from test_init.init_fct import plotMatches, decomposeEssentialMatrix, disambiguateRelativePose
from test_init.init_fct import fundamentalEightPointNormalized
from test_init.init_fct import ransacLocalization, drawCamera, getMatchedKeypoints

def initialization(img0, img1, dataset, left_images, K):
    
    print("ok ça fonctionne")
    corner_patch_size = 9
    harris_kappa = 0.08
        
    # Harris
    harris_scores = harris(img0, corner_patch_size, harris_kappa)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img0, cmap='gray')
    axs[0].axis('off')
    axs[1].imshow(harris_scores)
    axs[1].set_title('Harris Scores')
    axs[1].axis('off')  
    fig.tight_layout()
    plt.show()

    # Keypoints
    num_keypoints = 220
    nonmaximum_supression_radius = 8
    keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)
    print(keypoints)
    # plt.clf()
    # plt.close()
    # plt.imshow(img0, cmap='gray')
    # plt.plot(keypoints[1, :], keypoints[0, :], 'rx', linewidth=2)
    # plt.axis('off')
    # plt.show()

    descriptor_radius = 9
    match_lambda = 4
    # Describe keypoints 
    descriptors = describeKeypoints(img0, keypoints, descriptor_radius)

    # Match descriptors between images
    harris_scores_2 = harris(img1, corner_patch_size, harris_kappa)
    keypoints_2 = selectKeypoints(harris_scores_2, num_keypoints, nonmaximum_supression_radius)
    print(keypoints_2)
    descriptors_2 = describeKeypoints(img1, keypoints_2, descriptor_radius)
        
    matches = matchDescriptors(descriptors_2, descriptors, match_lambda)

    keypoints_1_matched, keypoints_2_matched = getMatchedKeypoints(matches, keypoints, keypoints_2)
    
    P, R, T = landmarks_3D(keypoints_1_matched, keypoints_2_matched, K)

    plt.clf()
    plt.close()
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(P[0, :], P[1, :], P[2, :], marker='o')
    ax1.set_title("3D Scatter Plot")
    fig.delaxes(axs[1])  # Remove the default axes of the second subplot
    ax2 = fig.add_subplot(122)
    ax2.imshow(img1, cmap='gray')
    ax2.plot(keypoints_2[1, :], keypoints_2[0, :], 'rx', linewidth=2)
    plotMatches(matches, keypoints_2, keypoints)  # This function plots matches
    ax2.set_title("Image with Keypoints")
    plt.tight_layout()

    #DOESNT WORK
    #drawCamera(ax1, np.zeros((3,)), np.eye(3), length_scale = 2)
    #ax1.text(-0.1,-0.1,-0.1,"Cam 1")

    #center_cam2_W = -R_best_W.T @ T_best_W
    #drawCamera(ax1, center_cam2_W, R_best_W.T, length_scale = 2)
    #ax1.text(center_cam2_W[0]-0.1, center_cam2_W[1]-0.1, center_cam2_W[2]-0.1,'Cam 2')
    plt.show()


    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(P[0,:], P[1,:], P[2,:], marker = 'o')
    #plt.axis('off')

    #matched_query_keypoints = keypoints[:, matches >= 0]
    #corresponding_matches = matches[matches >= 0]    
    #corresponding_landmarks = P[corresponding_matches, :]

    #out = ransacLocalization(matched_query_keypoints, P.T, K)
    #print(out)

    #plt.clf()
    #plt.close()
    #plt.imshow(img1, cmap='gray')
    #plt.plot(keypoints_2[1, :], keypoints_2[0, :], 'rx', linewidth=2)
    #plotMatches(matches, keypoints_2, keypoints)
    #plt.tight_layout()
    #plt.axis('off')
    #plt.show()
    
    # Part 4.5: Plot reprojection error
    print("3D Points, ", P)
    # reprojected_points = K [R T] @ P
    reprojection = K @ np.c_[R, T] @ P
    print(K)
    # reprojection = K @ np.eye(3,4) @ P
    reprojection_cartesian = reprojection[:2] / reprojection[2]
    # Plotting
    plt.clf()
    plt.close()
    plt.imshow(img1, cmap='gray')

    # Plot keypoints
    plt.plot(keypoints[1, :], keypoints[0, :], 'rx', linewidth=2, label='Keypoints')

    # Plot reprojected points
    plt.plot(reprojection_cartesian[1, :], reprojection_cartesian[0, :], 'go', linewidth=2, label='Reprojected 3D points')

    # Formatting the plot
    plt.axis('off')
    plt.legend()
    plt.show()
    
        
    # Part 5 - Match descriptors between all images
    # prev_desc = None
    # prev_kp = None
    # for i in range_frames:
    #     plt.clf()
    #     if dataset == 0:
    #         img = cv2.imread(os.path.join('data/kitti/05/image_0/', f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
    #     elif dataset == 1:
    #         img = cv2.imread(os.path.join('data/malaga/malaga-urban-dataset-extract-07_rectified_800x600_Images/', left_images[i]), cv2.IMREAD_GRAYSCALE)
    #     elif dataset == 2:
    #         img = cv2.imread(os.path.join(f"data/parking/images/img_{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
    #     else:
    #         raise AssertionError("Invalid dataset selection")
    #     scores = harris(img, corner_patch_size, harris_kappa)
    #     kp = selectKeypoints(scores, num_keypoints, nonmaximum_supression_radius)
    #     desc = describeKeypoints(img, kp, descriptor_radius)
        

    #     #if prev_desc is not None:
    #     #    P = landmarks_3D (prev_desc, desc, K)

    #     #fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    #     #ax1 = fig.add_subplot(121, projection='3d')
    #     #ax1.scatter(P[0, :], P[1, :], P[2, :], marker='o')
    #     #fig.delaxes(axs[1])  # Remove the default axes of the second subplot
    #     #ax2 = fig.add_subplot(122)
    #     #ax2.imshow(img, cmap='gray')
    #     #ax2.plot(kp[1, :], kp[0, :], 'rx', linewidth=2)
    #     #ax2.axis('off')
    
    #     plt.imshow(img, cmap='gray')
    #     plt.plot(kp[1, :], kp[0, :], 'rx', linewidth=2)
    #     plt.axis('off')

    #     if prev_desc is not None:
    #         matches = matchDescriptors(desc, prev_desc, match_lambda)
    #         plotMatches(matches, kp, prev_kp)
    #     prev_kp = kp
    #     prev_desc = desc
        
    #     plt.pause(0.1)

    return keypoints, P 


def initialization_cv2(img0, img1, dataset, K, left_images=0, verbosity=1):
    """
    Initialization using cv2 Harris detection, cv2 descriptor computation, 
    and cv2 feature matching. Replicates the same plots as the original 'initialization'.
    """
    # -----------------------------
    # 1) Harris Corner Detection NOT ACTUALLY USED
    # -----------------------------
    corner_patch_size = 9    # blockSize for cv2.cornerHarris
    harris_kappa      = 0.08 # k for cv2.cornerHarris
    ksize             = 3    # Aperture parameter for the Sobel operator in cornerHarris
    

    if verbosity > 0: # Only do harris computation, if verbosity is high for visualization
        # Convert images to float32 if necessary for Harris
        img0_float = np.float32(img0)
        img1_float = np.float32(img1)

        # Compute Harris response
        harris_scores_1 = cv2.cornerHarris(img0_float, blockSize=corner_patch_size, ksize=ksize, k=harris_kappa)
        harris_scores_2 = cv2.cornerHarris(img1_float, blockSize=corner_patch_size, ksize=ksize, k=harris_kappa)

        # For visualization, normalize/clip Harris scores so they're easier to see
        # (optional step, depends on your preference)
        harris_display_1 = np.clip(harris_scores_1, 0, None)
        harris_display_2 = np.clip(harris_scores_2, 0, None)

        # Plot: left = original image0, right = Harris scores
        fig, axs = plt.subplots(1, 2, figsize=(10,4))
        axs[0].imshow(img0, cmap='gray')
        axs[0].axis('off')
        axs[0].set_title('Image 0')
        axs[1].imshow(harris_display_1, cmap='jet')
        axs[1].axis('off')
        axs[1].set_title('Harris Scores (Image 0)')
        fig.tight_layout()
        plt.show()

    # -----------------------------
    # 2) Keypoint Detection (Harris-based)
    #    Using cv2.goodFeaturesToTrack with the Harris detector
    # -----------------------------
    maxCorners    = 400       
    qualityLevel  = 0.01
    minDistance   = 8         # Similar to 'nonmaximum_supression_radius'
    blockSize     = corner_patch_size  # same as above for consistency

    pts1 = cv2.goodFeaturesToTrack(
        image=img0, 
        maxCorners=maxCorners, 
        qualityLevel=qualityLevel, 
        minDistance=minDistance, 
        blockSize=blockSize,
        useHarrisDetector=True,
        k=harris_kappa
    )
    pts2 = cv2.goodFeaturesToTrack(
        image=img1, 
        maxCorners=maxCorners, 
        qualityLevel=qualityLevel, 
        minDistance=minDistance, 
        blockSize=blockSize,
        useHarrisDetector=True,
        k=harris_kappa
    )

    pts1 = np.squeeze(pts1).T  #  (2, N)
    pts2 = np.squeeze(pts2).T  #  (2, N)

    # -----------------------------
    # 3) Descriptor Extraction
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
    # 4) Descriptor Matching
    #    Using a simple Brute Force with Hamming distance
    # -----------------------------
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    # Sort by distance
    matches = sorted(matches, key=lambda x: x.distance)


    # Extract matched keypoints in (2,N) shape
    matched_pts1 = []
    matched_pts2 = []
    for m in matches:
        # queryIdx is index into desc1/kp1, trainIdx is index into desc2/kp2
        matched_pts1.append(kp1[m.queryIdx].pt)
        matched_pts2.append(kp2[m.trainIdx].pt)

    matched_pts1 = np.array(matched_pts1).T  # shape (2, numMatches)
    matched_pts2 = np.array(matched_pts2).T  # shape (2, numMatches)

    # -----------------------------
    # 5) Triangulation or 3D Landmarks
    #    Using landmarks_3D
    # -----------------------------
    # P is shape (3, N), R is (3,3) rotation, T is (3,1) translation
    P, R, T, matched_pts2 = landmarks_3D_cv2(matched_pts1, matched_pts2, K)

    # -----------------------------
    # 6) Visualization
    #    A) 3D scatter
    #    B) Keypoints in second image
    #    C) Matches 
    # -----------------------------
    # Make a figure with 2 subplots: one for the 3D scatter, one for the 2D image
    if verbosity > 0:
        plt.clf()
        plt.close()
        fig = plt.figure(figsize=(12, 6))

        # Subplot 1: 3D scatter
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.scatter(P[0, :], P[1, :], P[2, :], marker='o')
        ax1.set_title("3D Scatter Plot")

        # Subplot 2: second image with keypoints
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(img1, cmap='gray')
        # Plot keypoints
        ax2.plot(pts2[0, :], pts2[1, :], 'rx', linewidth=2, label='Keypoints (Img 1)')

        # matched_pts1 -> matched_pts2, 
        # cv2.drawMatches(img0, kp1, img1, kp2, matches, img1, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        ax2.set_title("Image 1 with Detected Keypoints")
        ax2.set_axis_off()
        plt.tight_layout()
        plt.show()

        # -----------------------------
        # 7) Reprojection of 3D points
        # -----------------------------
        # [R|T] is 3x4, P is 4xN 
        reprojection = K @ np.hstack((R, T)) @ P  # shape (3, N)
        reprojection_cartesian = reprojection[:2] / reprojection[2]

        # Plot reprojection vs. original keypoints 
        plt.clf()
        plt.close()
        plt.imshow(img1, cmap='gray')

        # plot matched_pts2 (the 2D corners from the second image in the match)
        plt.plot(matched_pts2[0, :], matched_pts2[1, :], 'rx', linewidth=2, label='Keypoints (matched)')

        # plot the reprojected 3D points
        plt.plot(reprojection_cartesian[0, :], reprojection_cartesian[1, :], 'go', linewidth=2, label='Reprojected 3D points')

        plt.axis('off')
        plt.legend()
        plt.show()

        print("3D Points:\n", P)
        print("Calibration Matrix K:\n", K)
        print("Rotation:\n", R)
        print("Translation:\n", T)

    # We return the *matched keypoints from image1* (2,N) and the 3D points
    return matched_pts2, P


    

def landmarks_3D (keypoints, keypoints_2, K ):
    #With Cv2 function: to compare : doesn't give the same 

    F, mask = cv2.findFundamentalMat(keypoints.T, keypoints_2.T, method=cv2.FM_8POINT)
    # print("F:\n",F)
    E, mask = cv2.findEssentialMat(keypoints.T, keypoints_2.T, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    # print("E:\n", E)
    _, R, T, _ = cv2.recoverPose(E, keypoints.T, keypoints_2.T, K)
    print(R, T)
    print(T)
    
    # #Without Cv2 function 
    # keypoints1 = np.r_[keypoints, np.ones((1, keypoints.shape[1]))]
    # keypoints2 = np.r_[keypoints_2, np.ones((1, keypoints_2.shape[1]))]

    # F = fundamentalEightPointNormalized(keypoints1, keypoints2)
    # E = K.T @ F @ K
    # print("E:\n", E)
    # R_1, u3 = decomposeEssentialMatrix(E)
    # R,T = disambiguateRelativePose(R_1, u3, keypoints1, keypoints2, K)
    # #print("Rotation Matrix R:", R)
    # #print("Translation Vector T, u3:", T)


    M1 = np.dot(K, np.eye(3, 4))
    M2 = np.dot (K, np.c_[R, T])
    # P = linearTriangulation(keypoints1, keypoints2, M1, M2)
    P = cv2.triangulatePoints( M1, M2 ,keypoints, keypoints_2)
    print(P)
    
    #z-score remove outliers could use whatever else
    # points_3d = P[:3, :] 
    # z_scores = np.abs(zscore(points_3d, axis=1))
    # outliers = np.any(z_scores > 3, axis=0)
    # P = P[:, ~outliers]
    
    return P, R, T

def landmarks_3D_cv2 (keypoints, keypoints_2, K, verbosity=1):
    E, mask = cv2.findEssentialMat(keypoints.T, keypoints_2.T, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    keypoints = np.expand_dims(keypoints.T, 1)[mask == 1].T
    keypoints_2 = np.expand_dims(keypoints_2.T, 1)[mask == 1].T
    _, R, T, _ = cv2.recoverPose(E, keypoints.T, keypoints_2.T, K)
    if verbosity > 0:
        print(R, T)
        print(T)
    M1 = np.dot(K, np.eye(3, 4))
    M2 = np.dot (K, np.c_[R, T])
    P = cv2.triangulatePoints( M1, M2, keypoints, keypoints_2)
    if verbosity > 0:
        print(P)
    
    return P, R, T, keypoints_2