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

def initialization(img0, img1, dataset, range_frames, left_images, K):
    
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
    plt.clf()
    plt.close()
    plt.imshow(img0, cmap='gray')
    plt.plot(keypoints[1, :], keypoints[0, :], 'rx', linewidth=2)
    plt.axis('off')
    plt.show()

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
    prev_desc = None
    prev_kp = None
    for i in range_frames:
        if i > 2:
            break
        plt.clf()
        if dataset == 0:
            img = cv2.imread(os.path.join('data/kitti/05/image_0/', f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        elif dataset == 1:
            img = cv2.imread(os.path.join('data/malaga/malaga-urban-dataset-extract-07_rectified_800x600_Images/', left_images[i]), cv2.IMREAD_GRAYSCALE)
        elif dataset == 2:
            img = cv2.imread(os.path.join(f"data/parking/images/img_{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        else:
            raise AssertionError("Invalid dataset selection")
        scores = harris(img, corner_patch_size, harris_kappa)
        kp = selectKeypoints(scores, num_keypoints, nonmaximum_supression_radius)
        desc = describeKeypoints(img, kp, descriptor_radius)
        

        #if prev_desc is not None:
        #    P = landmarks_3D (prev_desc, desc, K)

        #fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        #ax1 = fig.add_subplot(121, projection='3d')
        #ax1.scatter(P[0, :], P[1, :], P[2, :], marker='o')
        #fig.delaxes(axs[1])  # Remove the default axes of the second subplot
        #ax2 = fig.add_subplot(122)
        #ax2.imshow(img, cmap='gray')
        #ax2.plot(kp[1, :], kp[0, :], 'rx', linewidth=2)
        #ax2.axis('off')
    
        plt.imshow(img, cmap='gray')
        plt.plot(kp[1, :], kp[0, :], 'rx', linewidth=2)
        plt.axis('off')

        if prev_desc is not None:
            matches = matchDescriptors(desc, prev_desc, match_lambda)
            plotMatches(matches, kp, prev_kp)
        prev_kp = kp
        prev_desc = desc
        
        plt.pause(0.1)


def landmarks_3D (keypoints, keypoints_2, K ):
    #With Cv2 function: to compare : doesn't give the same 

    F, mask = cv2.findFundamentalMat(keypoints.T, keypoints_2.T, method=cv2.FM_8POINT)
    print("F:\n",F)
    E, mask = cv2.findEssentialMat(keypoints.T, keypoints_2.T, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    print("E:\n", E)
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

def landmarks_3D_with_masks(keypoints, keypoints_2, K):
    F, mask_F = cv2.findFundamentalMat(keypoints.T, keypoints_2.T, method=cv2.FM_8POINT)
    E, mask_E = cv2.findEssentialMat(keypoints.T, keypoints_2.T, K, method=cv2.RANSAC, prob=0.9, threshold=5.0)
    
    # Filter inliers for Essential
    inliers_E = (mask_E.ravel() == 1)
    pts1 = keypoints[:, inliers_E].T
    pts2 = keypoints_2[:, inliers_E].T
    # pts1 = keypoints.T
    # pts2 = keypoints_2.T

    print("points1 masked: ", pts1)
    print("points2 masked: ", pts2)
    
    # Recover pose only with inliers
    _, R, T, mask_pose = cv2.recoverPose(E, pts1, pts2, K, 5.0)
    inliers_pose = (mask_pose.ravel() == 1)
    # pts1 = pts1[inliers_pose, :]
    # pts2 = pts2[inliers_pose, :]

    
    M1 = np.dot(K, np.eye(3, 4))
    M2 = np.dot(K, np.c_[R, T])

    print("points1 masked: ", pts1)
    print("points2 masked: ", pts2)
    
    # Triangulate only the final inliers
    P = cv2.triangulatePoints(M1, M2, pts1.T, pts2.T)
    return P, R, T
