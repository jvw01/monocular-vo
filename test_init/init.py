import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from init_fct import harris, selectKeypoints
from init_fct import describeKeypoints, matchDescriptors
from init_fct import plotMatches


img0 = cv2.imread(('../data/parking/images/img_00000.png'), cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread(('../data/parking/images/img_00002.png'), cv2.IMREAD_GRAYSCALE)

#img0 = cv2.imread(('../data/malaga/malaga-urban-dataset-extract-07_rectified_800x600_Images/img_CAMERA1_1261229981.580023_left.jpg'), cv2.IMREAD_GRAYSCALE)
#img1 = cv2.imread(('../data/malaga/malaga-urban-dataset-extract-07_rectified_800x600_Images/img_CAMERA1_1261229981.680019_left.jpg'), cv2.IMREAD_GRAYSCALE)


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
num_keypoints = 200
nonmaximum_supression_radius = 8
keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)
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
descriptors_2 = describeKeypoints(img1, keypoints_2, descriptor_radius)
    
matches = matchDescriptors(descriptors_2, descriptors, match_lambda)
    
plt.clf()
plt.close()
plt.imshow(img1, cmap='gray')
plt.plot(keypoints_2[1, :], keypoints_2[0, :], 'rx', linewidth=2)
plotMatches(matches, keypoints_2, keypoints)
plt.tight_layout()
plt.axis('off')
plt.show()
    
# Part 5 - Match descriptors between all images
prev_desc = None
prev_kp = None
for i in range(200):
    plt.clf()
    img = cv2.imread('../data/parking/images/img_{0:05d}.png'.format(i), cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread('../data/parking/images/img_{0:05d}.png'.format(i), cv2.IMREAD_GRAYSCALE)
    scores = harris(img, corner_patch_size, harris_kappa)
    kp = selectKeypoints(scores, num_keypoints, nonmaximum_supression_radius)
    desc = describeKeypoints(img, kp, descriptor_radius)
    
    plt.imshow(img, cmap='gray')
    plt.plot(kp[1, :], kp[0, :], 'rx', linewidth=2)
    plt.axis('off')
    
    if prev_desc is not None:
        matches = matchDescriptors(desc, prev_desc, match_lambda)
        plotMatches(matches, kp, prev_kp)
    prev_kp = kp
    prev_desc = desc
    
    plt.pause(0.1)


