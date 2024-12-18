import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from test_init.init import initialization

# Setup
dataset = 2 # 0: KITTI, 1: Malaga, 2: parking
parking_path = "" 
malaga_path = ""
kitti_path = ""


if dataset == 0:
    assert 'kitti_path' in locals()
    ground_truth = np.loadtxt(os.path.join(kitti_path, 'data/kitti/poses/05.txt'))
    ground_truth = ground_truth[:, [-9, -1]] 
    left_images = 0
    print(ground_truth)
    print(len(ground_truth))
    plt.plot(ground_truth[:,0],ground_truth[:,1])
    plt.show()
    last_frame = 4540
    K = np.array([
        [718.856, 0, 607.1928],
        [0, 718.856, 185.2157],
        [0, 0, 1]
    ])
elif dataset == 1:
    assert 'malaga_path' in locals()
    images = sorted(os.listdir(os.path.join(malaga_path, 
                  'data/malaga/malaga-urban-dataset-extract-07_rectified_800x600_Images/')))
    left_images = images[0::2]
    #print(left_images) #correct
    last_frame = len(left_images) 
    #print(last_frame) #2121 correct
    K = np.array([
        [621.18428, 0, 404.0076],
        [0, 621.18428, 309.05989],
        [0, 0, 1]
    ])
elif dataset == 2:
    assert 'parking_path' in locals()
    last_frame = 598
    left_images = 0
    K = np.loadtxt(os.path.join(parking_path, 'data/parking/K.txt'))
    #print(K)
    ground_truth = np.loadtxt(os.path.join(parking_path, 'data/parking/poses.txt'))
    ground_truth = ground_truth[:, [-9, -1]] #correct (matlab=[end-8 end])
    print(ground_truth)
else:
    raise AssertionError("Invalid dataset selection")

# Bootstrap
# Need to set bootstrap_frames
bootstrap_frames = [0, 2]

if dataset == 0:
    img0 = cv2.imread(os.path.join(kitti_path, 'data/kitti/05/image_0/', 
                    f"{bootstrap_frames[0]:06d}.png"), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(os.path.join(kitti_path, 'data/kitti/05/image_0/', 
                    f"{bootstrap_frames[1]:06d}.png"), cv2.IMREAD_GRAYSCALE)
elif dataset == 1:
    img0 = cv2.imread(os.path.join(malaga_path, 
                    'data/malaga/malaga-urban-dataset-extract-07_rectified_800x600_Images/', 
                    left_images[bootstrap_frames[0]]), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(os.path.join(malaga_path, 
                    'data/malaga/malaga-urban-dataset-extract-07_rectified_800x600_Images/', 
                    left_images[bootstrap_frames[1]]), cv2.IMREAD_GRAYSCALE)
elif dataset == 2:
    img0 = cv2.imread(os.path.join(parking_path, 
                    f"data/parking/images/img_{bootstrap_frames[0]:05d}.png"), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(os.path.join(parking_path, 
                    f"data/parking/images/img_{bootstrap_frames[1]:05d}.png"), cv2.IMREAD_GRAYSCALE)
else:
    raise AssertionError("Invalid dataset selection")


# Continuous operation
range_frames = range(bootstrap_frames[1] + 1, last_frame + 1)
prev_img = None

initialization(img0,img1,dataset, range_frames, left_images, K)