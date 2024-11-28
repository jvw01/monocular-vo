import os
import numpy as np
import cv2

# Setup
dataset = 2  # 0: KITTI, 1: Malaga, 2: parking

if dataset == 0:
    assert 'kitti_path' in locals(), "Variable 'kitti_path' must be defined"
    ground_truth = np.loadtxt(os.path.join(kitti_path, 'poses/05.txt'))
    ground_truth = ground_truth[:, [-9, -1]] #[end-8 end8]
    last_frame = 4540
    K = np.array([
        [718.856, 0, 607.1928],
        [0, 718.856, 185.2157],
        [0, 0, 1]
    ])
elif dataset == 1:
    assert 'malaga_path' in locals(), "Variable 'malaga_path' must be defined"
    images = sorted(os.listdir(os.path.join(malaga_path, 
                  'malaga-urban-dataset-extract-07_rectified_800x600_Images')))
    left_images = images[2::2] #(3:2:end)
    last_frame = len(left_images)
    K = np.array([
        [621.18428, 0, 404.0076],
        [0, 621.18428, 309.05989],
        [0, 0, 1]
    ])
elif dataset == 2:
    assert 'parking_path' in locals(), "Variable 'parking_path' must be defined"
    last_frame = 598
    K = np.loadtxt(os.path.join(parking_path, 'K.txt'))
    
    ground_truth = np.loadtxt(os.path.join(parking_path, 'poses.txt'))
    ground_truth = ground_truth[:, [-9, -1]] #[end-8 end8]
else:
    raise AssertionError("Invalid dataset selection")

# Bootstrap
# Need to set bootstrap_frames
bootstrap_frames = [0, 2]

if dataset == 0:
    img0 = cv2.imread(os.path.join(kitti_path, '05/image_0', 
                    f"{bootstrap_frames[0]:06d}.png"), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(os.path.join(kitti_path, '05/image_0', 
                    f"{bootstrap_frames[1]:06d}.png"), cv2.IMREAD_GRAYSCALE)
elif dataset == 1:
    img0 = cv2.imread(os.path.join(malaga_path, 
                    'malaga-urban-dataset-extract-07_rectified_800x600_Images', 
                    left_images[bootstrap_frames[0]]), cv2.IMREAD_GRAYSCALE) #or /malaga-urban
    img1 = cv2.imread(os.path.join(malaga_path, 
                    'malaga-urban-dataset-extract-07_rectified_800x600_Images', 
                    left_images[bootstrap_frames[1]]), cv2.IMREAD_GRAYSCALE)
elif dataset == 2:
    img0 = cv2.imread(os.path.join(parking_path, 
                    f"images/img_{bootstrap_frames[0]:05d}.png"), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(os.path.join(parking_path, 
                    f"images/img_{bootstrap_frames[1]:05d}.png"), cv2.IMREAD_GRAYSCALE)
else:
    raise AssertionError("Invalid dataset selection")

# Continuous operation
range_frames = range(bootstrap_frames[1] + 1, last_frame + 1)
prev_img = None

for i in range_frames:
    print(f"\n\nProcessing frame {i}\n{'=' * 21}\n")
    if dataset == 0:
        image = cv2.imread(os.path.join(kitti_path, '05/image_0', f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
    elif dataset == 1:
        image = cv2.imread(os.path.join(malaga_path, 
                        'malaga-urban-dataset-extract-07_rectified_800x600_Images', 
                        left_images[i]), cv2.IMREAD_GRAYSCALE)
    elif dataset == 2:
        image = cv2.imread(os.path.join(parking_path, 
                        f"images/img_{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
    else:
        raise AssertionError("Invalid dataset selection")
    
    # Ensures plots refresh
    cv2.waitKey(10)
    
    prev_img = image
